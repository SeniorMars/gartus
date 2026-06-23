use super::colors::Rgb;
use crate::gmath::random::SampleRng;
use crate::gmath::ray::Ray;
use crate::gmath::{
    geometry::CameraPose,
    vector::{Point, Vector},
};
use crate::graphics::raytracing::{
    EnvironmentLight, HitRecord, Hittable, INFINITY, Interval, LinearColor, MaterialPdf,
    PdfContext, ScatterRecord, component_mul, degrees_to_radians,
};
use crate::graphics::raytracing::{
    HittablePdf, MixturePdf, Pdf, SHADOW_ACNE_EPSILON, scenes::normal_scene_color,
};
#[cfg(feature = "spectral")]
use crate::graphics::raytracing::{
    PolarizationFrame, SampledWavelength, SpectralImage, SpectralTransportMode, Spectrum,
    StokesVector,
};
use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::display::{Canvas, HdrImage},
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    io::{self, Write},
    ptr,
};

const RUSSIAN_ROULETTE_MIN_SURVIVAL_PROBABILITY: f64 = 0.05;
const RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY: f64 = 0.95;
const CLIP_VERTEX_EPSILON: f64 = 1e-12;
const DEFAULT_RENDER_TILE_SIZE: u32 = 16;
const PROGRESS_RENDER_CHUNK_ROWS: u32 = 8;

/// Pixel sampling pattern used for stochastic ray-camera renders.
///
/// Use [`PixelSampleMode::StratifiedGrid`] for deterministic final renders with an exact sample
/// count. Use [`PixelSampleMode::Random`] with [`RayCamera::with_adaptive_sampling`] for preview
/// renders that may stop easy pixels before the maximum sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelSampleMode {
    /// Place each sample randomly inside the pixel square.
    #[default]
    Random,
    /// Divide the pixel into a square grid and jitter one sample inside each cell.
    ///
    /// The renderer uses `floor(sqrt(samples_per_pixel))^2` samples per pixel in this mode.
    Stratified,
    /// Divide the pixel into an explicit square grid and jitter one sample inside each cell.
    StratifiedGrid {
        /// Number of strata along each pixel axis.
        grid_width: u32,
    },
}

/// Direct-lighting strategy used when rendering with an explicit light target set.
///
/// [`Self::CurrentPathContinuation`] preserves the existing renderer behavior: diffuse and volume
/// bounces sample a mixture of the material PDF and the light-target PDF, then collect light if
/// that continuation ray reaches an emitter. [`Self::NextEventEstimation`] samples one light
/// direction immediately, casts a visibility ray through the world, then continues the path with
/// the material PDF only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DirectLightingMode {
    /// Mix light-target sampling into the ordinary path-continuation direction.
    #[default]
    CurrentPathContinuation,
    /// Add direct lighting with an explicit shadow ray at each diffuse or volume scattering event.
    NextEventEstimation,
}

/// Transport backend used by the default Canvas/HDR render entrypoints.
#[cfg(feature = "spectral")]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RenderTransportMode {
    /// Use the RGB path tracer.
    Rgb,
    /// Use sampled-wavelength spectral transport, then reconstruct linear RGB output.
    Spectral,
}

/// Path-continuation and direct-light sampling policy for ray-camera renders.
///
/// The default matches the book-style path continuation used by
/// [`DirectLightingMode::CurrentPathContinuation`]: when explicit light targets are available,
/// continuation rays sample a weighted light/material PDF mixture. Use
/// [`Self::next_event_estimation`] to move direct-light sampling out of the continuation PDF and
/// into explicit shadow rays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingStrategy {
    direct_lighting_mode: DirectLightingMode,
    light_pdf_weight: f64,
}

impl SamplingStrategy {
    /// Creates the default continuation strategy with a 50/50 light/material PDF mixture.
    #[must_use]
    pub const fn current_path_continuation() -> Self {
        Self {
            direct_lighting_mode: DirectLightingMode::CurrentPathContinuation,
            light_pdf_weight: 0.5,
        }
    }

    /// Creates a strategy that uses direct-light shadow rays and material-only continuation.
    #[must_use]
    pub const fn next_event_estimation() -> Self {
        Self {
            direct_lighting_mode: DirectLightingMode::NextEventEstimation,
            light_pdf_weight: 0.0,
        }
    }

    /// Creates the strategy corresponding to the legacy direct-lighting mode.
    #[must_use]
    pub const fn from_direct_lighting_mode(mode: DirectLightingMode) -> Self {
        match mode {
            DirectLightingMode::CurrentPathContinuation => Self::current_path_continuation(),
            DirectLightingMode::NextEventEstimation => Self::next_event_estimation(),
        }
    }

    /// Sets the continuation probability assigned to the light-target PDF.
    ///
    /// The remaining probability is assigned to the material PDF. A non-default light-target
    /// weight only has meaning for current-path continuation, so this method selects
    /// [`DirectLightingMode::CurrentPathContinuation`] even if it is called on a next-event
    /// estimation strategy.
    ///
    /// # Panics
    ///
    /// Panics if `light_pdf_weight` is not finite.
    #[must_use]
    pub fn with_light_pdf_weight(mut self, light_pdf_weight: f64) -> Self {
        assert!(
            light_pdf_weight.is_finite(),
            "light PDF weight must be finite"
        );
        self.direct_lighting_mode = DirectLightingMode::CurrentPathContinuation;
        self.light_pdf_weight = light_pdf_weight.clamp(0.0, 1.0);
        self
    }

    /// Returns the direct-lighting mode represented by this strategy.
    #[must_use]
    pub const fn direct_lighting_mode(self) -> DirectLightingMode {
        self.direct_lighting_mode
    }

    /// Returns the continuation probability assigned to the light-target PDF.
    #[must_use]
    pub const fn light_pdf_weight(self) -> f64 {
        self.light_pdf_weight
    }

    fn uses_next_event_estimation(self) -> bool {
        matches!(
            self.direct_lighting_mode,
            DirectLightingMode::NextEventEstimation
        )
    }

    fn continuation_sample(
        self,
        ray: &Ray,
        hit: &HitRecord<'_>,
        material_pdf: MaterialPdf,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
        rng: &mut SampleRng,
    ) -> ContinuationSample {
        if self.direct_lighting_mode == DirectLightingMode::CurrentPathContinuation
            && let Some(lights) = lights
        {
            let light_pdf = HittablePdf::new(lights, PdfContext::new(hit.point, ray.time()));
            let mixture_pdf = MixturePdf::weighted(light_pdf, material_pdf, self.light_pdf_weight);
            let direction = mixture_pdf.generate(rng);
            return ContinuationSample {
                direction,
                pdf_value: mixture_pdf.value(direction),
                suppress_next_emission: false,
                weight_environment_miss: false,
            };
        }

        let direction = material_pdf.generate(rng);
        ContinuationSample {
            direction,
            pdf_value: material_pdf.value(direction),
            suppress_next_emission: self.uses_next_event_estimation() && lights.is_some(),
            weight_environment_miss: self.uses_next_event_estimation() && environment.is_some(),
        }
    }
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::current_path_continuation()
    }
}

/// Source of radiance for rays that miss all scene geometry.
pub trait RayBackgroundSource: Send + Sync {
    /// Returns background radiance for a world-space ray direction.
    fn radiance(&self, direction: Vector) -> LinearColor;
}

/// Built-in background sources that can be stored directly on a [`RayCamera`].
#[derive(Clone, Copy, Debug)]
pub enum RayBackground {
    /// Constant radiance independent of direction.
    Constant(LinearColor),
    /// Vertical gradient blended by normalized ray direction `y`.
    VerticalGradient {
        /// Radiance below the horizon.
        nadir: LinearColor,
        /// Radiance at the horizon.
        horizon: LinearColor,
        /// Radiance above the camera.
        zenith: LinearColor,
    },
    /// Function-pointer background radiance.
    Function(fn(Vector) -> LinearColor),
}

impl RayBackground {
    /// Creates a constant background.
    #[must_use]
    pub const fn constant(color: LinearColor) -> Self {
        Self::Constant(color)
    }

    /// Creates a vertical gradient background.
    #[must_use]
    pub const fn vertical_gradient(
        nadir: LinearColor,
        horizon: LinearColor,
        zenith: LinearColor,
    ) -> Self {
        Self::VerticalGradient {
            nadir,
            horizon,
            zenith,
        }
    }

    /// Creates a background from a function pointer.
    #[must_use]
    pub const fn function(sample: fn(Vector) -> LinearColor) -> Self {
        Self::Function(sample)
    }

    /// Returns true when the built-in source stores only finite constant color data.
    #[must_use]
    pub fn is_finite(self) -> bool {
        match self {
            Self::Constant(color) => color.is_finite(),
            Self::VerticalGradient {
                nadir,
                horizon,
                zenith,
            } => nadir.is_finite() && horizon.is_finite() && zenith.is_finite(),
            Self::Function(_) => true,
        }
    }

    /// Returns the constant color, if this is a constant background.
    #[must_use]
    pub const fn constant_color(self) -> Option<LinearColor> {
        match self {
            Self::Constant(color) => Some(color),
            Self::VerticalGradient { .. } | Self::Function(_) => None,
        }
    }
}

impl PartialEq for RayBackground {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Constant(lhs), Self::Constant(rhs)) => lhs == rhs,
            (
                Self::VerticalGradient {
                    nadir: lhs_nadir,
                    horizon: lhs_horizon,
                    zenith: lhs_zenith,
                },
                Self::VerticalGradient {
                    nadir: rhs_nadir,
                    horizon: rhs_horizon,
                    zenith: rhs_zenith,
                },
            ) => lhs_nadir == rhs_nadir && lhs_horizon == rhs_horizon && lhs_zenith == rhs_zenith,
            (Self::Function(lhs), Self::Function(rhs)) => ptr::fn_addr_eq(lhs, rhs),
            _ => false,
        }
    }
}

impl RayBackgroundSource for RayBackground {
    fn radiance(&self, direction: Vector) -> LinearColor {
        match *self {
            Self::Constant(color) => color,
            Self::VerticalGradient {
                nadir,
                horizon,
                zenith,
            } => {
                let t = 0.5 * (direction.normalized().y() + 1.0);
                if t < 0.5 {
                    let blend = 2.0 * t;
                    nadir * (1.0 - blend) + horizon * blend
                } else {
                    let blend = 2.0 * (t - 0.5);
                    horizon * (1.0 - blend) + zenith * blend
                }
            }
            Self::Function(sample) => sample(direction),
        }
    }
}

impl RayBackgroundSource for EnvironmentLight {
    fn radiance(&self, direction: Vector) -> LinearColor {
        self.radiance(direction)
    }
}

impl<F> RayBackgroundSource for F
where
    F: Fn(Vector) -> LinearColor + Send + Sync,
{
    fn radiance(&self, direction: Vector) -> LinearColor {
        self(direction)
    }
}

#[derive(Clone, Copy)]
enum RayBackgroundContext<'a> {
    BuiltIn(RayBackground),
    Borrowed(&'a dyn RayBackgroundSource),
}

impl RayBackgroundContext<'_> {
    fn radiance(self, direction: Vector) -> LinearColor {
        match self {
            Self::BuiltIn(background) => background.radiance(direction),
            Self::Borrowed(background) => background.radiance(direction),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ContinuationSample {
    direction: Vector,
    pdf_value: f64,
    suppress_next_emission: bool,
    weight_environment_miss: bool,
}

/// Per-pixel adaptive sampling settings for random world renders.
///
/// Adaptive sampling is only applied to [`PixelSampleMode::Random`]. Stratified modes keep their
/// exact grid sample count so jittered comparisons and final renders stay deterministic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveSampling {
    /// Minimum samples to take before checking convergence.
    pub min_samples: u32,
    /// Maximum samples to take if the pixel has not converged.
    pub max_samples: u32,
    /// Stop once the largest channel standard error of the mean is below this threshold.
    pub error_threshold: f64,
}

impl AdaptiveSampling {
    /// Creates validated adaptive sampling settings.
    ///
    /// # Panics
    ///
    /// Panics if `error_threshold` is not positive and finite.
    #[must_use]
    pub fn new(min_samples: u32, max_samples: u32, error_threshold: f64) -> Self {
        assert!(
            error_threshold.is_finite() && error_threshold > 0.0,
            "adaptive sampling error threshold must be positive and finite"
        );
        let min_samples = min_samples.max(1);
        let max_samples = max_samples.max(min_samples);
        Self {
            min_samples,
            max_samples,
            error_threshold,
        }
    }
}

/// Rectangular image tile rendered as one scheduling unit by the ray camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTile {
    /// Left edge of the tile in pixel coordinates.
    pub x: u32,
    /// Top edge of the tile in pixel coordinates.
    pub y: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

impl RenderTile {
    /// Returns the exclusive right edge of the tile.
    #[must_use]
    pub const fn x_end(self) -> u32 {
        self.x.saturating_add(self.width)
    }

    /// Returns the exclusive bottom edge of the tile.
    #[must_use]
    pub const fn y_end(self) -> u32 {
        self.y.saturating_add(self.height)
    }

    fn pixel_count(self) -> usize {
        let count = u64::from(self.width) * u64::from(self.height);
        usize::try_from(count).expect("tile pixel count should fit usize")
    }
}

/// Progress counters for a tiled render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderProgress {
    /// Tile that just finished and was copied into the output image.
    pub tile: RenderTile,
    /// Number of completed tiles, including `tile`.
    pub completed_tiles: usize,
    /// Total number of tiles in the render.
    pub total_tiles: usize,
}

impl RenderProgress {
    /// Returns true once all tiles have been completed.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        self.completed_tiles >= self.total_tiles
    }
}

/// Partial image view passed to tiled progressive render callbacks.
#[derive(Debug)]
pub struct ProgressiveRenderUpdate<'a> {
    progress: RenderProgress,
    image_width: u32,
    image_height: u32,
    pixels: &'a [Rgb],
}

impl<'a> ProgressiveRenderUpdate<'a> {
    /// Returns the tile progress counters.
    #[must_use]
    pub const fn progress(&self) -> RenderProgress {
        self.progress
    }

    /// Returns the current image width in pixels.
    #[must_use]
    pub const fn image_width(&self) -> u32 {
        self.image_width
    }

    /// Returns the current image height in pixels.
    #[must_use]
    pub const fn image_height(&self) -> u32 {
        self.image_height
    }

    /// Returns the current partial pixel buffer.
    #[must_use]
    pub const fn pixels(&self) -> &'a [Rgb] {
        self.pixels
    }

    /// Copies the current partial buffer into a display canvas.
    pub fn to_canvas(&self) -> Canvas {
        RayCamera::image_canvas(self.image_width, self.image_height, self.pixels.to_vec())
    }
}

/// Beauty, albedo, and normal buffers for denoising workflows.
#[derive(Clone, Debug)]
pub struct DenoisingAovs {
    /// Gamma-encoded path-traced beauty preview.
    pub beauty: Canvas,
    /// First-hit material albedo preview encoded as raw linear RGB bytes.
    pub albedo: Canvas,
    /// First-hit shading-normal preview encoded from `[-1, 1]` to raw `[0, 1]` RGB bytes.
    pub normal: Canvas,
    /// Linear floating-point beauty samples, row-major.
    pub beauty_linear: Vec<LinearColor>,
    /// Linear floating-point first-hit albedo samples, row-major.
    pub albedo_linear: Vec<LinearColor>,
    /// Linear floating-point first-hit normal samples encoded in `[0, 1]`, row-major.
    pub normal_linear: Vec<LinearColor>,
}

impl DenoisingAovs {
    /// Creates a denoising AOV bundle.
    #[must_use]
    pub fn new(
        beauty: Canvas,
        albedo: Canvas,
        normal: Canvas,
        beauty_linear: Vec<LinearColor>,
        albedo_linear: Vec<LinearColor>,
        normal_linear: Vec<LinearColor>,
    ) -> Self {
        Self {
            beauty,
            albedo,
            normal,
            beauty_linear,
            albedo_linear,
            normal_linear,
        }
    }
}

/// A simple perspective camera for projecting 3D points onto a 2D canvas.
#[derive(Debug, Clone, Copy)]
pub struct Camera3D {
    width: u32,
    height: u32,
    camera_distance: f64,
    focal_length: f64,
    center_y_factor: f64,
    near_depth: f64,
    lookfrom: Option<Point>,
    lookat: Point,
    vup: Vector,
}

/// A projected 2D point plus its camera-space depth.
#[derive(Debug, Clone, Copy)]
pub struct ScreenPoint {
    /// Horizontal screen coordinate.
    pub x: f64,
    /// Vertical screen coordinate.
    pub y: f64,
    /// Camera-space depth used for sorting and shading.
    pub depth: f64,
}

/// A projected colored line segment.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedSegment {
    /// First projected endpoint.
    pub a: ScreenPoint,
    /// Second projected endpoint.
    pub b: ScreenPoint,
    /// Segment draw color.
    pub color: Rgb,
}

#[derive(Debug, Clone, Copy)]
struct ProjectionFrame {
    origin: Point,
    right: Vector,
    up: Vector,
    forward: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CameraSpacePoint {
    x: f64,
    y: f64,
    depth: f64,
}

impl CameraSpacePoint {
    fn interpolate(self, other: Self, depth: f64) -> Self {
        let depth_delta = other.depth - self.depth;
        if depth_delta.abs() <= f64::EPSILON {
            return Self { depth, ..self };
        }

        let t = (depth - self.depth) / depth_delta;
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            depth,
        }
    }

    fn is_in_front_of(self, near_depth: f64) -> bool {
        self.depth >= near_depth
    }

    fn is_close_to(self, other: Self) -> bool {
        (self.x - other.x).abs() <= CLIP_VERTEX_EPSILON
            && (self.y - other.y).abs() <= CLIP_VERTEX_EPSILON
            && (self.depth - other.depth).abs() <= CLIP_VERTEX_EPSILON
    }
}

/// Perspective path-tracing camera with stochastic pixel, lens, and time sampling.
///
/// `RayCamera` is the low-level camera used by [`crate::graphics::raytracing::PathTracer`]. It
/// supports fixed random sampling, stratified jittered grids, optional adaptive sampling, defocus
/// blur, motion-blur shutter intervals, and Russian roulette path termination. For expensive final
/// renders, use the crate's `render` cargo profile and keep [`Self::max_depth`] as a safety cap;
/// Russian roulette is enabled after five bounces by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCamera {
    aspect_ratio: f64,
    image_width: u32,
    image_height: u32,
    samples_per_pixel: u32,
    pixel_sample_mode: PixelSampleMode,
    adaptive_sampling: Option<AdaptiveSampling>,
    sampling_strategy: SamplingStrategy,
    #[cfg(feature = "spectral")]
    render_transport_mode: RenderTransportMode,
    #[cfg(feature = "spectral")]
    spectral_transport_mode: SpectralTransportMode,
    max_depth: u32,
    russian_roulette_min_depth: Option<u32>,
    rng_seed: u64,
    vertical_fov: f64,
    lookfrom: Point,
    lookat: Point,
    view_up: Vector,
    defocus_angle: f64,
    focus_distance: f64,
    shutter_start: f64,
    shutter_end: f64,
    background: RayBackground,
    camera_center: Point,
    pixel00_loc: Point,
    pixel_delta_u: Vector,
    pixel_delta_v: Vector,
    defocus_disk_u: Vector,
    defocus_disk_v: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RayCameraParams {
    image_width: u32,
    aspect_ratio: f64,
    samples_per_pixel: u32,
    pixel_sample_mode: PixelSampleMode,
    adaptive_sampling: Option<AdaptiveSampling>,
    sampling_strategy: SamplingStrategy,
    #[cfg(feature = "spectral")]
    render_transport_mode: RenderTransportMode,
    #[cfg(feature = "spectral")]
    spectral_transport_mode: SpectralTransportMode,
    max_depth: u32,
    russian_roulette_min_depth: Option<u32>,
    rng_seed: u64,
    vertical_fov: f64,
    lookfrom: Point,
    lookat: Point,
    view_up: Vector,
    defocus_angle: f64,
    focus_distance: f64,
    shutter_start: f64,
    shutter_end: f64,
    background: RayBackground,
}

#[derive(Clone, Copy)]
struct RayColorContext<'a> {
    world: &'a dyn Hittable,
    lights: Option<&'a dyn Hittable>,
    environment: Option<&'a EnvironmentLight>,
    sampling_strategy: SamplingStrategy,
    background: RayBackgroundContext<'a>,
    russian_roulette_min_depth: Option<u32>,
}

#[derive(Clone, Copy)]
struct PixelRenderContext<'a> {
    world: &'a dyn Hittable,
    lights: Option<&'a dyn Hittable>,
    environment: Option<&'a EnvironmentLight>,
    background: RayBackgroundContext<'a>,
}

#[cfg(feature = "spectral")]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SpectralPixel {
    linear_rgb: LinearColor,
    polarization: Option<StokesVector>,
}

impl RayColorContext<'_> {
    fn miss_radiance(&self, direction: Vector) -> LinearColor {
        self.background.radiance(direction)
    }
}

#[derive(Debug)]
struct RenderedTile {
    tile: RenderTile,
    pixels: Vec<Rgb>,
}

#[derive(Debug)]
struct RenderedValueTile<T> {
    tile: RenderTile,
    values: Vec<T>,
}

#[derive(Clone, Copy, Debug)]
enum DenoisingAovKind {
    Albedo,
    Normal,
}

impl Camera3D {
    /// Creates a camera centered in a canvas.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            camera_distance: 900.0,
            focal_length: 700.0,
            center_y_factor: 0.5,
            near_depth: 80.0,
            lookfrom: None,
            lookat: Point::new(0.0, 0.0, 0.0),
            vup: Vector::new(0.0, 1.0, 0.0),
        }
    }

    /// Returns the target canvas width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the target canvas height.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Sets the distance added to incoming z values before projection.
    ///
    /// # Panics
    ///
    /// Panics if `camera_distance` is not finite.
    #[must_use]
    pub fn with_camera_distance(mut self, camera_distance: f64) -> Self {
        assert!(
            camera_distance.is_finite(),
            "camera distance must be finite"
        );
        self.camera_distance = camera_distance;
        self
    }

    /// Sets the focal length used for perspective scaling.
    ///
    /// # Panics
    ///
    /// Panics if `focal_length` is not positive and finite.
    #[must_use]
    pub fn with_focal_length(mut self, focal_length: f64) -> Self {
        assert!(
            focal_length.is_finite() && focal_length > 0.0,
            "focal length must be positive and finite"
        );
        self.focal_length = focal_length;
        self
    }

    /// Sets the focal length from a vertical field-of-view angle in degrees.
    ///
    /// # Panics
    ///
    /// Panics if `vertical_fov` is not finite or is outside `0..180` degrees.
    #[must_use]
    pub fn with_vertical_fov(mut self, vertical_fov: f64) -> Self {
        assert!(
            vertical_fov.is_finite() && 0.0 < vertical_fov && vertical_fov < 180.0,
            "vertical field of view must be finite and in 0..180 degrees"
        );
        let theta = vertical_fov.to_radians();
        self.focal_length = f64::from(self.height) * 0.5 / (theta * 0.5).tan();
        self
    }

    /// Sets the vertical screen center as a fraction of canvas height.
    ///
    /// # Panics
    ///
    /// Panics if `center_y_factor` is not finite.
    #[must_use]
    pub fn with_center_y_factor(mut self, center_y_factor: f64) -> Self {
        assert!(
            center_y_factor.is_finite(),
            "center-y factor must be finite"
        );
        self.center_y_factor = center_y_factor;
        self
    }

    /// Sets the minimum projected depth.
    ///
    /// # Panics
    ///
    /// Panics if `near_depth` is not positive and finite.
    #[must_use]
    pub fn with_near_depth(mut self, near_depth: f64) -> Self {
        assert!(
            near_depth.is_finite() && near_depth > 0.0,
            "near depth must be positive and finite"
        );
        self.near_depth = near_depth;
        self
    }

    /// Positions the projection camera at `lookfrom`, aimed at `lookat`.
    ///
    /// The default camera is equivalent to looking from `(0, 0, -camera_distance)`
    /// toward the origin, preserving the historical projection behavior.
    ///
    /// # Panics
    ///
    /// Panics if `lookfrom` and `lookat` are the same point.
    #[must_use]
    pub fn with_look_at(mut self, lookfrom: Point, lookat: Point) -> Self {
        assert!(
            (lookat - lookfrom).length_squared() > f64::EPSILON,
            "lookfrom and lookat must be distinct"
        );
        self.lookfrom = Some(lookfrom);
        self.lookat = lookat;
        self
    }

    /// Sets the camera-relative up direction.
    ///
    /// # Panics
    ///
    /// Panics if `vup` is zero.
    #[must_use]
    pub fn with_view_up(mut self, vup: Vector) -> Self {
        assert!(
            vup.length_squared() > f64::EPSILON,
            "view-up vector must be nonzero"
        );
        self.vup = vup;
        self
    }

    fn effective_lookfrom(&self) -> Point {
        self.lookfrom
            .unwrap_or_else(|| Point::new(0.0, 0.0, -self.camera_distance))
    }

    fn camera_frame(&self) -> Option<ProjectionFrame> {
        let lookfrom = self.effective_lookfrom();
        let frame = CameraPose::new(lookfrom, self.lookat, self.vup).frame()?;
        Some(ProjectionFrame {
            origin: frame.origin,
            right: -frame.right,
            up: frame.up,
            forward: frame.forward,
        })
    }

    fn camera_space_point(point: &[f64], frame: ProjectionFrame) -> Option<CameraSpacePoint> {
        if point.len() < 3 {
            return None;
        }

        let point = Point::new(point[0], point[1], point[2]);
        let camera_relative = point - frame.origin;
        Some(CameraSpacePoint {
            x: camera_relative.dot(frame.right),
            y: camera_relative.dot(frame.up),
            depth: camera_relative.dot(frame.forward),
        })
    }

    fn project_camera_space_point(&self, point: CameraSpacePoint) -> ScreenPoint {
        let scale = self.focal_length / point.depth;
        ScreenPoint {
            x: f64::from(self.width) * 0.5 + point.x * scale,
            y: f64::from(self.height) * self.center_y_factor - point.y * scale,
            depth: point.depth,
        }
    }

    /// Projects a homogeneous point into 2D screen coordinates.
    #[must_use]
    pub fn project(&self, point: &[f64]) -> Option<ScreenPoint> {
        let frame = self.camera_frame()?;
        let point = Self::camera_space_point(point, frame)?;
        if !point.is_in_front_of(self.near_depth) {
            return None;
        }
        Some(self.project_camera_space_point(point))
    }

    /// Projects a triangle after clipping it against the near plane.
    pub(crate) fn project_clipped_triangle(&self, points: [&[f64]; 3]) -> Vec<[ScreenPoint; 3]> {
        let Some(frame) = self.camera_frame() else {
            return Vec::new();
        };
        let Some(p0) = Self::camera_space_point(points[0], frame) else {
            return Vec::new();
        };
        let Some(p1) = Self::camera_space_point(points[1], frame) else {
            return Vec::new();
        };
        let Some(p2) = Self::camera_space_point(points[2], frame) else {
            return Vec::new();
        };

        let clipped = clip_camera_triangle_to_near([p0, p1, p2], self.near_depth);
        if clipped.len() < 3 {
            return Vec::new();
        }

        let mut triangles = Vec::with_capacity(clipped.len() - 2);
        for vertex in 1..clipped.len() - 1 {
            triangles.push([
                self.project_camera_space_point(clipped[0]),
                self.project_camera_space_point(clipped[vertex]),
                self.project_camera_space_point(clipped[vertex + 1]),
            ]);
        }
        triangles
    }

    /// Projects transformed mesh triangle edges into colored wireframe segments.
    ///
    /// `color_for_triangle` receives the triangle index and average projected triangle depth.
    pub fn project_mesh_wireframe_segments<F>(
        &self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        stride: usize,
        mut color_for_triangle: F,
    ) -> Vec<ProjectedSegment>
    where
        F: FnMut(usize, f64) -> Rgb,
    {
        let stride = stride.max(1);
        let mut segments = Vec::new();
        for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
            if idx % stride != 0 {
                continue;
            }
            let Some(a) = self.project(&p0) else {
                continue;
            };
            let Some(b) = self.project(&p1) else {
                continue;
            };
            let Some(c) = self.project(&p2) else {
                continue;
            };
            let depth = (a.depth + b.depth + c.depth) / 3.0;
            let color = color_for_triangle(idx, depth);
            segments.push(ProjectedSegment { a, b, color });
            segments.push(ProjectedSegment { a: b, b: c, color });
            segments.push(ProjectedSegment { a: c, b: a, color });
        }
        segments
    }
}

fn clip_camera_triangle_to_near(
    vertices: [CameraSpacePoint; 3],
    near_depth: f64,
) -> Vec<CameraSpacePoint> {
    let mut clipped = Vec::with_capacity(4);
    let mut previous = vertices[2];
    let mut previous_inside = previous.is_in_front_of(near_depth);

    for current in vertices {
        let current_inside = current.is_in_front_of(near_depth);
        if current_inside != previous_inside {
            push_clipped_vertex(&mut clipped, previous.interpolate(current, near_depth));
        }
        if current_inside {
            push_clipped_vertex(&mut clipped, current);
        }

        previous = current;
        previous_inside = current_inside;
    }

    if clipped
        .last()
        .is_some_and(|last| clipped[0].is_close_to(*last))
    {
        clipped.pop();
    }

    clipped
}

fn push_clipped_vertex(vertices: &mut Vec<CameraSpacePoint>, vertex: CameraSpacePoint) {
    if vertices.last().is_some_and(|last| last.is_close_to(vertex)) {
        return;
    }
    vertices.push(vertex);
}

impl Default for RayCamera {
    fn default() -> Self {
        Self::new(100, 1.0)
    }
}

impl RayCamera {
    /// Creates a camera with the requested image width and ideal aspect ratio.
    ///
    /// The image height is rounded down from `image_width / aspect_ratio`, with a
    /// minimum height of one pixel. The viewport is sized from the actual integer
    /// image dimensions so pixel spacing remains square.
    ///
    /// # Panics
    ///
    /// Panics if `image_width` is zero or `aspect_ratio` is not positive and finite.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(image_width: u32, aspect_ratio: f64) -> Self {
        assert!(image_width > 0, "image width must be positive");
        assert!(
            aspect_ratio.is_finite() && aspect_ratio > 0.0,
            "aspect ratio must be positive and finite"
        );

        Self::initialized(&RayCameraParams {
            image_width,
            aspect_ratio,
            samples_per_pixel: 1,
            pixel_sample_mode: PixelSampleMode::Random,
            adaptive_sampling: None,
            sampling_strategy: SamplingStrategy::current_path_continuation(),
            #[cfg(feature = "spectral")]
            render_transport_mode: RenderTransportMode::Rgb,
            #[cfg(feature = "spectral")]
            spectral_transport_mode: SpectralTransportMode::Polarized,
            max_depth: 10,
            russian_roulette_min_depth: Some(5),
            rng_seed: 1,
            vertical_fov: 90.0,
            lookfrom: Point::new(0.0, 0.0, 0.0),
            lookat: Point::new(0.0, 0.0, -1.0),
            view_up: Vector::new(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_distance: 1.0,
            shutter_start: 0.0,
            shutter_end: 1.0,
            background: RayBackground::constant(LinearColor::new(0.70, 0.80, 1.00)),
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn initialized(params: &RayCameraParams) -> Self {
        Self::validate_view(params);

        let image_height = ((f64::from(params.image_width) / params.aspect_ratio) as u32).max(1);
        let theta = degrees_to_radians(params.vertical_fov);
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h * params.focus_distance;
        let viewport_width =
            viewport_height * (f64::from(params.image_width) / f64::from(image_height));
        let camera_center = params.lookfrom;

        let frame = CameraPose::new(params.lookfrom, params.lookat, params.view_up)
            .frame()
            .expect("validated camera basis");
        let w = frame.backward();
        let u = frame.right;
        let v = frame.up;

        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;
        let pixel_delta_u = viewport_u / f64::from(params.image_width);
        let pixel_delta_v = viewport_v / f64::from(image_height);

        let viewport_upper_left =
            camera_center - params.focus_distance * w - viewport_u / 2.0 - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        let defocus_radius =
            params.focus_distance * degrees_to_radians(params.defocus_angle * 0.5).tan();
        let defocus_disk_u = defocus_radius * u;
        let defocus_disk_v = defocus_radius * v;

        Self {
            aspect_ratio: params.aspect_ratio,
            image_width: params.image_width,
            image_height,
            samples_per_pixel: params.samples_per_pixel.max(1),
            pixel_sample_mode: params.pixel_sample_mode,
            adaptive_sampling: params.adaptive_sampling,
            sampling_strategy: params.sampling_strategy,
            #[cfg(feature = "spectral")]
            render_transport_mode: params.render_transport_mode,
            #[cfg(feature = "spectral")]
            spectral_transport_mode: params.spectral_transport_mode,
            max_depth: params.max_depth,
            russian_roulette_min_depth: params.russian_roulette_min_depth,
            rng_seed: params.rng_seed,
            vertical_fov: params.vertical_fov,
            lookfrom: params.lookfrom,
            lookat: params.lookat,
            view_up: params.view_up,
            defocus_angle: params.defocus_angle,
            focus_distance: params.focus_distance,
            shutter_start: params.shutter_start,
            shutter_end: params.shutter_end,
            background: params.background,
            camera_center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
            defocus_disk_u,
            defocus_disk_v,
        }
    }

    /// Returns a copy of the camera initialized from its current public render parameters.
    #[must_use]
    fn initialize(self) -> Self {
        Self::initialized(&RayCameraParams {
            image_width: self.image_width,
            aspect_ratio: self.aspect_ratio,
            samples_per_pixel: self.samples_per_pixel,
            pixel_sample_mode: self.pixel_sample_mode,
            adaptive_sampling: self.adaptive_sampling,
            sampling_strategy: self.sampling_strategy,
            #[cfg(feature = "spectral")]
            render_transport_mode: self.render_transport_mode,
            #[cfg(feature = "spectral")]
            spectral_transport_mode: self.spectral_transport_mode,
            max_depth: self.max_depth,
            russian_roulette_min_depth: self.russian_roulette_min_depth,
            rng_seed: self.rng_seed,
            vertical_fov: self.vertical_fov,
            lookfrom: self.lookfrom,
            lookat: self.lookat,
            view_up: self.view_up,
            defocus_angle: self.defocus_angle,
            focus_distance: self.focus_distance,
            shutter_start: self.shutter_start,
            shutter_end: self.shutter_end,
            background: self.background,
        })
    }

    fn validate_view(params: &RayCameraParams) {
        assert!(
            params.vertical_fov.is_finite()
                && 0.0 < params.vertical_fov
                && params.vertical_fov < 180.0,
            "vertical field of view must be finite and in 0..180 degrees"
        );
        let w = params.lookfrom - params.lookat;
        assert!(
            w.length_squared() > f64::EPSILON,
            "lookfrom and lookat must be distinct"
        );
        assert!(
            params.view_up.length_squared() > f64::EPSILON,
            "view-up vector must be nonzero"
        );
        assert!(
            CameraPose::new(params.lookfrom, params.lookat, params.view_up)
                .frame()
                .is_some(),
            "view-up vector must not be parallel to the viewing direction"
        );
        assert!(
            params.defocus_angle.is_finite() && (0.0..180.0).contains(&params.defocus_angle),
            "defocus angle must be finite and in 0..180 degrees"
        );
        assert!(
            params.focus_distance.is_finite() && params.focus_distance > 0.0,
            "focus distance must be positive and finite"
        );
        assert!(
            params.shutter_start.is_finite()
                && params.shutter_end.is_finite()
                && params.shutter_start <= params.shutter_end,
            "shutter interval must be finite and ordered"
        );
        assert!(
            params.background.is_finite(),
            "background color components must be finite"
        );
    }

    /// Sets the target image width and recomputes derived camera values.
    ///
    /// # Panics
    ///
    /// Panics if `image_width` is zero.
    #[must_use]
    pub fn with_image_width(mut self, image_width: u32) -> Self {
        assert!(image_width > 0, "image width must be positive");
        self.image_width = image_width;
        self.initialize()
    }

    /// Sets the target aspect ratio and recomputes derived camera values.
    ///
    /// # Panics
    ///
    /// Panics if `aspect_ratio` is not positive and finite.
    #[must_use]
    pub fn with_aspect_ratio(mut self, aspect_ratio: f64) -> Self {
        assert!(
            aspect_ratio.is_finite() && aspect_ratio > 0.0,
            "aspect ratio must be positive and finite"
        );
        self.aspect_ratio = aspect_ratio;
        self.initialize()
    }

    /// Sets the random samples taken per pixel for world rendering.
    #[must_use]
    pub fn with_samples_per_pixel(mut self, samples_per_pixel: u32) -> Self {
        self.samples_per_pixel = samples_per_pixel.max(1);
        self.initialize()
    }

    /// Sets the pixel sampling pattern for stochastic world rendering.
    #[must_use]
    pub fn with_pixel_sample_mode(mut self, mode: PixelSampleMode) -> Self {
        self.pixel_sample_mode = mode;
        self.initialize()
    }

    /// Enables adaptive per-pixel sampling for random stochastic world renders.
    ///
    /// The camera takes at least `min_samples`, up to `max_samples`, and stops early when the
    /// largest channel standard error of the mean drops below `error_threshold`. Adaptive sampling
    /// is ignored for stratified modes because those modes promise an exact jittered grid.
    ///
    /// # Panics
    ///
    /// Panics if `error_threshold` is not positive and finite.
    #[must_use]
    pub fn with_adaptive_sampling(
        mut self,
        min_samples: u32,
        max_samples: u32,
        error_threshold: f64,
    ) -> Self {
        let adaptive_sampling = AdaptiveSampling::new(min_samples, max_samples, error_threshold);
        self.adaptive_sampling = Some(adaptive_sampling);
        self.samples_per_pixel = adaptive_sampling.max_samples;
        self
    }

    /// Disables adaptive per-pixel sampling.
    #[must_use]
    pub fn without_adaptive_sampling(mut self) -> Self {
        self.adaptive_sampling = None;
        self
    }

    /// Sets the direct-lighting strategy used by [`Self::render_world_with_lights`].
    #[must_use]
    pub fn with_direct_lighting_mode(mut self, mode: DirectLightingMode) -> Self {
        self.sampling_strategy = SamplingStrategy::from_direct_lighting_mode(mode);
        self
    }

    /// Sets the path sampling strategy used by world renders.
    #[must_use]
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Sets the transport backend used by default Canvas/HDR render entrypoints.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn with_render_transport_mode(mut self, mode: RenderTransportMode) -> Self {
        self.render_transport_mode = mode;
        self
    }

    /// Uses sampled-wavelength spectral transport for default Canvas/HDR render entrypoints.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn with_spectral_render_transport(self) -> Self {
        self.with_render_transport_mode(RenderTransportMode::Spectral)
    }

    /// Sets the sampled-wavelength transport mode.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn with_spectral_transport_mode(mut self, mode: SpectralTransportMode) -> Self {
        self.spectral_transport_mode = mode;
        self
    }

    /// Enables scalar sampled-wavelength transport without Stokes/Mueller polarization state.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn with_unpolarized_spectral_transport(self) -> Self {
        self.with_spectral_transport_mode(SpectralTransportMode::Unpolarized)
    }

    /// Enables Stokes/Mueller polarized sampled-wavelength transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn with_polarized_spectral_transport(self) -> Self {
        self.with_spectral_transport_mode(SpectralTransportMode::Polarized)
    }

    /// Enables stratified jittered samples inside each rendered pixel.
    ///
    /// This follows the sampling pattern from *Ray Tracing: The Rest of Your Life* and uses
    /// `floor(sqrt(samples_per_pixel))^2` samples per pixel.
    #[must_use]
    pub fn with_stratified_sampling(self) -> Self {
        self.with_pixel_sample_mode(PixelSampleMode::Stratified)
    }

    /// Enables stratified sampling with an explicit square grid width.
    ///
    /// A `grid_width` of 32 gives exactly 1024 samples per pixel, regardless of the value last
    /// passed to [`Self::with_samples_per_pixel`].
    #[must_use]
    pub fn with_stratified_grid_width(mut self, grid_width: u32) -> Self {
        let grid_width = grid_width.max(1);
        self.samples_per_pixel = grid_width.saturating_mul(grid_width);
        self.with_pixel_sample_mode(PixelSampleMode::StratifiedGrid { grid_width })
    }

    /// Sets the maximum ray-bounce recursion depth for diffuse world rendering.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Starts Russian roulette path termination after `min_depth` completed bounces.
    ///
    /// Surviving paths are scaled by the inverse survival probability, keeping the estimator
    /// unbiased while avoiding low-energy long tails. Use [`Self::without_russian_roulette`] for
    /// exact fixed-depth debugging renders.
    #[must_use]
    pub fn with_russian_roulette_min_depth(mut self, min_depth: u32) -> Self {
        self.russian_roulette_min_depth = Some(min_depth);
        self
    }

    /// Disables Russian roulette path termination.
    #[must_use]
    pub fn without_russian_roulette(mut self) -> Self {
        self.russian_roulette_min_depth = None;
        self
    }

    /// Sets the deterministic random seed used for antialiasing samples.
    #[must_use]
    pub fn with_rng_seed(mut self, rng_seed: u64) -> Self {
        self.rng_seed = rng_seed;
        self
    }

    /// Sets the vertical field of view in degrees and recomputes derived camera values.
    ///
    /// # Panics
    ///
    /// Panics if `vertical_fov` is not finite or is outside `0..180` degrees.
    #[must_use]
    pub fn with_vertical_fov(mut self, vertical_fov: f64) -> Self {
        self.vertical_fov = vertical_fov;
        self.initialize()
    }

    /// Positions the camera at `lookfrom`, aimed at `lookat`.
    ///
    /// # Panics
    ///
    /// Panics if `lookfrom` and `lookat` are the same point, or if the current view-up vector is
    /// parallel to the new viewing direction.
    #[must_use]
    pub fn with_look_at(mut self, lookfrom: Point, lookat: Point) -> Self {
        self.lookfrom = lookfrom;
        self.lookat = lookat;
        self.initialize()
    }

    /// Sets the camera-relative up direction.
    ///
    /// # Panics
    ///
    /// Panics if `view_up` is zero or parallel to the current viewing direction.
    #[must_use]
    pub fn with_view_up(mut self, view_up: Vector) -> Self {
        self.view_up = view_up;
        self.initialize()
    }

    /// Sets the variation angle of rays through each pixel for defocus blur.
    ///
    /// A zero angle keeps the camera as a pinhole camera.
    ///
    /// # Panics
    ///
    /// Panics if `defocus_angle` is not finite or is outside `0..180` degrees.
    #[must_use]
    pub fn with_defocus_angle(mut self, defocus_angle: f64) -> Self {
        self.defocus_angle = defocus_angle;
        self.initialize()
    }

    /// Sets the distance from the camera origin to the plane of perfect focus.
    ///
    /// # Panics
    ///
    /// Panics if `focus_distance` is not positive and finite.
    #[must_use]
    pub fn with_focus_distance(mut self, focus_distance: f64) -> Self {
        self.focus_distance = focus_distance;
        self.initialize()
    }

    /// Sets the camera shutter interval used for sampled rays.
    ///
    /// # Panics
    ///
    /// Panics if either endpoint is non-finite, or if `start > end`.
    #[must_use]
    pub fn with_shutter_interval(mut self, start: f64, end: f64) -> Self {
        assert!(
            start.is_finite() && end.is_finite() && start <= end,
            "shutter interval must be finite and ordered"
        );
        self.shutter_start = start;
        self.shutter_end = end;
        self.initialize()
    }

    /// Sets the color returned by world rendering when a ray misses all scene objects.
    ///
    /// # Panics
    ///
    /// Panics if any color component is non-finite.
    #[must_use]
    pub fn with_background(mut self, background: LinearColor) -> Self {
        self.background = RayBackground::constant(background);
        self.with_background_source(self.background)
    }

    /// Sets the built-in source used when rendered rays miss all scene objects.
    ///
    /// # Panics
    ///
    /// Panics if stored background colors are not finite.
    #[must_use]
    pub fn with_background_source(mut self, background: RayBackground) -> Self {
        assert!(
            background.is_finite(),
            "background color components must be finite"
        );
        self.background = background;
        self
    }

    /// Sets a function-pointer background source.
    #[must_use]
    pub fn with_background_fn(self, background: fn(Vector) -> LinearColor) -> Self {
        self.with_background_source(RayBackground::function(background))
    }

    /// Returns the camera's ideal aspect ratio.
    #[must_use]
    pub fn aspect_ratio(self) -> f64 {
        self.aspect_ratio
    }

    /// Returns the rendered image width in pixels.
    #[must_use]
    pub fn image_width(self) -> u32 {
        self.image_width
    }

    /// Returns the rendered image height in pixels.
    #[must_use]
    pub fn image_height(self) -> u32 {
        self.image_height
    }

    /// Returns the number of random samples per pixel used by [`Self::render_world`].
    #[must_use]
    pub fn samples_per_pixel(self) -> u32 {
        self.samples_per_pixel
    }

    /// Returns the pixel sampling pattern used by stochastic world renders.
    #[must_use]
    pub const fn pixel_sample_mode(self) -> PixelSampleMode {
        self.pixel_sample_mode
    }

    /// Returns adaptive sampling settings, if enabled.
    #[must_use]
    pub const fn adaptive_sampling(self) -> Option<AdaptiveSampling> {
        self.adaptive_sampling
    }

    /// Returns the direct-lighting strategy used for renders with explicit light targets.
    #[must_use]
    pub const fn direct_lighting_mode(self) -> DirectLightingMode {
        self.sampling_strategy.direct_lighting_mode()
    }

    /// Returns the path sampling strategy used by world renders.
    #[must_use]
    pub const fn sampling_strategy(self) -> SamplingStrategy {
        self.sampling_strategy
    }

    /// Returns the transport backend used by default Canvas/HDR render entrypoints.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn render_transport_mode(self) -> RenderTransportMode {
        self.render_transport_mode
    }

    /// Returns the sampled-wavelength transport mode.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub const fn spectral_transport_mode(self) -> SpectralTransportMode {
        self.spectral_transport_mode
    }

    /// Returns the actual number of samples used per pixel for the current sampling mode.
    #[must_use]
    pub fn effective_samples_per_pixel(self) -> u32 {
        match self.pixel_sample_mode {
            PixelSampleMode::Random => self.samples_per_pixel,
            PixelSampleMode::Stratified => {
                let sqrt_spp = Self::stratified_grid_width(self.samples_per_pixel);
                sqrt_spp * sqrt_spp
            }
            PixelSampleMode::StratifiedGrid { grid_width } => grid_width.saturating_mul(grid_width),
        }
    }

    /// Returns the maximum ray-bounce recursion depth used by [`Self::render_world`].
    #[must_use]
    pub fn max_depth(self) -> u32 {
        self.max_depth
    }

    /// Returns the bounce count after which Russian roulette termination starts.
    #[must_use]
    pub const fn russian_roulette_min_depth(self) -> Option<u32> {
        self.russian_roulette_min_depth
    }

    /// Returns the vertical field of view in degrees.
    #[must_use]
    pub fn vertical_fov(self) -> f64 {
        self.vertical_fov
    }

    /// Returns the defocus cone angle in degrees.
    #[must_use]
    pub fn defocus_angle(self) -> f64 {
        self.defocus_angle
    }

    /// Returns the distance from the camera origin to the plane of perfect focus.
    #[must_use]
    pub fn focus_distance(self) -> f64 {
        self.focus_distance
    }

    /// Returns the camera shutter interval used for sampled rays.
    #[must_use]
    pub fn shutter_interval(self) -> (f64, f64) {
        (self.shutter_start, self.shutter_end)
    }

    /// Returns the radiance used by the camera background along the center view direction.
    ///
    /// For constant backgrounds this is the configured color. For gradient or function
    /// backgrounds, this is a directional sample; use [`Self::constant_background`] when callers
    /// need to distinguish a configured constant color from a directional source.
    #[must_use]
    pub fn background(self) -> LinearColor {
        self.background_source()
            .radiance(self.lookat - self.lookfrom)
    }

    /// Returns the configured constant background color, if the built-in background is constant.
    #[must_use]
    pub const fn constant_background(self) -> Option<LinearColor> {
        self.background.constant_color()
    }

    /// Returns the built-in background source used when rays miss the scene.
    #[must_use]
    pub const fn background_source(self) -> RayBackground {
        self.background
    }

    /// Returns the camera origin point.
    #[must_use]
    pub fn camera_center(self) -> Point {
        self.camera_center
    }

    /// Returns the point this camera is aimed at.
    #[must_use]
    pub fn lookat(self) -> Point {
        self.lookat
    }

    /// Returns the camera-relative up direction.
    #[must_use]
    pub fn view_up(self) -> Vector {
        self.view_up
    }

    /// Returns a ray from the camera center through the center of pixel `(x, y)`.
    ///
    /// Pixel coordinates are in storage order: `(0, 0)` is the upper-left pixel,
    /// rows scan left to right, and rows advance downward.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is outside the camera image dimensions.
    #[must_use]
    pub fn ray_for_pixel(self, x: u32, y: u32) -> Ray {
        assert!(x < self.image_width, "pixel x must be inside the image");
        assert!(y < self.image_height, "pixel y must be inside the image");

        let pixel_center = self.pixel00_loc
            + f64::from(x) * self.pixel_delta_u
            + f64::from(y) * self.pixel_delta_v;
        Ray::with_time(
            self.camera_center,
            pixel_center - self.camera_center,
            self.shutter_start,
        )
    }

    fn ray_for_pixel_sample(self, x: u32, y: u32, rng: &mut SampleRng) -> Ray {
        let offset = Self::sample_square(rng);
        let pixel_sample = self.pixel00_loc
            + (f64::from(x) + offset.x()) * self.pixel_delta_u
            + (f64::from(y) + offset.y()) * self.pixel_delta_v;
        let ray_origin = if self.defocus_angle <= 0.0 {
            self.camera_center
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_time = rng.random_range(self.shutter_start, self.shutter_end);
        Ray::with_time(ray_origin, pixel_sample - ray_origin, ray_time)
    }

    fn ray_for_pixel_stratified_sample(
        self,
        x: u32,
        y: u32,
        sample_x: u32,
        sample_y: u32,
        grid_width: u32,
        rng: &mut SampleRng,
    ) -> Ray {
        let offset = Self::sample_square_stratified(sample_x, sample_y, grid_width, rng);
        let pixel_sample = self.pixel00_loc
            + (f64::from(x) + offset.x()) * self.pixel_delta_u
            + (f64::from(y) + offset.y()) * self.pixel_delta_v;
        let ray_origin = if self.defocus_angle <= 0.0 {
            self.camera_center
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_time = rng.random_range(self.shutter_start, self.shutter_end);
        Ray::with_time(ray_origin, pixel_sample - ray_origin, ray_time)
    }

    fn sample_square(rng: &mut SampleRng) -> Vector {
        Vector::new(rng.random_double() - 0.5, rng.random_double() - 0.5, 0.0)
    }

    fn sample_square_stratified(
        sample_x: u32,
        sample_y: u32,
        grid_width: u32,
        rng: &mut SampleRng,
    ) -> Vector {
        let reciprocal = 1.0 / f64::from(grid_width);
        let x = ((f64::from(sample_x) + rng.random_double()) * reciprocal) - 0.5;
        let y = ((f64::from(sample_y) + rng.random_double()) * reciprocal) - 0.5;
        Vector::new(x, y, 0.0)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn stratified_grid_width(samples_per_pixel: u32) -> u32 {
        (f64::from(samples_per_pixel).sqrt() as u32).max(1)
    }

    fn defocus_disk_sample(self, rng: &mut SampleRng) -> Point {
        let point = rng.random_in_unit_disk();
        self.camera_center + point.x() * self.defocus_disk_u + point.y() * self.defocus_disk_v
    }

    fn pixel_seed(seed: u64, x: u32, y: u32) -> u64 {
        let mut z = seed
            ^ u64::from(x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(y).wrapping_mul(0xD1B5_4A32_D192_ED03);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::too_many_lines)]
    fn ray_color(
        ray: &Ray,
        depth: u32,
        context: RayColorContext<'_>,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let mut current_ray = *ray;
        let mut attenuation = LinearColor::new(1.0, 1.0, 1.0);
        let mut color = LinearColor::default();
        let mut allow_emitted = true;
        let mut miss_radiance_weight = 1.0;

        for bounce_index in 0..depth {
            let Some(record) = context.world.hit_with_rng(
                &current_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            ) else {
                let miss = context.miss_radiance(*current_ray.direction()) * miss_radiance_weight;
                return color + component_mul(attenuation, miss);
            };

            if allow_emitted {
                color += component_mul(attenuation, Self::emitted_at(&current_ray, &record));
            }

            let Some(scatter) = record.material.scatter(&current_ray, &record, rng) else {
                return color;
            };

            match scatter {
                ScatterRecord::Specular {
                    ray,
                    attenuation: scatter_attenuation,
                } => {
                    attenuation = component_mul(attenuation, scatter_attenuation);
                    current_ray = ray;
                    allow_emitted = true;
                    miss_radiance_weight = 1.0;
                    if !Self::russian_roulette_survives(
                        bounce_index,
                        context.russian_roulette_min_depth,
                        &mut attenuation,
                        rng,
                    ) {
                        return color;
                    }
                }
                ScatterRecord::Scattering {
                    attenuation: scatter_attenuation,
                    pdf: material_pdf,
                } => {
                    if context.sampling_strategy.uses_next_event_estimation() {
                        if let Some(lights) = context.lights {
                            let direct = Self::estimate_direct_lighting(
                                &current_ray,
                                &record,
                                context.world,
                                lights,
                                scatter_attenuation,
                                rng,
                            );
                            color += component_mul(attenuation, direct);
                        }
                        if let Some(environment) = context.environment {
                            let direct = Self::estimate_environment_lighting(
                                &current_ray,
                                &record,
                                context.world,
                                environment,
                                scatter_attenuation,
                                rng,
                            );
                            color += component_mul(attenuation, direct);
                        }
                    }

                    let continuation = context.sampling_strategy.continuation_sample(
                        &current_ray,
                        &record,
                        material_pdf,
                        context.lights,
                        context.environment,
                        rng,
                    );

                    if !continuation.pdf_value.is_finite() || continuation.pdf_value <= f64::EPSILON
                    {
                        return color;
                    }

                    let scattered_ray =
                        Ray::with_time(record.point, continuation.direction, current_ray.time());
                    let scattering_pdf =
                        record
                            .material
                            .scattering_pdf(&current_ray, &record, &scattered_ray);
                    if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
                        return color;
                    }

                    let next_miss_radiance_weight = if continuation.weight_environment_miss {
                        context.environment.map_or(1.0, |environment| {
                            Self::power_heuristic(
                                continuation.pdf_value,
                                environment.pdf_value(continuation.direction),
                            )
                        })
                    } else {
                        1.0
                    };

                    attenuation = component_mul(
                        attenuation,
                        scatter_attenuation * (scattering_pdf / continuation.pdf_value),
                    );
                    current_ray = scattered_ray;
                    allow_emitted = !continuation.suppress_next_emission;
                    miss_radiance_weight = next_miss_radiance_weight;
                    if !Self::russian_roulette_survives(
                        bounce_index,
                        context.russian_roulette_min_depth,
                        &mut attenuation,
                        rng,
                    ) {
                        return color;
                    }
                }
            }
        }

        color
    }

    #[cfg(feature = "spectral")]
    #[allow(clippy::too_many_lines)]
    fn ray_sampled_wavelength_radiance(
        ray: &Ray,
        depth: u32,
        context: RayColorContext<'_>,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> f64 {
        let mut current_ray = *ray;
        let mut attenuation = 1.0;
        let mut radiance = 0.0;
        let mut allow_emitted = true;
        let mut miss_radiance_weight = 1.0;

        for bounce_index in 0..depth {
            let Some(record) = context.world.hit_with_rng(
                &current_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            ) else {
                let miss = Self::sample_spectrum(
                    context.miss_radiance(*current_ray.direction()),
                    wavelength,
                ) * miss_radiance_weight;
                return radiance + attenuation * miss;
            };

            if allow_emitted {
                radiance +=
                    attenuation * Self::spectral_emitted_at(&current_ray, &record, wavelength);
            }

            let Some(scatter) =
                record
                    .material
                    .scatter_spectral(&current_ray, &record, wavelength, rng)
            else {
                return radiance;
            };

            match scatter {
                ScatterRecord::Specular {
                    ray,
                    attenuation: scatter_attenuation,
                } => {
                    attenuation *=
                        Self::sample_material_spectrum(&record, scatter_attenuation, wavelength);
                    current_ray = ray;
                    allow_emitted = true;
                    miss_radiance_weight = 1.0;
                    if !Self::russian_roulette_survives_scalar(
                        bounce_index,
                        context.russian_roulette_min_depth,
                        &mut attenuation,
                        rng,
                    ) {
                        return radiance;
                    }
                }
                ScatterRecord::Scattering {
                    attenuation: scatter_attenuation,
                    pdf: material_pdf,
                } => {
                    if context.sampling_strategy.uses_next_event_estimation() {
                        if let Some(lights) = context.lights {
                            radiance += attenuation
                                * Self::estimate_direct_lighting_spectral(
                                    &current_ray,
                                    &record,
                                    context.world,
                                    lights,
                                    scatter_attenuation,
                                    wavelength,
                                    rng,
                                );
                        }
                        if let Some(environment) = context.environment {
                            radiance += attenuation
                                * Self::estimate_environment_lighting_spectral(
                                    &current_ray,
                                    &record,
                                    context.world,
                                    environment,
                                    scatter_attenuation,
                                    wavelength,
                                    rng,
                                );
                        }
                    }

                    let continuation = context.sampling_strategy.continuation_sample(
                        &current_ray,
                        &record,
                        material_pdf,
                        context.lights,
                        context.environment,
                        rng,
                    );

                    if !continuation.pdf_value.is_finite() || continuation.pdf_value <= f64::EPSILON
                    {
                        return radiance;
                    }

                    let scattered_ray =
                        Ray::with_time(record.point, continuation.direction, current_ray.time());
                    let scattering_pdf =
                        record
                            .material
                            .scattering_pdf(&current_ray, &record, &scattered_ray);
                    if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
                        return radiance;
                    }

                    let next_miss_radiance_weight = if continuation.weight_environment_miss {
                        context.environment.map_or(1.0, |environment| {
                            Self::power_heuristic(
                                continuation.pdf_value,
                                environment.pdf_value(continuation.direction),
                            )
                        })
                    } else {
                        1.0
                    };

                    attenuation *=
                        Self::sample_material_spectrum(&record, scatter_attenuation, wavelength)
                            * (scattering_pdf / continuation.pdf_value);
                    current_ray = scattered_ray;
                    allow_emitted = !continuation.suppress_next_emission;
                    miss_radiance_weight = next_miss_radiance_weight;
                    if !Self::russian_roulette_survives_scalar(
                        bounce_index,
                        context.russian_roulette_min_depth,
                        &mut attenuation,
                        rng,
                    ) {
                        return radiance;
                    }
                }
            }
        }

        radiance
    }

    #[cfg(feature = "spectral")]
    #[allow(clippy::too_many_lines)]
    fn ray_sampled_wavelength_stokes(
        ray: &Ray,
        depth: u32,
        context: RayColorContext<'_>,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> StokesVector {
        let mut current_ray = *ray;
        let mut current_frame = PolarizationFrame::from_direction(*current_ray.direction());
        let mut throughput = StokesVector::unpolarized(1.0);
        let mut radiance = StokesVector::default();
        let mut allow_emitted = true;
        let mut miss_radiance_weight = 1.0;

        for bounce_index in 0..depth {
            let Some(record) = context.world.hit_with_rng(
                &current_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            ) else {
                let miss = Self::sample_spectrum(
                    context.miss_radiance(*current_ray.direction()),
                    wavelength,
                ) * miss_radiance_weight;
                return radiance + throughput * miss;
            };

            if allow_emitted {
                radiance +=
                    throughput * Self::spectral_emitted_at(&current_ray, &record, wavelength);
            }

            let Some(scatter) =
                record
                    .material
                    .scatter_spectral(&current_ray, &record, wavelength, rng)
            else {
                return radiance;
            };

            match scatter {
                ScatterRecord::Specular {
                    ray,
                    attenuation: scatter_attenuation,
                } => {
                    let outgoing_frame = PolarizationFrame::from_direction(*ray.direction());
                    let mueller = record.material.polarized_scatter_mueller(
                        &current_ray,
                        &record,
                        &ray,
                        current_frame,
                        outgoing_frame,
                        wavelength,
                    );
                    throughput = mueller.apply(throughput)
                        * Self::sample_material_spectrum(&record, scatter_attenuation, wavelength);
                    current_ray = ray;
                    current_frame = outgoing_frame;
                    allow_emitted = true;
                    miss_radiance_weight = 1.0;
                    if !Self::russian_roulette_survives_stokes(
                        bounce_index,
                        context.russian_roulette_min_depth,
                        &mut throughput,
                        rng,
                    ) {
                        return radiance;
                    }
                }
                ScatterRecord::Scattering {
                    attenuation: scatter_attenuation,
                    pdf: material_pdf,
                } => {
                    if context.sampling_strategy.uses_next_event_estimation() {
                        if let Some(lights) = context.lights {
                            radiance += Self::estimate_direct_lighting_polarized(
                                &current_ray,
                                &record,
                                current_frame,
                                throughput,
                                context.world,
                                lights,
                                scatter_attenuation,
                                wavelength,
                                rng,
                            );
                        }
                        if let Some(environment) = context.environment {
                            radiance += Self::estimate_environment_lighting_polarized(
                                &current_ray,
                                &record,
                                current_frame,
                                throughput,
                                context.world,
                                environment,
                                scatter_attenuation,
                                wavelength,
                                rng,
                            );
                        }
                    }

                    let continuation = context.sampling_strategy.continuation_sample(
                        &current_ray,
                        &record,
                        material_pdf,
                        context.lights,
                        context.environment,
                        rng,
                    );

                    if !continuation.pdf_value.is_finite() || continuation.pdf_value <= f64::EPSILON
                    {
                        return radiance;
                    }

                    let scattered_ray =
                        Ray::with_time(record.point, continuation.direction, current_ray.time());
                    let scattering_pdf =
                        record
                            .material
                            .scattering_pdf(&current_ray, &record, &scattered_ray);
                    if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
                        return radiance;
                    }

                    let next_miss_radiance_weight = if continuation.weight_environment_miss {
                        context.environment.map_or(1.0, |environment| {
                            Self::power_heuristic(
                                continuation.pdf_value,
                                environment.pdf_value(continuation.direction),
                            )
                        })
                    } else {
                        1.0
                    };

                    let outgoing_frame =
                        PolarizationFrame::from_direction(*scattered_ray.direction());
                    let mueller = record.material.polarized_scatter_mueller(
                        &current_ray,
                        &record,
                        &scattered_ray,
                        current_frame,
                        outgoing_frame,
                        wavelength,
                    );
                    let scalar_weight =
                        Self::sample_material_spectrum(&record, scatter_attenuation, wavelength)
                            * (scattering_pdf / continuation.pdf_value);
                    throughput = mueller.apply(throughput) * scalar_weight;
                    current_ray = scattered_ray;
                    current_frame = outgoing_frame;
                    allow_emitted = !continuation.suppress_next_emission;
                    miss_radiance_weight = next_miss_radiance_weight;
                    if !Self::russian_roulette_survives_stokes(
                        bounce_index,
                        context.russian_roulette_min_depth,
                        &mut throughput,
                        rng,
                    ) {
                        return radiance;
                    }
                }
            }
        }

        radiance
    }

    fn ray_color_context_with_background<'a>(
        self,
        world: &'a dyn Hittable,
        lights: Option<&'a dyn Hittable>,
        environment: Option<&'a EnvironmentLight>,
        background: RayBackgroundContext<'a>,
    ) -> RayColorContext<'a> {
        RayColorContext {
            world,
            lights,
            environment,
            sampling_strategy: self.sampling_strategy,
            background,
            russian_roulette_min_depth: self.russian_roulette_min_depth,
        }
    }

    fn emitted_at(ray: &Ray, hit: &HitRecord<'_>) -> LinearColor {
        hit.material.emitted(ray, hit, hit.u, hit.v, hit.point)
    }

    #[cfg(feature = "spectral")]
    fn sample_spectrum(color: LinearColor, wavelength: SampledWavelength) -> f64 {
        Spectrum::from_linear_rgb(color).sample(wavelength)
    }

    #[cfg(feature = "spectral")]
    fn sample_material_spectrum(
        hit: &HitRecord<'_>,
        attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        hit.material
            .spectral_attenuation(hit, attenuation, wavelength)
    }

    #[cfg(feature = "spectral")]
    fn spectral_emitted_at(ray: &Ray, hit: &HitRecord<'_>, wavelength: SampledWavelength) -> f64 {
        hit.material
            .spectral_emitted(ray, hit, hit.u, hit.v, hit.point, wavelength)
    }

    fn estimate_direct_lighting(
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        scatter_attenuation: LinearColor,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let context = PdfContext::new(hit.point, ray_in.time());
        let light_pdf = HittablePdf::new(lights, context);
        let direction_to_light = light_pdf.generate(rng);
        let light_pdf_value = light_pdf.value(direction_to_light);

        if !light_pdf_value.is_finite() || light_pdf_value <= f64::EPSILON {
            return LinearColor::default();
        }

        let shadow_ray = Ray::with_time(hit.point, direction_to_light, ray_in.time());
        let scattering_pdf = hit.material.scattering_pdf(ray_in, hit, &shadow_ray);
        if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
            return LinearColor::default();
        }

        let Some(light_hit) = world.hit_with_rng(
            &shadow_ray,
            Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
            rng,
        ) else {
            return LinearColor::default();
        };

        let emitted = Self::emitted_at(&shadow_ray, &light_hit);
        if !emitted.is_finite() {
            return LinearColor::default();
        }

        component_mul(scatter_attenuation, emitted) * (scattering_pdf / light_pdf_value)
    }

    fn estimate_environment_lighting(
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        scatter_attenuation: LinearColor,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let (direction, environment_pdf) = environment.sample_direction(rng);
        if !environment_pdf.is_finite() || environment_pdf <= f64::EPSILON {
            return LinearColor::default();
        }

        let shadow_ray = Ray::with_time(hit.point, direction, ray_in.time());
        let scattering_pdf = hit.material.scattering_pdf(ray_in, hit, &shadow_ray);
        if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
            return LinearColor::default();
        }
        let mis_weight = Self::power_heuristic(environment_pdf, scattering_pdf);

        if world
            .hit_with_rng(
                &shadow_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            )
            .is_some()
        {
            return LinearColor::default();
        }

        component_mul(scatter_attenuation, environment.radiance(direction))
            * (mis_weight * scattering_pdf / environment_pdf)
    }

    #[cfg(feature = "spectral")]
    fn estimate_direct_lighting_spectral(
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        scatter_attenuation: LinearColor,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> f64 {
        let context = PdfContext::new(hit.point, ray_in.time());
        let light_pdf = HittablePdf::new(lights, context);
        let direction_to_light = light_pdf.generate(rng);
        let light_pdf_value = light_pdf.value(direction_to_light);

        if !light_pdf_value.is_finite() || light_pdf_value <= f64::EPSILON {
            return 0.0;
        }

        let shadow_ray = Ray::with_time(hit.point, direction_to_light, ray_in.time());
        let scattering_pdf = hit.material.scattering_pdf(ray_in, hit, &shadow_ray);
        if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
            return 0.0;
        }

        let Some(light_hit) = world.hit_with_rng(
            &shadow_ray,
            Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
            rng,
        ) else {
            return 0.0;
        };

        let emitted = Self::spectral_emitted_at(&shadow_ray, &light_hit, wavelength);
        if !emitted.is_finite() {
            return 0.0;
        }

        Self::sample_material_spectrum(hit, scatter_attenuation, wavelength)
            * emitted
            * (scattering_pdf / light_pdf_value)
    }

    #[cfg(feature = "spectral")]
    fn estimate_environment_lighting_spectral(
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        scatter_attenuation: LinearColor,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> f64 {
        let (direction, environment_pdf) = environment.sample_direction(rng);
        if !environment_pdf.is_finite() || environment_pdf <= f64::EPSILON {
            return 0.0;
        }

        let shadow_ray = Ray::with_time(hit.point, direction, ray_in.time());
        let scattering_pdf = hit.material.scattering_pdf(ray_in, hit, &shadow_ray);
        if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
            return 0.0;
        }
        let mis_weight = Self::power_heuristic(environment_pdf, scattering_pdf);

        if world
            .hit_with_rng(
                &shadow_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            )
            .is_some()
        {
            return 0.0;
        }

        Self::sample_material_spectrum(hit, scatter_attenuation, wavelength)
            * Self::sample_spectrum(environment.radiance(direction), wavelength)
            * (mis_weight * scattering_pdf / environment_pdf)
    }

    #[cfg(feature = "spectral")]
    #[allow(clippy::too_many_arguments)]
    fn estimate_direct_lighting_polarized(
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        incoming_frame: PolarizationFrame,
        throughput: StokesVector,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        scatter_attenuation: LinearColor,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> StokesVector {
        let context = PdfContext::new(hit.point, ray_in.time());
        let light_pdf = HittablePdf::new(lights, context);
        let direction_to_light = light_pdf.generate(rng);
        let light_pdf_value = light_pdf.value(direction_to_light);

        if !light_pdf_value.is_finite() || light_pdf_value <= f64::EPSILON {
            return StokesVector::default();
        }

        let shadow_ray = Ray::with_time(hit.point, direction_to_light, ray_in.time());
        let scattering_pdf = hit.material.scattering_pdf(ray_in, hit, &shadow_ray);
        if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
            return StokesVector::default();
        }

        let Some(light_hit) = world.hit_with_rng(
            &shadow_ray,
            Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
            rng,
        ) else {
            return StokesVector::default();
        };

        let emitted = Self::spectral_emitted_at(&shadow_ray, &light_hit, wavelength);
        if !emitted.is_finite() {
            return StokesVector::default();
        }

        let outgoing_frame = PolarizationFrame::from_direction(direction_to_light);
        let mueller = hit.material.polarized_scatter_mueller(
            ray_in,
            hit,
            &shadow_ray,
            incoming_frame,
            outgoing_frame,
            wavelength,
        );
        let scatter_weight = Self::sample_material_spectrum(hit, scatter_attenuation, wavelength)
            * (scattering_pdf / light_pdf_value);
        mueller.apply(throughput) * (scatter_weight * emitted)
    }

    #[cfg(feature = "spectral")]
    #[allow(clippy::too_many_arguments)]
    fn estimate_environment_lighting_polarized(
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        incoming_frame: PolarizationFrame,
        throughput: StokesVector,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        scatter_attenuation: LinearColor,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> StokesVector {
        let (direction, environment_pdf) = environment.sample_direction(rng);
        if !environment_pdf.is_finite() || environment_pdf <= f64::EPSILON {
            return StokesVector::default();
        }

        let shadow_ray = Ray::with_time(hit.point, direction, ray_in.time());
        let scattering_pdf = hit.material.scattering_pdf(ray_in, hit, &shadow_ray);
        if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
            return StokesVector::default();
        }
        let mis_weight = Self::power_heuristic(environment_pdf, scattering_pdf);

        if world
            .hit_with_rng(
                &shadow_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            )
            .is_some()
        {
            return StokesVector::default();
        }

        let outgoing_frame = PolarizationFrame::from_direction(direction);
        let mueller = hit.material.polarized_scatter_mueller(
            ray_in,
            hit,
            &shadow_ray,
            incoming_frame,
            outgoing_frame,
            wavelength,
        );
        let scatter_weight = Self::sample_material_spectrum(hit, scatter_attenuation, wavelength)
            * (mis_weight * scattering_pdf / environment_pdf);
        let emitted = Self::sample_spectrum(environment.radiance(direction), wavelength);
        mueller.apply(throughput) * (scatter_weight * emitted)
    }

    fn power_heuristic(sampled_pdf: f64, other_pdf: f64) -> f64 {
        if !sampled_pdf.is_finite() || sampled_pdf <= f64::EPSILON {
            return 0.0;
        }
        if !other_pdf.is_finite() || other_pdf <= f64::EPSILON {
            return 1.0;
        }
        let sampled = sampled_pdf * sampled_pdf;
        let other = other_pdf * other_pdf;
        sampled / (sampled + other)
    }

    fn russian_roulette_survives(
        bounce_index: u32,
        min_depth: Option<u32>,
        attenuation: &mut LinearColor,
        rng: &mut SampleRng,
    ) -> bool {
        if min_depth.is_none_or(|min_depth| bounce_index < min_depth) {
            return true;
        }

        let max_component = attenuation.max_component();
        if !max_component.is_finite() || max_component <= f64::EPSILON {
            return false;
        }

        let survival_probability = max_component.clamp(
            RUSSIAN_ROULETTE_MIN_SURVIVAL_PROBABILITY,
            RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY,
        );
        if rng.random_double() >= survival_probability {
            return false;
        }

        *attenuation = *attenuation / survival_probability;
        true
    }

    #[cfg(feature = "spectral")]
    fn russian_roulette_survives_scalar(
        bounce_index: u32,
        min_depth: Option<u32>,
        attenuation: &mut f64,
        rng: &mut SampleRng,
    ) -> bool {
        if min_depth.is_none_or(|min_depth| bounce_index < min_depth) {
            return true;
        }

        if !attenuation.is_finite() || *attenuation <= f64::EPSILON {
            return false;
        }

        let survival_probability = (*attenuation).clamp(
            RUSSIAN_ROULETTE_MIN_SURVIVAL_PROBABILITY,
            RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY,
        );
        if rng.random_double() >= survival_probability {
            return false;
        }

        *attenuation /= survival_probability;
        true
    }

    #[cfg(feature = "spectral")]
    fn russian_roulette_survives_stokes(
        bounce_index: u32,
        min_depth: Option<u32>,
        throughput: &mut StokesVector,
        rng: &mut SampleRng,
    ) -> bool {
        if min_depth.is_none_or(|min_depth| bounce_index < min_depth) {
            return true;
        }

        if !throughput.is_finite() {
            return false;
        }

        let survival_signal = throughput.i.abs().max(throughput.polarization_magnitude());
        if survival_signal <= f64::EPSILON {
            return false;
        }

        let survival_probability = survival_signal.clamp(
            RUSSIAN_ROULETTE_MIN_SURVIVAL_PROBABILITY,
            RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY,
        );
        if rng.random_double() >= survival_probability {
            return false;
        }

        *throughput = *throughput / survival_probability;
        true
    }

    fn render_world_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
    ) -> Rgb {
        Rgb::from_linear_color(self.render_world_linear_pixel(x, y, world, lights))
    }

    fn render_world_display_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
    ) -> Rgb {
        #[cfg(feature = "spectral")]
        if self.render_transport_mode == RenderTransportMode::Spectral {
            let pixel = self.render_world_spectral_pixel(
                x,
                y,
                world,
                lights,
                None,
                RayBackgroundContext::BuiltIn(self.background),
            );
            return Rgb::from_linear_color(pixel.linear_rgb);
        }

        self.render_world_pixel(x, y, world, lights)
    }

    fn render_world_linear_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
    ) -> LinearColor {
        self.render_world_linear_pixel_with_environment(x, y, world, lights, None)
    }

    fn render_world_linear_pixel_with_environment(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
    ) -> LinearColor {
        let background = environment.map_or(
            RayBackgroundContext::BuiltIn(self.background),
            |environment| RayBackgroundContext::Borrowed(environment),
        );
        self.render_world_linear_pixel_with_context(x, y, world, lights, environment, background)
    }

    fn render_world_linear_pixel_with_background_source(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        background: &dyn RayBackgroundSource,
    ) -> LinearColor {
        self.render_world_linear_pixel_with_context(
            x,
            y,
            world,
            lights,
            None,
            RayBackgroundContext::Borrowed(background),
        )
    }

    fn render_world_linear_pixel_with_context(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
        background: RayBackgroundContext<'_>,
    ) -> LinearColor {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        let sample_count = self.effective_samples_per_pixel();
        let pixel_context = PixelRenderContext {
            world,
            lights,
            environment,
            background,
        };

        match self.pixel_sample_mode {
            PixelSampleMode::Random => {
                if let Some(settings) = self.adaptive_sampling {
                    pixel_color = self.render_world_pixel_adaptive(x, y, pixel_context, settings);
                } else {
                    let mut accepted_samples = 0;
                    for _ in 0..sample_count {
                        accepted_samples += u32::from(Self::add_finite_sample(
                            &mut pixel_color,
                            self.sample_world_color(x, y, pixel_context, &mut rng),
                        ));
                    }
                    pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
                }
            }
            PixelSampleMode::Stratified | PixelSampleMode::StratifiedGrid { .. } => {
                let grid_width = self.active_stratified_grid_width();
                let mut accepted_samples = 0;
                for sample_y in 0..grid_width {
                    for sample_x in 0..grid_width {
                        let ray = self.ray_for_pixel_stratified_sample(
                            x, y, sample_x, sample_y, grid_width, &mut rng,
                        );
                        accepted_samples += u32::from(Self::add_finite_sample(
                            &mut pixel_color,
                            Self::ray_color(
                                &ray,
                                self.max_depth,
                                self.ray_color_context_with_background(
                                    pixel_context.world,
                                    pixel_context.lights,
                                    pixel_context.environment,
                                    pixel_context.background,
                                ),
                                &mut rng,
                            ),
                        ));
                    }
                }
                pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
            }
        }

        pixel_color
    }

    #[cfg(feature = "spectral")]
    fn render_world_spectral_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
        background: RayBackgroundContext<'_>,
    ) -> SpectralPixel {
        match self.spectral_transport_mode {
            SpectralTransportMode::Unpolarized => self.render_world_unpolarized_spectral_pixel(
                x,
                y,
                world,
                lights,
                environment,
                background,
            ),
            SpectralTransportMode::Polarized => self.render_world_polarized_spectral_pixel(
                x,
                y,
                world,
                lights,
                environment,
                background,
            ),
        }
    }

    #[cfg(feature = "spectral")]
    fn render_world_unpolarized_spectral_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
        background: RayBackgroundContext<'_>,
    ) -> SpectralPixel {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        let sample_count = self.effective_samples_per_pixel();

        match self.pixel_sample_mode {
            PixelSampleMode::Random => {
                let mut accepted_samples = 0;
                for _ in 0..sample_count {
                    let ray = self.ray_for_pixel_sample(x, y, &mut rng);
                    let wavelength = SampledWavelength::sample_visible(&mut rng);
                    let radiance = Self::ray_sampled_wavelength_radiance(
                        &ray,
                        self.max_depth,
                        self.ray_color_context_with_background(
                            world,
                            lights,
                            environment,
                            background,
                        ),
                        wavelength,
                        &mut rng,
                    );
                    accepted_samples += u32::from(Self::add_finite_sample(
                        &mut pixel_color,
                        wavelength.reconstruct_linear_rgb(radiance),
                    ));
                }
                pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
            }
            PixelSampleMode::Stratified | PixelSampleMode::StratifiedGrid { .. } => {
                let grid_width = self.active_stratified_grid_width();
                let mut accepted_samples = 0;
                for sample_y in 0..grid_width {
                    for sample_x in 0..grid_width {
                        let ray = self.ray_for_pixel_stratified_sample(
                            x, y, sample_x, sample_y, grid_width, &mut rng,
                        );
                        let wavelength = SampledWavelength::sample_visible(&mut rng);
                        let radiance = Self::ray_sampled_wavelength_radiance(
                            &ray,
                            self.max_depth,
                            self.ray_color_context_with_background(
                                world,
                                lights,
                                environment,
                                background,
                            ),
                            wavelength,
                            &mut rng,
                        );
                        accepted_samples += u32::from(Self::add_finite_sample(
                            &mut pixel_color,
                            wavelength.reconstruct_linear_rgb(radiance),
                        ));
                    }
                }
                pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
            }
        }

        SpectralPixel {
            linear_rgb: pixel_color,
            polarization: None,
        }
    }

    #[cfg(feature = "spectral")]
    fn render_world_polarized_spectral_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
        background: RayBackgroundContext<'_>,
    ) -> SpectralPixel {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        let mut pixel_stokes = StokesVector::default();
        let sample_count = self.effective_samples_per_pixel();

        match self.pixel_sample_mode {
            PixelSampleMode::Random => {
                let mut accepted_samples = 0;
                for _ in 0..sample_count {
                    let ray = self.ray_for_pixel_sample(x, y, &mut rng);
                    let wavelength = SampledWavelength::sample_visible(&mut rng);
                    let stokes = Self::ray_sampled_wavelength_stokes(
                        &ray,
                        self.max_depth,
                        self.ray_color_context_with_background(
                            world,
                            lights,
                            environment,
                            background,
                        ),
                        wavelength,
                        &mut rng,
                    );
                    accepted_samples += u32::from(Self::add_finite_spectral_sample(
                        &mut pixel_color,
                        &mut pixel_stokes,
                        wavelength,
                        stokes,
                    ));
                }
                pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
                pixel_stokes =
                    Self::average_accepted_stokes_samples(pixel_stokes, accepted_samples);
            }
            PixelSampleMode::Stratified | PixelSampleMode::StratifiedGrid { .. } => {
                let grid_width = self.active_stratified_grid_width();
                let mut accepted_samples = 0;
                for sample_y in 0..grid_width {
                    for sample_x in 0..grid_width {
                        let ray = self.ray_for_pixel_stratified_sample(
                            x, y, sample_x, sample_y, grid_width, &mut rng,
                        );
                        let wavelength = SampledWavelength::sample_visible(&mut rng);
                        let stokes = Self::ray_sampled_wavelength_stokes(
                            &ray,
                            self.max_depth,
                            self.ray_color_context_with_background(
                                world,
                                lights,
                                environment,
                                background,
                            ),
                            wavelength,
                            &mut rng,
                        );
                        accepted_samples += u32::from(Self::add_finite_spectral_sample(
                            &mut pixel_color,
                            &mut pixel_stokes,
                            wavelength,
                            stokes,
                        ));
                    }
                }
                pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
                pixel_stokes =
                    Self::average_accepted_stokes_samples(pixel_stokes, accepted_samples);
            }
        }

        SpectralPixel {
            linear_rgb: pixel_color,
            polarization: Some(pixel_stokes),
        }
    }

    fn sample_world_color(
        self,
        x: u32,
        y: u32,
        context: PixelRenderContext<'_>,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let ray = self.ray_for_pixel_sample(x, y, rng);
        Self::ray_color(
            &ray,
            self.max_depth,
            self.ray_color_context_with_background(
                context.world,
                context.lights,
                context.environment,
                context.background,
            ),
            rng,
        )
    }

    fn render_world_pixel_adaptive(
        self,
        x: u32,
        y: u32,
        context: PixelRenderContext<'_>,
        settings: AdaptiveSampling,
    ) -> LinearColor {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut mean = LinearColor::default();
        let mut m2 = LinearColor::default();
        let mut accepted_samples = 0;

        for _ in 0..settings.max_samples {
            let sample = self.sample_world_color(x, y, context, &mut rng);
            if !sample.is_finite() {
                continue;
            }

            accepted_samples += 1;
            let count = f64::from(accepted_samples);
            let delta = sample - mean;
            mean += delta / count;
            let delta2 = sample - mean;
            m2 += delta.component_mul(delta2);

            if accepted_samples >= settings.min_samples
                && Self::adaptive_error(m2, accepted_samples) < settings.error_threshold
            {
                break;
            }
        }

        mean
    }

    fn adaptive_error(m2: LinearColor, sample_count: u32) -> f64 {
        if sample_count <= 1 {
            return f64::INFINITY;
        }

        let variance = m2 / f64::from(sample_count - 1);
        (variance / f64::from(sample_count))
            .max_component()
            .max(0.0)
            .sqrt()
    }

    fn add_finite_sample(pixel_color: &mut LinearColor, sample: LinearColor) -> bool {
        if sample.is_finite() {
            *pixel_color += sample;
            true
        } else {
            false
        }
    }

    fn average_accepted_samples(pixel_color: LinearColor, accepted_samples: u32) -> LinearColor {
        if accepted_samples == 0 {
            LinearColor::default()
        } else {
            pixel_color / f64::from(accepted_samples)
        }
    }

    #[cfg(feature = "spectral")]
    fn add_finite_spectral_sample(
        pixel_color: &mut LinearColor,
        pixel_stokes: &mut StokesVector,
        wavelength: SampledWavelength,
        stokes: StokesVector,
    ) -> bool {
        if !stokes.is_finite() {
            return false;
        }
        *pixel_color += wavelength.reconstruct_linear_rgb(stokes.i);
        *pixel_stokes += wavelength.reconstruct_stokes(stokes);
        true
    }

    #[cfg(feature = "spectral")]
    fn average_accepted_stokes_samples(
        pixel_stokes: StokesVector,
        accepted_samples: u32,
    ) -> StokesVector {
        if accepted_samples == 0 {
            StokesVector::default()
        } else {
            pixel_stokes / f64::from(accepted_samples)
        }
    }

    fn active_stratified_grid_width(self) -> u32 {
        match self.pixel_sample_mode {
            PixelSampleMode::Random => 1,
            PixelSampleMode::Stratified => Self::stratified_grid_width(self.samples_per_pixel),
            PixelSampleMode::StratifiedGrid { grid_width } => grid_width.max(1),
        }
    }

    fn render_normal_pixel(self, x: u32, y: u32, world: &dyn Hittable) -> Rgb {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        let sample_count = self.effective_samples_per_pixel();

        match self.pixel_sample_mode {
            PixelSampleMode::Random => {
                for _ in 0..sample_count {
                    let ray = self.ray_for_pixel_sample(x, y, &mut rng);
                    pixel_color += normal_scene_color(&ray, world);
                }
            }
            PixelSampleMode::Stratified | PixelSampleMode::StratifiedGrid { .. } => {
                let grid_width = self.active_stratified_grid_width();
                for sample_y in 0..grid_width {
                    for sample_x in 0..grid_width {
                        let ray = self.ray_for_pixel_stratified_sample(
                            x, y, sample_x, sample_y, grid_width, &mut rng,
                        );
                        pixel_color += normal_scene_color(&ray, world);
                    }
                }
            }
        }

        Rgb::from_linear_color(pixel_color / f64::from(sample_count))
    }

    fn render_denoising_aov_linear_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        kind: DenoisingAovKind,
    ) -> LinearColor {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        let sample_count = self.effective_samples_per_pixel();

        match self.pixel_sample_mode {
            PixelSampleMode::Random => {
                for _ in 0..sample_count {
                    let ray = self.ray_for_pixel_sample(x, y, &mut rng);
                    pixel_color += Self::denoising_aov_sample(&ray, world, kind, &mut rng);
                }
            }
            PixelSampleMode::Stratified | PixelSampleMode::StratifiedGrid { .. } => {
                let grid_width = self.active_stratified_grid_width();
                for sample_y in 0..grid_width {
                    for sample_x in 0..grid_width {
                        let ray = self.ray_for_pixel_stratified_sample(
                            x, y, sample_x, sample_y, grid_width, &mut rng,
                        );
                        pixel_color += Self::denoising_aov_sample(&ray, world, kind, &mut rng);
                    }
                }
            }
        }

        pixel_color / f64::from(sample_count)
    }

    fn denoising_aov_sample(
        ray: &Ray,
        world: &dyn Hittable,
        kind: DenoisingAovKind,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let Some(record) =
            world.hit_with_rng(ray, Interval::new(SHADOW_ACNE_EPSILON, INFINITY), rng)
        else {
            return LinearColor::default();
        };

        match kind {
            DenoisingAovKind::Albedo => record.material.denoise_albedo(&record),
            DenoisingAovKind::Normal => {
                0.5 * (LinearColor::from(record.shading_normal) + LinearColor::new(1.0, 1.0, 1.0))
            }
        }
    }

    fn image_canvas(width: u32, height: u32, pixels: Vec<Rgb>) -> Canvas {
        Canvas::from_pixels_rgb_only(width, height, pixels, true, false)
    }

    fn render_pixels_tiled<F>(width: u32, height: u32, tile_size: u32, pixel: F) -> Vec<Rgb>
    where
        F: Fn(u32, u32) -> Rgb + Sync,
    {
        let result: Result<Vec<Rgb>, std::convert::Infallible> =
            Self::render_pixels_tiled_progressive(width, height, tile_size, pixel, |_| Ok(()));
        match result {
            Ok(pixels) => pixels,
            Err(error) => match error {},
        }
    }

    fn render_values_tiled<T, F>(width: u32, height: u32, tile_size: u32, value: F) -> Vec<T>
    where
        T: Clone + Default + Send,
        F: Fn(u32, u32) -> T + Sync,
    {
        let mut values = vec![T::default(); Canvas::pixel_count(width, height)];
        let tiles = Self::render_tiles(width, height, tile_size);
        let image_width = usize::try_from(width).expect("image width should fit usize");

        #[cfg(feature = "rayon")]
        {
            let rendered_tiles: Vec<_> = tiles
                .par_iter()
                .copied()
                .map(|tile| RenderedValueTile {
                    tile,
                    values: Self::render_tile_values(tile, &value),
                })
                .collect();
            for rendered_tile in rendered_tiles {
                Self::copy_tile_values(&mut values, image_width, &rendered_tile);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for tile in tiles {
                let rendered_tile = RenderedValueTile {
                    tile,
                    values: Self::render_tile_values(tile, &value),
                };
                Self::copy_tile_values(&mut values, image_width, &rendered_tile);
            }
        }

        values
    }

    fn render_pixels_tiled_progressive<F, P, E>(
        width: u32,
        height: u32,
        tile_size: u32,
        pixel: F,
        mut progress: P,
    ) -> Result<Vec<Rgb>, E>
    where
        F: Fn(u32, u32) -> Rgb + Sync,
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        let mut pixels = vec![Rgb::default(); Canvas::pixel_count(width, height)];
        let tiles = Self::render_tiles(width, height, tile_size);
        let total_tiles = tiles.len();

        #[cfg(feature = "rayon")]
        {
            use std::sync::{
                atomic::{AtomicBool, Ordering},
                mpsc,
            };

            let (sender, receiver) = mpsc::channel();
            let cancelled = AtomicBool::new(false);
            let mut progress_error = None;
            let image_width = usize::try_from(width).expect("image width should fit usize");

            std::thread::scope(|scope| {
                let tiles = &tiles;
                let pixel = &pixel;
                let cancelled = &cancelled;
                let worker = scope.spawn(move || {
                    tiles
                        .par_iter()
                        .copied()
                        .for_each_with(sender, |sender, tile| {
                            if cancelled.load(Ordering::Relaxed) {
                                return;
                            }
                            let rendered_tile = RenderedTile {
                                tile,
                                pixels: Self::render_tile_pixels(tile, pixel),
                            };
                            let _ = sender.send(rendered_tile);
                        });
                });

                for (tile_index, rendered_tile) in receiver.into_iter().enumerate() {
                    let tile = rendered_tile.tile;
                    Self::copy_tile_pixels(&mut pixels, image_width, &rendered_tile);
                    let update = ProgressiveRenderUpdate {
                        progress: RenderProgress {
                            tile,
                            completed_tiles: tile_index + 1,
                            total_tiles,
                        },
                        image_width: width,
                        image_height: height,
                        pixels: &pixels,
                    };
                    if let Err(error) = progress(update) {
                        cancelled.store(true, Ordering::Relaxed);
                        progress_error = Some(error);
                        break;
                    }
                }
                worker.join().expect("tile render worker should not panic");
            });

            if let Some(error) = progress_error {
                return Err(error);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            let image_width = usize::try_from(width).expect("image width should fit usize");
            for (tile_index, tile) in tiles.into_iter().enumerate() {
                let rendered_tile = RenderedTile {
                    tile,
                    pixels: Self::render_tile_pixels(tile, &pixel),
                };
                Self::copy_tile_pixels(&mut pixels, image_width, &rendered_tile);
                let update = ProgressiveRenderUpdate {
                    progress: RenderProgress {
                        tile,
                        completed_tiles: tile_index + 1,
                        total_tiles,
                    },
                    image_width: width,
                    image_height: height,
                    pixels: &pixels,
                };
                progress(update)?;
            }
        }

        Ok(pixels)
    }

    fn render_tiles(width: u32, height: u32, tile_size: u32) -> Vec<RenderTile> {
        let image_width = usize::try_from(width).expect("image width should fit usize");
        let image_height = usize::try_from(height).expect("image height should fit usize");
        let tile_size = usize::try_from(tile_size.max(1)).expect("tile size should fit usize");
        let tiles_x = image_width.div_ceil(tile_size);
        let tiles_y = image_height.div_ceil(tile_size);
        let mut tiles = Vec::with_capacity(tiles_x.saturating_mul(tiles_y));

        for y in (0..image_height).step_by(tile_size) {
            let y_end = y.saturating_add(tile_size).min(image_height);
            for x in (0..image_width).step_by(tile_size) {
                let x_end = x.saturating_add(tile_size).min(image_width);
                tiles.push(RenderTile {
                    x: u32::try_from(x).expect("tile x should fit u32"),
                    y: u32::try_from(y).expect("tile y should fit u32"),
                    width: u32::try_from(x_end - x).expect("tile width should fit u32"),
                    height: u32::try_from(y_end - y).expect("tile height should fit u32"),
                });
            }
        }

        tiles
    }

    fn render_tile_pixels<F>(tile: RenderTile, pixel: &F) -> Vec<Rgb>
    where
        F: Fn(u32, u32) -> Rgb,
    {
        let mut pixels = Vec::with_capacity(tile.pixel_count());
        for y in tile.y..tile.y_end() {
            for x in tile.x..tile.x_end() {
                pixels.push(pixel(x, y));
            }
        }
        pixels
    }

    fn render_tile_values<T, F>(tile: RenderTile, value: &F) -> Vec<T>
    where
        F: Fn(u32, u32) -> T,
    {
        let mut values = Vec::with_capacity(tile.pixel_count());
        for y in tile.y..tile.y_end() {
            for x in tile.x..tile.x_end() {
                values.push(value(x, y));
            }
        }
        values
    }

    fn copy_tile_pixels(
        image_pixels: &mut [Rgb],
        image_width: usize,
        rendered_tile: &RenderedTile,
    ) {
        debug_assert_eq!(rendered_tile.pixels.len(), rendered_tile.tile.pixel_count());
        let tile = rendered_tile.tile;
        let tile_width = usize::try_from(tile.width).expect("tile width should fit usize");
        let tile_x = usize::try_from(tile.x).expect("tile x should fit usize");
        let tile_y = usize::try_from(tile.y).expect("tile y should fit usize");

        for (row, tile_row) in rendered_tile.pixels.chunks_exact(tile_width).enumerate() {
            let start = (tile_y + row)
                .checked_mul(image_width)
                .and_then(|row_start| row_start.checked_add(tile_x))
                .expect("tile pixel offset should fit usize");
            let end = start + tile_width;
            image_pixels[start..end].copy_from_slice(tile_row);
        }
    }

    fn copy_tile_values<T: Clone>(
        image_values: &mut [T],
        image_width: usize,
        rendered_tile: &RenderedValueTile<T>,
    ) {
        debug_assert_eq!(rendered_tile.values.len(), rendered_tile.tile.pixel_count());
        let tile = rendered_tile.tile;
        let tile_width = usize::try_from(tile.width).expect("tile width should fit usize");
        let tile_x = usize::try_from(tile.x).expect("tile x should fit usize");
        let tile_y = usize::try_from(tile.y).expect("tile y should fit usize");

        for (row, tile_row) in rendered_tile.values.chunks_exact(tile_width).enumerate() {
            let start = (tile_y + row)
                .checked_mul(image_width)
                .and_then(|row_start| row_start.checked_add(tile_x))
                .expect("tile value offset should fit usize");
            let end = start + tile_width;
            image_values[start..end].clone_from_slice(tile_row);
        }
    }

    /// Renders a canvas by evaluating `ray_color` for each emitted camera ray.
    pub fn render<F>(self, mut ray_color: F) -> Canvas
    where
        F: FnMut(&Ray) -> LinearColor,
    {
        let mut pixels =
            Vec::with_capacity(Canvas::pixel_count(self.image_width, self.image_height));
        for y in 0..self.image_height {
            for x in 0..self.image_width {
                pixels.push(Rgb::from(ray_color(&self.ray_for_pixel(x, y))));
            }
        }
        Canvas::from_pixels_rgb_only(self.image_width, self.image_height, pixels, true, false)
    }

    /// Renders a hittable world using this camera's antialiasing sample count.
    pub fn render_world(self, world: &dyn Hittable) -> Canvas {
        self.render_world_hdr_image(world).to_canvas()
    }

    /// Renders a hittable world to linear floating-point HDR samples.
    #[must_use]
    pub fn render_world_hdr_image(self, world: &dyn Hittable) -> HdrImage {
        self.render_world_hdr_image_tiled(world, DEFAULT_RENDER_TILE_SIZE)
    }

    /// Renders a hittable world to linear floating-point HDR samples with explicit tile size.
    #[must_use]
    pub fn render_world_hdr_image_tiled(self, world: &dyn Hittable, tile_size: u32) -> HdrImage {
        self.render_world_with_optional_lights_hdr_image_tiled(world, None, tile_size)
    }

    /// Renders beauty plus denoising-friendly first-hit albedo and normal AOVs.
    #[must_use]
    pub fn render_world_denoising_aovs(self, world: &dyn Hittable) -> DenoisingAovs {
        self.render_world_with_optional_lights_denoising_aovs(world, None)
    }

    /// Renders beauty plus denoising AOVs with an explicit tile size.
    #[must_use]
    pub fn render_world_denoising_aovs_tiled(
        self,
        world: &dyn Hittable,
        tile_size: u32,
    ) -> DenoisingAovs {
        self.render_world_with_optional_lights_denoising_aovs_tiled(world, None, tile_size)
    }

    /// Renders a hittable world in tile-height bands.
    ///
    /// When the `rayon` feature is enabled, tile bands are rendered independently in parallel. Use
    /// this to tune tile size for previews or cache behavior; [`Self::render_world`] uses a
    /// conservative default tile size.
    pub fn render_world_tiled(self, world: &dyn Hittable, tile_size: u32) -> Canvas {
        self.render_world_hdr_image_tiled(world, tile_size)
            .to_canvas()
    }

    /// Renders a hittable world with the feature-gated sampled-wavelength prototype.
    ///
    /// This path samples one visible wavelength per camera sample, evaluates existing RGB
    /// materials through [`Spectrum`], and converts the linear spectral output to an RGB canvas.
    /// Enable it with the `spectral` cargo feature. Spectral renders use Stokes/Mueller polarized
    /// transport by default; use [`Self::with_unpolarized_spectral_transport`] to select scalar
    /// spectral transport, and [`Self::render_world_spectral_image`] when you need linear
    /// floating-point output before display encoding. [`Spectrum::Rgb`] keeps RGB assets
    /// compatible; measured inputs should use [`crate::graphics::raytracing::MeasuredSpectrum`].
    #[cfg(feature = "spectral")]
    pub fn render_world_spectral(self, world: &dyn Hittable) -> Canvas {
        self.render_world_spectral_image(world).to_canvas()
    }

    /// Renders a hittable world to a linear sampled-wavelength spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_spectral_image(self, world: &dyn Hittable) -> SpectralImage {
        self.render_world_spectral_image_tiled(world, DEFAULT_RENDER_TILE_SIZE)
    }

    /// Renders a hittable world with the sampled-wavelength prototype and explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_spectral_tiled(self, world: &dyn Hittable, tile_size: u32) -> Canvas {
        self.render_world_spectral_image_tiled(world, tile_size)
            .to_canvas()
    }

    /// Renders a hittable world to a linear spectral image with explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        tile_size: u32,
    ) -> SpectralImage {
        self.render_world_with_optional_lights_spectral_image_tiled(world, None, None, tile_size)
    }

    /// Renders a hittable world in tiles and calls `progress` after each tile is copied.
    ///
    /// The callback runs on the caller thread and receives a partial image view. When the `rayon`
    /// feature is enabled, tile rendering work is still parallel; tile completion order is not
    /// guaranteed.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_world_progressive<P, E>(
        self,
        world: &dyn Hittable,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        self.render_world_tiled_progressive(world, DEFAULT_RENDER_TILE_SIZE, progress)
    }

    /// Renders a hittable world in tiles and calls `progress` after each tile is copied.
    ///
    /// This variant lets callers choose a tile size explicitly.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_world_tiled_progressive<P, E>(
        self,
        world: &dyn Hittable,
        tile_size: u32,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        self.render_world_with_optional_lights_tiled_progressive(world, None, tile_size, progress)
    }

    /// Renders a hittable world while importance-sampling directions toward `lights`.
    ///
    /// Pass a lights-only or otherwise important target set here; passing the full scene is valid
    /// but usually raises variance by sampling non-emissive geometry.
    pub fn render_world_with_lights(self, world: &dyn Hittable, lights: &dyn Hittable) -> Canvas {
        self.render_world_with_lights_hdr_image(world, lights)
            .to_canvas()
    }

    /// Renders a lit hittable world to linear floating-point HDR samples.
    #[must_use]
    pub fn render_world_with_lights_hdr_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> HdrImage {
        self.render_world_with_lights_hdr_image_tiled(world, lights, DEFAULT_RENDER_TILE_SIZE)
    }

    /// Renders a lit hittable world to linear floating-point HDR samples with explicit tile size.
    #[must_use]
    pub fn render_world_with_lights_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
    ) -> HdrImage {
        self.render_world_with_optional_lights_hdr_image_tiled(world, Some(lights), tile_size)
    }

    /// Renders `world` with a lat-long environment light used for miss radiance and direct sampling.
    pub fn render_world_with_environment(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        self.render_world_with_environment_hdr_image(world, environment)
            .to_canvas()
    }

    /// Renders `world` with an environment light to linear floating-point HDR samples.
    #[must_use]
    pub fn render_world_with_environment_hdr_image(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> HdrImage {
        self.render_world_with_environment_hdr_image_tiled(
            world,
            environment,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with an environment light to HDR samples and explicit tile size.
    #[must_use]
    pub fn render_world_with_environment_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> HdrImage {
        self.render_world_with_optional_lights_and_environment_hdr_image_tiled(
            world,
            None,
            environment,
            tile_size,
        )
    }

    /// Renders `world` with a custom background source for miss radiance.
    pub fn render_world_with_background(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        self.render_world_with_background_hdr_image(world, background)
            .to_canvas()
    }

    /// Renders `world` with a custom background source to linear HDR samples.
    #[must_use]
    pub fn render_world_with_background_hdr_image(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> HdrImage {
        self.render_world_with_background_hdr_image_tiled(
            world,
            background,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with a custom background source to HDR samples and explicit tile size.
    #[must_use]
    pub fn render_world_with_background_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> HdrImage {
        self.render_world_with_optional_lights_and_background_hdr_image_tiled(
            world, None, background, tile_size,
        )
    }

    /// Renders `world` with a custom background source and explicit tile size.
    pub fn render_world_with_background_tiled(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_background_hdr_image_tiled(world, background, tile_size)
            .to_canvas()
    }

    /// Renders `world` with a custom background source using sampled-wavelength spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_background_spectral(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        self.render_world_with_background_spectral_image(world, background)
            .to_canvas()
    }

    /// Renders `world` with a custom background source to a linear spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_background_spectral_image(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> SpectralImage {
        self.render_world_with_background_spectral_image_tiled(
            world,
            background,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with a custom background source spectrally and explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_background_spectral_tiled(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_background_spectral_image_tiled(world, background, tile_size)
            .to_canvas()
    }

    /// Renders `world` with a custom background source to a spectral image and explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_background_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> SpectralImage {
        self.render_world_with_optional_lights_and_background_spectral_image_tiled(
            world, None, background, tile_size,
        )
    }

    /// Renders `world` with an environment light using sampled-wavelength spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_environment_spectral(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        self.render_world_with_environment_spectral_image(world, environment)
            .to_canvas()
    }

    /// Renders `world` with an environment light to a linear spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_environment_spectral_image(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> SpectralImage {
        self.render_world_with_environment_spectral_image_tiled(
            world,
            environment,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with explicit geometry lights plus an importance-sampled environment light.
    pub fn render_world_with_lights_and_environment(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        self.render_world_with_lights_and_environment_hdr_image(world, lights, environment)
            .to_canvas()
    }

    /// Renders `world` with geometry lights and an environment light to HDR samples.
    #[must_use]
    pub fn render_world_with_lights_and_environment_hdr_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> HdrImage {
        self.render_world_with_lights_and_environment_hdr_image_tiled(
            world,
            lights,
            environment,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with geometry lights and an environment light to HDR samples and tile size.
    #[must_use]
    pub fn render_world_with_lights_and_environment_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> HdrImage {
        self.render_world_with_optional_lights_and_environment_hdr_image_tiled(
            world,
            Some(lights),
            environment,
            tile_size,
        )
    }

    /// Renders `world` with explicit geometry lights plus a custom background source.
    pub fn render_world_with_lights_and_background(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        self.render_world_with_lights_and_background_hdr_image(world, lights, background)
            .to_canvas()
    }

    /// Renders `world` with geometry lights and a custom background to HDR samples.
    #[must_use]
    pub fn render_world_with_lights_and_background_hdr_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> HdrImage {
        self.render_world_with_lights_and_background_hdr_image_tiled(
            world,
            lights,
            background,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with geometry lights and a custom background to HDR samples and tile size.
    #[must_use]
    pub fn render_world_with_lights_and_background_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> HdrImage {
        self.render_world_with_optional_lights_and_background_hdr_image_tiled(
            world,
            Some(lights),
            background,
            tile_size,
        )
    }

    /// Renders `world` with explicit geometry lights, custom background, and tile size.
    pub fn render_world_with_lights_and_background_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_lights_and_background_hdr_image_tiled(
            world, lights, background, tile_size,
        )
        .to_canvas()
    }

    /// Renders `world` with explicit geometry lights plus a custom spectral background source.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_background_spectral(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        self.render_world_with_lights_and_background_spectral_image(world, lights, background)
            .to_canvas()
    }

    /// Renders `world` with explicit lights plus a custom background to a spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_background_spectral_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> SpectralImage {
        self.render_world_with_lights_and_background_spectral_image_tiled(
            world,
            lights,
            background,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders `world` with explicit lights, custom spectral background, and tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_background_spectral_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_lights_and_background_spectral_image_tiled(
            world, lights, background, tile_size,
        )
        .to_canvas()
    }

    /// Renders `world` with explicit lights and custom background to a spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_background_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> SpectralImage {
        self.render_world_with_optional_lights_and_background_spectral_image_tiled(
            world,
            Some(lights),
            background,
            tile_size,
        )
    }

    /// Renders `world` with geometry lights and an environment light using spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_environment_spectral(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        self.render_world_with_lights_and_environment_spectral_image(world, lights, environment)
            .to_canvas()
    }

    /// Renders `world` with geometry lights and an environment light to a linear spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_environment_spectral_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> SpectralImage {
        self.render_world_with_lights_and_environment_spectral_image_tiled(
            world,
            lights,
            environment,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    /// Renders lit beauty plus denoising-friendly first-hit albedo and normal AOVs.
    #[must_use]
    pub fn render_world_with_lights_denoising_aovs(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> DenoisingAovs {
        self.render_world_with_optional_lights_denoising_aovs(world, Some(lights))
    }

    /// Renders lit beauty plus denoising AOVs with an explicit tile size.
    #[must_use]
    pub fn render_world_with_lights_denoising_aovs_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
    ) -> DenoisingAovs {
        self.render_world_with_optional_lights_denoising_aovs_tiled(world, Some(lights), tile_size)
    }

    /// Renders a hittable world in tile-height bands while importance-sampling `lights`.
    ///
    /// When the `rayon` feature is enabled, tile bands are rendered independently in parallel.
    pub fn render_world_with_lights_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_lights_hdr_image_tiled(world, lights, tile_size)
            .to_canvas()
    }

    /// Renders `world` with an importance-sampled environment light and explicit tile size.
    pub fn render_world_with_environment_tiled(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_environment_hdr_image_tiled(world, environment, tile_size)
            .to_canvas()
    }

    /// Renders `world` with an environment light spectrally and explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_environment_spectral_tiled(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_environment_spectral_image_tiled(world, environment, tile_size)
            .to_canvas()
    }

    /// Renders `world` with an environment light to a spectral image and explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_environment_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> SpectralImage {
        self.render_world_with_optional_lights_spectral_image_tiled(
            world,
            None,
            Some(environment),
            tile_size,
        )
    }

    /// Renders `world` with geometry lights, an environment light, and explicit tile size.
    pub fn render_world_with_lights_and_environment_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_lights_and_environment_hdr_image_tiled(
            world,
            lights,
            environment,
            tile_size,
        )
        .to_canvas()
    }

    /// Renders `world` with geometry lights, environment, and spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_environment_spectral_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_lights_and_environment_spectral_image_tiled(
            world,
            lights,
            environment,
            tile_size,
        )
        .to_canvas()
    }

    /// Renders `world` with geometry lights and environment to a spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_and_environment_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> SpectralImage {
        self.render_world_with_optional_lights_spectral_image_tiled(
            world,
            Some(lights),
            Some(environment),
            tile_size,
        )
    }

    /// Renders `world` with explicit next-event light connections.
    ///
    /// This forces next-event estimation for callers that built a camera with
    /// [`DirectLightingMode::CurrentPathContinuation`]. It is not a bidirectional path tracer.
    pub fn render_world_with_light_connections(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> Canvas {
        self.light_connection_camera()
            .render_world_with_lights(world, lights)
    }

    /// Renders `world` with explicit next-event light connections and explicit tile size.
    pub fn render_world_with_light_connections_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
    ) -> Canvas {
        self.light_connection_camera()
            .render_world_with_lights_tiled(world, lights, tile_size)
    }

    /// Renders a lit hittable world with the feature-gated sampled-wavelength prototype.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_spectral(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> Canvas {
        self.render_world_with_lights_spectral_image(world, lights)
            .to_canvas()
    }

    /// Renders a lit hittable world to a linear spectral image.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_spectral_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> SpectralImage {
        self.render_world_with_lights_spectral_image_tiled(world, lights, DEFAULT_RENDER_TILE_SIZE)
    }

    /// Renders a lit hittable world with the sampled-wavelength prototype and explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_spectral_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
    ) -> Canvas {
        self.render_world_with_lights_spectral_image_tiled(world, lights, tile_size)
            .to_canvas()
    }

    /// Renders a lit hittable world to a linear spectral image with explicit tile size.
    #[cfg(feature = "spectral")]
    pub fn render_world_with_lights_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
    ) -> SpectralImage {
        self.render_world_with_optional_lights_spectral_image_tiled(
            world,
            Some(lights),
            None,
            tile_size,
        )
    }

    /// Renders a lit hittable world in tiles and calls `progress` after each tile is copied.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_world_with_lights_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        self.render_world_with_lights_tiled_progressive(
            world,
            lights,
            DEFAULT_RENDER_TILE_SIZE,
            progress,
        )
    }

    /// Renders a lit hittable world in tiles and calls `progress` after each tile is copied.
    ///
    /// This variant lets callers choose a tile size explicitly.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_world_with_lights_tiled_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        self.render_world_with_optional_lights_tiled_progressive(
            world,
            Some(lights),
            tile_size,
            progress,
        )
    }

    /// Renders `world` with explicit light connections and progressive tile callbacks.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_world_with_light_connections_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        self.render_world_with_light_connections_tiled_progressive(
            world,
            lights,
            DEFAULT_RENDER_TILE_SIZE,
            progress,
        )
    }

    /// Renders `world` with explicit light connections, explicit tile size, and callbacks.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_world_with_light_connections_tiled_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tile_size: u32,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        self.light_connection_camera()
            .render_world_with_lights_tiled_progressive(world, lights, tile_size, progress)
    }

    fn light_connection_camera(self) -> Self {
        self.with_direct_lighting_mode(DirectLightingMode::NextEventEstimation)
    }

    fn environment_importance_camera(self) -> Self {
        self.with_direct_lighting_mode(DirectLightingMode::NextEventEstimation)
    }

    fn render_world_with_optional_lights_denoising_aovs(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
    ) -> DenoisingAovs {
        self.render_world_with_optional_lights_denoising_aovs_tiled(
            world,
            lights,
            DEFAULT_RENDER_TILE_SIZE,
        )
    }

    fn render_world_with_optional_lights_denoising_aovs_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        tile_size: u32,
    ) -> DenoisingAovs {
        let camera = self.initialize();
        let beauty_linear = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_world_linear_pixel(x, y, world, lights),
        );
        let albedo_linear = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_denoising_aov_linear_pixel(x, y, world, DenoisingAovKind::Albedo),
        );
        let normal_linear = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_denoising_aov_linear_pixel(x, y, world, DenoisingAovKind::Normal),
        );
        DenoisingAovs::new(
            Self::image_canvas(
                camera.image_width,
                camera.image_height,
                beauty_linear
                    .iter()
                    .copied()
                    .map(Rgb::from_linear_color)
                    .collect(),
            ),
            Self::image_canvas(
                camera.image_width,
                camera.image_height,
                albedo_linear
                    .iter()
                    .copied()
                    .map(Rgb::from_raw_linear_color)
                    .collect(),
            ),
            Self::image_canvas(
                camera.image_width,
                camera.image_height,
                normal_linear
                    .iter()
                    .copied()
                    .map(Rgb::from_raw_linear_color)
                    .collect(),
            ),
            beauty_linear,
            albedo_linear,
            normal_linear,
        )
    }

    fn render_world_with_optional_lights_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        tile_size: u32,
    ) -> HdrImage {
        #[cfg(feature = "spectral")]
        if self.render_transport_mode == RenderTransportMode::Spectral {
            return self
                .render_world_with_optional_lights_spectral_image_tiled(
                    world, lights, None, tile_size,
                )
                .to_hdr_image();
        }

        let camera = self.initialize();
        let pixels = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_world_linear_pixel(x, y, world, lights),
        );
        HdrImage::from_pixels(camera.image_width, camera.image_height, pixels)
    }

    fn render_world_with_optional_lights_and_environment_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: &EnvironmentLight,
        tile_size: u32,
    ) -> HdrImage {
        #[cfg(feature = "spectral")]
        if self.render_transport_mode == RenderTransportMode::Spectral {
            return self
                .render_world_with_optional_lights_spectral_image_tiled(
                    world,
                    lights,
                    Some(environment),
                    tile_size,
                )
                .to_hdr_image();
        }

        let camera = self.environment_importance_camera().initialize();
        let pixels = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| {
                camera.render_world_linear_pixel_with_environment(
                    x,
                    y,
                    world,
                    lights,
                    Some(environment),
                )
            },
        );
        HdrImage::from_pixels(camera.image_width, camera.image_height, pixels)
    }

    fn render_world_with_optional_lights_and_background_hdr_image_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> HdrImage {
        #[cfg(feature = "spectral")]
        if self.render_transport_mode == RenderTransportMode::Spectral {
            return self
                .render_world_with_optional_lights_and_background_spectral_image_tiled(
                    world, lights, background, tile_size,
                )
                .to_hdr_image();
        }

        let camera = self.initialize();
        let pixels = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| {
                camera.render_world_linear_pixel_with_background_source(
                    x, y, world, lights, background,
                )
            },
        );
        HdrImage::from_pixels(camera.image_width, camera.image_height, pixels)
    }

    #[cfg(feature = "spectral")]
    fn render_world_with_optional_lights_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        environment: Option<&EnvironmentLight>,
        tile_size: u32,
    ) -> SpectralImage {
        let camera = if environment.is_some() {
            self.environment_importance_camera().initialize()
        } else {
            self.initialize()
        };
        let background = environment.map_or(
            RayBackgroundContext::BuiltIn(camera.background),
            |environment| RayBackgroundContext::Borrowed(environment),
        );
        let spectral_pixels = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_world_spectral_pixel(x, y, world, lights, environment, background),
        );
        camera.spectral_image_from_pixels(spectral_pixels)
    }

    #[cfg(feature = "spectral")]
    fn render_world_with_optional_lights_and_background_spectral_image_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        background: &dyn RayBackgroundSource,
        tile_size: u32,
    ) -> SpectralImage {
        let camera = self.initialize();
        let background = RayBackgroundContext::Borrowed(background);
        let spectral_pixels = Self::render_values_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_world_spectral_pixel(x, y, world, lights, None, background),
        );
        camera.spectral_image_from_pixels(spectral_pixels)
    }

    #[cfg(feature = "spectral")]
    fn spectral_image_from_pixels(self, spectral_pixels: Vec<SpectralPixel>) -> SpectralImage {
        let linear_rgb = spectral_pixels
            .iter()
            .map(|pixel| pixel.linear_rgb)
            .collect();
        if self.spectral_transport_mode == SpectralTransportMode::Polarized {
            let polarization = spectral_pixels
                .into_iter()
                .map(|pixel| {
                    pixel
                        .polarization
                        .expect("polarized spectral pixel should store Stokes output")
                })
                .collect();
            SpectralImage::new_with_polarization(
                self.image_width,
                self.image_height,
                linear_rgb,
                polarization,
            )
        } else {
            SpectralImage::new(
                self.image_width,
                self.image_height,
                linear_rgb,
                SpectralTransportMode::Unpolarized,
            )
        }
    }

    fn render_world_with_optional_lights_tiled_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        tile_size: u32,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        let camera = self.initialize();
        let pixels = Self::render_pixels_tiled_progressive(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_world_display_pixel(x, y, world, lights),
            progress,
        )?;
        Ok(Self::image_canvas(
            camera.image_width,
            camera.image_height,
            pixels,
        ))
    }

    /// Renders a hittable world as surface-normal colors for debugging.
    pub fn render_world_normals(self, world: &dyn Hittable) -> Canvas {
        let camera = self.initialize();
        let pixels = Self::render_pixels_tiled(
            camera.image_width,
            camera.image_height,
            DEFAULT_RENDER_TILE_SIZE,
            |x, y| camera.render_normal_pixel(x, y, world),
        );
        Self::image_canvas(camera.image_width, camera.image_height, pixels)
    }

    /// Renders a canvas while writing scanline progress messages to `log`.
    ///
    /// Use `std::io::stderr()` for book-style progress reporting that stays separate
    /// from generated PPM image output.
    ///
    /// # Errors
    ///
    /// Returns any write error produced by `log`.
    pub fn render_with_progress<F, W>(self, mut log: W, mut ray_color: F) -> io::Result<Canvas>
    where
        F: FnMut(&Ray) -> LinearColor,
        W: Write,
    {
        let mut pixels =
            Vec::with_capacity(Canvas::pixel_count(self.image_width, self.image_height));
        for y in 0..self.image_height {
            write!(log, "\rScanlines remaining: {} ", self.image_height - y)?;
            log.flush()?;
            for x in 0..self.image_width {
                pixels.push(Rgb::from(ray_color(&self.ray_for_pixel(x, y))));
            }
        }
        writeln!(log, "\rDone.                 ")?;

        Ok(Self::image_canvas(
            self.image_width,
            self.image_height,
            pixels,
        ))
    }

    /// Renders a hittable world with antialiasing while writing scanline progress messages.
    ///
    /// Rows are rendered in chunks so progress updates stay ordered. With the `rayon` feature
    /// enabled, pixels inside each chunk are rendered independently in parallel.
    ///
    /// # Panics
    ///
    /// Panics on platforms where the image width or row indices cannot be represented as `usize`.
    ///
    /// # Errors
    ///
    /// Returns any write error produced by `log`.
    pub fn render_world_with_progress<W>(
        self,
        world: &dyn Hittable,
        mut log: W,
    ) -> io::Result<Canvas>
    where
        W: Write,
    {
        let camera = self.initialize();
        let mut pixels =
            vec![Rgb::default(); Canvas::pixel_count(camera.image_width, camera.image_height)];
        let width = usize::try_from(camera.image_width).expect("image width should fit usize");

        let mut y_start = 0;
        while y_start < camera.image_height {
            write!(
                log,
                "\rScanlines remaining: {} ",
                camera.image_height - y_start
            )?;
            log.flush()?;
            let y_end = y_start
                .saturating_add(PROGRESS_RENDER_CHUNK_ROWS)
                .min(camera.image_height);
            let start = usize::try_from(y_start).expect("row index should fit usize") * width;
            let end = usize::try_from(y_end).expect("row index should fit usize") * width;
            let chunk = &mut pixels[start..end];

            #[cfg(feature = "rayon")]
            {
                chunk
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(row_offset, row_pixels)| {
                        let y =
                            y_start + u32::try_from(row_offset).expect("chunk row should fit u32");
                        for (x, pixel) in row_pixels.iter_mut().enumerate() {
                            *pixel = camera.render_world_display_pixel(
                                u32::try_from(x).expect("pixel x should fit u32"),
                                y,
                                world,
                                None,
                            );
                        }
                    });
            }

            #[cfg(not(feature = "rayon"))]
            {
                for (row_offset, row_pixels) in chunk.chunks_mut(width).enumerate() {
                    let y = y_start + u32::try_from(row_offset).expect("chunk row should fit u32");
                    for (x, pixel) in row_pixels.iter_mut().enumerate() {
                        *pixel = camera.render_world_display_pixel(
                            u32::try_from(x).expect("pixel x should fit u32"),
                            y,
                            world,
                            None,
                        );
                    }
                }
            }

            y_start = y_end;
        }
        writeln!(log, "\rDone.                 ")?;

        Ok(Self::image_canvas(
            camera.image_width,
            camera.image_height,
            pixels,
        ))
    }
}

impl ProjectedSegment {
    /// Creates a projected segment if both source points project in front of the camera.
    #[must_use]
    pub fn from_points(camera: &Camera3D, p0: &[f64], p1: &[f64], color: Rgb) -> Option<Self> {
        Some(Self {
            a: camera.project(p0)?,
            b: camera.project(p1)?,
            color,
        })
    }

    /// Returns the average projected depth of the segment.
    #[must_use]
    pub fn average_depth(&self) -> f64 {
        (self.a.depth + self.b.depth) * 0.5
    }
}

/// Sorts projected segments back-to-front for painter-style wireframe rendering.
pub fn sort_segments_back_to_front(segments: &mut [ProjectedSegment]) {
    segments.sort_by(|a, b| {
        b.average_depth()
            .partial_cmp(&a.average_depth())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

impl Canvas {
    /// Draws already-projected colored segments.
    pub fn draw_projected_segments<I>(&mut self, segments: I)
    where
        I: IntoIterator<Item = ProjectedSegment>,
    {
        for segment in segments {
            self.draw_line(
                segment.color,
                segment.a.x,
                segment.a.y,
                segment.b.x,
                segment.b.y,
            );
        }
    }

    /// Projects and draws transformed edge lines without allocating a transformed edge matrix.
    pub fn draw_projected_edges(
        &mut self,
        edges: &EdgeMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        color: Rgb,
    ) {
        for (p0, p1) in edges.transformed_edges(transform) {
            if let Some(segment) = ProjectedSegment::from_points(camera, &p0, &p1, color) {
                self.draw_projected_segments([segment]);
            }
        }
    }

    /// Projects and draws transformed mesh triangle wireframes without allocating a transformed mesh.
    pub fn draw_projected_mesh_wireframe(
        &mut self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        color: Rgb,
        stride: usize,
    ) {
        let stride = stride.max(1);
        for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
            if idx % stride != 0 {
                continue;
            }
            let Some(ab) = ProjectedSegment::from_points(camera, &p0, &p1, color) else {
                continue;
            };
            let Some(bc) = ProjectedSegment::from_points(camera, &p1, &p2, color) else {
                continue;
            };
            let Some(ca) = ProjectedSegment::from_points(camera, &p2, &p0, color) else {
                continue;
            };
            self.draw_projected_segments([ab, bc, ca]);
        }
    }

    /// Projects, depth-sorts, and draws a transformed mesh as triangle wireframe segments.
    ///
    /// `color_for_triangle` receives the triangle index and average projected triangle depth.
    pub fn draw_projected_mesh_wireframe_depth_sorted<F>(
        &mut self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        stride: usize,
        color_for_triangle: F,
    ) where
        F: FnMut(usize, f64) -> Rgb,
    {
        let mut segments =
            camera.project_mesh_wireframe_segments(mesh, transform, stride, color_for_triangle);
        sort_segments_back_to_front(&mut segments);
        self.draw_projected_segments(segments);
    }

    /// Projects and draws a filled mesh without allocating a projected [`PolygonMatrix`].
    pub fn draw_projected_mesh(&mut self, camera: &Camera3D, mesh: &PolygonMatrix, color: Rgb) {
        self.set_line_color(color);
        for (p0, p1, p2) in mesh.triangles() {
            for [a, b, c] in camera.project_clipped_triangle([p0, p1, p2]) {
                self.draw_triangle_culled(
                    color,
                    (a.x, a.y, -a.depth),
                    (b.x, b.y, -b.depth),
                    (c.x, c.y, -c.depth),
                );
            }
        }
    }

    /// Projects and draws a filled mesh with the canvas's current lighting state.
    pub fn draw_lit_projected_mesh(&mut self, camera: &Camera3D, mesh: &PolygonMatrix) {
        for (p0, p1, p2) in mesh.triangles() {
            for [a, b, c] in camera.project_clipped_triangle([p0, p1, p2]) {
                self.draw_lit_triangle_culled(
                    (a.x, a.y, -a.depth),
                    (b.x, b.y, -b.depth),
                    (c.x, c.y, -c.depth),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::raytracing::{HitRecord, Lambertian, Material, SpherePdf, SurfaceHit};
    use std::sync::atomic::{AtomicU32, Ordering};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[derive(Debug, Default)]
    struct CountingMissWorld {
        samples: AtomicU32,
    }

    impl CountingMissWorld {
        fn sample_count(&self) -> u32 {
            self.samples.load(Ordering::SeqCst)
        }
    }

    impl Hittable for CountingMissWorld {
        fn hit_with_rng(
            &self,
            _ray: &Ray,
            _ray_t: Interval,
            _rng: &mut SampleRng,
        ) -> Option<HitRecord<'_>> {
            self.samples.fetch_add(1, Ordering::SeqCst);
            None
        }
    }

    #[derive(Debug)]
    struct FixedPdfTarget {
        direction: Vector,
        pdf_value: f64,
    }

    impl Hittable for FixedPdfTarget {
        fn hit_with_rng(
            &self,
            _ray: &Ray,
            _ray_t: Interval,
            _rng: &mut SampleRng,
        ) -> Option<HitRecord<'_>> {
            None
        }

        fn pdf_value(&self, _context: PdfContext, _direction: Vector) -> f64 {
            self.pdf_value
        }

        fn random_direction(&self, _context: PdfContext, _rng: &mut SampleRng) -> Vector {
            self.direction
        }
    }

    fn sampling_strategy_hit(material: &dyn Material) -> (Ray, HitRecord<'_>) {
        let ray = Ray::new(Point::new(0.0, 0.0, -1.0), Vector::new(0.0, 0.0, 1.0));
        let hit = HitRecord::new(
            &ray,
            Point::new(0.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
            1.0,
            material,
        );
        (ray, hit)
    }

    #[derive(Debug, Default)]
    struct AlternatingEmissionMaterial;

    impl Material for AlternatingEmissionMaterial {
        fn emitted(
            &self,
            _ray_in: &Ray,
            _hit: &HitRecord<'_>,
            u: f64,
            _v: f64,
            _point: Point,
        ) -> LinearColor {
            if u < 0.5 {
                LinearColor::default()
            } else {
                LinearColor::new(1.0, 1.0, 1.0)
            }
        }
    }

    #[derive(Debug, Default)]
    struct AlternatingEmissionWorld {
        samples: AtomicU32,
        material: AlternatingEmissionMaterial,
    }

    impl AlternatingEmissionWorld {
        fn sample_count(&self) -> u32 {
            self.samples.load(Ordering::SeqCst)
        }
    }

    impl Hittable for AlternatingEmissionWorld {
        fn hit_with_rng(
            &self,
            ray: &Ray,
            _ray_t: Interval,
            _rng: &mut SampleRng,
        ) -> Option<HitRecord<'_>> {
            let sample = self.samples.fetch_add(1, Ordering::SeqCst);
            let u = if sample.is_multiple_of(2) { 0.0 } else { 1.0 };
            Some(HitRecord::from_surface(
                SurfaceHit {
                    point: ray.at(1.0),
                    normal: Vector::new(0.0, 0.0, 1.0),
                    t: 1.0,
                    u,
                    v: 0.0,
                    front_face: true,
                },
                &self.material,
            ))
        }
    }

    #[test]
    fn projected_mesh_wireframe_returns_three_segments_per_visible_triangle() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        let camera = Camera3D::new(100, 100);
        let segments = camera.project_mesh_wireframe_segments(
            &mesh,
            &Matrix::identity_matrix(4),
            1,
            |_, _| Rgb::WHITE,
        );

        assert_eq!(segments.len(), 3);
    }

    #[test]
    fn camera3d_default_projection_matches_legacy_camera_distance() {
        let camera = Camera3D::new(100, 100);
        let point = camera.project(&[0.0, 0.0, 0.0]).expect("visible");

        assert_close(point.x, 50.0);
        assert_close(point.y, 50.0);
        assert_close(point.depth, 900.0);
    }

    #[test]
    fn camera3d_can_be_positioned_with_look_at() {
        let camera = Camera3D::new(100, 100)
            .with_look_at(Point::new(0.0, 0.0, -10.0), Point::new(0.0, 0.0, 0.0))
            .with_focal_length(10.0)
            .with_near_depth(0.1);

        let center = camera.project(&[0.0, 0.0, 0.0]).expect("center visible");
        let right = camera.project(&[1.0, 0.0, 0.0]).expect("right visible");

        assert_close(center.x, 50.0);
        assert_close(center.y, 50.0);
        assert_close(center.depth, 10.0);
        assert!(right.x > center.x);
    }

    #[test]
    fn camera3d_vertical_fov_sets_projection_scale() {
        let camera = Camera3D::new(100, 100)
            .with_look_at(Point::new(0.0, 0.0, -10.0), Point::new(0.0, 0.0, 0.0))
            .with_vertical_fov(90.0)
            .with_near_depth(0.1);

        let top = camera.project(&[0.0, 10.0, 0.0]).expect("top visible");

        assert_close(top.y, 0.0);
    }
    #[test]
    fn ray_camera_uses_actual_integer_image_ratio() {
        let camera = RayCamera::new(400, 16.0 / 9.0);
        assert_eq!(camera.image_width(), 400);
        assert_eq!(camera.image_height(), 225);
    }
    #[test]
    fn ray_camera_sends_center_pixel_forward() {
        let camera = RayCamera::new(400, 16.0 / 9.0);
        let ray = camera.ray_for_pixel(200, 112);

        assert_close(ray.origin().x(), 0.0);
        assert_close(ray.origin().y(), 0.0);
        assert_close(ray.origin().z(), 0.0);
        assert!(ray.direction().z() < 0.0);
        assert!(ray.direction().x().abs() < 0.01);
        assert!(ray.direction().y().abs() < 0.01);
    }
    #[test]
    fn ray_camera_tracks_antialiasing_sample_count() {
        let camera = RayCamera::default()
            .with_image_width(40)
            .with_aspect_ratio(16.0 / 9.0)
            .with_samples_per_pixel(25)
            .with_max_depth(50);

        assert_eq!(camera.image_width(), 40);
        assert_eq!(camera.image_height(), 22);
        assert_eq!(camera.samples_per_pixel(), 25);
        assert_eq!(camera.max_depth(), 50);
        assert_eq!(camera.russian_roulette_min_depth(), Some(5));
        assert_close(camera.defocus_angle(), 0.0);
        assert_close(camera.focus_distance(), 1.0);
    }

    #[test]
    fn ray_camera_tracks_russian_roulette_setting() {
        let camera = RayCamera::default()
            .with_russian_roulette_min_depth(3)
            .without_russian_roulette();

        assert_eq!(camera.russian_roulette_min_depth(), None);
        assert_eq!(
            camera
                .with_russian_roulette_min_depth(7)
                .russian_roulette_min_depth(),
            Some(7)
        );
    }

    #[test]
    fn ray_camera_tracks_pixel_sample_mode() {
        let camera = RayCamera::new(40, 1.0)
            .with_samples_per_pixel(50)
            .with_stratified_sampling();

        assert_eq!(camera.samples_per_pixel(), 50);
        assert_eq!(camera.pixel_sample_mode(), PixelSampleMode::Stratified);
        assert_eq!(camera.effective_samples_per_pixel(), 49);
    }

    #[test]
    fn ray_camera_tracks_adaptive_sampling_settings() {
        let camera = RayCamera::new(40, 1.0).with_adaptive_sampling(16, 128, 0.01);

        assert_eq!(
            camera.adaptive_sampling(),
            Some(AdaptiveSampling {
                min_samples: 16,
                max_samples: 128,
                error_threshold: 0.01,
            })
        );
        assert_eq!(camera.samples_per_pixel(), 128);
        assert_eq!(camera.effective_samples_per_pixel(), 128);
        assert_eq!(camera.without_adaptive_sampling().adaptive_sampling(), None);
    }

    #[test]
    fn adaptive_sampling_settings_are_sanitized() {
        let settings = AdaptiveSampling::new(0, 0, 0.5);

        assert_eq!(settings.min_samples, 1);
        assert_eq!(settings.max_samples, 1);
        assert_close(settings.error_threshold, 0.5);
    }

    #[test]
    fn adaptive_error_uses_largest_channel_standard_error() {
        let m2 = LinearColor::new(1.0, 4.0, 9.0);

        assert_close(RayCamera::adaptive_error(m2, 4), 0.75_f64.sqrt());
        assert!(RayCamera::adaptive_error(m2, 1).is_infinite());
    }

    #[test]
    fn adaptive_sampling_stops_at_min_samples_for_zero_variance_pixel() {
        let world = CountingMissWorld::default();
        let camera = RayCamera::new(1, 1.0)
            .with_background(LinearColor::new(0.25, 0.25, 0.25))
            .with_adaptive_sampling(4, 16, 0.001);

        let color = camera.render_world_pixel(0, 0, &world, None);

        assert_eq!(world.sample_count(), 4);
        assert_eq!(
            color,
            Rgb::from_linear_color(LinearColor::new(0.25, 0.25, 0.25))
        );
    }

    #[test]
    fn ray_camera_hdr_render_preserves_values_above_display_white() {
        let world = CountingMissWorld::default();
        let camera = RayCamera::new(1, 1.0)
            .with_background(LinearColor::new(4.0, 2.0, 0.25))
            .with_samples_per_pixel(1);

        let image = camera.render_world_hdr_image(&world);

        assert_eq!(image.width(), 1);
        assert_eq!(image.height(), 1);
        assert_eq!(image.pixels()[0], LinearColor::new(4.0, 2.0, 0.25));
        assert_eq!(image.to_canvas().pixels()[0], Rgb::new(255, 255, 128));
    }

    #[test]
    fn adaptive_sampling_reaches_max_samples_for_noisy_pixel() {
        let world = AlternatingEmissionWorld::default();
        let camera = RayCamera::new(1, 1.0).with_adaptive_sampling(4, 16, 0.001);

        let color = camera.render_world_pixel(0, 0, &world, None);

        assert_eq!(world.sample_count(), 16);
        assert_eq!(
            color,
            Rgb::from_linear_color(LinearColor::new(0.5, 0.5, 0.5))
        );
    }

    #[test]
    fn ray_camera_supports_explicit_stratified_grid_width() {
        let camera = RayCamera::new(40, 1.0).with_stratified_grid_width(32);

        assert_eq!(camera.samples_per_pixel(), 1024);
        assert_eq!(
            camera.pixel_sample_mode(),
            PixelSampleMode::StratifiedGrid { grid_width: 32 }
        );
        assert_eq!(camera.effective_samples_per_pixel(), 1024);
    }

    #[test]
    fn stratified_sample_offsets_stay_inside_pixel_square() {
        let mut rng = SampleRng::new(23);
        let grid_width = 4;

        for sample_y in 0..grid_width {
            for sample_x in 0..grid_width {
                let offset =
                    RayCamera::sample_square_stratified(sample_x, sample_y, grid_width, &mut rng);
                assert!((-0.5..0.5).contains(&offset.x()));
                assert!((-0.5..0.5).contains(&offset.y()));
                assert_close(offset.z(), 0.0);
            }
        }
    }

    #[test]
    fn russian_roulette_skips_until_configured_depth() {
        let mut rng = SampleRng::new(41);
        let mut attenuation = LinearColor::default();

        assert!(RayCamera::russian_roulette_survives(
            3,
            Some(5),
            &mut attenuation,
            &mut rng
        ));
        assert_eq!(attenuation, LinearColor::default());

        assert!(RayCamera::russian_roulette_survives(
            5,
            None,
            &mut attenuation,
            &mut rng
        ));
    }

    #[test]
    fn russian_roulette_terminates_zero_throughput() {
        let mut rng = SampleRng::new(43);
        let mut attenuation = LinearColor::default();

        assert!(!RayCamera::russian_roulette_survives(
            5,
            Some(5),
            &mut attenuation,
            &mut rng
        ));
    }

    #[test]
    fn russian_roulette_scales_surviving_throughput() {
        let survived = (0..100).find_map(|seed| {
            let mut rng = SampleRng::new(seed);
            let mut attenuation = LinearColor::new(2.0, 1.0, 0.5);
            RayCamera::russian_roulette_survives(5, Some(5), &mut attenuation, &mut rng)
                .then_some(attenuation)
        });

        let attenuation = survived.expect("at least one deterministic seed should survive");
        assert_close(
            attenuation.x(),
            2.0 / RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY,
        );
        assert_close(
            attenuation.y(),
            1.0 / RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY,
        );
        assert_close(
            attenuation.z(),
            0.5 / RUSSIAN_ROULETTE_MAX_SURVIVAL_PROBABILITY,
        );
    }

    #[test]
    fn ray_camera_tracks_background_color() {
        let background = LinearColor::new(0.1, 0.2, 0.3);
        let camera = RayCamera::new(20, 1.0).with_background(background);
        let empty_world = crate::graphics::raytracing::HittableList::new();

        let canvas = camera.render_world(&empty_world);

        assert_eq!(camera.background(), background);
        assert_eq!(camera.constant_background(), Some(background));
        assert_eq!(canvas.pixels()[0], Rgb::from_linear_color(background));
    }

    #[test]
    fn ray_camera_environment_render_uses_environment_on_miss() {
        let environment = crate::graphics::raytracing::EnvironmentLight::constant(
            LinearColor::new(0.25, 0.5, 0.75),
        );
        let empty_world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0)
            .with_background(LinearColor::default())
            .with_samples_per_pixel(1);

        let canvas = camera.render_world_with_environment(&empty_world, &environment);

        assert_eq!(canvas.width(), 2);
        assert!(canvas.pixels().iter().any(|pixel| *pixel != Rgb::BLACK));
    }

    fn test_background_fn(_direction: Vector) -> LinearColor {
        LinearColor::new(0.25, 0.0, 0.0)
    }

    #[test]
    fn ray_camera_supports_background_sources() {
        let empty_world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0).with_samples_per_pixel(1);
        let gradient = RayBackground::vertical_gradient(
            LinearColor::new(0.0, 0.0, 0.1),
            LinearColor::new(0.0, 0.1, 0.0),
            LinearColor::new(0.1, 0.0, 0.0),
        );

        let gradient_canvas = camera
            .with_background_source(gradient)
            .render_world(&empty_world);
        let function_canvas = camera
            .with_background_fn(test_background_fn)
            .render_world(&empty_world);
        let closure_background = |_direction: Vector| LinearColor::new(0.0, 0.25, 0.0);
        let borrowed_canvas =
            camera.render_world_with_background(&empty_world, &closure_background);

        assert_eq!(
            camera.with_background_source(gradient).background_source(),
            gradient
        );
        assert_eq!(
            camera
                .with_background_source(gradient)
                .constant_background(),
            None
        );
        assert!(
            gradient_canvas
                .pixels()
                .iter()
                .any(|pixel| *pixel != Rgb::BLACK)
        );
        assert_eq!(function_canvas.pixels()[0], Rgb::new(128, 0, 0));
        assert_eq!(borrowed_canvas.pixels()[0], Rgb::new(0, 128, 0));
    }

    #[test]
    fn ray_camera_exposes_sampling_strategy_policy() {
        let strategy = SamplingStrategy::current_path_continuation().with_light_pdf_weight(0.25);
        let camera = RayCamera::new(2, 1.0).with_sampling_strategy(strategy);

        assert_eq!(camera.sampling_strategy(), strategy);
        assert_eq!(
            camera.direct_lighting_mode(),
            DirectLightingMode::CurrentPathContinuation
        );
        assert!((camera.sampling_strategy().light_pdf_weight() - 0.25).abs() < 1.0e-12);
        assert_eq!(
            RayCamera::new(2, 1.0)
                .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation)
                .sampling_strategy(),
            SamplingStrategy::next_event_estimation()
        );
        assert_eq!(
            SamplingStrategy::next_event_estimation()
                .with_light_pdf_weight(0.25)
                .direct_lighting_mode(),
            DirectLightingMode::CurrentPathContinuation
        );
    }

    #[test]
    fn sampling_strategy_light_pdf_weight_controls_continuation_pdf() {
        let material = Lambertian::new(LinearColor::new(0.5, 0.5, 0.5));
        let (ray, hit) = sampling_strategy_hit(&material);
        let material_pdf = MaterialPdf::Sphere(SpherePdf);
        let light = FixedPdfTarget {
            direction: Vector::new(1.0, 0.0, 0.0),
            pdf_value: 0.75,
        };

        for (seed_offset, weight) in [(0_u64, 0.0), (1, 0.5), (2, 1.0)] {
            let strategy =
                SamplingStrategy::current_path_continuation().with_light_pdf_weight(weight);
            let mut rng = SampleRng::new(100 + seed_offset);
            let sample = strategy.continuation_sample(
                &ray,
                &hit,
                material_pdf,
                Some(&light),
                None,
                &mut rng,
            );
            let material_pdf_value = material_pdf.value(sample.direction);
            let expected_pdf = (1.0 - weight) * material_pdf_value + weight * light.pdf_value;

            assert_close(sample.pdf_value, expected_pdf);
            assert!(!sample.suppress_next_emission);
            assert!(!sample.weight_environment_miss);
        }
    }

    #[test]
    fn ray_camera_denoising_aovs_include_albedo_and_normals() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let camera = RayCamera::new(4, 1.0)
            .with_samples_per_pixel(1)
            .with_background(LinearColor::default());

        let aovs = camera.render_world_denoising_aovs(&world);
        let tiled = camera.render_world_denoising_aovs_tiled(&world, 2);

        assert_eq!(aovs.beauty.width(), 4);
        assert_eq!(aovs.albedo.width(), 4);
        assert_eq!(aovs.normal.width(), 4);
        assert_eq!(aovs.beauty_linear.len(), 16);
        assert_eq!(aovs.albedo_linear.len(), 16);
        assert_eq!(aovs.normal_linear.len(), 16);
        assert_eq!(tiled.beauty.pixels(), aovs.beauty.pixels());
        assert_eq!(tiled.albedo_linear, aovs.albedo_linear);
        assert!(
            aovs.albedo
                .pixels()
                .iter()
                .any(|pixel| *pixel != Rgb::BLACK)
        );
        assert!(
            aovs.normal
                .pixels()
                .iter()
                .any(|pixel| *pixel != Rgb::BLACK)
        );
    }

    #[test]
    fn ray_camera_vertical_fov_controls_ray_spread() {
        let wide = RayCamera::new(101, 1.0).with_vertical_fov(90.0);
        let narrow = RayCamera::new(101, 1.0).with_vertical_fov(20.0);

        let wide_top = wide.ray_for_pixel(50, 0).direction().normalized();
        let narrow_top = narrow.ray_for_pixel(50, 0).direction().normalized();

        assert!(wide_top.y().abs() > narrow_top.y().abs());
    }
    #[test]
    fn ray_camera_can_be_positioned_with_look_at() {
        let lookfrom = Point::new(-2.0, 2.0, 1.0);
        let lookat = Point::new(0.0, 0.0, -1.0);
        let camera = RayCamera::new(101, 1.0)
            .with_look_at(lookfrom, lookat)
            .with_view_up(Vector::new(0.0, 1.0, 0.0));

        let ray = camera.ray_for_pixel(50, 50);
        let expected_direction = (lookat - lookfrom).normalized();
        let actual_direction = ray.direction().normalized();

        assert_eq!(*ray.origin(), lookfrom);
        assert_close(actual_direction.dot(expected_direction), 1.0);
    }
    #[test]
    fn ray_camera_defocus_blur_offsets_sample_origin() {
        let mut rng = SampleRng::new(17);
        let pinhole = RayCamera::new(101, 1.0);
        let defocused = pinhole.with_defocus_angle(10.0).with_focus_distance(3.4);

        let pinhole_ray = pinhole.ray_for_pixel_sample(50, 50, &mut rng);
        let defocused_ray = defocused.ray_for_pixel_sample(50, 50, &mut rng);

        assert_eq!(*pinhole_ray.origin(), pinhole.camera_center());
        assert_ne!(*defocused_ray.origin(), defocused.camera_center());
    }

    #[test]
    fn ray_camera_samples_shutter_interval() {
        let mut rng = SampleRng::new(19);
        let camera = RayCamera::new(101, 1.0).with_shutter_interval(0.25, 0.75);
        let center_ray = camera.ray_for_pixel(50, 50);
        let sampled_ray = camera.ray_for_pixel_sample(50, 50, &mut rng);

        assert_close(center_ray.time(), 0.25);
        assert!((0.25..0.75).contains(&sampled_ray.time()));
        assert_eq!(camera.shutter_interval(), (0.25, 0.75));
    }

    #[test]
    fn ray_camera_world_render_is_seeded_and_deterministic() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let camera = RayCamera::new(20, 16.0 / 9.0)
            .with_samples_per_pixel(4)
            .with_max_depth(3)
            .with_rng_seed(123);

        let first = camera.render_world(&world);
        let second = camera.render_world(&world);

        assert_eq!(first.pixels(), second.pixels());
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_spectral_render_is_seeded_and_deterministic() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let camera = RayCamera::new(8, 1.0)
            .with_samples_per_pixel(4)
            .with_max_depth(3)
            .with_rng_seed(1234);

        let first = camera.render_world_spectral(&world);
        let second = camera.render_world_spectral(&world);

        assert_eq!(first.pixels(), second.pixels());
        assert!(first.pixels().iter().any(|pixel| *pixel != Rgb::BLACK));
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_default_render_transport_is_rgb_compatible() {
        let camera = RayCamera::new(2, 1.0);

        assert_eq!(
            camera.render_transport_mode(),
            crate::graphics::raytracing::RenderTransportMode::Rgb
        );
        assert_eq!(
            camera.spectral_transport_mode(),
            crate::graphics::raytracing::SpectralTransportMode::Polarized
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_default_render_can_use_spectral_transport() {
        let world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0)
            .with_samples_per_pixel(4)
            .with_rng_seed(81)
            .with_background(LinearColor::new(0.25, 0.5, 0.75))
            .with_spectral_render_transport();

        let default_hdr = camera.render_world_hdr_image(&world);
        let explicit_spectral = camera.render_world_spectral_image(&world).to_hdr_image();
        let default_canvas = camera.render_world(&world);
        let explicit_canvas = camera.render_world_spectral(&world);

        assert_eq!(
            camera.render_transport_mode(),
            crate::graphics::raytracing::RenderTransportMode::Spectral
        );
        assert_eq!(default_hdr.pixels(), explicit_spectral.pixels());
        assert_eq!(default_canvas.pixels(), explicit_canvas.pixels());
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_progress_render_paths_can_use_spectral_transport() {
        let world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0)
            .with_samples_per_pixel(4)
            .with_rng_seed(82)
            .with_background(LinearColor::new(0.25, 0.5, 0.75))
            .with_spectral_render_transport();
        let explicit_spectral = camera.render_world_spectral(&world);
        let mut log = Vec::new();
        let mut tile_updates = 0;

        let scanline_progress = camera
            .render_world_with_progress(&world, &mut log)
            .expect("scanline progress render should write");
        let tiled_progress = camera
            .render_world_tiled_progressive(&world, 1, |_| {
                tile_updates += 1;
                Ok::<_, std::convert::Infallible>(())
            })
            .expect("infallible progress callback");

        assert_eq!(scanline_progress.pixels(), explicit_spectral.pixels());
        assert_eq!(tiled_progress.pixels(), explicit_spectral.pixels());
        assert_eq!(tile_updates, 4);
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_spectral_image_exposes_linear_environment_render() {
        let environment = crate::graphics::raytracing::EnvironmentLight::constant(
            LinearColor::new(0.4, 0.4, 0.4),
        );
        let empty_world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0)
            .with_samples_per_pixel(4)
            .with_rng_seed(77)
            .with_background(LinearColor::default());

        let image = camera.render_world_with_environment_spectral_image(&empty_world, &environment);
        let canvas = camera.render_world_with_environment_spectral(&empty_world, &environment);

        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert_eq!(image.linear_rgb.len(), 4);
        assert_eq!(
            image.transport_mode,
            crate::graphics::raytracing::SpectralTransportMode::Polarized
        );
        assert_eq!(
            image
                .polarization
                .as_ref()
                .expect("default spectral render should store Stokes samples")
                .len(),
            4
        );
        assert_eq!(image.to_canvas().pixels(), canvas.pixels());
        assert!(
            image
                .linear_rgb
                .iter()
                .any(|pixel| pixel.max_component() > 0.0)
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_spectral_render_accepts_borrowed_background_source() {
        let empty_world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0)
            .with_samples_per_pixel(8)
            .with_rng_seed(79)
            .with_background(LinearColor::default())
            .with_unpolarized_spectral_transport();
        let background = |_direction: Vector| LinearColor::new(0.25, 0.25, 0.25);

        let image = camera.render_world_with_background_spectral_image(&empty_world, &background);
        let canvas = camera.render_world_with_background_spectral(&empty_world, &background);

        assert_eq!(
            image.transport_mode,
            crate::graphics::raytracing::SpectralTransportMode::Unpolarized
        );
        assert_eq!(image.to_canvas().pixels(), canvas.pixels());
        assert!(
            image
                .linear_rgb
                .iter()
                .any(|pixel| pixel.max_component() > 0.0)
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ray_camera_polarized_spectral_image_exposes_stokes_output() {
        let environment = crate::graphics::raytracing::EnvironmentLight::constant(
            LinearColor::new(0.4, 0.4, 0.4),
        );
        let empty_world = crate::graphics::raytracing::HittableList::new();
        let camera = RayCamera::new(2, 1.0)
            .with_samples_per_pixel(4)
            .with_rng_seed(78)
            .with_background(LinearColor::default())
            .with_polarized_spectral_transport();

        let image = camera.render_world_with_environment_spectral_image(&empty_world, &environment);

        assert_eq!(
            image.transport_mode,
            crate::graphics::raytracing::SpectralTransportMode::Polarized
        );
        let polarization = image
            .polarization
            .as_ref()
            .expect("polarized render should store Stokes samples");
        assert_eq!(polarization.len(), 4);
        assert!(polarization.iter().all(|stokes| stokes.is_finite()));
        assert!(
            image
                .polarization_pixel(0, 0)
                .is_some_and(|stokes| stokes.degree_of_polarization() <= 1.0e-12)
        );
    }

    #[test]
    fn ray_camera_world_render_with_progress_matches_default_render() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let camera = RayCamera::new(8, 1.0)
            .with_samples_per_pixel(2)
            .with_max_depth(3)
            .with_rng_seed(321);
        let expected = camera.render_world(&world);
        let mut log = Vec::new();

        let actual = camera
            .render_world_with_progress(&world, &mut log)
            .expect("progress render should write");

        assert_eq!(actual.pixels(), expected.pixels());
        assert!(String::from_utf8_lossy(&log).contains("Scanlines remaining"));
    }

    #[test]
    fn ray_camera_progressive_tiled_render_matches_default_render() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let camera = RayCamera::new(8, 1.0)
            .with_samples_per_pixel(2)
            .with_max_depth(3)
            .with_rng_seed(654);
        let expected = camera.render_world_tiled(&world, 3);
        let mut updates = Vec::new();

        let actual = camera
            .render_world_tiled_progressive(&world, 3, |update| {
                let progress = update.progress();
                assert_eq!(update.image_width(), 8);
                assert_eq!(update.image_height(), 8);
                assert_eq!(update.pixels().len(), expected.pixels().len());
                updates.push(progress);
                Ok::<_, std::convert::Infallible>(())
            })
            .expect("infallible progress callback");

        assert_eq!(actual.pixels(), expected.pixels());
        assert_eq!(updates.len(), 9);
        assert!(
            updates
                .last()
                .is_some_and(|progress| progress.is_complete())
        );
    }

    #[test]
    fn ray_camera_world_render_with_lights_is_seeded_and_deterministic() {
        use crate::graphics::raytracing::{DiffuseLight, HittableList, Lambertian, Quad, Sphere};

        let mut world = HittableList::new();
        world.add(Sphere::with_material(
            Point::new(0.0, 0.0, -1.0),
            0.5,
            Lambertian::new(LinearColor::new(0.7, 0.7, 0.7)),
        ));
        world.add(Quad::with_material(
            Point::new(-0.5, 1.0, -1.5),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
            DiffuseLight::new(LinearColor::new(4.0, 4.0, 4.0)),
        ));

        let mut lights = HittableList::new();
        lights.add(Quad::new(
            Point::new(-0.5, 1.0, -1.5),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ));

        let camera = RayCamera::new(6, 1.0)
            .with_samples_per_pixel(4)
            .with_max_depth(4)
            .with_background(LinearColor::default())
            .with_rng_seed(321);

        let first = camera.render_world_with_lights(&world, &lights);
        let second = camera.render_world_with_lights(&world, &lights);

        assert_eq!(first.pixels(), second.pixels());
    }

    #[test]
    fn ray_camera_next_event_estimation_render_is_seeded_and_deterministic() {
        use crate::graphics::raytracing::{DiffuseLight, HittableList, Lambertian, Quad, Sphere};

        let mut world = HittableList::new();
        world.add(Sphere::with_material(
            Point::new(0.0, 0.0, -1.0),
            0.5,
            Lambertian::new(LinearColor::new(0.7, 0.7, 0.7)),
        ));
        world.add(Quad::with_material(
            Point::new(-0.5, 1.0, -1.5),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
            DiffuseLight::new(LinearColor::new(4.0, 4.0, 4.0)),
        ));

        let mut lights = HittableList::new();
        lights.add(Quad::new(
            Point::new(-0.5, 1.0, -1.5),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ));

        let camera = RayCamera::new(6, 1.0)
            .with_samples_per_pixel(4)
            .with_max_depth(4)
            .with_background(LinearColor::default())
            .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation)
            .with_rng_seed(321);

        assert_eq!(
            camera.direct_lighting_mode(),
            DirectLightingMode::NextEventEstimation
        );
        let first = camera.render_world_with_lights(&world, &lights);
        let second = camera.render_world_with_lights(&world, &lights);

        assert_eq!(first.pixels(), second.pixels());
    }

    #[test]
    fn ray_camera_light_connection_render_forces_next_event_estimation() {
        use crate::graphics::raytracing::{DiffuseLight, HittableList, Lambertian, Quad, Sphere};

        let mut world = HittableList::new();
        world.add(Sphere::with_material(
            Point::new(0.0, 0.0, -1.0),
            0.5,
            Lambertian::new(LinearColor::new(0.7, 0.7, 0.7)),
        ));
        world.add(Quad::with_material(
            Point::new(-0.5, 1.0, -1.5),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
            DiffuseLight::new(LinearColor::new(4.0, 4.0, 4.0)),
        ));

        let mut lights = HittableList::new();
        lights.add(Quad::new(
            Point::new(-0.5, 1.0, -1.5),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ));

        let camera = RayCamera::new(6, 1.0)
            .with_samples_per_pixel(4)
            .with_max_depth(4)
            .with_background(LinearColor::default())
            .with_direct_lighting_mode(DirectLightingMode::CurrentPathContinuation)
            .with_rng_seed(321);
        let expected = camera
            .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation)
            .render_world_with_lights(&world, &lights);
        let actual = camera.render_world_with_light_connections(&world, &lights);
        let tiled = camera.render_world_with_light_connections_tiled(&world, &lights, 2);
        let mut updates = 0;
        let progressive = camera
            .render_world_with_light_connections_tiled_progressive(&world, &lights, 2, |update| {
                updates += 1;
                assert_eq!(update.image_width(), 6);
                Ok::<_, std::convert::Infallible>(())
            })
            .expect("infallible progress callback");

        assert_eq!(actual.pixels(), expected.pixels());
        assert_eq!(tiled.pixels(), expected.pixels());
        assert_eq!(progressive.pixels(), expected.pixels());
        assert_eq!(updates, 9);
    }

    #[test]
    fn ray_camera_world_render_uses_image_coordinate_canvas() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let canvas = RayCamera::new(4, 1.0).render_world(&world);

        assert!(canvas.upper_left_origin());
        assert!(!canvas.wrapped());
        assert!(canvas.zbuffer().is_empty());
    }
}

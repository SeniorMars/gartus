use super::colors::Rgb;
use crate::gmath::random::SampleRng;
use crate::gmath::ray::Ray;
use crate::gmath::{
    geometry::CameraPose,
    vector::{Point, Vector},
};
use crate::graphics::raytracing::{
    HitRecord, Hittable, INFINITY, Interval, LinearColor, PdfContext, ScatterRecord, component_mul,
    degrees_to_radians,
};
use crate::graphics::raytracing::{
    HittablePdf, MixturePdf, Pdf, SHADOW_ACNE_EPSILON, scenes::normal_scene_color,
};
use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::display::Canvas,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::io::{self, Write};

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
    direct_lighting_mode: DirectLightingMode,
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
    background: LinearColor,
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
    direct_lighting_mode: DirectLightingMode,
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
    background: LinearColor,
}

#[derive(Clone, Copy)]
struct RayColorContext<'a> {
    world: &'a dyn Hittable,
    lights: Option<&'a dyn Hittable>,
    direct_lighting_mode: DirectLightingMode,
    background: LinearColor,
    russian_roulette_min_depth: Option<u32>,
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

        Self::initialized(RayCameraParams {
            image_width,
            aspect_ratio,
            samples_per_pixel: 1,
            pixel_sample_mode: PixelSampleMode::Random,
            adaptive_sampling: None,
            direct_lighting_mode: DirectLightingMode::CurrentPathContinuation,
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
            background: LinearColor::new(0.70, 0.80, 1.00),
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn initialized(params: RayCameraParams) -> Self {
        Self::validate_view(&params);

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
            direct_lighting_mode: params.direct_lighting_mode,
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
        Self::initialized(RayCameraParams {
            image_width: self.image_width,
            aspect_ratio: self.aspect_ratio,
            samples_per_pixel: self.samples_per_pixel,
            pixel_sample_mode: self.pixel_sample_mode,
            adaptive_sampling: self.adaptive_sampling,
            direct_lighting_mode: self.direct_lighting_mode,
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
            params.background.x().is_finite()
                && params.background.y().is_finite()
                && params.background.z().is_finite(),
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
        self.direct_lighting_mode = mode;
        self
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
        assert!(
            background.x().is_finite() && background.y().is_finite() && background.z().is_finite(),
            "background color components must be finite"
        );
        self.background = background;
        self
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
        self.direct_lighting_mode
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

    /// Returns the color used when world-rendered rays miss the scene.
    #[must_use]
    pub fn background(self) -> LinearColor {
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

        for bounce_index in 0..depth {
            let Some(record) = context.world.hit_with_rng(
                &current_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            ) else {
                return color + component_mul(attenuation, context.background);
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
                    let (scattered_direction, pdf_value, suppress_next_emission) =
                        match (context.direct_lighting_mode, context.lights) {
                            (DirectLightingMode::CurrentPathContinuation, Some(lights)) => {
                                let light_pdf = HittablePdf::new(
                                    lights,
                                    PdfContext::new(record.point, current_ray.time()),
                                );
                                let mixture_pdf = MixturePdf::new(light_pdf, material_pdf);
                                let direction = mixture_pdf.generate(rng);
                                (direction, mixture_pdf.value(direction), false)
                            }
                            (DirectLightingMode::NextEventEstimation, Some(lights)) => {
                                let direct = Self::estimate_direct_lighting(
                                    &current_ray,
                                    &record,
                                    context.world,
                                    lights,
                                    scatter_attenuation,
                                    rng,
                                );
                                color += component_mul(attenuation, direct);

                                let direction = material_pdf.generate(rng);
                                (direction, material_pdf.value(direction), true)
                            }
                            (_, None) => {
                                let direction = material_pdf.generate(rng);
                                (direction, material_pdf.value(direction), false)
                            }
                        };

                    if !pdf_value.is_finite() || pdf_value <= f64::EPSILON {
                        return color;
                    }

                    let scattered_ray =
                        Ray::with_time(record.point, scattered_direction, current_ray.time());
                    let scattering_pdf =
                        record
                            .material
                            .scattering_pdf(&current_ray, &record, &scattered_ray);
                    if !scattering_pdf.is_finite() || scattering_pdf <= 0.0 {
                        return color;
                    }

                    attenuation = component_mul(
                        attenuation,
                        scatter_attenuation * (scattering_pdf / pdf_value),
                    );
                    current_ray = scattered_ray;
                    allow_emitted = !suppress_next_emission;
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

    fn ray_color_context<'a>(
        self,
        world: &'a dyn Hittable,
        lights: Option<&'a dyn Hittable>,
    ) -> RayColorContext<'a> {
        RayColorContext {
            world,
            lights,
            direct_lighting_mode: self.direct_lighting_mode,
            background: self.background,
            russian_roulette_min_depth: self.russian_roulette_min_depth,
        }
    }

    fn emitted_at(ray: &Ray, hit: &HitRecord<'_>) -> LinearColor {
        hit.material.emitted(ray, hit, hit.u, hit.v, hit.point)
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

    fn render_world_pixel(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
    ) -> Rgb {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        let sample_count = self.effective_samples_per_pixel();

        match self.pixel_sample_mode {
            PixelSampleMode::Random => {
                if let Some(settings) = self.adaptive_sampling {
                    pixel_color = self.render_world_pixel_adaptive(x, y, world, lights, settings);
                } else {
                    let mut accepted_samples = 0;
                    for _ in 0..sample_count {
                        accepted_samples += u32::from(Self::add_finite_sample(
                            &mut pixel_color,
                            self.sample_world_color(x, y, world, lights, &mut rng),
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
                                self.ray_color_context(world, lights),
                                &mut rng,
                            ),
                        ));
                    }
                }
                pixel_color = Self::average_accepted_samples(pixel_color, accepted_samples);
            }
        }

        Rgb::from_linear_color(pixel_color)
    }

    fn sample_world_color(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let ray = self.ray_for_pixel_sample(x, y, rng);
        Self::ray_color(
            &ray,
            self.max_depth,
            self.ray_color_context(world, lights),
            rng,
        )
    }

    fn render_world_pixel_adaptive(
        self,
        x: u32,
        y: u32,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        settings: AdaptiveSampling,
    ) -> LinearColor {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut mean = LinearColor::default();
        let mut m2 = LinearColor::default();
        let mut accepted_samples = 0;

        for _ in 0..settings.max_samples {
            let sample = self.sample_world_color(x, y, world, lights, &mut rng);
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

    fn image_canvas(width: u32, height: u32, pixels: Vec<Rgb>) -> Canvas {
        Canvas::from_pixels_rgb_only(width, height, pixels, true, false)
    }

    fn render_pixels_tiled<F>(width: u32, height: u32, tile_size: u32, pixel: F) -> Vec<Rgb>
    where
        F: Fn(u32, u32) -> Rgb + Sync,
    {
        let mut pixels = vec![Rgb::default(); Canvas::pixel_count(width, height)];
        let width = usize::try_from(width).expect("image width should fit usize");
        let height = usize::try_from(height).expect("image height should fit usize");
        let tile_size = usize::try_from(tile_size.max(1)).expect("tile size should fit usize");
        let band_len = width
            .checked_mul(tile_size)
            .expect("tile band pixel count should fit usize");

        #[cfg(feature = "rayon")]
        {
            pixels
                .par_chunks_mut(band_len)
                .enumerate()
                .for_each(|(band_index, band_pixels)| {
                    let y_start = band_index * tile_size;
                    Self::render_tile_band(band_pixels, width, height, y_start, tile_size, &pixel);
                });
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (band_index, band_pixels) in pixels.chunks_mut(band_len).enumerate() {
                let y_start = band_index * tile_size;
                Self::render_tile_band(band_pixels, width, height, y_start, tile_size, &pixel);
            }
        }

        pixels
    }

    fn render_tile_band<F>(
        band_pixels: &mut [Rgb],
        image_width: usize,
        image_height: usize,
        y_start: usize,
        tile_size: usize,
        pixel: &F,
    ) where
        F: Fn(u32, u32) -> Rgb,
    {
        debug_assert!(image_width > 0);
        debug_assert_eq!(band_pixels.len() % image_width, 0);
        let rows = band_pixels.len() / image_width;
        for tile_x in (0..image_width).step_by(tile_size) {
            let x_end = tile_x.saturating_add(tile_size).min(image_width);
            for row in 0..rows {
                let y = y_start + row;
                if y >= image_height {
                    return;
                }
                let row_offset = row * image_width;
                for x in tile_x..x_end {
                    band_pixels[row_offset + x] = pixel(
                        u32::try_from(x).expect("pixel x should fit u32"),
                        u32::try_from(y).expect("pixel y should fit u32"),
                    );
                }
            }
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
        self.render_world_with_optional_lights(world, None)
    }

    /// Renders a hittable world in tile-height bands.
    ///
    /// When the `rayon` feature is enabled, tile bands are rendered independently in parallel. Use
    /// this to tune tile size for previews or cache behavior; [`Self::render_world`] uses a
    /// conservative default tile size.
    pub fn render_world_tiled(self, world: &dyn Hittable, tile_size: u32) -> Canvas {
        self.render_world_with_optional_lights_tiled(world, None, tile_size)
    }

    /// Renders a hittable world while importance-sampling directions toward `lights`.
    ///
    /// Pass a lights-only or otherwise important target set here; passing the full scene is valid
    /// but usually raises variance by sampling non-emissive geometry.
    pub fn render_world_with_lights(self, world: &dyn Hittable, lights: &dyn Hittable) -> Canvas {
        self.render_world_with_optional_lights(world, Some(lights))
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
        self.render_world_with_optional_lights_tiled(world, Some(lights), tile_size)
    }

    fn render_world_with_optional_lights(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
    ) -> Canvas {
        self.render_world_with_optional_lights_tiled(world, lights, DEFAULT_RENDER_TILE_SIZE)
    }

    fn render_world_with_optional_lights_tiled(
        self,
        world: &dyn Hittable,
        lights: Option<&dyn Hittable>,
        tile_size: u32,
    ) -> Canvas {
        let camera = self.initialize();
        let pixels = Self::render_pixels_tiled(
            camera.image_width,
            camera.image_height,
            tile_size,
            |x, y| camera.render_world_pixel(x, y, world, lights),
        );
        Self::image_canvas(camera.image_width, camera.image_height, pixels)
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
                            *pixel = camera.render_world_pixel(
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
                        *pixel = camera.render_world_pixel(
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
    use crate::graphics::raytracing::{HitRecord, Material, SurfaceHit};
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
        assert_eq!(canvas.pixels()[0], Rgb::from_linear_color(background));
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
    fn ray_camera_world_render_uses_image_coordinate_canvas() {
        let world = crate::graphics::raytracing::scenes::normal_sphere_world();
        let canvas = RayCamera::new(4, 1.0).render_world(&world);

        assert!(canvas.upper_left_origin());
        assert!(!canvas.wrapped());
        assert!(canvas.zbuffer().is_empty());
    }
}

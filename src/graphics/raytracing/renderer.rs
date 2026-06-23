//! Path-tracing render entrypoints.
//!
//! [`PathTracer`] is the high-level wrapper around [`RayCamera`]. Use
//! [`PathTracer::render_scene`] as a one-shot convenience for renderer-neutral [`SurfaceScene`]
//! content. For repeated renders of the same surface scene, compile once with
//! [`SurfaceScene::to_ray_scene`] and pass the resulting [`RayScene`] to
//! [`PathTracer::render_ray_scene`], [`PathTracer::render`], or
//! [`PathTracer::render_with_lights`]. [`PathTracer::render_with_light_connections`] forces
//! next-event light connections for cameras configured with material-PDF path continuation.

#[cfg(feature = "spectral")]
use super::SpectralImage;
use super::{EnvironmentLight, Hittable, RayScene};
use crate::graphics::{
    camera::{DenoisingAovs, ProgressiveRenderUpdate, RayBackgroundSource, RayCamera},
    display::{Canvas, HdrImage, ToneMap},
    scene::SurfaceScene,
};

/// Path-tracer image traversal options.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderOptions {
    tile_size: Option<u32>,
}

impl RenderOptions {
    /// Uses the camera's default image traversal settings.
    #[must_use]
    pub const fn new() -> Self {
        Self { tile_size: None }
    }

    /// Sets a tile size for world renders.
    #[must_use]
    pub const fn tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = Some(if tile_size == 0 { 1 } else { tile_size });
        self
    }

    /// Returns the configured tile size, if any.
    #[must_use]
    pub const fn tile_size_override(self) -> Option<u32> {
        self.tile_size
    }
}

/// Path-tracing renderer wrapper around a ray camera.
///
/// Use [`Self::render_scene`] for one-shot renderer-neutral mesh scenes. Use [`Self::render`] and
/// [`Self::render_with_lights`] when you already have a low-level [`RayScene`] or custom
/// [`Hittable`] world, especially for repeated renders where the compiled `RayScene` can reuse its
/// cached BVH. Indoor or small-light scenes usually converge faster with
/// [`Self::render_with_lights`] and a lights-only
/// [`SamplingTargetList`](crate::graphics::raytracing::SamplingTargetList).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathTracer {
    camera: RayCamera,
    options: RenderOptions,
}

impl PathTracer {
    /// Creates a path tracer using `camera`.
    #[must_use]
    pub const fn new(camera: RayCamera) -> Self {
        Self {
            camera,
            options: RenderOptions::new(),
        }
    }

    /// Returns the camera used by this tracer.
    #[must_use]
    pub const fn camera(self) -> RayCamera {
        self.camera
    }

    /// Replaces the camera.
    #[must_use]
    pub const fn with_camera(mut self, camera: RayCamera) -> Self {
        self.camera = camera;
        self
    }

    /// Returns this tracer's render options.
    #[must_use]
    pub const fn options(self) -> RenderOptions {
        self.options
    }

    /// Replaces this tracer's render options.
    #[must_use]
    pub const fn with_options(mut self, options: RenderOptions) -> Self {
        self.options = options;
        self
    }

    /// Renders `world` with path-traced material scattering.
    ///
    /// This samples only the material PDFs. Prefer [`Self::render_with_lights`] for scenes where
    /// small emitters, windows, or glass caustic targets should drive importance sampling.
    pub fn render(self, world: &dyn Hittable) -> Canvas {
        self.render_hdr_image(world).to_canvas()
    }

    /// Renders `world` to display RGB using explicit tone-mapping controls.
    pub fn render_tone_mapped(self, world: &dyn Hittable, tone_map: ToneMap) -> Canvas {
        self.render_hdr_image(world).to_canvas_tone_mapped(tone_map)
    }

    /// Renders `world` to linear floating-point HDR samples.
    #[must_use]
    pub fn render_hdr_image(self, world: &dyn Hittable) -> HdrImage {
        self.options.tile_size.map_or_else(
            || self.camera.render_world_hdr_image(world),
            |tile_size| self.camera.render_world_hdr_image_tiled(world, tile_size),
        )
    }

    /// Renders beauty, albedo, and normal buffers for denoising.
    #[must_use]
    pub fn render_denoising_aovs(self, world: &dyn Hittable) -> DenoisingAovs {
        self.options.tile_size.map_or_else(
            || self.camera.render_world_denoising_aovs(world),
            |tile_size| {
                self.camera
                    .render_world_denoising_aovs_tiled(world, tile_size)
            },
        )
    }

    /// Renders `world` in tiles and calls `progress` as partial image tiles complete.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_progressive<P, E>(self, world: &dyn Hittable, progress: P) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_tiled_progressive(world, tile_size, progress),
            None => self.camera.render_world_progressive(world, progress),
        }
    }

    /// Renders `world` with the feature-gated sampled-wavelength spectral prototype.
    ///
    /// [`Spectrum::Rgb`](crate::graphics::raytracing::Spectrum::Rgb) keeps RGB assets compatible;
    /// measured reflectance, emission, eta, and k data should use
    /// [`MeasuredSpectrum`](crate::graphics::raytracing::MeasuredSpectrum).
    #[cfg(feature = "spectral")]
    pub fn render_spectral(self, world: &dyn Hittable) -> Canvas {
        match self.options.tile_size {
            Some(tile_size) => self.camera.render_world_spectral_tiled(world, tile_size),
            None => self.camera.render_world_spectral(world),
        }
    }

    /// Renders `world` to a linear sampled-wavelength spectral image.
    ///
    /// The returned image preserves linear spectral transport output. RGB-backed spectra remain
    /// available as a fallback, and measured spectra can be supplied through material constructors.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_spectral_image(self, world: &dyn Hittable) -> SpectralImage {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_spectral_image_tiled(world, tile_size),
            None => self.camera.render_world_spectral_image(world),
        }
    }

    /// Renders a compiled ray scene, automatically importance-sampling emissive primitives.
    ///
    /// This is the simplest entry point for [`RayScene`] content. It builds a light-target list
    /// from [`RayScene::emissive_targets`] and calls [`Self::render_with_lights`] when that list is
    /// non-empty; scenes without emissive primitives render through [`Self::render`].
    pub fn render_ray_scene(self, scene: &RayScene) -> Canvas {
        self.render_ray_scene_hdr_image(scene).to_canvas()
    }

    /// Renders a compiled ray scene to display RGB using explicit tone-mapping controls.
    pub fn render_ray_scene_tone_mapped(self, scene: &RayScene, tone_map: ToneMap) -> Canvas {
        self.render_ray_scene_hdr_image(scene)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Renders a compiled ray scene to linear floating-point HDR samples.
    #[must_use]
    pub fn render_ray_scene_hdr_image(self, scene: &RayScene) -> HdrImage {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render_hdr_image(scene)
        } else {
            self.render_with_lights_hdr_image(scene, &lights)
        }
    }

    /// Renders a compiled ray scene with forced next-event light connections.
    ///
    /// Emissive primitives are collected as light targets. Scenes without emissive primitives fall
    /// back to the ordinary path tracer.
    pub fn render_ray_scene_with_light_connections(self, scene: &RayScene) -> Canvas {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render(scene)
        } else {
            self.render_with_light_connections(scene, &lights)
        }
    }

    /// Renders a compiled ray scene with denoising AOVs and automatic emissive-target sampling.
    #[must_use]
    pub fn render_ray_scene_denoising_aovs(self, scene: &RayScene) -> DenoisingAovs {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render_denoising_aovs(scene)
        } else {
            self.render_with_lights_denoising_aovs(scene, &lights)
        }
    }

    /// Renders a compiled ray scene progressively with forced next-event light connections.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_ray_scene_with_light_connections_progressive<P, E>(
        self,
        scene: &RayScene,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render_progressive(scene, progress)
        } else {
            self.render_with_light_connections_progressive(scene, &lights, progress)
        }
    }

    /// Renders a compiled ray scene progressively, automatically importance-sampling emitters.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_ray_scene_progressive<P, E>(
        self,
        scene: &RayScene,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render_progressive(scene, progress)
        } else {
            self.render_with_lights_progressive(scene, &lights, progress)
        }
    }

    /// Renders a compiled ray scene with the sampled-wavelength spectral prototype.
    #[cfg(feature = "spectral")]
    pub fn render_ray_scene_spectral(self, scene: &RayScene) -> Canvas {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render_spectral(scene)
        } else {
            self.render_with_lights_spectral(scene, &lights)
        }
    }

    /// Renders a compiled ray scene to a linear sampled-wavelength spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_ray_scene_spectral_image(self, scene: &RayScene) -> SpectralImage {
        let lights = scene.emissive_targets();
        if lights.is_empty() {
            self.render_spectral_image(scene)
        } else {
            self.render_with_lights_spectral_image(scene, &lights)
        }
    }

    /// Compiles and renders a renderer-neutral surface scene.
    ///
    /// Surface meshes are converted to a [`RayScene`] with Lambertian materials derived from the
    /// shared [`SurfaceScene`] material data. This convenience helper recompiles the `RayScene` and
    /// its BVH every call; for animation, camera iteration, or sample-count iteration, prefer:
    ///
    /// ```rust
    /// use gartus::prelude::*;
    ///
    /// let surface_scene = SurfaceScene::new();
    /// let ray_scene = surface_scene.to_ray_scene();
    /// let camera = RayCamera::new(100, 1.0);
    /// let image = PathTracer::new(camera).render(&ray_scene);
    /// assert_eq!(image.width(), 100);
    /// ```
    ///
    /// Diffuse texture paths on surface materials are retained as metadata and are not loaded by
    /// this helper.
    pub fn render_scene(self, scene: &SurfaceScene) -> Canvas {
        self.render_scene_hdr_image(scene).to_canvas()
    }

    /// Compiles and renders a renderer-neutral surface scene with explicit tone mapping.
    pub fn render_scene_tone_mapped(self, scene: &SurfaceScene, tone_map: ToneMap) -> Canvas {
        self.render_scene_hdr_image(scene)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Compiles and renders a renderer-neutral surface scene to linear HDR samples.
    #[must_use]
    pub fn render_scene_hdr_image(self, scene: &SurfaceScene) -> HdrImage {
        let ray_scene = RayScene::from(scene);
        self.render_hdr_image(&ray_scene)
    }

    /// Compiles and renders a renderer-neutral surface scene with spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_scene_spectral(self, scene: &SurfaceScene) -> Canvas {
        let ray_scene = RayScene::from(scene);
        self.render_spectral(&ray_scene)
    }

    /// Compiles and renders a surface scene to a linear sampled-wavelength spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_scene_spectral_image(self, scene: &SurfaceScene) -> SpectralImage {
        let ray_scene = RayScene::from(scene);
        self.render_spectral_image(&ray_scene)
    }

    /// Compiles and renders a renderer-neutral surface scene with denoising AOVs.
    #[must_use]
    pub fn render_scene_denoising_aovs(self, scene: &SurfaceScene) -> DenoisingAovs {
        let ray_scene = RayScene::from(scene);
        self.render_denoising_aovs(&ray_scene)
    }

    /// Compiles and renders a surface scene while importance-sampling `lights`.
    ///
    /// Use this one-shot helper when a renderer-neutral scene still needs explicit path-tracing
    /// sampling targets. It recompiles the `RayScene` and its BVH every call; compile the scene
    /// once and call [`Self::render_with_lights`] for repeated renders.
    pub fn render_scene_with_lights(self, scene: &SurfaceScene, lights: &dyn Hittable) -> Canvas {
        let ray_scene = RayScene::from(scene);
        self.render_with_lights(&ray_scene, lights)
    }

    /// Compiles and renders a surface scene with forced next-event light connections.
    ///
    /// This one-shot helper uses the provided `lights` as explicit light sampling targets.
    pub fn render_scene_with_light_connections(
        self,
        scene: &SurfaceScene,
        lights: &dyn Hittable,
    ) -> Canvas {
        let ray_scene = RayScene::from(scene);
        self.render_with_light_connections(&ray_scene, lights)
    }

    /// Renders `world` while importance-sampling directions toward `lights`.
    ///
    /// Pass a lights-only or otherwise important target set here; passing the full scene is valid
    /// but usually raises variance by sampling non-emissive geometry.
    pub fn render_with_lights(self, world: &dyn Hittable, lights: &dyn Hittable) -> Canvas {
        self.render_with_lights_hdr_image(world, lights).to_canvas()
    }

    /// Renders `world` with explicit light sampling and tone-mapping controls.
    pub fn render_with_lights_tone_mapped(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        tone_map: ToneMap,
    ) -> Canvas {
        self.render_with_lights_hdr_image(world, lights)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Renders `world` with explicit light sampling to linear HDR samples.
    #[must_use]
    pub fn render_with_lights_hdr_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> HdrImage {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_lights_hdr_image(world, lights)
            },
            |tile_size| {
                self.camera
                    .render_world_with_lights_hdr_image_tiled(world, lights, tile_size)
            },
        )
    }

    /// Renders `world` with an importance-sampled environment light.
    pub fn render_with_environment(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        self.render_with_environment_hdr_image(world, environment)
            .to_canvas()
    }

    /// Renders `world` with an importance-sampled environment and tone-mapping controls.
    pub fn render_with_environment_tone_mapped(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
        tone_map: ToneMap,
    ) -> Canvas {
        self.render_with_environment_hdr_image(world, environment)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Renders `world` with an importance-sampled environment light to linear HDR samples.
    #[must_use]
    pub fn render_with_environment_hdr_image(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> HdrImage {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_environment_hdr_image(world, environment)
            },
            |tile_size| {
                self.camera.render_world_with_environment_hdr_image_tiled(
                    world,
                    environment,
                    tile_size,
                )
            },
        )
    }

    /// Renders `world` with a custom background source for miss radiance.
    pub fn render_with_background(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        self.render_with_background_hdr_image(world, background)
            .to_canvas()
    }

    /// Renders `world` with a custom background source and tone-mapping controls.
    pub fn render_with_background_tone_mapped(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tone_map: ToneMap,
    ) -> Canvas {
        self.render_with_background_hdr_image(world, background)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Renders `world` with a custom background source to linear HDR samples.
    #[must_use]
    pub fn render_with_background_hdr_image(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> HdrImage {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_background_hdr_image(world, background)
            },
            |tile_size| {
                self.camera
                    .render_world_with_background_hdr_image_tiled(world, background, tile_size)
            },
        )
    }

    /// Renders `world` with an importance-sampled environment light and spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_with_environment_spectral(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        match self.options.tile_size {
            Some(tile_size) => self.camera.render_world_with_environment_spectral_tiled(
                world,
                environment,
                tile_size,
            ),
            None => self
                .camera
                .render_world_with_environment_spectral(world, environment),
        }
    }

    /// Renders `world` with an environment light to a linear spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_with_environment_spectral_image(
        self,
        world: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> SpectralImage {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_environment_spectral_image_tiled(world, environment, tile_size),
            None => self
                .camera
                .render_world_with_environment_spectral_image(world, environment),
        }
    }

    /// Renders `world` with a custom background source and spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_with_background_spectral(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_background_spectral_tiled(world, background, tile_size),
            None => self
                .camera
                .render_world_with_background_spectral(world, background),
        }
    }

    /// Renders `world` with a custom background source to a linear spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_with_background_spectral_image(
        self,
        world: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> SpectralImage {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_background_spectral_image_tiled(world, background, tile_size),
            None => self
                .camera
                .render_world_with_background_spectral_image(world, background),
        }
    }

    /// Renders `world` with geometry lights plus an importance-sampled environment light.
    pub fn render_with_lights_and_environment(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        self.render_with_lights_and_environment_hdr_image(world, lights, environment)
            .to_canvas()
    }

    /// Renders `world` with geometry lights, environment sampling, and tone mapping.
    pub fn render_with_lights_and_environment_tone_mapped(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
        tone_map: ToneMap,
    ) -> Canvas {
        self.render_with_lights_and_environment_hdr_image(world, lights, environment)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Renders `world` with geometry lights and an environment light to linear HDR samples.
    #[must_use]
    pub fn render_with_lights_and_environment_hdr_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> HdrImage {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_lights_and_environment_hdr_image(world, lights, environment)
            },
            |tile_size| {
                self.camera
                    .render_world_with_lights_and_environment_hdr_image_tiled(
                        world,
                        lights,
                        environment,
                        tile_size,
                    )
            },
        )
    }

    /// Renders `world` with geometry lights plus a custom background source.
    pub fn render_with_lights_and_background(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        self.render_with_lights_and_background_hdr_image(world, lights, background)
            .to_canvas()
    }

    /// Renders `world` with geometry lights, a custom background, and tone mapping.
    pub fn render_with_lights_and_background_tone_mapped(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
        tone_map: ToneMap,
    ) -> Canvas {
        self.render_with_lights_and_background_hdr_image(world, lights, background)
            .to_canvas_tone_mapped(tone_map)
    }

    /// Renders `world` with geometry lights and a custom background to linear HDR samples.
    #[must_use]
    pub fn render_with_lights_and_background_hdr_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> HdrImage {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_lights_and_background_hdr_image(world, lights, background)
            },
            |tile_size| {
                self.camera
                    .render_world_with_lights_and_background_hdr_image_tiled(
                        world, lights, background, tile_size,
                    )
            },
        )
    }

    /// Renders `world` with geometry lights, environment, and spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_with_lights_and_environment_spectral(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> Canvas {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_and_environment_spectral_tiled(
                    world,
                    lights,
                    environment,
                    tile_size,
                ),
            None => self
                .camera
                .render_world_with_lights_and_environment_spectral(world, lights, environment),
        }
    }

    /// Renders `world` with geometry lights and environment to a linear spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_with_lights_and_environment_spectral_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        environment: &EnvironmentLight,
    ) -> SpectralImage {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_and_environment_spectral_image_tiled(
                    world,
                    lights,
                    environment,
                    tile_size,
                ),
            None => self
                .camera
                .render_world_with_lights_and_environment_spectral_image(
                    world,
                    lights,
                    environment,
                ),
        }
    }

    /// Renders `world` with geometry lights, a custom background source, and spectral transport.
    #[cfg(feature = "spectral")]
    pub fn render_with_lights_and_background_spectral(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> Canvas {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_and_background_spectral_tiled(
                    world, lights, background, tile_size,
                ),
            None => self
                .camera
                .render_world_with_lights_and_background_spectral(world, lights, background),
        }
    }

    /// Renders `world` with geometry lights and a custom background to a spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_with_lights_and_background_spectral_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        background: &dyn RayBackgroundSource,
    ) -> SpectralImage {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_and_background_spectral_image_tiled(
                    world, lights, background, tile_size,
                ),
            None => self
                .camera
                .render_world_with_lights_and_background_spectral_image(world, lights, background),
        }
    }

    /// Renders lit beauty, albedo, and normal buffers for denoising.
    #[must_use]
    pub fn render_with_lights_denoising_aovs(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> DenoisingAovs {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_lights_denoising_aovs(world, lights)
            },
            |tile_size| {
                self.camera
                    .render_world_with_lights_denoising_aovs_tiled(world, lights, tile_size)
            },
        )
    }

    /// Renders `world` with forced next-event light connections.
    ///
    /// This is ordinary camera-subpath next-event estimation. It does not implement bidirectional
    /// light subpaths, path-space MIS, or path mutation.
    pub fn render_with_light_connections(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> Canvas {
        self.options.tile_size.map_or_else(
            || {
                self.camera
                    .render_world_with_light_connections(world, lights)
            },
            |tile_size| {
                self.camera
                    .render_world_with_light_connections_tiled(world, lights, tile_size)
            },
        )
    }

    /// Renders `world` with explicit light sampling and progressive tile callbacks.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_with_lights_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_tiled_progressive(world, lights, tile_size, progress),
            None => self
                .camera
                .render_world_with_lights_progressive(world, lights, progress),
        }
    }

    /// Renders `world` with forced light connections and progressive tile callbacks.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `progress`.
    pub fn render_with_light_connections_progressive<P, E>(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        progress: P,
    ) -> Result<Canvas, E>
    where
        P: for<'a> FnMut(ProgressiveRenderUpdate<'a>) -> Result<(), E>,
    {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_light_connections_tiled_progressive(
                    world, lights, tile_size, progress,
                ),
            None => self
                .camera
                .render_world_with_light_connections_progressive(world, lights, progress),
        }
    }

    /// Renders `world` with explicit light sampling and the sampled-wavelength spectral prototype.
    #[cfg(feature = "spectral")]
    pub fn render_with_lights_spectral(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> Canvas {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_spectral_tiled(world, lights, tile_size),
            None => self.camera.render_world_with_lights_spectral(world, lights),
        }
    }

    /// Renders `world` with explicit light sampling to a linear spectral image.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn render_with_lights_spectral_image(
        self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
    ) -> SpectralImage {
        match self.options.tile_size {
            Some(tile_size) => self
                .camera
                .render_world_with_lights_spectral_image_tiled(world, lights, tile_size),
            None => self
                .camera
                .render_world_with_lights_spectral_image(world, lights),
        }
    }

    /// Renders `world` as normal-visualization colors.
    pub fn render_normals(self, world: &dyn Hittable) -> Canvas {
        self.camera.render_world_normals(world)
    }
}

impl Default for PathTracer {
    fn default() -> Self {
        Self::new(RayCamera::default())
    }
}

impl From<RayCamera> for PathTracer {
    fn from(camera: RayCamera) -> Self {
        Self::new(camera)
    }
}

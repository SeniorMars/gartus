//! Path-tracing render entrypoints.
//!
//! [`PathTracer`] is the high-level wrapper around [`RayCamera`]. Use
//! [`PathTracer::render_scene`] as a one-shot convenience for renderer-neutral [`SurfaceScene`]
//! content. For repeated renders of the same surface scene, compile once with
//! [`SurfaceScene::to_ray_scene`] and pass the resulting [`RayScene`] to
//! [`PathTracer::render`] or [`PathTracer::render_with_lights`].

use super::{Hittable, RayScene};
use crate::graphics::{camera::RayCamera, display::Canvas, scene::SurfaceScene};

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
        self.options.tile_size.map_or_else(
            || self.camera.render_world(world),
            |tile_size| self.camera.render_world_tiled(world, tile_size),
        )
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
        let ray_scene = RayScene::from(scene);
        self.render(&ray_scene)
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

    /// Renders `world` while importance-sampling directions toward `lights`.
    ///
    /// Pass a lights-only or otherwise important target set here; passing the full scene is valid
    /// but usually raises variance by sampling non-emissive geometry.
    pub fn render_with_lights(self, world: &dyn Hittable, lights: &dyn Hittable) -> Canvas {
        self.options.tile_size.map_or_else(
            || self.camera.render_world_with_lights(world, lights),
            |tile_size| {
                self.camera
                    .render_world_with_lights_tiled(world, lights, tile_size)
            },
        )
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

//! Path-tracing render entrypoints.

use super::{Hittable, RayScene};
use crate::graphics::{camera::RayCamera, display::Canvas, scene::SurfaceScene};

/// Path-tracing renderer wrapper around a ray camera.
///
/// Prefer [`Self::render_scene`] for renderer-neutral mesh scenes. Use [`Self::render`] and
/// [`Self::render_with_lights`] when you already have a low-level [`RayScene`] or custom
/// [`Hittable`] world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathTracer {
    camera: RayCamera,
}

impl PathTracer {
    /// Creates a path tracer using `camera`.
    #[must_use]
    pub const fn new(camera: RayCamera) -> Self {
        Self { camera }
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

    /// Renders `world` with path-traced material scattering.
    pub fn render(self, world: &dyn Hittable) -> Canvas {
        self.camera.render_world(world)
    }

    /// Compiles and renders a renderer-neutral surface scene.
    ///
    /// Surface meshes are converted to a [`RayScene`] with Lambertian materials derived from the
    /// shared [`SurfaceScene`] material data.
    pub fn render_scene(self, scene: &SurfaceScene) -> Canvas {
        let ray_scene = RayScene::from(scene);
        self.render(&ray_scene)
    }

    /// Compiles and renders a surface scene while importance-sampling `lights`.
    pub fn render_scene_with_lights(self, scene: &SurfaceScene, lights: &dyn Hittable) -> Canvas {
        let ray_scene = RayScene::from(scene);
        self.render_with_lights(&ray_scene, lights)
    }

    /// Renders `world` while importance-sampling directions toward `lights`.
    ///
    /// Pass a lights-only or otherwise important target set here; passing the full scene is valid
    /// but usually raises variance by sampling non-emissive geometry.
    pub fn render_with_lights(self, world: &dyn Hittable, lights: &dyn Hittable) -> Canvas {
        self.camera.render_world_with_lights(world, lights)
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

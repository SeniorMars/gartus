//! Path-tracing render entrypoints.

use super::Hittable;
use crate::graphics::{camera::RayCamera, display::Canvas};

/// Path-tracing renderer wrapper around a ray camera.
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

    /// Renders `world` while importance-sampling directions toward `lights`.
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

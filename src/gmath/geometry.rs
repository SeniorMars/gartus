//! Shared analytic geometry descriptors.

use super::vector::{Point, Vector};

/// Analytic sphere geometry shared by raster and ray-tracing paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereGeometry {
    center: Point,
    radius: f64,
}

impl SphereGeometry {
    /// Creates a sphere geometry descriptor.
    #[must_use]
    pub fn new(center: Point, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Creates a sphere geometry descriptor from tuple coordinates.
    #[must_use]
    pub fn from_tuple(center: (f64, f64, f64), radius: f64) -> Self {
        Self::new(Point::new(center.0, center.1, center.2), radius)
    }

    /// Returns the sphere center.
    #[must_use]
    pub fn center(self) -> Point {
        self.center
    }

    /// Returns the sphere center as tuple coordinates.
    #[must_use]
    pub fn center_tuple(self) -> (f64, f64, f64) {
        (self.center.x(), self.center.y(), self.center.z())
    }

    /// Returns the sphere radius.
    #[must_use]
    pub fn radius(self) -> f64 {
        self.radius
    }

    /// Returns the outward unit normal at `point`.
    #[must_use]
    pub fn outward_normal_at(self, point: Point) -> Vector {
        let outward = point - self.center;
        if self.radius.abs() <= f64::EPSILON {
            outward.normalized()
        } else {
            outward / self.radius
        }
    }
}

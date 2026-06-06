//! Shared analytic geometry descriptors.

use super::ray::Ray;
use super::{
    polygon_matrix::Bounds3,
    vector::{Point, Vector},
};

const TRIANGLE_BOUNDS_EPSILON: f64 = 1e-10;
const TRIANGLE_HIT_EPSILON: f64 = 1e-10;

/// Orthonormal camera frame derived from an eye point, target point, and view-up vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraBasis {
    /// Unit vector pointing to camera right for a conventional look-at matrix.
    pub right: Vector,
    /// Unit vector pointing camera up.
    pub up: Vector,
    /// Unit vector pointing from the camera eye toward the target.
    pub forward: Vector,
}

impl CameraBasis {
    /// Builds an orthonormal camera basis from `eye`, `target`, and a view-up vector.
    ///
    /// Returns `None` when the eye and target match, the up vector is zero, or the up vector is
    /// parallel to the viewing direction.
    #[must_use]
    pub fn looking_at(eye: Point, target: Point, vup: Vector) -> Option<Self> {
        let forward = (target - eye).normalized();
        if forward.length_squared() <= f64::EPSILON {
            return None;
        }

        let right = forward.cross(vup).normalized();
        if right.length_squared() <= f64::EPSILON {
            return None;
        }

        let up = right.cross(forward);
        Some(Self { right, up, forward })
    }
}

/// Camera pose shared by projection and ray-tracing cameras.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    /// Camera origin.
    pub lookfrom: Point,
    /// Point the camera is aimed at.
    pub lookat: Point,
    /// Requested camera-relative up direction.
    pub view_up: Vector,
}

impl CameraPose {
    /// Creates a camera pose.
    #[must_use]
    pub const fn new(lookfrom: Point, lookat: Point, view_up: Vector) -> Self {
        Self {
            lookfrom,
            lookat,
            view_up,
        }
    }

    /// Builds a complete camera frame from this pose.
    #[must_use]
    pub fn frame(self) -> Option<CameraFrame> {
        let basis = CameraBasis::looking_at(self.lookfrom, self.lookat, self.view_up)?;
        Some(CameraFrame {
            origin: self.lookfrom,
            right: basis.right,
            up: basis.up,
            forward: basis.forward,
        })
    }
}

/// Orthonormal camera frame with origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFrame {
    /// Camera origin.
    pub origin: Point,
    /// Unit vector pointing camera right.
    pub right: Vector,
    /// Unit vector pointing camera up.
    pub up: Vector,
    /// Unit vector pointing from camera origin toward the target.
    pub forward: Vector,
}

impl CameraFrame {
    /// Returns the camera backward vector used by right-handed ray cameras.
    #[must_use]
    pub fn backward(self) -> Vector {
        -self.forward
    }
}

/// Analytic triangle geometry shared by raster and ray-tracing paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleGeometry {
    vertices: [Point; 3],
}

impl TriangleGeometry {
    /// Creates triangle geometry from three vertices.
    #[must_use]
    pub const fn new(p0: Point, p1: Point, p2: Point) -> Self {
        Self {
            vertices: [p0, p1, p2],
        }
    }

    /// Creates triangle geometry from tuple vertices.
    #[must_use]
    pub fn from_tuples(vertices: [(f64, f64, f64); 3]) -> Self {
        Self::new(
            Point::new(vertices[0].0, vertices[0].1, vertices[0].2),
            Point::new(vertices[1].0, vertices[1].1, vertices[1].2),
            Point::new(vertices[2].0, vertices[2].1, vertices[2].2),
        )
    }

    /// Returns the triangle vertices.
    #[must_use]
    pub const fn vertices(self) -> [Point; 3] {
        self.vertices
    }

    /// Returns padded axis-aligned bounds for the triangle.
    #[must_use]
    pub fn bounds(self) -> Bounds3 {
        let [p0, p1, p2] = self.vertices;
        Bounds3::from_points(p0, p0)
            .union_point(p1)
            .union_point(p2)
            .padded(TRIANGLE_BOUNDS_EPSILON)
    }

    /// Returns the triangle centroid.
    #[must_use]
    pub fn centroid(self) -> Point {
        Point::new(
            (self.vertices[0].x() + self.vertices[1].x() + self.vertices[2].x()) / 3.0,
            (self.vertices[0].y() + self.vertices[1].y() + self.vertices[2].y()) / 3.0,
            (self.vertices[0].z() + self.vertices[1].z() + self.vertices[2].z()) / 3.0,
        )
    }

    /// Returns the unit geometric normal from vertex winding.
    #[must_use]
    pub fn geometric_normal(self) -> Vector {
        let [p0, p1, p2] = self.vertices;
        (p1 - p0).cross(p2 - p0).normalized()
    }

    /// Returns four times the triangle area squared.
    #[must_use]
    pub fn area_squared(self) -> f64 {
        let [p0, p1, p2] = self.vertices;
        (p1 - p0).cross(p2 - p0).length_squared()
    }

    /// Returns barycentric ray hit data for a two-sided Möller-Trumbore intersection.
    #[must_use]
    pub fn hit_ray(self, ray: &Ray, t_min: f64, t_max: f64) -> Option<TriangleHit> {
        let [p0, p1, p2] = self.vertices;
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let direction = *ray.direction();
        let pvec = direction.cross(e2);
        let det = e1.dot(pvec);
        if det.abs() < TRIANGLE_HIT_EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;
        let tvec = *ray.origin() - p0;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let qvec = tvec.cross(e1);
        let v = direction.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = e2.dot(qvec) * inv_det;
        (t_min < t && t < t_max).then_some(TriangleHit { t, u, v })
    }
}

/// Barycentric data for a triangle ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleHit {
    /// Ray parameter at the triangle hit.
    pub t: f64,
    /// Barycentric coordinate for the second vertex.
    pub u: f64,
    /// Barycentric coordinate for the third vertex.
    pub v: f64,
}

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
        Self {
            center,
            radius: radius.max(0.0),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn camera_basis_looking_down_negative_z_is_canonical() {
        let basis = CameraBasis::looking_at(
            Point::new(0.0, 0.0, 0.0),
            Point::new(0.0, 0.0, -1.0),
            Vector::new(0.0, 1.0, 0.0),
        )
        .expect("valid basis");

        assert_eq!(basis.right, Vector::new(1.0, -0.0, 0.0));
        assert_eq!(basis.up, Vector::new(0.0, 1.0, 0.0));
        assert_eq!(basis.forward, Vector::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn camera_basis_is_orthonormal() {
        let basis = CameraBasis::looking_at(
            Point::new(2.0, 3.0, 5.0),
            Point::new(-1.0, 0.5, 1.0),
            Vector::new(0.0, 1.0, 0.0),
        )
        .expect("valid basis");

        assert_close(basis.right.length(), 1.0);
        assert_close(basis.up.length(), 1.0);
        assert_close(basis.forward.length(), 1.0);
        assert_close(basis.right.dot(basis.up), 0.0);
        assert_close(basis.right.dot(basis.forward), 0.0);
        assert_close(basis.up.dot(basis.forward), 0.0);
    }

    #[test]
    fn camera_basis_rejects_degenerate_inputs() {
        assert!(
            CameraBasis::looking_at(
                Point::new(0.0, 0.0, 0.0),
                Point::new(0.0, 0.0, 0.0),
                Vector::new(0.0, 1.0, 0.0),
            )
            .is_none()
        );
        assert!(
            CameraBasis::looking_at(
                Point::new(0.0, 0.0, 0.0),
                Point::new(0.0, 1.0, 0.0),
                Vector::new(0.0, 2.0, 0.0),
            )
            .is_none()
        );
    }

    #[test]
    fn camera_pose_builds_frame_with_origin() {
        let frame = CameraPose::new(
            Point::new(1.0, 2.0, 3.0),
            Point::new(1.0, 2.0, 2.0),
            Vector::new(0.0, 1.0, 0.0),
        )
        .frame()
        .expect("valid frame");

        assert_eq!(frame.origin, Point::new(1.0, 2.0, 3.0));
        assert_eq!(frame.right, Vector::new(1.0, -0.0, 0.0));
        assert_eq!(frame.up, Vector::new(0.0, 1.0, 0.0));
        assert_eq!(frame.forward, Vector::new(0.0, 0.0, -1.0));
        assert_eq!(frame.backward(), Vector::new(-0.0, -0.0, 1.0));
    }

    #[test]
    fn triangle_geometry_reports_bounds_centroid_and_normal() {
        let triangle = TriangleGeometry::new(
            Point::new(0.0, 0.0, -1.0),
            Point::new(1.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
        );
        let bounds = triangle.bounds();

        assert_eq!(triangle.centroid(), Point::new(1.0 / 3.0, 1.0 / 3.0, -1.0));
        assert_eq!(triangle.geometric_normal(), Vector::new(0.0, 0.0, 1.0));
        assert_close(triangle.area_squared(), 1.0);
        assert_close(bounds.min.0, -0.5e-10);
        assert_close(bounds.max.2, -1.0 + 0.5e-10);
    }
    #[test]
    fn triangle_geometry_hit_ray_returns_barycentrics() {
        use crate::gmath::ray::Ray;

        let triangle = TriangleGeometry::new(
            Point::new(0.0, 0.0, -1.0),
            Point::new(1.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
        );
        let ray = Ray::new(Point::new(0.25, 0.25, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = triangle
            .hit_ray(&ray, 0.0, f64::INFINITY)
            .expect("triangle should be hit");

        assert_close(hit.t, 1.0);
        assert_close(hit.u, 0.25);
        assert_close(hit.v, 0.25);
    }

    #[test]
    fn sphere_geometry_clamps_negative_radius() {
        let sphere = SphereGeometry::new(Point::new(0.0, 0.0, 0.0), -2.0);

        assert_close(sphere.radius(), 0.0);
    }
}

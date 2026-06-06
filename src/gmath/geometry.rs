//! Shared analytic geometry descriptors.

use super::ray::Ray;
use super::{
    polygon_matrix::Bounds3,
    vector::{Point, Vector},
};

const TRIANGLE_BOUNDS_EPSILON: f64 = 1e-10;
const TRIANGLE_HIT_EPSILON: f64 = 1e-10;
const QUAD_BOUNDS_EPSILON: f64 = 1e-4;
const QUAD_HIT_EPSILON: f64 = 1e-8;

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
///
/// This is the camera-space companion to [`OrthonormalBasis`]: use `CameraFrame` when an origin
/// and view direction matter, and use [`OrthonormalBasis`] for direction-only local sample frames.
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

/// Orthonormal basis for mapping local sample directions into world space.
///
/// This is a direction-only frame. For camera placement and projection, use [`CameraPose`] /
/// [`CameraFrame`], which include the eye point and view-up constrained camera axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthonormalBasis {
    u: Vector,
    v: Vector,
    w: Vector,
}

impl OrthonormalBasis {
    /// Builds a basis with `w` aligned to `axis`.
    ///
    /// Returns `None` when `axis` is zero.
    #[must_use]
    pub fn from_w(axis: Vector) -> Option<Self> {
        let w = axis.normalized();
        if w.length_squared() <= f64::EPSILON {
            return None;
        }

        let a = if w.x().abs() > 0.9 {
            Vector::new(0.0, 1.0, 0.0)
        } else {
            Vector::new(1.0, 0.0, 0.0)
        };
        let v = w.cross(a).normalized();
        let u = w.cross(v);

        Some(Self { u, v, w })
    }

    /// Returns the first tangent axis.
    #[must_use]
    pub const fn u(self) -> Vector {
        self.u
    }

    /// Returns the second tangent axis.
    #[must_use]
    pub const fn v(self) -> Vector {
        self.v
    }

    /// Returns the normal axis.
    #[must_use]
    pub const fn w(self) -> Vector {
        self.w
    }

    /// Converts a local vector into this basis.
    #[must_use]
    pub fn local(self, local: Vector) -> Vector {
        local.x() * self.u + local.y() * self.v + local.z() * self.w
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
        self.area_weighted_normal().normalized()
    }

    /// Returns the unnormalized normal from vertex winding.
    ///
    /// The vector length is twice the triangle area, so callers can use it for area-weighted
    /// normal accumulation and winding tests without losing magnitude information.
    #[must_use]
    pub fn area_weighted_normal(self) -> Vector {
        let [p0, p1, p2] = self.vertices;
        (p1 - p0).cross(p2 - p0)
    }

    /// Returns four times the triangle area squared.
    #[must_use]
    pub fn area_squared(self) -> f64 {
        self.area_weighted_normal().length_squared()
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

/// Analytic parallelogram geometry defined by one corner and two side vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadGeometry {
    corner: Point,
    u: Vector,
    v: Vector,
    normal: Vector,
    plane_d: f64,
    w: Vector,
}

impl QuadGeometry {
    /// Creates quad geometry from starting corner `corner` and side vectors `u` and `v`.
    #[must_use]
    pub fn new(corner: Point, u: Vector, v: Vector) -> Self {
        let n = u.cross(v);
        let normal = n.normalized();
        let plane_d = normal.dot(point_to_vector(corner));
        let normal_length_squared = n.length_squared();
        let w = if normal_length_squared <= f64::EPSILON {
            Vector::default()
        } else {
            n / normal_length_squared
        };

        Self {
            corner,
            u,
            v,
            normal,
            plane_d,
            w,
        }
    }

    /// Creates quad geometry from tuple coordinates.
    #[must_use]
    pub fn from_tuples(corner: (f64, f64, f64), u: (f64, f64, f64), v: (f64, f64, f64)) -> Self {
        Self::new(
            Point::new(corner.0, corner.1, corner.2),
            Vector::new(u.0, u.1, u.2),
            Vector::new(v.0, v.1, v.2),
        )
    }

    /// Returns the starting corner.
    #[must_use]
    pub const fn corner(self) -> Point {
        self.corner
    }

    /// Returns the first side vector.
    #[must_use]
    pub const fn u(self) -> Vector {
        self.u
    }

    /// Returns the second side vector.
    #[must_use]
    pub const fn v(self) -> Vector {
        self.v
    }

    /// Returns the four quad vertices.
    #[must_use]
    pub fn vertices(self) -> [Point; 4] {
        [
            self.corner,
            self.corner + self.u,
            self.corner + self.v,
            self.corner + self.u + self.v,
        ]
    }

    /// Returns padded axis-aligned bounds for the quad.
    #[must_use]
    pub fn bounds(self) -> Bounds3 {
        let [p0, p1, p2, p3] = self.vertices();
        Bounds3::from_points(p0, p0)
            .union_point(p1)
            .union_point(p2)
            .union_point(p3)
            .padded(QUAD_BOUNDS_EPSILON)
    }

    /// Returns the quad centroid.
    #[must_use]
    pub fn centroid(self) -> Point {
        self.corner + 0.5 * (self.u + self.v)
    }

    /// Returns the unit geometric normal from side-vector winding.
    #[must_use]
    pub const fn geometric_normal(self) -> Vector {
        self.normal
    }

    /// Returns the squared parallelogram area.
    #[must_use]
    pub fn area_squared(self) -> f64 {
        self.u.cross(self.v).length_squared()
    }

    /// Returns the planar `(u, v)` coordinates for `point`.
    #[must_use]
    pub fn plane_coordinates(self, point: Point) -> (f64, f64) {
        let planar_point = point - self.corner;
        (
            self.w.dot(planar_point.cross(self.v)),
            self.w.dot(self.u.cross(planar_point)),
        )
    }

    /// Returns ray hit data for a two-sided ray/quad intersection.
    #[must_use]
    pub fn hit_ray(self, ray: &Ray, t_min: f64, t_max: f64) -> Option<QuadHit> {
        if self.area_squared() <= f64::EPSILON {
            return None;
        }

        let denom = self.normal.dot(*ray.direction());
        if denom.abs() < QUAD_HIT_EPSILON {
            return None;
        }

        let t = (self.plane_d - self.normal.dot(point_to_vector(*ray.origin()))) / denom;
        if !(t_min < t && t < t_max) {
            return None;
        }

        let (u, v) = self.plane_coordinates(ray.at(t));
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }

        Some(QuadHit { t, u, v })
    }
}

/// Planar coordinate data for a quad ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadHit {
    /// Ray parameter at the quad hit.
    pub t: f64,
    /// Coordinate along the quad's first side vector.
    pub u: f64,
    /// Coordinate along the quad's second side vector.
    pub v: f64,
}

fn point_to_vector(point: Point) -> Vector {
    Vector::new(point.x(), point.y(), point.z())
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

/// Analytic sphere geometry moving linearly from `center_start` to `center_end`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovingSphereGeometry {
    center_start: Point,
    center_end: Point,
    radius: f64,
}

impl MovingSphereGeometry {
    /// Creates a moving sphere geometry descriptor.
    #[must_use]
    pub fn new(center_start: Point, center_end: Point, radius: f64) -> Self {
        Self {
            center_start,
            center_end,
            radius: radius.max(0.0),
        }
    }

    /// Returns the sphere center at shutter time `time`.
    #[must_use]
    pub fn center_at(self, time: f64) -> Point {
        self.center_start + time * (self.center_end - self.center_start)
    }

    /// Returns the sphere center at time zero.
    #[must_use]
    pub fn center_start(self) -> Point {
        self.center_start
    }

    /// Returns the sphere center at time one.
    #[must_use]
    pub fn center_end(self) -> Point {
        self.center_end
    }

    /// Returns the sphere radius.
    #[must_use]
    pub fn radius(self) -> f64 {
        self.radius
    }

    /// Returns the outward unit normal at `point` and shutter `time`.
    #[must_use]
    pub fn outward_normal_at(self, point: Point, time: f64) -> Vector {
        let outward = point - self.center_at(time);
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
    fn orthonormal_basis_maps_local_z_to_axis() {
        let basis = OrthonormalBasis::from_w(Vector::new(0.0, 2.0, 0.0)).expect("valid normal");

        assert_eq!(basis.w(), Vector::new(0.0, 1.0, 0.0));
        assert_close(basis.u().length(), 1.0);
        assert_close(basis.v().length(), 1.0);
        assert_close(basis.u().dot(basis.v()), 0.0);
        assert_eq!(basis.local(Vector::new(0.0, 0.0, 1.0)), basis.w());
        assert!(OrthonormalBasis::from_w(Vector::default()).is_none());
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
    fn quad_geometry_reports_bounds_centroid_and_normal() {
        let quad = QuadGeometry::new(
            Point::new(-2.0, -2.0, 0.0),
            Vector::new(4.0, 0.0, 0.0),
            Vector::new(0.0, 4.0, 0.0),
        );
        let bounds = quad.bounds();

        assert_eq!(quad.centroid(), Point::new(0.0, 0.0, 0.0));
        assert_eq!(quad.geometric_normal(), Vector::new(0.0, 0.0, 1.0));
        assert_close(quad.area_squared(), 256.0);
        assert_close(bounds.min.0, -2.0);
        assert_close(bounds.max.1, 2.0);
        assert_close(bounds.min.2, -0.5e-4);
        assert_close(bounds.max.2, 0.5e-4);
    }

    #[test]
    fn quad_geometry_hit_ray_returns_planar_coordinates() {
        use crate::gmath::ray::Ray;

        let quad = QuadGeometry::new(
            Point::new(-2.0, -2.0, -1.0),
            Vector::new(4.0, 0.0, 0.0),
            Vector::new(0.0, 4.0, 0.0),
        );
        let ray = Ray::new(Point::new(0.0, 1.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = quad
            .hit_ray(&ray, 0.0, f64::INFINITY)
            .expect("quad should be hit");

        assert_close(hit.t, 1.0);
        assert_close(hit.u, 0.5);
        assert_close(hit.v, 0.75);
    }

    #[test]
    fn quad_geometry_rejects_outside_parallel_and_degenerate_rays() {
        use crate::gmath::ray::Ray;

        let quad = QuadGeometry::new(
            Point::new(-1.0, -1.0, -1.0),
            Vector::new(2.0, 0.0, 0.0),
            Vector::new(0.0, 2.0, 0.0),
        );

        let outside = Ray::new(Point::new(2.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        assert!(quad.hit_ray(&outside, 0.0, f64::INFINITY).is_none());

        let parallel = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(1.0, 0.0, 0.0));
        assert!(quad.hit_ray(&parallel, 0.0, f64::INFINITY).is_none());

        let degenerate = QuadGeometry::new(
            Point::new(0.0, 0.0, 0.0),
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(2.0, 0.0, 0.0),
        );
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        assert!(degenerate.hit_ray(&ray, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn sphere_geometry_clamps_negative_radius() {
        let sphere = SphereGeometry::new(Point::new(0.0, 0.0, 0.0), -2.0);

        assert_close(sphere.radius(), 0.0);
    }
}

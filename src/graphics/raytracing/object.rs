//! Core hittable objects, hit records, and analytic ray intersections.

use super::{
    Aabb, INFINITY, PI, SampleRng,
    material::{Material, MaterialRef, default_material},
    scene::HittableList,
};
use crate::gmath::{
    geometry::{MovingSphereGeometry, QuadGeometry, SphereGeometry, TriangleGeometry},
    ray::Ray,
    vector::{Point, Vector},
};
use std::{fmt, sync::Arc};

/// A closed interval of real-valued ray parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Minimum interval endpoint.
    pub min: f64,
    /// Maximum interval endpoint.
    pub max: f64,
}

impl Interval {
    /// Empty interval.
    pub const EMPTY: Self = Self {
        min: INFINITY,
        max: -INFINITY,
    };

    /// Interval covering all finite and infinite real values.
    pub const UNIVERSE: Self = Self {
        min: -INFINITY,
        max: INFINITY,
    };

    /// Creates an interval from inclusive endpoints.
    #[must_use]
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Creates a finite interval when `min <= max`.
    #[must_use]
    pub fn try_new(min: f64, max: f64) -> Option<Self> {
        (min.is_finite() && max.is_finite() && min <= max).then_some(Self { min, max })
    }

    /// Returns the interval width.
    #[must_use]
    pub fn size(self) -> f64 {
        self.max - self.min
    }

    /// Returns true when `x` lies inside the closed interval.
    #[must_use]
    pub fn contains(self, x: f64) -> bool {
        self.min <= x && x <= self.max
    }

    /// Returns true when `x` lies strictly inside the interval.
    #[must_use]
    pub fn surrounds(self, x: f64) -> bool {
        self.min < x && x < self.max
    }

    /// Clamps `x` to the interval endpoints.
    #[must_use]
    pub fn clamp(self, x: f64) -> f64 {
        x.clamp(self.min, self.max)
    }
}

impl Default for Interval {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Context for evaluating or generating a directional PDF sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PdfContext {
    /// Point from which outgoing directions are sampled.
    pub origin: Point,
    /// Ray time used by motion-blurred sampling targets.
    pub time: f64,
}

impl PdfContext {
    /// Creates a sampling context.
    #[must_use]
    pub const fn new(origin: Point, time: f64) -> Self {
        Self { origin, time }
    }
}

/// Geometry-only ray intersection information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceHit {
    /// Hit point.
    pub point: Point,
    /// Unit-length surface normal, always oriented against the incident ray.
    pub normal: Vector,
    /// Ray parameter at the hit point.
    pub t: f64,
    /// Horizontal texture coordinate at the hit point.
    pub u: f64,
    /// Vertical texture coordinate at the hit point.
    pub v: f64,
    /// True when the ray hit the outside face of the surface.
    pub front_face: bool,
}

impl SurfaceHit {
    /// Creates a surface hit and orients `outward_normal` against `ray`.
    ///
    /// `outward_normal` is expected to have unit length.
    #[must_use]
    pub fn new(ray: &Ray, point: Point, outward_normal: Vector, t: f64) -> Self {
        Self::with_uv(ray, point, outward_normal, t, 0.0, 0.0)
    }

    /// Creates a surface hit with texture coordinates and orients `outward_normal` against `ray`.
    ///
    /// `outward_normal` is expected to have unit length.
    #[must_use]
    pub fn with_uv(
        ray: &Ray,
        point: Point,
        outward_normal: Vector,
        t: f64,
        u: f64,
        v: f64,
    ) -> Self {
        let front_face = ray.direction().dot(outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };
        Self {
            point,
            normal,
            t,
            u,
            v,
            front_face,
        }
    }
}

/// Geometry that can be intersected by a ray without owning material data.
pub trait Intersect: Send + Sync {
    /// Returns the closest surface hit inside `ray_t`, if any.
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit>;

    /// Returns an axis-aligned bounding box for this geometry, if available.
    fn bounding_box(&self) -> Option<Aabb> {
        None
    }
}

impl Intersect for SphereGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        let root = hit_sphere_in_interval(self.center(), self.radius(), ray, ray_t)?;
        let point = ray.at(root);
        let outward_normal = self.outward_normal_at(point);
        let (u, v) = sphere_uv(outward_normal);
        Some(SurfaceHit::with_uv(ray, point, outward_normal, root, u, v))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(sphere_bounds(*self))
    }
}

impl Intersect for MovingSphereGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        let center = self.center_at(ray.time());
        let root = hit_sphere_in_interval(center, self.radius(), ray, ray_t)?;
        let point = ray.at(root);
        let outward_normal = self.outward_normal_at(point, ray.time());
        let (u, v) = sphere_uv(outward_normal);
        Some(SurfaceHit::with_uv(ray, point, outward_normal, root, u, v))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(moving_sphere_bounds(*self))
    }
}

impl Intersect for TriangleGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        let hit = self.hit_ray(ray, ray_t.min, ray_t.max)?;
        let normal = self.geometric_normal();
        if normal.length_squared() <= f64::EPSILON {
            return None;
        }
        Some(SurfaceHit::with_uv(
            ray,
            ray.at(hit.t),
            normal,
            hit.t,
            hit.u,
            hit.v,
        ))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(self.bounds())
    }
}

impl Intersect for QuadGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        let hit = self.hit_ray(ray, ray_t.min, ray_t.max)?;
        let normal = self.geometric_normal();
        if normal.length_squared() <= f64::EPSILON {
            return None;
        }
        Some(SurfaceHit::with_uv(
            ray,
            ray.at(hit.t),
            normal,
            hit.t,
            hit.u,
            hit.v,
        ))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(self.bounds())
    }
}

/// Returns `(t, u, v)` for a two-sided Möller-Trumbore triangle hit.
#[must_use]
pub fn hit_triangle(
    p0: Point,
    p1: Point,
    p2: Point,
    ray: &Ray,
    ray_t: Interval,
) -> Option<(f64, f64, f64)> {
    TriangleGeometry::new(p0, p1, p2)
        .hit_ray(ray, ray_t.min, ray_t.max)
        .map(|hit| (hit.t, hit.u, hit.v))
}

/// Geometry variants supported by the data-oriented ray scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RayGeometry {
    /// Analytic sphere geometry.
    Sphere(SphereGeometry),
    /// Analytic moving sphere geometry.
    MovingSphere(MovingSphereGeometry),
    /// Analytic triangle geometry.
    Triangle(TriangleGeometry),
    /// Analytic parallelogram geometry.
    Quad(QuadGeometry),
}

impl RayGeometry {
    /// Creates a sphere geometry variant.
    #[must_use]
    pub fn sphere(center: Point, radius: f64) -> Self {
        Self::Sphere(SphereGeometry::new(center, radius))
    }

    /// Creates a moving sphere geometry variant.
    #[must_use]
    pub fn moving_sphere(center_start: Point, center_end: Point, radius: f64) -> Self {
        Self::MovingSphere(MovingSphereGeometry::new(center_start, center_end, radius))
    }

    /// Creates a triangle geometry variant.
    #[must_use]
    pub const fn triangle(p0: Point, p1: Point, p2: Point) -> Self {
        Self::Triangle(TriangleGeometry::new(p0, p1, p2))
    }

    /// Creates a quad geometry variant.
    #[must_use]
    pub fn quad(corner: Point, u: Vector, v: Vector) -> Self {
        Self::Quad(QuadGeometry::new(corner, u, v))
    }

    pub(crate) fn pdf_value(self, context: PdfContext, direction: Vector) -> f64 {
        match self {
            Self::Sphere(geometry) => sphere_pdf_value(geometry, context.origin, direction),
            Self::MovingSphere(geometry) => {
                let sphere = moving_sphere_at(geometry, context.time);
                sphere_pdf_value(sphere, context.origin, direction)
            }
            Self::Triangle(geometry) => triangle_pdf_value(geometry, context.origin, direction),
            Self::Quad(geometry) => quad_pdf_value(geometry, context.origin, direction),
        }
    }

    pub(crate) fn random_direction(self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        match self {
            Self::Sphere(geometry) => random_direction_to_sphere(geometry, context.origin, rng),
            Self::MovingSphere(geometry) => random_direction_to_sphere(
                moving_sphere_at(geometry, context.time),
                context.origin,
                rng,
            ),
            Self::Triangle(geometry) => random_direction_to_triangle(geometry, context.origin, rng),
            Self::Quad(geometry) => random_direction_to_quad(geometry, context.origin, rng),
        }
    }
}

impl From<SphereGeometry> for RayGeometry {
    fn from(geometry: SphereGeometry) -> Self {
        Self::Sphere(geometry)
    }
}

impl From<MovingSphereGeometry> for RayGeometry {
    fn from(geometry: MovingSphereGeometry) -> Self {
        Self::MovingSphere(geometry)
    }
}

impl From<TriangleGeometry> for RayGeometry {
    fn from(geometry: TriangleGeometry) -> Self {
        Self::Triangle(geometry)
    }
}

impl From<QuadGeometry> for RayGeometry {
    fn from(geometry: QuadGeometry) -> Self {
        Self::Quad(geometry)
    }
}

impl Intersect for RayGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        match self {
            Self::Sphere(geometry) => geometry.intersect(ray, ray_t),
            Self::MovingSphere(geometry) => geometry.intersect(ray, ray_t),
            Self::Triangle(geometry) => geometry.intersect(ray, ray_t),
            Self::Quad(geometry) => geometry.intersect(ray, ray_t),
        }
    }

    fn bounding_box(&self) -> Option<Aabb> {
        match self {
            Self::Sphere(geometry) => geometry.bounding_box(),
            Self::MovingSphere(geometry) => geometry.bounding_box(),
            Self::Triangle(geometry) => geometry.bounding_box(),
            Self::Quad(geometry) => geometry.bounding_box(),
        }
    }
}

/// Information recorded when a ray intersects a hittable object.
#[derive(Clone, Copy)]
pub struct HitRecord<'a> {
    /// Hit point.
    pub point: Point,
    /// Unit-length surface normal, always oriented against the incident ray.
    pub normal: Vector,
    /// Unit-length geometric surface normal, always oriented against the incident ray.
    pub geometric_normal: Vector,
    /// Unit-length shading normal, always oriented against the incident ray.
    pub shading_normal: Vector,
    /// Optional tangent vector for tangent-space shading, oriented around [`Self::shading_normal`].
    pub tangent: Option<Vector>,
    /// Optional bitangent vector for tangent-space shading, oriented around [`Self::shading_normal`].
    pub bitangent: Option<Vector>,
    /// Tangent-space handedness. `-1.0` marks mirrored UV orientation.
    pub tangent_handedness: f64,
    /// Ray parameter at the hit point.
    pub t: f64,
    /// Horizontal texture coordinate at the hit point.
    pub u: f64,
    /// Vertical texture coordinate at the hit point.
    pub v: f64,
    /// True when the ray hit the outside face of the surface.
    pub front_face: bool,
    /// Material associated with the hit surface.
    pub material: &'a dyn Material,
}

impl fmt::Debug for HitRecord<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HitRecord")
            .field("point", &self.point)
            .field("normal", &self.normal)
            .field("geometric_normal", &self.geometric_normal)
            .field("shading_normal", &self.shading_normal)
            .field("tangent", &self.tangent)
            .field("bitangent", &self.bitangent)
            .field("tangent_handedness", &self.tangent_handedness)
            .field("t", &self.t)
            .field("u", &self.u)
            .field("v", &self.v)
            .field("front_face", &self.front_face)
            .finish_non_exhaustive()
    }
}

impl<'a> HitRecord<'a> {
    /// Creates a hit record and orients `outward_normal` against `ray`.
    ///
    /// `outward_normal` is expected to have unit length.
    #[must_use]
    pub fn new(
        ray: &Ray,
        point: Point,
        outward_normal: Vector,
        t: f64,
        material: &'a dyn Material,
    ) -> Self {
        let mut record = Self::from_surface(
            SurfaceHit {
                point,
                normal: outward_normal,
                t,
                u: 0.0,
                v: 0.0,
                front_face: false,
            },
            material,
        );
        record.set_face_normal(ray, outward_normal);
        record
    }

    /// Adds material information to a geometry-only surface hit.
    #[must_use]
    pub fn from_surface(surface: SurfaceHit, material: &'a dyn Material) -> Self {
        Self {
            point: surface.point,
            normal: surface.normal,
            geometric_normal: surface.normal,
            shading_normal: surface.normal,
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: surface.t,
            u: surface.u,
            v: surface.v,
            front_face: surface.front_face,
            material,
        }
    }

    /// Orients `outward_normal` against `ray` and stores which side was hit.
    ///
    /// `outward_normal` is expected to have unit length.
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vector) {
        self.front_face = ray.direction().dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
        self.geometric_normal = self.normal;
        self.shading_normal = self.normal;
        self.tangent = None;
        self.bitangent = None;
        self.tangent_handedness = 1.0;
    }

    /// Replaces the shading normal while preserving geometric front-face state.
    pub fn set_shading_normal(&mut self, outward_normal: Vector) {
        self.shading_normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
        self.tangent = None;
        self.bitangent = None;
        self.tangent_handedness = 1.0;
    }

    /// Replaces the shading normal with an already-oriented world-space normal.
    pub fn set_oriented_shading_normal(&mut self, normal: Vector) {
        if normal.length_squared() > f64::EPSILON {
            let tangent_frame = self.tangent.zip(self.bitangent);
            self.shading_normal = normal.normalized();
            self.tangent = None;
            self.bitangent = None;
            self.tangent_handedness = 1.0;
            if let Some((tangent, bitangent)) = tangent_frame {
                self.set_tangent_frame(tangent, bitangent);
            }
        }
    }

    /// Stores a tangent frame orthonormalized around the current shading normal.
    pub fn set_tangent_frame(&mut self, tangent: Vector, bitangent: Vector) {
        let normal = self.shading_normal;
        let tangent = tangent - normal * tangent.dot(normal);
        if tangent.length_squared() <= f64::EPSILON {
            return;
        }
        let tangent = tangent.normalized();
        let reference_bitangent = normal.cross(tangent);
        let mut bitangent = bitangent - normal * bitangent.dot(normal);
        bitangent = bitangent - tangent * bitangent.dot(tangent);
        if bitangent.length_squared() <= f64::EPSILON {
            bitangent = reference_bitangent;
        }
        let handedness = if bitangent.dot(reference_bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        self.tangent = Some(tangent);
        self.bitangent = Some((reference_bitangent * handedness).normalized());
        self.tangent_handedness = handedness;
    }
}

/// A material-bearing scene object.
#[derive(Clone)]
pub struct SceneObject<G> {
    geometry: G,
    material: MaterialRef,
}

impl<G> SceneObject<G> {
    /// Creates a scene object from geometry and a concrete material.
    #[must_use]
    pub fn new(geometry: G, material: impl Material + 'static) -> Self {
        Self::with_shared_material(geometry, Arc::new(material))
    }

    /// Creates a scene object from geometry and a shared material handle.
    #[must_use]
    pub fn with_shared_material(geometry: G, material: MaterialRef) -> Self {
        Self { geometry, material }
    }

    /// Returns the object's geometry.
    #[must_use]
    pub fn geometry(&self) -> &G {
        &self.geometry
    }

    /// Returns the material associated with this object.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }
}

impl<G: fmt::Debug> fmt::Debug for SceneObject<G> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SceneObject")
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl<G: Intersect> Hittable for SceneObject<G> {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let surface = self.geometry.intersect(ray, ray_t)?;
        Some(HitRecord::from_surface(surface, self.material()))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.geometry.bounding_box()
    }
}

impl<T: Hittable + ?Sized> Hittable for Arc<T> {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        (**self).hit_with_rng(ray, ray_t, rng)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        (**self).bounding_box()
    }
}

/// A scene object that can be intersected by a ray.
pub trait Hittable: Send + Sync {
    /// Returns the closest hit inside `ray_t`, if any.
    ///
    /// This is a deterministic convenience wrapper. Hot paths and stochastic objects such as
    /// volumes should call [`Self::hit_with_rng`] so they can reuse the caller's sample stream.
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut rng = SampleRng::default();
        self.hit_with_rng(ray, ray_t, &mut rng)
    }

    /// Returns the closest hit inside `ray_t`, using `rng` for probabilistic volumes.
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>>;

    /// Returns an axis-aligned bounding box for acceleration structures, if available.
    fn bounding_box(&self) -> Option<Aabb> {
        None
    }

    /// Returns the probability density for sampling `direction` from `origin`.
    fn pdf_value(&self, _context: PdfContext, _direction: Vector) -> f64 {
        0.0
    }

    /// Returns a random direction from `origin` toward this object.
    fn random_direction(&self, _context: PdfContext, _rng: &mut SampleRng) -> Vector {
        Vector::new(1.0, 0.0, 0.0)
    }
}

/// A sphere hittable.
#[derive(Clone)]
pub struct Sphere {
    object: SceneObject<SphereGeometry>,
}

impl fmt::Debug for Sphere {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Sphere")
            .field("geometry", self.object.geometry())
            .finish_non_exhaustive()
    }
}

impl Sphere {
    /// Creates a gray diffuse sphere. Negative radii are clamped to zero.
    #[must_use]
    pub fn new(center: Point, radius: f64) -> Self {
        Self::with_shared_material(center, radius, default_material())
    }

    /// Creates a sphere with a material. Negative radii are clamped to zero.
    #[must_use]
    pub fn with_material(center: Point, radius: f64, material: impl Material + 'static) -> Self {
        Self::with_shared_material(center, radius, Arc::new(material))
    }

    /// Creates a sphere with a shared material handle. Negative radii are clamped to zero.
    #[must_use]
    pub fn with_shared_material(center: Point, radius: f64, material: MaterialRef) -> Self {
        Self::from_shared_geometry(SphereGeometry::new(center, radius.max(0.0)), material)
    }

    /// Creates a sphere from shared analytic geometry and material.
    #[must_use]
    pub fn from_geometry(geometry: SphereGeometry, material: impl Material + 'static) -> Self {
        Self::from_shared_geometry(geometry, Arc::new(material))
    }

    /// Creates a sphere from shared analytic geometry and a shared material handle.
    #[must_use]
    pub fn from_shared_geometry(geometry: SphereGeometry, material: MaterialRef) -> Self {
        Self {
            object: SceneObject::with_shared_material(geometry, material),
        }
    }

    /// Returns the shared analytic sphere geometry.
    #[must_use]
    pub fn geometry(&self) -> SphereGeometry {
        *self.object.geometry()
    }

    /// Returns the sphere center.
    #[must_use]
    pub fn center(&self) -> Point {
        self.geometry().center()
    }

    /// Returns the sphere radius.
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.geometry().radius()
    }

    /// Returns the material associated with this sphere.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.object.material()
    }
}

impl Hittable for Sphere {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.object.hit_with_rng(ray, ray_t, rng)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box()
    }

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        sphere_pdf_value(self.geometry(), context.origin, direction)
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        random_direction_to_sphere(self.geometry(), context.origin, rng)
    }
}

/// A linearly moving sphere hittable.
#[derive(Clone)]
pub struct MovingSphere {
    object: SceneObject<MovingSphereGeometry>,
}

impl fmt::Debug for MovingSphere {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MovingSphere")
            .field("geometry", self.object.geometry())
            .finish_non_exhaustive()
    }
}

impl MovingSphere {
    /// Creates a gray diffuse moving sphere.
    #[must_use]
    pub fn new(center_start: Point, center_end: Point, radius: f64) -> Self {
        Self::with_shared_material(center_start, center_end, radius, default_material())
    }

    /// Creates a moving sphere with a concrete material.
    #[must_use]
    pub fn with_material(
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: impl Material + 'static,
    ) -> Self {
        Self::with_shared_material(center_start, center_end, radius, Arc::new(material))
    }

    /// Creates a moving sphere with a shared material handle.
    #[must_use]
    pub fn with_shared_material(
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: MaterialRef,
    ) -> Self {
        Self::from_shared_geometry(
            MovingSphereGeometry::new(center_start, center_end, radius),
            material,
        )
    }

    /// Creates a moving sphere from shared analytic geometry and a concrete material.
    #[must_use]
    pub fn from_geometry(
        geometry: MovingSphereGeometry,
        material: impl Material + 'static,
    ) -> Self {
        Self::from_shared_geometry(geometry, Arc::new(material))
    }

    /// Creates a moving sphere from shared analytic geometry and a shared material handle.
    #[must_use]
    pub fn from_shared_geometry(geometry: MovingSphereGeometry, material: MaterialRef) -> Self {
        Self {
            object: SceneObject::with_shared_material(geometry, material),
        }
    }

    /// Returns the shared analytic moving sphere geometry.
    #[must_use]
    pub fn geometry(&self) -> MovingSphereGeometry {
        *self.object.geometry()
    }

    /// Returns the material associated with this moving sphere.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.object.material()
    }
}

impl Hittable for MovingSphere {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.object.hit_with_rng(ray, ray_t, rng)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box()
    }

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        let geometry = self.geometry();
        sphere_pdf_value(
            moving_sphere_at(geometry, context.time),
            context.origin,
            direction,
        )
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        let geometry = self.geometry();
        random_direction_to_sphere(
            moving_sphere_at(geometry, context.time),
            context.origin,
            rng,
        )
    }
}

/// A parallelogram hittable.
#[derive(Clone)]
pub struct Quad {
    object: SceneObject<QuadGeometry>,
}

impl fmt::Debug for Quad {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Quad")
            .field("geometry", self.object.geometry())
            .finish_non_exhaustive()
    }
}

impl Quad {
    /// Creates a gray diffuse quad from a corner and two side vectors.
    #[must_use]
    pub fn new(corner: Point, u: Vector, v: Vector) -> Self {
        Self::with_shared_material(corner, u, v, default_material())
    }

    /// Creates a quad with a concrete material.
    #[must_use]
    pub fn with_material(
        corner: Point,
        u: Vector,
        v: Vector,
        material: impl Material + 'static,
    ) -> Self {
        Self::with_shared_material(corner, u, v, Arc::new(material))
    }

    /// Creates a quad with a shared material handle.
    #[must_use]
    pub fn with_shared_material(
        corner: Point,
        u: Vector,
        v: Vector,
        material: MaterialRef,
    ) -> Self {
        Self::from_shared_geometry(QuadGeometry::new(corner, u, v), material)
    }

    /// Creates a quad from shared analytic geometry and a concrete material.
    #[must_use]
    pub fn from_geometry(geometry: QuadGeometry, material: impl Material + 'static) -> Self {
        Self::from_shared_geometry(geometry, Arc::new(material))
    }

    /// Creates a quad from shared analytic geometry and a shared material handle.
    #[must_use]
    pub fn from_shared_geometry(geometry: QuadGeometry, material: MaterialRef) -> Self {
        Self {
            object: SceneObject::with_shared_material(geometry, material),
        }
    }

    /// Returns the shared analytic quad geometry.
    #[must_use]
    pub fn geometry(&self) -> QuadGeometry {
        *self.object.geometry()
    }

    /// Returns the material associated with this quad.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.object.material()
    }
}

impl Hittable for Quad {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.object.hit_with_rng(ray, ray_t, rng)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box()
    }

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        quad_pdf_value(self.geometry(), context.origin, direction)
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        random_direction_to_quad(self.geometry(), context.origin, rng)
    }
}

/// Creates an axis-aligned box from six quad sides.
#[must_use]
pub fn box_object(a: Point, b: Point, material: MaterialRef) -> HittableList {
    let min = Point::new(a.x().min(b.x()), a.y().min(b.y()), a.z().min(b.z()));
    let max = Point::new(a.x().max(b.x()), a.y().max(b.y()), a.z().max(b.z()));

    let dx = Vector::new(max.x() - min.x(), 0.0, 0.0);
    let dy = Vector::new(0.0, max.y() - min.y(), 0.0);
    let dz = Vector::new(0.0, 0.0, max.z() - min.z());

    let mut sides = HittableList::with_capacity(6);
    sides.add(Quad::with_shared_material(
        Point::new(min.x(), min.y(), max.z()),
        dx,
        dy,
        material.clone(),
    ));
    sides.add(Quad::with_shared_material(
        Point::new(max.x(), min.y(), max.z()),
        -dz,
        dy,
        material.clone(),
    ));
    sides.add(Quad::with_shared_material(
        Point::new(max.x(), min.y(), min.z()),
        -dx,
        dy,
        material.clone(),
    ));
    sides.add(Quad::with_shared_material(
        Point::new(min.x(), min.y(), min.z()),
        dz,
        dy,
        material.clone(),
    ));
    sides.add(Quad::with_shared_material(
        Point::new(min.x(), max.y(), max.z()),
        dx,
        -dz,
        material.clone(),
    ));
    sides.add(Quad::with_shared_material(
        Point::new(min.x(), min.y(), min.z()),
        dx,
        dz,
        material,
    ));
    sides
}

/// Returns the nearest ray parameter where `ray` intersects the sphere.
#[must_use]
pub fn hit_sphere(center: Point, radius: f64, ray: &Ray) -> Option<f64> {
    hit_sphere_in_interval(center, radius, ray, Interval::UNIVERSE)
}

/// Returns the nearest ray parameter where `ray` intersects the sphere inside `ray_t`.
#[must_use]
pub fn hit_sphere_in_interval(
    center: Point,
    radius: f64,
    ray: &Ray,
    ray_t: Interval,
) -> Option<f64> {
    let oc = center - *ray.origin();
    let a = ray.direction().length_squared();
    if a <= f64::EPSILON {
        return None;
    }

    let h = ray.direction().dot(oc);
    let c = oc.length_squared() - radius * radius;
    let discriminant = h * h - a * c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrtd = discriminant.sqrt();
    let first = (h - sqrtd) / a;
    if ray_t.surrounds(first) {
        return Some(first);
    }

    let second = (h + sqrtd) / a;
    ray_t.surrounds(second).then_some(second)
}

fn sphere_bounds(geometry: SphereGeometry) -> Aabb {
    let radius = geometry.radius();
    let radius_vector = Vector::new(radius, radius, radius);
    Aabb::from_points(
        geometry.center() - radius_vector,
        geometry.center() + radius_vector,
    )
}

fn moving_sphere_bounds(geometry: MovingSphereGeometry) -> Aabb {
    let radius = geometry.radius();
    let radius_vector = Vector::new(radius, radius, radius);
    let start = Aabb::from_points(
        geometry.center_start() - radius_vector,
        geometry.center_start() + radius_vector,
    );
    let end = Aabb::from_points(
        geometry.center_end() - radius_vector,
        geometry.center_end() + radius_vector,
    );
    start.union(end)
}

pub(crate) fn sphere_uv(point_on_unit_sphere: Vector) -> (f64, f64) {
    let theta = (-point_on_unit_sphere.y()).acos();
    let phi = (-point_on_unit_sphere.z()).atan2(point_on_unit_sphere.x()) + PI;
    (phi / (2.0 * PI), theta / PI)
}

fn moving_sphere_at(geometry: MovingSphereGeometry, time: f64) -> SphereGeometry {
    SphereGeometry::new(geometry.center_at(time), geometry.radius())
}

fn sphere_pdf_value(geometry: SphereGeometry, origin: Point, direction: Vector) -> f64 {
    if geometry
        .intersect(
            &Ray::new(origin, direction),
            Interval::new(SHADOW_ACNE_PDF_EPSILON, INFINITY),
        )
        .is_none()
    {
        return 0.0;
    }

    let center_direction = geometry.center() - origin;
    let distance_squared = center_direction.length_squared();
    let radius_squared = geometry.radius() * geometry.radius();
    if distance_squared <= radius_squared {
        return 1.0 / (4.0 * PI);
    }

    let cos_theta_max = (1.0 - radius_squared / distance_squared).sqrt();
    let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
    if solid_angle <= f64::EPSILON {
        0.0
    } else {
        1.0 / solid_angle
    }
}

fn quad_pdf_value(geometry: QuadGeometry, origin: Point, direction: Vector) -> f64 {
    let Some(hit) = geometry.hit_ray(
        &Ray::new(origin, direction),
        SHADOW_ACNE_PDF_EPSILON,
        INFINITY,
    ) else {
        return 0.0;
    };

    let area = geometry.area_squared().sqrt();
    area_pdf_value(direction, hit.t, geometry.geometric_normal(), area)
}

fn triangle_pdf_value(geometry: TriangleGeometry, origin: Point, direction: Vector) -> f64 {
    let Some(hit) = geometry.hit_ray(
        &Ray::new(origin, direction),
        SHADOW_ACNE_PDF_EPSILON,
        INFINITY,
    ) else {
        return 0.0;
    };

    let area = 0.5 * geometry.area_squared().sqrt();
    area_pdf_value(direction, hit.t, geometry.geometric_normal(), area)
}

fn area_pdf_value(direction: Vector, hit_t: f64, normal: Vector, area: f64) -> f64 {
    if area <= f64::EPSILON {
        return 0.0;
    }

    let distance_squared = (hit_t * direction).length_squared();
    let cosine = direction.normalized().dot(normal).abs();
    if cosine <= f64::EPSILON {
        return 0.0;
    }

    distance_squared / (cosine * area)
}

fn random_direction_to_sphere(
    geometry: SphereGeometry,
    origin: Point,
    rng: &mut SampleRng,
) -> Vector {
    let direction = geometry.center() - origin;
    let distance_squared = direction.length_squared();
    let radius_squared = geometry.radius() * geometry.radius();
    if distance_squared <= radius_squared || distance_squared <= f64::EPSILON {
        return rng.random_unit_vector_spherical();
    }

    let local = random_to_sphere(geometry.radius(), distance_squared, rng);
    crate::gmath::geometry::OrthonormalBasis::from_w(direction)
        .map_or(local, |basis| basis.local(local))
}

fn random_direction_to_quad(geometry: QuadGeometry, origin: Point, rng: &mut SampleRng) -> Vector {
    let random_point =
        geometry.corner() + rng.random_double() * geometry.u() + rng.random_double() * geometry.v();
    random_point - origin
}

fn random_direction_to_triangle(
    geometry: TriangleGeometry,
    origin: Point,
    rng: &mut SampleRng,
) -> Vector {
    let [p0, p1, p2] = geometry.vertices();
    let mut a = rng.random_double();
    let mut b = rng.random_double();
    if a + b > 1.0 {
        a = 1.0 - a;
        b = 1.0 - b;
    }
    let random_point = p0 + a * (p1 - p0) + b * (p2 - p0);
    random_point - origin
}

fn random_to_sphere(radius: f64, distance_squared: f64, rng: &mut SampleRng) -> Vector {
    let r1 = rng.random_double();
    let r2 = rng.random_double();
    let z = 1.0 + r2 * ((1.0 - radius * radius / distance_squared).sqrt() - 1.0);
    let phi = 2.0 * PI * r1;
    let radius_at_z = (1.0 - z * z).sqrt();
    Vector::new(phi.cos() * radius_at_z, phi.sin() * radius_at_z, z)
}

const SHADOW_ACNE_PDF_EPSILON: f64 = 0.001;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_interval_constructor_requires_finite_ordered_bounds() {
        assert_eq!(Interval::try_new(0.0, 1.0), Some(Interval::new(0.0, 1.0)));
        assert_eq!(Interval::try_new(1.0, 0.0), None);
        assert_eq!(Interval::try_new(0.0, f64::INFINITY), None);
        assert_eq!(Interval::try_new(f64::NAN, 1.0), None);
    }
}

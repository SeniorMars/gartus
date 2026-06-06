//! Minimal ray-tracing helpers following the early "Ray Tracing in One Weekend" steps.

pub use crate::gmath::random::SampleRng;
pub mod scenes;
pub mod weekend;

use crate::{
    gmath::{
        geometry::{SphereGeometry, TriangleGeometry},
        polygon_matrix::{Bounds3, PolygonMatrix},
        ray::Ray,
        vector::{Point, Vector},
    },
    graphics::{
        camera::RayCamera,
        colors::{LinearRgb, Rgb},
        display::Canvas,
        lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex, SurfaceMaterial},
    },
};
pub use scenes::*;
use std::{fmt, sync::Arc};

/// Floating-point infinity for ray intervals.
pub const INFINITY: f64 = f64::INFINITY;

/// Pi, provided with the book's common ray-tracing constants.
pub const PI: f64 = std::f64::consts::PI;

/// The 16:9 aspect ratio used by the first weekend camera setup.
pub const WIDESCREEN_ASPECT_RATIO: f64 = 16.0 / 9.0;

/// A color represented as linear floating-point RGB components in `0.0..=1.0`.
pub type LinearColor = LinearRgb;

/// Axis-aligned bounding box used by ray-tracing acceleration structures.
pub type Aabb = Bounds3;

/// Minimum ray parameter accepted for secondary rays to avoid self-intersections.
pub const SHADOW_ACNE_EPSILON: f64 = 0.001;

/// Converts degrees to radians.
#[must_use]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

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

/// A ray scattered by a material, with the color attenuation applied to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterRecord {
    /// Scattered ray.
    pub ray: Ray,
    /// Per-channel color attenuation.
    pub attenuation: LinearColor,
}

/// A surface material that can scatter rays.
pub trait Material: Send + Sync {
    /// Produces a scattered ray and attenuation for a surface hit.
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord>;
}

/// Lambertian diffuse material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lambertian {
    /// Diffuse reflectance.
    pub albedo: LinearColor,
}

impl Lambertian {
    /// Creates a Lambertian material with the supplied albedo.
    #[must_use]
    pub fn new(albedo: LinearColor) -> Self {
        Self { albedo }
    }

    /// Creates a Lambertian material from display RGB bytes.
    #[must_use]
    pub fn from_rgb(color: Rgb) -> Self {
        Self::new(rgb_to_linear_color(color))
    }

    /// Creates a Lambertian material from existing diffuse reflection constants.
    #[must_use]
    pub fn from_reflectance(reflectance: ReflectionConstants) -> Self {
        Self::new(LinearColor::new(
            reflectance.red,
            reflectance.green,
            reflectance.blue,
        ))
    }

    /// Creates a Lambertian material from the diffuse component of a Phong material.
    #[must_use]
    pub fn from_phong_diffuse(material: PhongMaterial) -> Self {
        Self::from_reflectance(material.diffuse)
    }
}

impl From<Rgb> for Lambertian {
    fn from(color: Rgb) -> Self {
        Self::from_rgb(color)
    }
}

impl From<ReflectionConstants> for Lambertian {
    fn from(reflectance: ReflectionConstants) -> Self {
        Self::from_reflectance(reflectance)
    }
}

impl From<PhongMaterial> for Lambertian {
    fn from(material: PhongMaterial) -> Self {
        Self::from_phong_diffuse(material)
    }
}

impl From<SurfaceMaterial> for Lambertian {
    fn from(material: SurfaceMaterial) -> Self {
        Self::new(material.base_color)
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        _ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let mut scatter_direction = hit.normal + rng.random_unit_vector();
        if scatter_direction.length_squared() < 1e-20 {
            scatter_direction = hit.normal;
        }

        Some(ScatterRecord {
            ray: Ray::new(hit.point, scatter_direction),
            attenuation: self.albedo,
        })
    }
}

/// Reflective metal material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metal {
    /// Reflected ray attenuation.
    pub albedo: LinearColor,
    /// Reflection fuzziness in `0.0..=1.0`.
    pub fuzz: f64,
}

impl Metal {
    /// Creates a metal material with clamped fuzziness.
    #[must_use]
    pub fn new(albedo: LinearColor, fuzz: f64) -> Self {
        Self {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
        }
    }

    /// Creates a sharp metal material from display RGB bytes.
    #[must_use]
    pub fn from_rgb(color: Rgb) -> Self {
        Self::new(rgb_to_linear_color(color), 0.0)
    }

    /// Creates a metal material from existing reflection constants.
    #[must_use]
    pub fn from_reflectance(reflectance: ReflectionConstants, fuzz: f64) -> Self {
        Self::new(
            LinearColor::new(reflectance.red, reflectance.green, reflectance.blue),
            fuzz,
        )
    }

    /// Creates a metal material from the specular component of a Phong material.
    #[must_use]
    pub fn from_phong_specular(material: PhongMaterial, fuzz: f64) -> Self {
        Self::from_reflectance(material.specular, fuzz)
    }
}

impl From<Rgb> for Metal {
    fn from(color: Rgb) -> Self {
        Self::from_rgb(color)
    }
}

impl From<ReflectionConstants> for Metal {
    fn from(reflectance: ReflectionConstants) -> Self {
        Self::from_reflectance(reflectance, 0.0)
    }
}

impl From<PhongMaterial> for Metal {
    fn from(material: PhongMaterial) -> Self {
        Self::from_phong_specular(material, 0.0)
    }
}

impl From<SurfaceMaterial> for Metal {
    fn from(material: SurfaceMaterial) -> Self {
        Self::new(material.specular_color, 0.0)
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let reflected = ray_in.direction().normalized().reflected(hit.normal);
        let scattered_direction = reflected + self.fuzz * rng.random_unit_vector();
        if scattered_direction.dot(hit.normal) <= 0.0 {
            return None;
        }

        Some(ScatterRecord {
            ray: Ray::new(hit.point, scattered_direction),
            attenuation: self.albedo,
        })
    }
}

/// Transparent dielectric material such as glass, water, or diamond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dielectric {
    /// Refractive index in air/vacuum, or relative to the enclosing medium.
    pub refraction_index: RefractiveIndex,
}

impl Dielectric {
    /// Creates a dielectric material.
    #[must_use]
    pub const fn new(refraction_index: RefractiveIndex) -> Self {
        Self { refraction_index }
    }

    /// Creates a dielectric material from a raw refractive-index ratio.
    #[must_use]
    pub fn from_ratio(refraction_index: f64) -> Self {
        Self::new(RefractiveIndex::new(refraction_index))
    }

    /// Returns Schlick's approximation for angle-dependent reflectance.
    #[must_use]
    pub fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
        let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl From<RefractiveIndex> for Dielectric {
    fn from(refraction_index: RefractiveIndex) -> Self {
        Self::new(refraction_index)
    }
}

impl TryFrom<SurfaceMaterial> for Dielectric {
    type Error = &'static str;

    fn try_from(material: SurfaceMaterial) -> Result<Self, Self::Error> {
        material
            .refractive_index
            .map(Self::new)
            .ok_or("surface material has no refractive index")
    }
}

impl SurfaceMaterial {
    /// Derives a raytracing Lambertian material from this shared surface material.
    #[must_use]
    pub fn as_lambertian(&self) -> Lambertian {
        Lambertian::new(self.base_color)
    }

    /// Derives a raytracing metal material from this shared surface material.
    #[must_use]
    pub fn as_metal(&self, fuzz: f64) -> Metal {
        Metal::new(self.specular_color, fuzz)
    }

    /// Derives a raytracing dielectric material when this surface has a refractive index.
    #[must_use]
    pub fn as_dielectric(&self) -> Option<Dielectric> {
        self.refractive_index.map(Dielectric::new)
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let attenuation = LinearColor::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit.front_face {
            1.0 / self.refraction_index.0
        } else {
            self.refraction_index.0
        };

        let unit_direction = ray_in.direction().normalized();
        let cos_theta = (-unit_direction).dot(hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction = if cannot_refract
            || Self::reflectance(cos_theta, refraction_ratio) > rng.random_double()
        {
            unit_direction.reflected(hit.normal)
        } else {
            unit_direction.refracted(hit.normal, refraction_ratio)
        };

        Some(ScatterRecord {
            ray: Ray::new(hit.point, direction),
            attenuation,
        })
    }
}

/// Material variants supported by the data-oriented ray scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RayMaterial {
    /// Lambertian diffuse material.
    Lambertian(Lambertian),
    /// Reflective metal material.
    Metal(Metal),
    /// Transparent dielectric material.
    Dielectric(Dielectric),
}

impl RayMaterial {
    /// Creates a Lambertian material variant.
    #[must_use]
    pub const fn lambertian(albedo: LinearColor) -> Self {
        Self::Lambertian(Lambertian { albedo })
    }

    /// Creates a metal material variant.
    #[must_use]
    pub fn metal(albedo: LinearColor, fuzz: f64) -> Self {
        Self::Metal(Metal::new(albedo, fuzz))
    }

    /// Creates a dielectric material variant.
    #[must_use]
    pub const fn dielectric(refraction_index: RefractiveIndex) -> Self {
        Self::Dielectric(Dielectric::new(refraction_index))
    }

    /// Derives a Lambertian ray material from shared surface material data.
    #[must_use]
    pub fn from_surface_lambertian(material: &SurfaceMaterial) -> Self {
        Self::Lambertian(material.as_lambertian())
    }

    /// Derives a metal ray material from shared surface material data.
    #[must_use]
    pub fn from_surface_metal(material: &SurfaceMaterial, fuzz: f64) -> Self {
        Self::Metal(material.as_metal(fuzz))
    }

    /// Derives a dielectric ray material when shared surface data has a refractive index.
    #[must_use]
    pub fn from_surface_dielectric(material: &SurfaceMaterial) -> Option<Self> {
        material.as_dielectric().map(Self::Dielectric)
    }
}

impl From<Lambertian> for RayMaterial {
    fn from(material: Lambertian) -> Self {
        Self::Lambertian(material)
    }
}

impl From<Metal> for RayMaterial {
    fn from(material: Metal) -> Self {
        Self::Metal(material)
    }
}

impl From<Dielectric> for RayMaterial {
    fn from(material: Dielectric) -> Self {
        Self::Dielectric(material)
    }
}

impl Material for RayMaterial {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        match self {
            Self::Lambertian(material) => material.scatter(ray_in, hit, rng),
            Self::Metal(material) => material.scatter(ray_in, hit, rng),
            Self::Dielectric(material) => material.scatter(ray_in, hit, rng),
        }
    }
}

/// Shared material handle used by hittable objects.
pub type MaterialRef = Arc<dyn Material>;

fn default_material() -> MaterialRef {
    Arc::new(Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)))
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
    /// True when the ray hit the outside face of the surface.
    pub front_face: bool,
}

impl SurfaceHit {
    /// Creates a surface hit and orients `outward_normal` against `ray`.
    ///
    /// `outward_normal` is expected to have unit length.
    #[must_use]
    pub fn new(ray: &Ray, point: Point, outward_normal: Vector, t: f64) -> Self {
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
        Some(SurfaceHit::new(ray, point, outward_normal, root))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(sphere_bounds(*self))
    }
}

impl Intersect for TriangleGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        let hit = self.hit_ray(ray, ray_t.min, ray_t.max)?;
        let normal = self.geometric_normal();
        if normal.length_squared() <= f64::EPSILON {
            return None;
        }
        Some(SurfaceHit::new(ray, ray.at(hit.t), normal, hit.t))
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
    /// Analytic triangle geometry.
    Triangle(TriangleGeometry),
}

impl RayGeometry {
    /// Creates a sphere geometry variant.
    #[must_use]
    pub fn sphere(center: Point, radius: f64) -> Self {
        Self::Sphere(SphereGeometry::new(center, radius))
    }

    /// Creates a triangle geometry variant.
    #[must_use]
    pub const fn triangle(p0: Point, p1: Point, p2: Point) -> Self {
        Self::Triangle(TriangleGeometry::new(p0, p1, p2))
    }
}

impl From<SphereGeometry> for RayGeometry {
    fn from(geometry: SphereGeometry) -> Self {
        Self::Sphere(geometry)
    }
}

impl From<TriangleGeometry> for RayGeometry {
    fn from(geometry: TriangleGeometry) -> Self {
        Self::Triangle(geometry)
    }
}

impl Intersect for RayGeometry {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<SurfaceHit> {
        match self {
            Self::Sphere(geometry) => geometry.intersect(ray, ray_t),
            Self::Triangle(geometry) => geometry.intersect(ray, ray_t),
        }
    }

    fn bounding_box(&self) -> Option<Aabb> {
        match self {
            Self::Sphere(geometry) => geometry.bounding_box(),
            Self::Triangle(geometry) => geometry.bounding_box(),
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
    /// Ray parameter at the hit point.
    pub t: f64,
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
            .field("t", &self.t)
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
            t: surface.t,
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
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let surface = self.geometry.intersect(ray, ray_t)?;
        Some(HitRecord::from_surface(surface, self.material()))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.geometry.bounding_box()
    }
}

/// A scene object that can be intersected by a ray.
pub trait Hittable: Send + Sync {
    /// Returns the closest hit inside `ray_t`, if any.
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>>;

    /// Returns an axis-aligned bounding box for acceleration structures, if available.
    fn bounding_box(&self) -> Option<Aabb> {
        None
    }
}

/// A sphere hittable.
#[derive(Clone)]
pub struct Sphere {
    geometry: SphereGeometry,
    material: MaterialRef,
}

impl fmt::Debug for Sphere {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Sphere")
            .field("geometry", &self.geometry)
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
        Self {
            geometry: SphereGeometry::new(center, radius.max(0.0)),
            material,
        }
    }

    /// Creates a sphere from shared analytic geometry and material.
    #[must_use]
    pub fn from_geometry(geometry: SphereGeometry, material: impl Material + 'static) -> Self {
        Self::from_shared_geometry(geometry, Arc::new(material))
    }

    /// Creates a sphere from shared analytic geometry and a shared material handle.
    #[must_use]
    pub fn from_shared_geometry(geometry: SphereGeometry, material: MaterialRef) -> Self {
        Self { geometry, material }
    }

    /// Returns the shared analytic sphere geometry.
    #[must_use]
    pub fn geometry(&self) -> SphereGeometry {
        self.geometry
    }

    /// Returns the sphere center.
    #[must_use]
    pub fn center(&self) -> Point {
        self.geometry.center()
    }

    /// Returns the sphere radius.
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.geometry.radius()
    }

    /// Returns the material associated with this sphere.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let surface = self.geometry.intersect(ray, ray_t)?;
        Some(HitRecord::from_surface(surface, self.material()))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(sphere_bounds(self.geometry))
    }
}

/// A collection of hittable scene objects.
#[derive(Default)]
pub struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl fmt::Debug for HittableList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HittableList")
            .field("len", &self.objects.len())
            .finish()
    }
}

impl HittableList {
    /// Creates an empty hittable list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty hittable list with space for at least `capacity` objects.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
        }
    }

    /// Creates a hittable list containing one object.
    #[must_use]
    pub fn with_object(object: impl Hittable + 'static) -> Self {
        let mut list = Self::new();
        list.add(object);
        list
    }

    /// Removes all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    /// Adds an object to the scene.
    pub fn add(&mut self, object: impl Hittable + 'static) {
        self.objects.push(Box::new(object));
    }

    /// Adds a boxed hittable object to the scene.
    pub fn add_box(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }

    /// Builds a BVH from this list, returning `None` if any object lacks bounds.
    #[must_use]
    pub fn into_bvh(self) -> Option<BvhNode> {
        BvhNode::from_hittables(self.objects)
    }

    /// Returns the number of objects in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns true when the scene has no objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for object in &self.objects {
            if let Some(record) = object.hit(ray, Interval::new(ray_t.min, closest_so_far)) {
                closest_so_far = record.t;
                closest_hit = Some(record);
            }
        }

        closest_hit
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let mut objects = self.objects.iter();
        let first = objects.next()?.bounding_box()?;
        objects.try_fold(first, |bounds, object| {
            object.bounding_box().map(|other| bounds.surrounding(other))
        })
    }
}

/// Bounding-volume hierarchy over arbitrary bounded hittables.
pub struct BvhNode {
    objects: Vec<Box<dyn Hittable>>,
    bvh: ObjectBvh,
    bounds: Aabb,
}

impl fmt::Debug for BvhNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BvhNode")
            .field("objects", &self.objects.len())
            .field("bounds", &self.bounds)
            .field("nodes", &self.bvh.nodes.len())
            .finish_non_exhaustive()
    }
}

impl BvhNode {
    /// Builds a BVH from bounded hittable objects.
    #[must_use]
    pub fn from_hittables(objects: Vec<Box<dyn Hittable>>) -> Option<Self> {
        if objects.is_empty() || objects.iter().any(|object| object.bounding_box().is_none()) {
            return None;
        }
        let bvh = ObjectBvh::build(&objects)?;
        let bounds = bvh.bounds();
        Some(Self {
            objects,
            bvh,
            bounds,
        })
    }

    /// Brute-force hit path used as a correctness oracle for the object BVH.
    #[must_use]
    pub fn hit_bruteforce(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        hit_object_indices(&self.objects, 0..self.objects.len(), ray, ray_t)
    }
}

impl Hittable for BvhNode {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        self.bvh.hit(&self.objects, ray, ray_t)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(self.bounds)
    }
}

#[derive(Clone, Debug)]
struct ObjectBvh {
    nodes: Vec<ObjectBvhNode>,
    indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct ObjectBvhNode {
    bounds: Aabb,
    kind: ObjectBvhNodeKind,
}

#[derive(Clone, Copy, Debug)]
enum ObjectBvhNodeKind {
    Leaf { first: usize, count: usize },
    Internal { left: usize, right: usize },
}

impl ObjectBvh {
    const LEAF_SIZE: usize = 4;

    fn build(objects: &[Box<dyn Hittable>]) -> Option<Self> {
        if objects.is_empty() {
            return None;
        }

        let primitive_info = objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                let bounds = object.bounding_box().expect("bounded object");
                BvhPrimitiveInfo::new(index, bounds)
            })
            .collect::<Vec<_>>();
        let mut bvh = Self {
            nodes: Vec::with_capacity(objects.len().saturating_mul(2).saturating_sub(1)),
            indices: (0..objects.len()).collect(),
        };
        bvh.build_range(&primitive_info, 0, objects.len());
        Some(bvh)
    }

    fn bounds(&self) -> Aabb {
        self.nodes[0].bounds
    }

    fn build_range(
        &mut self,
        primitive_info: &[BvhPrimitiveInfo],
        first: usize,
        count: usize,
    ) -> usize {
        let bounds = bounds_for_primitive_indices(
            primitive_info,
            self.indices[first..first + count].iter().copied(),
        )
        .expect("BVH node has at least one object");
        let node_index = self.nodes.len();
        self.nodes.push(ObjectBvhNode {
            bounds,
            kind: ObjectBvhNodeKind::Leaf { first, count },
        });

        if let Some(left_count) = split_bvh_indices(
            &mut self.indices[first..first + count],
            primitive_info,
            Self::LEAF_SIZE,
        ) {
            let right_count = count - left_count;
            let left = self.build_range(primitive_info, first, left_count);
            let right = self.build_range(primitive_info, first + left_count, right_count);
            self.nodes[node_index].kind = ObjectBvhNodeKind::Internal { left, right };
        }

        node_index
    }

    fn hit<'a>(
        &'a self,
        objects: &'a [Box<dyn Hittable>],
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        self.hit_node(0, objects, ray, ray_t)
    }

    fn hit_node<'a>(
        &'a self,
        node_index: usize,
        objects: &'a [Box<dyn Hittable>],
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        let node = self.nodes[node_index];
        if !node.bounds.hit_ray(ray, ray_t.min, ray_t.max) {
            return None;
        }

        match node.kind {
            ObjectBvhNodeKind::Leaf { first, count } => hit_object_indices(
                objects,
                self.indices[first..first + count].iter().copied(),
                ray,
                ray_t,
            ),
            ObjectBvhNodeKind::Internal { left, right } => {
                let left_hit = self.hit_node(left, objects, ray, ray_t);
                let closest = left_hit.as_ref().map_or(ray_t.max, |hit| hit.t);
                let right_hit =
                    self.hit_node(right, objects, ray, Interval::new(ray_t.min, closest));
                right_hit.or(left_hit)
            }
        }
    }
}

fn hit_object_indices<'a>(
    objects: &'a [Box<dyn Hittable>],
    indices: impl IntoIterator<Item = usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        if let Some(record) = objects[index].hit(ray, Interval::new(ray_t.min, closest_so_far)) {
            closest_so_far = record.t;
            closest_hit = Some(record);
        }
    }

    closest_hit
}

/// Index into a [`RayScene`] material table.
pub type MaterialId = usize;

/// One data-oriented scene primitive: compact geometry plus a material table index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayPrimitive {
    /// Primitive geometry.
    pub geometry: RayGeometry,
    /// Material table index.
    pub material: MaterialId,
}

/// Data-oriented ray scene with enum geometry and a compact material table.
#[derive(Clone, Debug, Default)]
pub struct RayScene {
    materials: Vec<RayMaterial>,
    primitives: Vec<RayPrimitive>,
}

impl RayScene {
    /// Creates an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty scene with reserved material and primitive capacity.
    #[must_use]
    pub fn with_capacity(materials: usize, primitives: usize) -> Self {
        Self {
            materials: Vec::with_capacity(materials),
            primitives: Vec::with_capacity(primitives),
        }
    }

    /// Adds a material and returns its table index.
    pub fn add_material(&mut self, material: impl Into<RayMaterial>) -> MaterialId {
        let id = self.materials.len();
        self.materials.push(material.into());
        id
    }

    /// Adds a primitive using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_primitive(&mut self, geometry: impl Into<RayGeometry>, material: MaterialId) {
        assert!(
            material < self.materials.len(),
            "ray scene material id out of bounds"
        );
        self.primitives.push(RayPrimitive {
            geometry: geometry.into(),
            material,
        });
    }

    /// Adds a sphere using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_sphere(&mut self, center: Point, radius: f64, material: MaterialId) {
        self.add_primitive(RayGeometry::sphere(center, radius), material);
    }

    /// Adds a material and a sphere that references it.
    pub fn add_sphere_with_material(
        &mut self,
        center: Point,
        radius: f64,
        material: impl Into<RayMaterial>,
    ) -> MaterialId {
        let material = self.add_material(material);
        self.add_sphere(center, radius, material);
        material
    }

    /// Returns a material by id.
    #[must_use]
    pub fn material(&self, id: MaterialId) -> Option<&RayMaterial> {
        self.materials.get(id)
    }

    /// Returns the material table.
    #[must_use]
    pub fn materials(&self) -> &[RayMaterial] {
        &self.materials
    }

    /// Returns scene primitives.
    #[must_use]
    pub fn primitives(&self) -> &[RayPrimitive] {
        &self.primitives
    }

    /// Returns the number of primitives in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Returns true when the scene has no primitives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Returns the number of stored materials.
    #[must_use]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
}

impl Hittable for RayScene {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for primitive in &self.primitives {
            if let Some(surface) = primitive
                .geometry
                .intersect(ray, Interval::new(ray_t.min, closest_so_far))
            {
                closest_so_far = surface.t;
                closest_hit = Some(HitRecord::from_surface(
                    surface,
                    &self.materials[primitive.material],
                ));
            }
        }

        closest_hit
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let mut primitives = self.primitives.iter();
        let first = primitives.next()?.geometry.bounding_box()?;
        primitives.try_fold(first, |bounds, primitive| {
            primitive
                .geometry
                .bounding_box()
                .map(|other| bounds.surrounding(other))
        })
    }
}

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

/// Triangle mesh with a monomorphic internal BVH.
#[derive(Clone)]
pub struct TriangleMesh {
    triangles: Vec<TriangleGeometry>,
    material: MaterialRef,
    bounds: Option<Aabb>,
    bvh: Option<TriangleBvh>,
}

impl fmt::Debug for TriangleMesh {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TriangleMesh")
            .field("triangles", &self.triangles.len())
            .field("bounds", &self.bounds)
            .field("has_bvh", &self.bvh.is_some())
            .finish_non_exhaustive()
    }
}

impl TriangleMesh {
    /// Creates a triangle mesh from geometry and a concrete material.
    #[must_use]
    pub fn new(triangles: Vec<TriangleGeometry>, material: impl Material + 'static) -> Self {
        Self::with_shared_material(triangles, Arc::new(material))
    }

    /// Creates a triangle mesh from geometry and a shared material handle.
    #[must_use]
    pub fn with_shared_material(triangles: Vec<TriangleGeometry>, material: MaterialRef) -> Self {
        let bounds = triangle_bounds_for_slice(&triangles);
        let bvh = TriangleBvh::build(&triangles);
        Self {
            triangles,
            material,
            bounds,
            bvh,
        }
    }

    /// Creates a triangle mesh from a polygon matrix and a concrete material.
    #[must_use]
    pub fn from_polygon_matrix(mesh: &PolygonMatrix, material: impl Material + 'static) -> Self {
        Self::from_shared_polygon_matrix(mesh, Arc::new(material))
    }

    /// Creates a triangle mesh from a polygon matrix and shared material handle.
    #[must_use]
    pub fn from_shared_polygon_matrix(mesh: &PolygonMatrix, material: MaterialRef) -> Self {
        let triangles = mesh
            .iter_triangles()
            .map(|(p0, p1, p2)| {
                TriangleGeometry::new(
                    Point::new(p0[0], p0[1], p0[2]),
                    Point::new(p1[0], p1[1], p1[2]),
                    Point::new(p2[0], p2[1], p2[2]),
                )
            })
            .collect();
        Self::with_shared_material(triangles, material)
    }

    /// Creates a triangle mesh from one material mesh group and a shared material handle.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn from_material_mesh_group_with_shared_material(
        group: &crate::external::MaterialMeshGroup,
        material: MaterialRef,
    ) -> Self {
        Self::from_shared_polygon_matrix(&group.polygons, material)
    }

    /// Creates one triangle mesh per material group using a caller-supplied material policy.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn from_material_mesh_with_policy<F>(
        mesh: &crate::external::MaterialMesh,
        mut policy: F,
    ) -> Vec<Self>
    where
        F: FnMut(&crate::external::MaterialMeshGroup) -> MaterialRef,
    {
        mesh.groups
            .iter()
            .filter(|group| !group.polygons.is_empty())
            .map(|group| Self::from_material_mesh_group_with_shared_material(group, policy(group)))
            .collect()
    }

    /// Creates one Lambertian triangle mesh per material group.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn from_material_mesh_lambertian(mesh: &crate::external::MaterialMesh) -> Vec<Self> {
        Self::from_material_mesh_with_policy(mesh, default_material_for_mesh_group)
    }

    /// Returns the triangle geometry slice.
    #[must_use]
    pub fn triangles(&self) -> &[TriangleGeometry] {
        &self.triangles
    }

    /// Returns the number of triangles in this mesh.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    /// Returns true if this mesh has no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Returns the material associated with this mesh.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }

    /// Brute-force hit path used for testing and diagnostics.
    #[must_use]
    pub fn hit_bruteforce(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        hit_triangle_range(
            &self.triangles,
            self.material(),
            0..self.triangles.len(),
            ray,
            ray_t,
        )
    }
}

#[cfg(feature = "external")]
fn default_material_for_mesh_group(group: &crate::external::MaterialMeshGroup) -> MaterialRef {
    if let Some(material) = group.material.clone() {
        Arc::new(Lambertian::from(SurfaceMaterial::from(material)))
    } else if let Some(diffuse_color) = group.diffuse_color {
        Arc::new(Lambertian::from(diffuse_color))
    } else {
        default_material()
    }
}

impl Hittable for TriangleMesh {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        self.bvh.as_ref().map_or_else(
            || self.hit_bruteforce(ray, ray_t),
            |bvh| bvh.hit(&self.triangles, self.material(), ray, ray_t),
        )
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

const BVH_BUCKETS: usize = 12;
const BVH_BUCKETS_F64: f64 = 12.0;

#[derive(Clone, Copy, Debug)]
struct BvhPrimitiveInfo {
    bounds: Aabb,
    centroid: Point,
}

impl BvhPrimitiveInfo {
    fn new(_index: usize, bounds: Aabb) -> Self {
        Self {
            bounds,
            centroid: bounds.centroid(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BvhBucket {
    count: usize,
    bounds: Option<Aabb>,
}

impl BvhBucket {
    fn add(&mut self, bounds: Aabb) {
        self.count += 1;
        self.bounds = Some(self.bounds.map_or(bounds, |current| current.union(bounds)));
    }
}

fn split_bvh_indices(
    indices: &mut [usize],
    primitive_info: &[BvhPrimitiveInfo],
    leaf_size: usize,
) -> Option<usize> {
    if indices.len() <= leaf_size {
        return None;
    }

    let centroid_bounds =
        centroid_bounds_for_primitive_indices(primitive_info, indices.iter().copied())
            .expect("BVH split range has centroid bounds");
    let axis = centroid_bounds.largest_axis();
    let centroid_extent = centroid_bounds.axis_max(axis) - centroid_bounds.axis_min(axis);
    if centroid_extent <= f64::EPSILON {
        return Some(midpoint_split(indices, primitive_info, axis));
    }

    let mut buckets = [BvhBucket::default(); BVH_BUCKETS];
    for &index in indices.iter() {
        let offset = (point_axis(primitive_info[index].centroid, axis)
            - centroid_bounds.axis_min(axis))
            / centroid_extent;
        let bucket = bucket_index(offset);
        buckets[bucket].add(primitive_info[index].bounds);
    }

    let mut best_split = 0;
    let mut best_cost = f64::INFINITY;
    for split in 0..BVH_BUCKETS - 1 {
        let (left_count, left_bounds) = merge_buckets(&buckets[..=split]);
        let (right_count, right_bounds) = merge_buckets(&buckets[split + 1..]);
        if left_count == 0 || right_count == 0 {
            continue;
        }

        let cost = left_bounds.expect("left bounds").surface_area() * count_as_f64(left_count)
            + right_bounds.expect("right bounds").surface_area() * count_as_f64(right_count);
        if cost < best_cost {
            best_cost = cost;
            best_split = split;
        }
    }

    if !best_cost.is_finite() {
        return Some(midpoint_split(indices, primitive_info, axis));
    }

    let min_axis = centroid_bounds.axis_min(axis);
    let mut left_count = 0;
    for next in 0..indices.len() {
        let index = indices[next];
        let offset =
            (point_axis(primitive_info[index].centroid, axis) - min_axis) / centroid_extent;
        let bucket = bucket_index(offset);
        if bucket <= best_split {
            indices.swap(left_count, next);
            left_count += 1;
        }
    }

    if left_count == 0 || left_count == indices.len() {
        Some(midpoint_split(indices, primitive_info, axis))
    } else {
        Some(left_count)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bucket_index(offset: f64) -> usize {
    ((offset * BVH_BUCKETS_F64) as usize).min(BVH_BUCKETS - 1)
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}

fn midpoint_split(
    indices: &mut [usize],
    primitive_info: &[BvhPrimitiveInfo],
    axis: usize,
) -> usize {
    indices.sort_by(|left, right| {
        let left_axis = point_axis(primitive_info[*left].centroid, axis);
        let right_axis = point_axis(primitive_info[*right].centroid, axis);
        left_axis
            .partial_cmp(&right_axis)
            .expect("BVH centroids should be finite")
            .then_with(|| left.cmp(right))
    });
    indices.len() / 2
}

fn merge_buckets(buckets: &[BvhBucket]) -> (usize, Option<Aabb>) {
    buckets.iter().fold((0, None), |(count, bounds), bucket| {
        let count = count + bucket.count;
        let bounds = match (bounds, bucket.bounds) {
            (Some(left), Some(right)) => Some(left.union(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        (count, bounds)
    })
}

fn bounds_for_primitive_indices(
    primitive_info: &[BvhPrimitiveInfo],
    indices: impl IntoIterator<Item = usize>,
) -> Option<Aabb> {
    indices
        .into_iter()
        .map(|index| primitive_info[index].bounds)
        .reduce(Aabb::union)
}

fn centroid_bounds_for_primitive_indices(
    primitive_info: &[BvhPrimitiveInfo],
    indices: impl IntoIterator<Item = usize>,
) -> Option<Aabb> {
    let mut centroids = indices
        .into_iter()
        .map(|index| primitive_info[index].centroid);
    let first = centroids.next()?;
    Some(centroids.fold(Aabb::from_points(first, first), Aabb::union_point))
}

#[derive(Clone, Debug)]
struct TriangleBvh {
    nodes: Vec<TriangleBvhNode>,
    indices: Vec<usize>,
    primitive_info: Vec<BvhPrimitiveInfo>,
}

#[derive(Clone, Copy, Debug)]
struct TriangleBvhNode {
    bounds: Aabb,
    kind: TriangleBvhNodeKind,
}

#[derive(Clone, Copy, Debug)]
enum TriangleBvhNodeKind {
    Leaf { first: usize, count: usize },
    Internal { left: usize, right: usize },
}

impl TriangleBvh {
    const LEAF_SIZE: usize = 4;

    fn build(triangles: &[TriangleGeometry]) -> Option<Self> {
        if triangles.is_empty() {
            return None;
        }
        let mut bvh = Self {
            nodes: Vec::with_capacity(triangles.len().saturating_mul(2).saturating_sub(1)),
            indices: (0..triangles.len()).collect(),
            primitive_info: triangles
                .iter()
                .enumerate()
                .map(|(index, triangle)| {
                    BvhPrimitiveInfo::new(index, triangle.bounding_box().expect("bounded triangle"))
                })
                .collect(),
        };
        bvh.build_range(0, triangles.len());
        Some(bvh)
    }

    fn build_range(&mut self, first: usize, count: usize) -> usize {
        let bounds = bounds_for_primitive_indices(
            &self.primitive_info,
            self.indices[first..first + count].iter().copied(),
        )
        .expect("BVH node has at least one triangle");
        let node_index = self.nodes.len();
        self.nodes.push(TriangleBvhNode {
            bounds,
            kind: TriangleBvhNodeKind::Leaf { first, count },
        });

        if let Some(left_count) = split_bvh_indices(
            &mut self.indices[first..first + count],
            &self.primitive_info,
            Self::LEAF_SIZE,
        ) {
            let right_count = count - left_count;
            let left = self.build_range(first, left_count);
            let right = self.build_range(first + left_count, right_count);
            self.nodes[node_index].kind = TriangleBvhNodeKind::Internal { left, right };
        }

        node_index
    }

    fn hit<'a>(
        &'a self,
        triangles: &'a [TriangleGeometry],
        material: &'a dyn Material,
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        self.hit_node(0, triangles, material, ray, ray_t)
    }

    fn hit_node<'a>(
        &'a self,
        node_index: usize,
        triangles: &'a [TriangleGeometry],
        material: &'a dyn Material,
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        let node = self.nodes[node_index];
        if !node.bounds.hit_ray(ray, ray_t.min, ray_t.max) {
            return None;
        }

        match node.kind {
            TriangleBvhNodeKind::Leaf { first, count } => hit_triangle_indices(
                triangles,
                material,
                self.indices[first..first + count].iter().copied(),
                ray,
                ray_t,
            ),
            TriangleBvhNodeKind::Internal { left, right } => {
                let left_hit = self.hit_node(left, triangles, material, ray, ray_t);
                let closest = left_hit.as_ref().map_or(ray_t.max, |hit| hit.t);
                let right_hit = self.hit_node(
                    right,
                    triangles,
                    material,
                    ray,
                    Interval::new(ray_t.min, closest),
                );
                right_hit.or(left_hit)
            }
        }
    }
}

fn point_axis(point: Point, axis: usize) -> f64 {
    match axis {
        0 => point.x(),
        1 => point.y(),
        2 => point.z(),
        _ => panic!("point axis index out of bounds"),
    }
}

fn triangle_bounds_for_slice(triangles: &[TriangleGeometry]) -> Option<Aabb> {
    triangle_bounds_for_indices(triangles, 0..triangles.len())
}

fn triangle_bounds_for_indices(
    triangles: &[TriangleGeometry],
    indices: impl IntoIterator<Item = usize>,
) -> Option<Aabb> {
    indices
        .into_iter()
        .filter_map(|index| triangles[index].bounding_box())
        .reduce(Aabb::union)
}

fn hit_triangle_range<'a>(
    triangles: &'a [TriangleGeometry],
    material: &'a dyn Material,
    range: std::ops::Range<usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    hit_triangle_indices(triangles, material, range, ray, ray_t)
}

fn hit_triangle_indices<'a>(
    triangles: &'a [TriangleGeometry],
    material: &'a dyn Material,
    indices: impl IntoIterator<Item = usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        if let Some(surface) =
            triangles[index].intersect(ray, Interval::new(ray_t.min, closest_so_far))
        {
            closest_so_far = surface.t;
            closest_hit = Some(HitRecord::from_surface(surface, material));
        }
    }

    closest_hit
}

/// Sphere-only hittable list that avoids boxed geometry dispatch in hit loops.
#[derive(Clone, Debug, Default)]
pub struct SphereList {
    spheres: Vec<Sphere>,
}

impl SphereList {
    /// Creates an empty sphere list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty sphere list with space for `capacity` spheres.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            spheres: Vec::with_capacity(capacity),
        }
    }

    /// Adds a sphere.
    pub fn add(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    /// Returns the number of spheres.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spheres.len()
    }

    /// Returns true when there are no spheres.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spheres.is_empty()
    }
}

impl Hittable for SphereList {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for sphere in &self.spheres {
            if let Some(record) = sphere.hit(ray, Interval::new(ray_t.min, closest_so_far)) {
                closest_so_far = record.t;
                closest_hit = Some(record);
            }
        }

        closest_hit
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let mut spheres = self.spheres.iter();
        let first = spheres.next()?.bounding_box()?;
        spheres.try_fold(first, |bounds, sphere| {
            sphere.bounding_box().map(|other| bounds.surrounding(other))
        })
    }
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

pub(crate) fn component_mul(lhs: LinearColor, rhs: LinearColor) -> LinearColor {
    lhs.component_mul(rhs)
}

/// Converts a linear color component to gamma space using gamma 2.
#[must_use]
pub fn linear_to_gamma(linear_component: f64) -> f64 {
    Rgb::linear_to_gamma_component(linear_component)
}

/// Converts a linear ray-traced color to display RGB with gamma correction.
#[must_use]
pub fn linear_color_to_rgb(color: LinearColor) -> Rgb {
    Rgb::from_linear_color(color)
}

/// Converts display RGB bytes to linear RGB using the library's gamma-2 approximation.
#[must_use]
pub fn rgb_to_linear_color(color: Rgb) -> LinearColor {
    LinearColor::from_rgb_srgb(color)
}

/// Converts RGB bytes to unit channel values without gamma decoding.
#[must_use]
pub fn rgb_bytes_to_unit_color(color: Rgb) -> LinearColor {
    LinearColor::from_rgb_linear_units(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn unit_gradient_matches_first_ppm_image_corners() {
        let canvas = render_unit_gradient(3, 3);
        assert_eq!(canvas.pixels()[0], Rgb::BLACK);
        assert_eq!(canvas.pixels()[2], Rgb::new(255, 0, 0));
        assert_eq!(canvas.pixels()[8], Rgb::YELLOW);
    }

    #[test]
    fn sphere_intersection_detects_direct_hit_and_miss() {
        let origin = Point::new(0.0, 0.0, 0.0);
        let center = Point::new(0.0, 0.0, -1.0);

        let hit = Ray::new(origin, Vector::new(0.0, 0.0, -1.0));
        assert_close(hit_sphere(center, 0.5, &hit).expect("hit"), 0.5);
        assert_close(
            hit_sphere_in_interval(center, 0.5, &hit, Interval::new(0.6, INFINITY))
                .expect("far hit"),
            1.5,
        );

        let miss = Ray::new(origin, Vector::new(0.0, 1.0, -1.0));
        assert!(hit_sphere(center, 0.5, &miss).is_none());
        let zero_direction = Ray::new(origin, Vector::default());
        assert!(hit_sphere(center, 0.5, &zero_direction).is_none());
    }

    #[test]
    fn sky_gradient_blends_white_to_blue() {
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let color = sky_gradient(&ray);

        assert_close(color.x(), 0.75);
        assert_close(color.y(), 0.85);
        assert_close(color.z(), 1.0);
    }

    #[test]
    fn first_sphere_render_has_red_center() {
        let canvas = render_first_sphere(40);
        let center = canvas
            .get_pixel(20, 11)
            .expect("center pixel should be inside the canvas");

        assert_eq!(*center, Rgb::RED);
    }

    #[test]
    fn interval_contains_and_surrounds_values() {
        let interval = Interval::new(1.0, 2.0);

        assert!(interval.contains(1.0));
        assert!(interval.contains(2.0));
        assert!(!interval.surrounds(1.0));
        assert!(interval.surrounds(1.5));
        assert_close(interval.clamp(3.0), 2.0);
    }

    #[test]
    fn sample_rng_returns_values_in_half_open_range() {
        let mut rng = SampleRng::new(7);

        for _ in 0..100 {
            let value = rng.random_double();
            assert!((0.0..1.0).contains(&value));
            let ranged = rng.random_range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&ranged));
        }
    }

    #[test]
    fn random_unit_vector_is_unit_length() {
        let mut rng = SampleRng::new(11);

        for _ in 0..20 {
            let vector = rng.random_unit_vector();
            assert!((vector.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_on_hemisphere_matches_normal_side() {
        let mut rng = SampleRng::new(13);
        let normal = Vector::new(0.0, 1.0, 0.0);

        for _ in 0..20 {
            assert!(rng.random_on_hemisphere(normal).dot(normal) > 0.0);
        }
    }

    #[test]
    fn random_in_unit_disk_stays_in_xy_unit_disk() {
        let mut rng = SampleRng::new(17);

        for _ in 0..20 {
            let point = rng.random_in_unit_disk();
            assert!(point.length_squared() < 1.0);
            assert_close(point.z(), 0.0);
        }
    }

    #[test]
    fn linear_color_to_rgb_applies_gamma_two() {
        let rgb = linear_color_to_rgb(LinearColor::new(0.25, 0.0, 1.0));

        assert_eq!(rgb, Rgb::new(128, 0, 255));
    }

    #[test]
    fn lambertian_reuses_existing_color_and_material_types() {
        let red = Lambertian::from(Rgb::RED);
        assert_eq!(red.albedo, LinearColor::new(1.0, 0.0, 0.0));

        let reflectance = ReflectionConstants::new(0.2, 0.4, 0.6);
        let from_reflectance = Lambertian::from(reflectance);
        assert_eq!(from_reflectance.albedo, LinearColor::new(0.2, 0.4, 0.6));

        let from_phong = Lambertian::from(PhongMaterial::SILVER);
        assert_eq!(
            from_phong.albedo,
            LinearColor::new(
                PhongMaterial::SILVER.diffuse.red,
                PhongMaterial::SILVER.diffuse.green,
                PhongMaterial::SILVER.diffuse.blue,
            )
        );
    }

    #[test]
    fn metal_reuses_existing_material_types_and_clamps_fuzz() {
        let silver = Metal::from(PhongMaterial::SILVER);
        assert_eq!(
            silver.albedo,
            LinearColor::new(
                PhongMaterial::SILVER.specular.red,
                PhongMaterial::SILVER.specular.green,
                PhongMaterial::SILVER.specular.blue,
            )
        );
        assert_close(silver.fuzz, 0.0);

        let fuzzy = Metal::from_reflectance(ReflectionConstants::new(0.8, 0.6, 0.2), 2.0);
        assert_close(fuzzy.fuzz, 1.0);
    }

    #[test]
    fn sphere_hit_records_front_face_and_unit_normal() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");

        assert!(record.front_face);
        assert_close(record.t, 0.5);
        assert_close(record.normal.length(), 1.0);
        assert_eq!(record.normal, Vector::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn ray_sphere_can_share_existing_sphere_geometry() {
        let geometry = SphereGeometry::new(Point::new(1.0, 2.0, 3.0), 4.0);
        let sphere =
            Sphere::from_geometry(geometry, Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)));

        assert_eq!(sphere.geometry(), geometry);
        assert_eq!(sphere.center(), Point::new(1.0, 2.0, 3.0));
        assert_close(sphere.radius(), 4.0);
    }

    #[test]
    fn lambertian_scatter_returns_attenuated_ray_from_hit_point() {
        let material = Lambertian::new(LinearColor::new(0.2, 0.4, 0.6));
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");
        let mut rng = SampleRng::new(17);

        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("lambertian should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
    }

    #[test]
    fn metal_scatter_reflects_incoming_ray() {
        let material = Metal::new(LinearColor::new(0.8, 0.8, 0.8), 0.0);
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");
        let mut rng = SampleRng::new(19);

        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("front-face metal hit should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_eq!(scatter.ray.direction(), &Vector::new(0.0, 0.0, 1.0));
        assert_eq!(scatter.attenuation, LinearColor::new(0.8, 0.8, 0.8));
    }

    #[test]
    fn dielectric_scatter_refracts_perpendicular_ray() {
        let material = Dielectric::new(RefractiveIndex::GLASS);
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");
        let mut rng = SampleRng::new(23);

        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("dielectric should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_eq!(scatter.ray.direction(), &Vector::new(0.0, 0.0, -1.0));
        assert_eq!(scatter.attenuation, LinearColor::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn dielectric_reflectance_increases_at_grazing_angles() {
        let straight = Dielectric::reflectance(1.0, RefractiveIndex::GLASS.0);
        let grazing = Dielectric::reflectance(0.1, RefractiveIndex::GLASS.0);

        assert!(grazing > straight);
    }

    #[test]
    fn metal_sphere_scene_contains_four_objects() {
        let world = metal_sphere_world();

        assert_eq!(world.len(), 4);
    }

    #[test]
    fn wide_angle_sphere_scene_contains_two_objects() {
        let world = wide_angle_sphere_world();

        assert_eq!(world.len(), 2);
    }

    #[test]
    fn dielectric_sphere_scene_contains_hollow_glass_setup() {
        let world = dielectric_sphere_world();

        assert_eq!(world.len(), 5);
    }

    #[test]
    fn final_scene_world_contains_many_random_spheres() {
        let world = final_scene_world();

        assert!(world.len() > 470);
    }

    #[test]
    fn ray_scene_uses_material_table_for_hits() {
        let mut scene = RayScene::new();
        let material = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.2, 0.4, 0.6)));
        scene.add_sphere(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(31);

        let hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("scene should be hit");
        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("lambertian should scatter");

        assert_eq!(scene.len(), 1);
        assert_eq!(scene.material_count(), 1);
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
    }

    #[test]
    fn final_scene_ray_scene_matches_compatibility_scene_size() {
        let compatibility_world = final_scene_world();
        let ray_scene = final_scene_ray_scene();

        assert_eq!(ray_scene.len(), compatibility_world.len());
        assert_eq!(ray_scene.material_count(), ray_scene.len());
        assert!(ray_scene.bounding_box().is_some());
    }

    #[test]
    fn final_scene_render_accepts_custom_sample_count() {
        let canvas = render_final_scene_with_samples(1, 1);

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 1);
    }

    #[test]
    fn path_tracer_wraps_camera_render_entrypoint() {
        let world = normal_sphere_world();
        let tracer = PathTracer::new(RayCamera::new(4, 1.0).with_samples_per_pixel(1));

        let canvas = tracer.render(&world);

        assert_eq!(tracer.camera().image_width(), 4);
        assert_eq!(canvas.width(), 4);
        assert!(canvas.upper_left_origin);
    }

    #[test]
    fn sphere_hit_flips_normal_for_inside_ray() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, 0.0), 1.0);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, 1.0));

        let record = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("inside ray should exit sphere");

        assert!(!record.front_face);
        assert_eq!(record.normal, Vector::new(-0.0, -0.0, -1.0));
    }

    #[test]
    fn triangle_hit_reports_barycentrics() {
        let p0 = Point::new(0.0, 0.0, -1.0);
        let p1 = Point::new(1.0, 0.0, -1.0);
        let p2 = Point::new(0.0, 1.0, -1.0);
        let ray = Ray::new(Point::new(0.25, 0.25, 0.0), Vector::new(0.0, 0.0, -1.0));

        let (t, u, v) = hit_triangle(p0, p1, p2, &ray, Interval::new(0.0, INFINITY))
            .expect("triangle should be hit");

        assert_close(t, 1.0);
        assert_close(u, 0.25);
        assert_close(v, 0.25);
        assert_eq!(ray.at(t), Point::new(0.25, 0.25, -1.0));
    }

    #[test]
    fn triangle_hit_rejects_edge_parallel_behind_and_degenerate_cases() {
        let p0 = Point::new(0.0, 0.0, -1.0);
        let p1 = Point::new(1.0, 0.0, -1.0);
        let p2 = Point::new(0.0, 1.0, -1.0);

        let outside = Ray::new(Point::new(1.1, 0.1, 0.0), Vector::new(0.0, 0.0, -1.0));
        assert!(hit_triangle(p0, p1, p2, &outside, Interval::new(0.0, INFINITY)).is_none());

        let parallel = Ray::new(Point::new(0.25, 0.25, -1.0), Vector::new(1.0, 0.0, 0.0));
        assert!(hit_triangle(p0, p1, p2, &parallel, Interval::new(0.0, INFINITY)).is_none());

        let behind = Ray::new(Point::new(0.25, 0.25, -2.0), Vector::new(0.0, 0.0, -1.0));
        assert!(hit_triangle(p0, p1, p2, &behind, Interval::new(0.0, INFINITY)).is_none());

        let degenerate = Ray::new(Point::new(0.25, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        assert!(
            hit_triangle(
                p0,
                Point::new(0.5, 0.0, -1.0),
                p1,
                &degenerate,
                Interval::new(0.0, INFINITY)
            )
            .is_none()
        );
    }

    #[test]
    fn triangle_scene_object_is_two_sided_and_flips_backface_normal() {
        let triangle = SceneObject::new(
            TriangleGeometry::new(
                Point::new(0.0, 0.0, -1.0),
                Point::new(1.0, 0.0, -1.0),
                Point::new(0.0, 1.0, -1.0),
            ),
            Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)),
        );
        let ray = Ray::new(Point::new(0.25, 0.25, -2.0), Vector::new(0.0, 0.0, 1.0));

        let record = triangle
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("backface should still be hit");

        assert!(!record.front_face);
        assert_eq!(record.normal, Vector::new(-0.0, -0.0, -1.0));
    }

    #[test]
    fn triangle_mesh_bvh_matches_bruteforce_hits() {
        let mut polygons = PolygonMatrix::new();
        polygons.push_polygons(&[
            [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (0.0, 1.0, -1.0)],
            [(-1.0, 0.0, -2.0), (0.0, 0.0, -2.0), (-1.0, 1.0, -2.0)],
            [(0.0, -1.0, -3.0), (1.0, -1.0, -3.0), (0.0, 0.0, -3.0)],
            [(-1.0, -1.0, -4.0), (0.0, -1.0, -4.0), (-1.0, 0.0, -4.0)],
            [(0.25, 0.25, -5.0), (1.25, 0.25, -5.0), (0.25, 1.25, -5.0)],
        ]);
        let mesh = TriangleMesh::from_polygon_matrix(
            &polygons,
            Lambertian::new(LinearColor::new(0.2, 0.2, 0.2)),
        );

        for x in [-0.75, -0.25, 0.25, 0.75, 1.5] {
            for y in [-0.75, -0.25, 0.25, 0.75, 1.5] {
                let ray = Ray::new(Point::new(x, y, 0.0), Vector::new(0.0, 0.0, -1.0));
                let bvh_hit = mesh
                    .hit(&ray, Interval::new(0.0, INFINITY))
                    .map(|hit| hit.t);
                let brute_hit = mesh
                    .hit_bruteforce(&ray, Interval::new(0.0, INFINITY))
                    .map(|hit| hit.t);
                assert_eq!(bvh_hit, brute_hit);
            }
        }
    }

    #[cfg(feature = "external")]
    #[test]
    fn material_mesh_converts_to_triangle_mesh_groups() {
        let mesh = crate::external::meshify_with_materials("examples/data/meshes/teapot.obj")
            .expect("load teapot mesh");
        let triangle_meshes = TriangleMesh::from_material_mesh_lambertian(&mesh);

        assert!(!triangle_meshes.is_empty());
        assert_eq!(
            triangle_meshes.iter().map(TriangleMesh::len).sum::<usize>(),
            mesh.triangle_count()
        );
        assert!(
            triangle_meshes
                .iter()
                .all(|triangle_mesh| triangle_mesh.bounding_box().is_some())
        );
    }

    #[test]
    fn hittable_list_returns_closest_hit() {
        let mut world = HittableList::new();
        world.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        world.add(Sphere::new(Point::new(0.0, 0.0, -1.0), 0.25));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = world
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("world should be hit");

        assert_close(record.t, 0.75);
    }

    #[test]
    fn object_bvh_returns_closest_hit() {
        let mut world = HittableList::new();
        world.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        world.add(Sphere::new(Point::new(0.0, 0.0, -1.0), 0.25));
        let bvh = world.into_bvh().expect("bounded world should build bvh");
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = bvh
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("BVH world should be hit");

        assert_close(record.t, 0.75);
    }

    #[test]
    fn object_bvh_matches_bruteforce_hits() {
        let mut world = HittableList::new();
        for z in 1..8 {
            let x = if z % 2 == 0 { -0.5 } else { 0.5 };
            world.add(Sphere::new(Point::new(x, 0.0, -f64::from(z)), 0.35));
        }
        let bvh = world.into_bvh().expect("bounded world should build bvh");

        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for y in [-0.5, 0.0, 0.5] {
                let ray = Ray::new(Point::new(x, y, 0.0), Vector::new(0.0, 0.0, -1.0));
                let bvh_hit = bvh.hit(&ray, Interval::new(0.0, INFINITY)).map(|hit| hit.t);
                let brute_hit = bvh
                    .hit_bruteforce(&ray, Interval::new(0.0, INFINITY))
                    .map(|hit| hit.t);
                assert_eq!(bvh_hit, brute_hit);
            }
        }
    }

    #[test]
    fn normal_scene_render_colors_sphere_by_normal() {
        let canvas = render_normal_sphere_scene(40);
        let center = canvas
            .get_pixel(20, 11)
            .expect("center pixel should be inside the canvas");

        assert_ne!(*center, Rgb::RED);
        assert!(center.blue > center.red);
    }

    #[test]
    fn diffuse_scene_render_is_gamma_corrected() {
        let canvas = render_diffuse_sphere_scene(20);
        let center = canvas
            .get_pixel(10, 5)
            .expect("center pixel should be inside the canvas");

        assert!(center.red > 0 || center.green > 0 || center.blue > 0);
    }
}

//! Minimal ray-tracing helpers following the early "Ray Tracing in One Weekend" steps.

use crate::{
    gmath::{
        geometry::SphereGeometry,
        ray::Ray,
        vector::{Point, Vector},
    },
    graphics::{
        camera::RayCamera,
        colors::Rgb,
        display::Canvas,
        lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex},
    },
};
use std::{fmt, sync::Arc};

/// Floating-point infinity for ray intervals.
pub const INFINITY: f64 = f64::INFINITY;

/// Pi, provided with the book's common ray-tracing constants.
pub const PI: f64 = std::f64::consts::PI;

/// The 16:9 aspect ratio used by the first weekend camera setup.
pub const WIDESCREEN_ASPECT_RATIO: f64 = 16.0 / 9.0;

/// A color represented as linear floating-point RGB components in `0.0..=1.0`.
pub type LinearColor = Vector;

/// Minimum ray parameter accepted for secondary rays to avoid self-intersections.
pub const SHADOW_ACNE_EPSILON: f64 = 0.001;

/// Converts degrees to radians.
#[must_use]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Small deterministic random-number generator for ray-tracing samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleRng {
    state: u64,
}

impl SampleRng {
    /// Creates a sample RNG from a seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a random real in `[0, 1)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn random_double(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = self.state >> 11;
        bits as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Returns a random real in `[min, max)`.
    #[must_use]
    pub fn random_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.random_double()
    }

    /// Returns a random vector with each component in `[0, 1)`.
    #[must_use]
    pub fn random_vector(&mut self) -> Vector {
        Vector::new(
            self.random_double(),
            self.random_double(),
            self.random_double(),
        )
    }

    /// Returns a random vector with each component in `[min, max)`.
    #[must_use]
    pub fn random_vector_range(&mut self, min: f64, max: f64) -> Vector {
        Vector::new(
            self.random_range(min, max),
            self.random_range(min, max),
            self.random_range(min, max),
        )
    }

    /// Returns a uniformly random unit vector using rejection sampling.
    #[must_use]
    pub fn random_unit_vector(&mut self) -> Vector {
        loop {
            let point = self.random_vector_range(-1.0, 1.0);
            let length_squared = point.length_squared();
            if 1e-160 < length_squared && length_squared <= 1.0 {
                return point / length_squared.sqrt();
            }
        }
    }

    /// Returns a random unit vector on the same hemisphere as `normal`.
    #[must_use]
    pub fn random_on_hemisphere(&mut self, normal: Vector) -> Vector {
        let on_unit_sphere = self.random_unit_vector();
        if on_unit_sphere.dot(normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    /// Returns a random point inside the unit disk in the xy plane.
    #[must_use]
    pub fn random_in_unit_disk(&mut self) -> Vector {
        loop {
            let point = Vector::new(
                self.random_range(-1.0, 1.0),
                self.random_range(-1.0, 1.0),
                0.0,
            );
            if point.length_squared() < 1.0 {
                return point;
            }
        }
    }
}

impl Default for SampleRng {
    fn default() -> Self {
        Self::new(1)
    }
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
    pub const fn from_ratio(refraction_index: f64) -> Self {
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

/// Shared material handle used by hittable objects.
pub type MaterialRef = Arc<dyn Material>;

fn default_material() -> MaterialRef {
    Arc::new(Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)))
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
        let mut record = Self {
            point,
            normal: outward_normal,
            t,
            front_face: false,
            material,
        };
        record.set_face_normal(ray, outward_normal);
        record
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

/// A scene object that can be intersected by a ray.
pub trait Hittable {
    /// Returns the closest hit inside `ray_t`, if any.
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>>;
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
        let root = hit_sphere(self.center(), self.radius(), ray)?;
        let root = if ray_t.surrounds(root) {
            root
        } else {
            let oc = self.center() - *ray.origin();
            let a = ray.direction().length_squared();
            let h = ray.direction().dot(oc);
            let c = oc.length_squared() - self.radius() * self.radius();
            let discriminant = h * h - a * c;
            let sqrtd = discriminant.sqrt();
            let second_root = (h + sqrtd) / a;
            if !ray_t.surrounds(second_root) {
                return None;
            }
            second_root
        };

        let point = ray.at(root);
        let outward_normal = self.geometry.outward_normal_at(point);

        Some(HitRecord::new(
            ray,
            point,
            outward_normal,
            root,
            self.material(),
        ))
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
}

/// Renders the book's first red/green PPM gradient into a [`Canvas`].
pub fn render_unit_gradient(width: u32, height: u32) -> Canvas {
    let denom_x = f64::from(width.saturating_sub(1)).max(1.0);
    let denom_y = f64::from(height.saturating_sub(1)).max(1.0);
    Canvas::from_fn(width, height, |x, y| {
        Rgb::from(LinearColor::new(
            f64::from(x) / denom_x,
            f64::from(y) / denom_y,
            0.0,
        ))
    })
}

/// Returns the nearest ray parameter where `ray` intersects the sphere.
#[must_use]
pub fn hit_sphere(center: Point, radius: f64, ray: &Ray) -> Option<f64> {
    let oc = center - *ray.origin();
    let a = ray.direction().length_squared();
    let h = ray.direction().dot(oc);
    let c = oc.length_squared() - radius * radius;
    let discriminant = h * h - a * c;

    if discriminant < 0.0 {
        None
    } else {
        Some((h - discriminant.sqrt()) / a)
    }
}

/// Computes the blue-to-white background gradient for a ray.
#[must_use]
pub fn sky_gradient(ray: &Ray) -> LinearColor {
    let unit_direction = ray.direction().normalized();
    let a = 0.5 * (unit_direction.y() + 1.0);
    (1.0 - a) * LinearColor::new(1.0, 1.0, 1.0) + a * LinearColor::new(0.5, 0.7, 1.0)
}

/// Computes the first sphere scene from the book: a red sphere over a blue sky.
#[must_use]
pub fn first_sphere_color(ray: &Ray) -> LinearColor {
    if hit_sphere(Point::new(0.0, 0.0, -1.0), 0.5, ray).is_some() {
        LinearColor::new(1.0, 0.0, 0.0)
    } else {
        sky_gradient(ray)
    }
}

/// Renders the first sphere scene from the book with a 16:9 camera.
pub fn render_first_sphere(image_width: u32) -> Canvas {
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO).render(first_sphere_color)
}

/// Computes a normal-visualization color for a ray cast into `world`.
#[must_use]
pub fn normal_scene_color(ray: &Ray, world: &dyn Hittable) -> LinearColor {
    if let Some(record) = world.hit(ray, Interval::new(0.0, INFINITY)) {
        0.5 * (record.normal + LinearColor::new(1.0, 1.0, 1.0))
    } else {
        sky_gradient(ray)
    }
}

/// Returns the book's first multi-object world: a sphere over a large ground sphere.
#[must_use]
pub fn normal_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    let material = Lambertian::new(LinearColor::new(0.5, 0.5, 0.5));
    world.add(Sphere::with_material(
        Point::new(0.0, 0.0, -1.0),
        0.5,
        material,
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        material,
    ));
    world
}

/// Renders the normals-colored sphere and ground scene from the book.
pub fn render_normal_sphere_scene(image_width: u32) -> Canvas {
    let world = normal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .render_world_normals(&world)
}

/// Renders the diffuse sphere and ground scene from the book.
pub fn render_diffuse_sphere_scene(image_width: u32) -> Canvas {
    let world = normal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .render_world(&world)
}

/// Returns the book's first mixed diffuse/metal sphere world.
#[must_use]
pub fn metal_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        Lambertian::new(LinearColor::new(0.8, 0.8, 0.0)),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 0.0, -1.2),
        0.5,
        Lambertian::new(LinearColor::new(0.1, 0.2, 0.5)),
    ));
    world.add(Sphere::with_material(
        Point::new(-1.0, 0.0, -1.0),
        0.5,
        Metal::from_phong_specular(PhongMaterial::SILVER, 0.0),
    ));
    world.add(Sphere::with_material(
        Point::new(1.0, 0.0, -1.0),
        0.5,
        Metal::from_reflectance(ReflectionConstants::new(0.8, 0.6, 0.2), 0.3),
    ));
    world
}

/// Renders the mixed diffuse/metal sphere scene from the book.
pub fn render_metal_sphere_scene(image_width: u32) -> Canvas {
    let world = metal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .render_world(&world)
}

/// Returns the book's two-sphere scene for checking camera field of view.
#[must_use]
pub fn wide_angle_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    let radius = (PI / 4.0).cos();

    world.add(Sphere::with_material(
        Point::new(-radius, 0.0, -1.0),
        radius,
        Lambertian::new(LinearColor::new(0.0, 0.0, 1.0)),
    ));
    world.add(Sphere::with_material(
        Point::new(radius, 0.0, -1.0),
        radius,
        Lambertian::new(LinearColor::new(1.0, 0.0, 0.0)),
    ));
    world
}

/// Renders the wide-angle camera test scene from the book.
pub fn render_wide_angle_sphere_scene(image_width: u32) -> Canvas {
    let world = wide_angle_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .with_vertical_fov(90.0)
        .render_world(&world)
}

/// Returns the book's diffuse/metal scene with a hollow glass sphere.
#[must_use]
pub fn dielectric_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        Lambertian::new(LinearColor::new(0.8, 0.8, 0.0)),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 0.0, -1.2),
        0.5,
        Lambertian::new(LinearColor::new(0.1, 0.2, 0.5)),
    ));
    world.add(Sphere::with_material(
        Point::new(-1.0, 0.0, -1.0),
        0.5,
        Dielectric::new(RefractiveIndex::GLASS),
    ));
    world.add(Sphere::with_material(
        Point::new(-1.0, 0.0, -1.0),
        0.4,
        Dielectric::new(RefractiveIndex::AIR.relative_to(RefractiveIndex::GLASS)),
    ));
    world.add(Sphere::with_material(
        Point::new(1.0, 0.0, -1.0),
        0.5,
        Metal::from_reflectance(ReflectionConstants::new(0.8, 0.6, 0.2), 0.0),
    ));
    world
}

/// Renders the hollow-glass dielectric sphere scene from the book.
pub fn render_dielectric_sphere_scene(image_width: u32) -> Canvas {
    let world = dielectric_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .with_vertical_fov(20.0)
        .with_look_at(Point::new(-2.0, 2.0, 1.0), Point::new(0.0, 0.0, -1.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .render_world(&world)
}

/// Renders the dielectric scene with defocus blur enabled.
pub fn render_defocus_sphere_scene(image_width: u32) -> Canvas {
    let world = dielectric_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .with_vertical_fov(20.0)
        .with_look_at(Point::new(-2.0, 2.0, 1.0), Point::new(0.0, 0.0, -1.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_defocus_angle(10.0)
        .with_focus_distance(3.4)
        .render_world(&world)
}

fn component_mul(lhs: LinearColor, rhs: LinearColor) -> LinearColor {
    LinearColor::new(lhs.x() * rhs.x(), lhs.y() * rhs.y(), lhs.z() * rhs.z())
}

/// Returns the final random-spheres scene from the book.
#[must_use]
pub fn final_scene_world() -> HittableList {
    let mut rng = SampleRng::new(61);
    let mut world = HittableList::new();

    world.add(Sphere::with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random_double();
            let center = Point::new(
                f64::from(a) + 0.9 * rng.random_double(),
                0.2,
                f64::from(b) + 0.9 * rng.random_double(),
            );

            if (center - Point::new(4.0, 0.2, 0.0)).length() <= 0.9 {
                continue;
            }

            if choose_material < 0.8 {
                let albedo = component_mul(rng.random_vector(), rng.random_vector());
                world.add(Sphere::with_material(center, 0.2, Lambertian::new(albedo)));
            } else if choose_material < 0.95 {
                let albedo = rng.random_vector_range(0.5, 1.0);
                let fuzz = rng.random_range(0.0, 0.5);
                world.add(Sphere::with_material(center, 0.2, Metal::new(albedo, fuzz)));
            } else {
                world.add(Sphere::with_material(
                    center,
                    0.2,
                    Dielectric::new(RefractiveIndex::GLASS),
                ));
            }
        }
    }

    world.add(Sphere::with_material(
        Point::new(0.0, 1.0, 0.0),
        1.0,
        Dielectric::new(RefractiveIndex::GLASS),
    ));
    world.add(Sphere::with_material(
        Point::new(-4.0, 1.0, 0.0),
        1.0,
        Lambertian::new(LinearColor::new(0.4, 0.2, 0.1)),
    ));
    world.add(Sphere::with_material(
        Point::new(4.0, 1.0, 0.0),
        1.0,
        Metal::new(LinearColor::new(0.7, 0.6, 0.5), 0.0),
    ));

    world
}

/// Renders the final random-spheres scene from the book.
pub fn render_final_scene(image_width: u32) -> Canvas {
    render_final_scene_with_samples(image_width, 10)
}

/// Renders the final random-spheres scene with a caller-selected sample count.
///
/// The book uses 500 samples per pixel for the cover-quality image. The shorter
/// [`render_final_scene`] helper uses 10 samples so examples complete quickly.
pub fn render_final_scene_with_samples(image_width: u32, samples_per_pixel: u32) -> Canvas {
    let world = final_scene_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(samples_per_pixel)
        .with_max_depth(50)
        .with_vertical_fov(20.0)
        .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_defocus_angle(0.6)
        .with_focus_distance(10.0)
        .render_world(&world)
}

/// Converts a linear color component to gamma space using gamma 2.
#[must_use]
pub fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0.0 {
        linear_component.sqrt()
    } else {
        0.0
    }
}

/// Converts a linear ray-traced color to display RGB with gamma correction.
#[must_use]
pub fn linear_color_to_rgb(color: LinearColor) -> Rgb {
    let intensity = Interval::new(0.0, 0.999);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let channel = |component: f64| (256.0 * intensity.clamp(linear_to_gamma(component))) as u8;
    Rgb::new(channel(color.x()), channel(color.y()), channel(color.z()))
}

/// Converts display RGB bytes to a linear color vector in `0.0..=1.0`.
#[must_use]
pub fn rgb_to_linear_color(color: Rgb) -> LinearColor {
    LinearColor::new(
        f64::from(color.red) / 255.0,
        f64::from(color.green) / 255.0,
        f64::from(color.blue) / 255.0,
    )
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

        let miss = Ray::new(origin, Vector::new(0.0, 1.0, -1.0));
        assert!(hit_sphere(center, 0.5, &miss).is_none());
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
    fn final_scene_render_accepts_custom_sample_count() {
        let canvas = render_final_scene_with_samples(1, 1);

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 1);
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

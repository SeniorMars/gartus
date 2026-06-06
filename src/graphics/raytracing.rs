//! Minimal ray-tracing helpers following the early "Ray Tracing in One Weekend" steps.

pub use crate::gmath::random::SampleRng;
pub mod scenes;
pub mod weekend;

use crate::{
    gmath::{
        geometry::{MovingSphereGeometry, QuadGeometry, SphereGeometry, TriangleGeometry},
        matrix::Matrix,
        perlin::{Perlin, scale_point},
        polygon_matrix::{Bounds3, PolygonMatrix},
        ray::Ray,
        vector::{Point, Vector},
    },
    graphics::{
        camera::RayCamera,
        colors::{LinearRgb, Rgb},
        display::Canvas,
        lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex, SurfaceMaterial},
        texture::Texture as BitmapTexture,
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
    /// Returns emitted light for this material at a surface point.
    fn emitted(&self, _u: f64, _v: f64, _point: Point) -> LinearColor {
        LinearColor::default()
    }

    /// Produces a scattered ray and attenuation for a surface hit.
    fn scatter(
        &self,
        _ray_in: &Ray,
        _hit: &HitRecord<'_>,
        _rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        None
    }
}

/// Procedural or image-backed texture sampled by ray-tracing materials.
pub trait RayTexture: fmt::Debug + Send + Sync {
    /// Returns the linear color at texture coordinate `(u, v)` and surface point `point`.
    fn value(&self, u: f64, v: f64, point: Point) -> LinearColor;
}

/// Shared ray-texture handle.
pub type TextureRef = Arc<dyn RayTexture>;

/// A texture that always returns one linear color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidColor {
    /// Constant texture color.
    pub color: LinearColor,
}

impl SolidColor {
    /// Creates a constant linear-color texture.
    #[must_use]
    pub const fn new(color: LinearColor) -> Self {
        Self { color }
    }

    /// Creates a constant texture from display RGB bytes treated as linear unit values.
    #[must_use]
    pub fn from_rgb(color: Rgb) -> Self {
        Self::new(rgb_to_linear_color(color))
    }
}

impl From<LinearColor> for SolidColor {
    fn from(color: LinearColor) -> Self {
        Self::new(color)
    }
}

impl From<Rgb> for SolidColor {
    fn from(color: Rgb) -> Self {
        Self::from_rgb(color)
    }
}

impl RayTexture for SolidColor {
    fn value(&self, _u: f64, _v: f64, _point: Point) -> LinearColor {
        self.color
    }
}

/// A 3D checker texture that alternates between two child textures in world space.
#[derive(Clone, Debug)]
pub struct CheckerTexture {
    inv_scale: f64,
    even: TextureRef,
    odd: TextureRef,
}

impl CheckerTexture {
    /// Creates a checker texture from two child textures.
    #[must_use]
    pub fn new(scale: f64, even: TextureRef, odd: TextureRef) -> Self {
        let inv_scale = if scale.is_finite() && scale.abs() > f64::EPSILON {
            1.0 / scale.abs()
        } else {
            1.0
        };
        Self {
            inv_scale,
            even,
            odd,
        }
    }

    /// Creates a checker texture from two constant colors.
    #[must_use]
    pub fn from_colors(scale: f64, even: LinearColor, odd: LinearColor) -> Self {
        Self::new(
            scale,
            Arc::new(SolidColor::new(even)),
            Arc::new(SolidColor::new(odd)),
        )
    }
}

impl RayTexture for CheckerTexture {
    #[allow(clippy::cast_possible_truncation)]
    fn value(&self, texture_u: f64, texture_v: f64, point: Point) -> LinearColor {
        let x_integer = (self.inv_scale * point.x()).floor() as i64;
        let y_integer = (self.inv_scale * point.y()).floor() as i64;
        let z_integer = (self.inv_scale * point.z()).floor() as i64;
        if (x_integer + y_integer + z_integer) % 2 == 0 {
            self.even.value(texture_u, texture_v, point)
        } else {
            self.odd.value(texture_u, texture_v, point)
        }
    }
}

/// A ray-tracing texture backed by the library's 2D bitmap texture sampler.
#[derive(Clone, Debug)]
pub struct ImageTexture {
    texture: BitmapTexture,
}

impl ImageTexture {
    /// Creates an image texture from an existing bitmap texture.
    #[must_use]
    pub const fn new(texture: BitmapTexture) -> Self {
        Self { texture }
    }

    /// Creates an image texture from an existing canvas.
    #[must_use]
    pub const fn from_canvas(canvas: Canvas) -> Self {
        Self::new(BitmapTexture::from_canvas(canvas))
    }

    /// Returns the underlying bitmap texture.
    #[must_use]
    pub const fn texture(&self) -> &BitmapTexture {
        &self.texture
    }

    /// Loads an image file through the library's external image loader.
    ///
    /// # Errors
    ///
    /// Returns an error if the image cannot be loaded or converted into a canvas.
    #[cfg(feature = "external")]
    pub fn from_file(path: impl AsRef<str>) -> Result<Self, Box<dyn std::error::Error>> {
        let canvas = crate::external::ppmify(path.as_ref(), false)?;
        Ok(Self::from_canvas(canvas))
    }
}

impl RayTexture for ImageTexture {
    fn value(&self, u: f64, v: f64, _point: Point) -> LinearColor {
        if self.texture.image().is_empty() {
            return rgb_to_linear_color(Rgb::CYAN);
        }
        rgb_to_linear_color(self.texture.sample(u, v))
    }
}

/// Procedural Perlin-noise texture.
#[derive(Clone, Debug)]
pub struct NoiseTexture {
    noise: Perlin,
    scale: f64,
    kind: NoiseTextureKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NoiseTextureKind {
    Noise,
    Turbulence { depth: usize },
    Marble { depth: usize },
}

impl NoiseTexture {
    /// Creates a shifted Perlin noise texture from `seed`.
    #[must_use]
    pub fn new(scale: f64, seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed),
            scale,
            kind: NoiseTextureKind::Noise,
        }
    }

    /// Creates a turbulence texture from `seed`.
    #[must_use]
    pub fn turbulence(scale: f64, depth: usize, seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed),
            scale,
            kind: NoiseTextureKind::Turbulence { depth },
        }
    }

    /// Creates a marble-like texture from `seed`.
    #[must_use]
    pub fn marble(scale: f64, depth: usize, seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed),
            scale,
            kind: NoiseTextureKind::Marble { depth },
        }
    }

    /// Returns the coordinate scale applied by this texture.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for NoiseTexture {
    fn default() -> Self {
        Self::new(1.0, 1)
    }
}

impl RayTexture for NoiseTexture {
    fn value(&self, _u: f64, _v: f64, point: Point) -> LinearColor {
        let intensity = match self.kind {
            NoiseTextureKind::Noise => {
                0.5 * (1.0 + self.noise.noise(scale_point(point, self.scale)))
            }
            NoiseTextureKind::Turbulence { depth } => {
                self.noise.turbulence(scale_point(point, self.scale), depth)
            }
            NoiseTextureKind::Marble { depth } => {
                0.5 * (1.0
                    + (self.scale * point.z() + 10.0 * self.noise.turbulence(point, depth)).sin())
            }
        };
        LinearColor::new(intensity, intensity, intensity)
    }
}

/// Lambertian diffuse material.
#[derive(Clone)]
pub struct Lambertian {
    /// Representative diffuse reflectance for constant-color compatibility.
    pub albedo: LinearColor,
    texture: TextureRef,
}

impl fmt::Debug for Lambertian {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Lambertian")
            .field("albedo", &self.albedo)
            .field("texture", &self.texture)
            .finish()
    }
}

impl Lambertian {
    /// Creates a Lambertian material with the supplied albedo.
    #[must_use]
    pub fn new(albedo: LinearColor) -> Self {
        Self {
            albedo,
            texture: Arc::new(SolidColor::new(albedo)),
        }
    }

    /// Creates a Lambertian material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self {
            albedo: LinearColor::new(0.5, 0.5, 0.5),
            texture,
        }
    }

    /// Creates a Lambertian material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl RayTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Creates a Lambertian material using a 3D checker texture.
    #[must_use]
    pub fn checker(scale: f64, even: LinearColor, odd: LinearColor) -> Self {
        Self::from_texture(CheckerTexture::from_colors(scale, even, odd))
    }

    /// Creates a Lambertian material using a Perlin noise texture.
    #[must_use]
    pub fn noise(scale: f64, seed: u64) -> Self {
        Self::from_texture(NoiseTexture::new(scale, seed))
    }

    /// Creates a Lambertian material using a marble-like Perlin texture.
    #[must_use]
    pub fn marble(scale: f64, seed: u64) -> Self {
        Self::from_texture(NoiseTexture::marble(scale, 7, seed))
    }

    /// Returns the texture sampled for diffuse attenuation.
    #[must_use]
    pub fn texture(&self) -> &dyn RayTexture {
        self.texture.as_ref()
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
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let mut scatter_direction = hit.normal + rng.random_unit_vector();
        if scatter_direction.length_squared() < 1e-20 {
            scatter_direction = hit.normal;
        }

        Some(ScatterRecord {
            ray: Ray::with_time(hit.point, scatter_direction, ray_in.time()),
            attenuation: self.texture.value(hit.u, hit.v, hit.point),
        })
    }
}

/// Diffuse light-emitting material.
#[derive(Clone, Debug)]
pub struct DiffuseLight {
    texture: TextureRef,
}

impl DiffuseLight {
    /// Creates a light material with a constant emitted color.
    #[must_use]
    pub fn new(emit: LinearColor) -> Self {
        Self::from_texture(SolidColor::new(emit))
    }

    /// Creates a light material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self { texture }
    }

    /// Creates a light material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl RayTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Returns the texture sampled for emitted radiance.
    #[must_use]
    pub fn texture(&self) -> &dyn RayTexture {
        self.texture.as_ref()
    }
}

impl Material for DiffuseLight {
    fn emitted(&self, u: f64, v: f64, point: Point) -> LinearColor {
        self.texture.value(u, v, point)
    }
}

/// Isotropic phase-function material for constant-density volumes.
#[derive(Clone, Debug)]
pub struct Isotropic {
    texture: TextureRef,
}

impl Isotropic {
    /// Creates an isotropic material with constant attenuation.
    #[must_use]
    pub fn new(albedo: LinearColor) -> Self {
        Self::from_texture(SolidColor::new(albedo))
    }

    /// Creates an isotropic material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self { texture }
    }

    /// Creates an isotropic material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl RayTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Returns the texture sampled for medium attenuation.
    #[must_use]
    pub fn texture(&self) -> &dyn RayTexture {
        self.texture.as_ref()
    }
}

impl Material for Isotropic {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        Some(ScatterRecord {
            ray: Ray::with_time(hit.point, rng.random_unit_vector(), ray_in.time()),
            attenuation: self.texture.value(hit.u, hit.v, hit.point),
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
            ray: Ray::with_time(hit.point, scattered_direction, ray_in.time()),
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
            ray: Ray::with_time(hit.point, direction, ray_in.time()),
            attenuation,
        })
    }
}

/// Material variants supported by the data-oriented ray scene.
#[derive(Clone, Debug)]
pub enum RayMaterial {
    /// Lambertian diffuse material.
    Lambertian(Lambertian),
    /// Diffuse light-emitting material.
    DiffuseLight(DiffuseLight),
    /// Isotropic phase-function material.
    Isotropic(Isotropic),
    /// Reflective metal material.
    Metal(Metal),
    /// Transparent dielectric material.
    Dielectric(Dielectric),
}

impl RayMaterial {
    /// Creates a Lambertian material variant.
    #[must_use]
    pub fn lambertian(albedo: LinearColor) -> Self {
        Self::Lambertian(Lambertian::new(albedo))
    }

    /// Creates a textured Lambertian material variant.
    #[must_use]
    pub fn textured_lambertian(texture: impl RayTexture + 'static) -> Self {
        Self::Lambertian(Lambertian::from_texture(texture))
    }

    /// Creates a diffuse light material variant.
    #[must_use]
    pub fn diffuse_light(emit: LinearColor) -> Self {
        Self::DiffuseLight(DiffuseLight::new(emit))
    }

    /// Creates a textured diffuse light material variant.
    #[must_use]
    pub fn textured_diffuse_light(texture: impl RayTexture + 'static) -> Self {
        Self::DiffuseLight(DiffuseLight::from_texture(texture))
    }

    /// Creates an isotropic phase-function material variant.
    #[must_use]
    pub fn isotropic(albedo: LinearColor) -> Self {
        Self::Isotropic(Isotropic::new(albedo))
    }

    /// Creates a textured isotropic phase-function material variant.
    #[must_use]
    pub fn textured_isotropic(texture: impl RayTexture + 'static) -> Self {
        Self::Isotropic(Isotropic::from_texture(texture))
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

impl From<DiffuseLight> for RayMaterial {
    fn from(material: DiffuseLight) -> Self {
        Self::DiffuseLight(material)
    }
}

impl From<Isotropic> for RayMaterial {
    fn from(material: Isotropic) -> Self {
        Self::Isotropic(material)
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
    fn emitted(&self, u: f64, v: f64, point: Point) -> LinearColor {
        match self {
            Self::Lambertian(material) => material.emitted(u, v, point),
            Self::DiffuseLight(material) => material.emitted(u, v, point),
            Self::Isotropic(material) => material.emitted(u, v, point),
            Self::Metal(material) => material.emitted(u, v, point),
            Self::Dielectric(material) => material.emitted(u, v, point),
        }
    }

    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        match self {
            Self::Lambertian(material) => material.scatter(ray_in, hit, rng),
            Self::DiffuseLight(material) => material.scatter(ray_in, hit, rng),
            Self::Isotropic(material) => material.scatter(ray_in, hit, rng),
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

/// A scene object that can be intersected by a ray.
pub trait Hittable: Send + Sync {
    /// Returns the closest hit inside `ray_t`, if any.
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
        Some(sphere_bounds(self.geometry))
    }
}

/// A linearly moving sphere hittable.
#[derive(Clone)]
pub struct MovingSphere {
    geometry: MovingSphereGeometry,
    material: MaterialRef,
}

impl fmt::Debug for MovingSphere {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MovingSphere")
            .field("geometry", &self.geometry)
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
        Self { geometry, material }
    }

    /// Returns the shared analytic moving sphere geometry.
    #[must_use]
    pub const fn geometry(&self) -> MovingSphereGeometry {
        self.geometry
    }

    /// Returns the material associated with this moving sphere.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }
}

impl Hittable for MovingSphere {
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

/// A parallelogram hittable.
#[derive(Clone)]
pub struct Quad {
    geometry: QuadGeometry,
    material: MaterialRef,
}

impl fmt::Debug for Quad {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Quad")
            .field("geometry", &self.geometry)
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
        Self { geometry, material }
    }

    /// Returns the shared analytic quad geometry.
    #[must_use]
    pub const fn geometry(&self) -> QuadGeometry {
        self.geometry
    }

    /// Returns the material associated with this quad.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }
}

impl Hittable for Quad {
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

/// A translated instance of a hittable object.
pub struct Translate {
    object: Box<dyn Hittable>,
    offset: Vector,
    bounds: Option<Aabb>,
}

impl fmt::Debug for Translate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Translate")
            .field("offset", &self.offset)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl Translate {
    /// Creates a translated instance.
    #[must_use]
    pub fn new(object: impl Hittable + 'static, offset: Vector) -> Self {
        let bounds = object
            .bounding_box()
            .map(|bounds| bounds.translated(offset));
        Self {
            object: Box::new(object),
            offset,
            bounds,
        }
    }

    /// Creates a translated instance from a boxed hittable.
    #[must_use]
    pub fn from_box(object: Box<dyn Hittable>, offset: Vector) -> Self {
        let bounds = object
            .bounding_box()
            .map(|bounds| bounds.translated(offset));
        Self {
            object,
            offset,
            bounds,
        }
    }

    /// Returns the translation offset.
    #[must_use]
    pub const fn offset(&self) -> Vector {
        self.offset
    }
}

impl Hittable for Translate {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let offset_ray = Ray::with_time(*ray.origin() - self.offset, *ray.direction(), ray.time());
        let mut record = self.object.hit_with_rng(&offset_ray, ray_t, rng)?;
        record.point = record.point + self.offset;
        Some(record)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

/// A Y-axis rotated instance of a hittable object.
pub struct RotateY {
    object: Box<dyn Hittable>,
    sin_theta: f64,
    cos_theta: f64,
    bounds: Option<Aabb>,
}

impl fmt::Debug for RotateY {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RotateY")
            .field("sin_theta", &self.sin_theta)
            .field("cos_theta", &self.cos_theta)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl RotateY {
    /// Creates a Y-axis rotated instance.
    #[must_use]
    pub fn new(object: impl Hittable + 'static, angle_degrees: f64) -> Self {
        Self::from_box(Box::new(object), angle_degrees)
    }

    /// Creates a Y-axis rotated instance from a boxed hittable.
    #[must_use]
    pub fn from_box(object: Box<dyn Hittable>, angle_degrees: f64) -> Self {
        let radians = degrees_to_radians(angle_degrees);
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let bounds = object
            .bounding_box()
            .map(|bounds| rotate_y_bounds(bounds, sin_theta, cos_theta));
        Self {
            object,
            sin_theta,
            cos_theta,
            bounds,
        }
    }

    /// Returns the sine of this instance rotation.
    #[must_use]
    pub const fn sin_theta(&self) -> f64 {
        self.sin_theta
    }

    /// Returns the cosine of this instance rotation.
    #[must_use]
    pub const fn cos_theta(&self) -> f64 {
        self.cos_theta
    }
}

impl Hittable for RotateY {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let rotated_ray = Ray::with_time(
            rotate_y_point_inverse(*ray.origin(), self.sin_theta, self.cos_theta),
            rotate_y_vector_inverse(*ray.direction(), self.sin_theta, self.cos_theta),
            ray.time(),
        );
        let mut record = self.object.hit_with_rng(&rotated_ray, ray_t, rng)?;
        record.point = rotate_y_point(record.point, self.sin_theta, self.cos_theta);
        record.normal = rotate_y_vector(record.normal, self.sin_theta, self.cos_theta);
        Some(record)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

/// A generic matrix-transformed instance of a hittable object.
///
/// The transform maps object space into world space. Rays are transformed by the cached inverse
/// before hitting the child object, then hit points and normals are transformed back to world
/// space. Normals use the inverse-transpose transform, which keeps them correct for non-uniform
/// scales as well as rotations and translations.
pub struct MatrixInstance {
    object: Box<dyn Hittable>,
    transform: Matrix,
    inverse: Matrix,
    normal_transform: Matrix,
    bounds: Option<Aabb>,
}

impl fmt::Debug for MatrixInstance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MatrixInstance")
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl MatrixInstance {
    /// Creates a transformed instance.
    ///
    /// Returns `None` if `transform` is not an invertible 4x4 matrix.
    #[must_use]
    pub fn new(object: impl Hittable + 'static, transform: Matrix) -> Option<Self> {
        Self::from_box(Box::new(object), transform)
    }

    /// Creates a transformed instance from a boxed hittable.
    ///
    /// Returns `None` if `transform` is not an invertible 4x4 matrix.
    #[must_use]
    pub fn from_box(object: Box<dyn Hittable>, transform: Matrix) -> Option<Self> {
        if transform.rows() != 4 || transform.cols() != 4 {
            return None;
        }
        let inverse = transform.inverse()?;
        let normal_transform = inverse.transpose();
        let bounds = object
            .bounding_box()
            .map(|bounds| transform_bounds(bounds, &transform));
        Some(Self {
            object,
            transform,
            inverse,
            normal_transform,
            bounds,
        })
    }

    /// Returns the object-to-world transform.
    pub const fn transform(&self) -> &Matrix {
        &self.transform
    }

    /// Returns the world-to-object transform.
    pub const fn inverse(&self) -> &Matrix {
        &self.inverse
    }
}

impl Hittable for MatrixInstance {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let object_ray = Ray::with_time(
            transform_point(*ray.origin(), &self.inverse),
            transform_vector(*ray.direction(), &self.inverse),
            ray.time(),
        );
        let mut record = self.object.hit_with_rng(&object_ray, ray_t, rng)?;
        let object_outward_normal = if record.front_face {
            record.normal
        } else {
            -record.normal
        };
        record.point = transform_point(record.point, &self.transform);
        let outward_normal =
            transform_vector(object_outward_normal, &self.normal_transform).normalized();
        record.set_face_normal(ray, outward_normal);
        Some(record)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

fn transform_bounds(bounds: Aabb, transform: &Matrix) -> Aabb {
    let mut transformed_bounds = None;

    for x in [bounds.min.0, bounds.max.0] {
        for y in [bounds.min.1, bounds.max.1] {
            for z in [bounds.min.2, bounds.max.2] {
                let point = transform_point(Point::new(x, y, z), transform);
                transformed_bounds = Some(transformed_bounds.map_or_else(
                    || Aabb::from_points(point, point),
                    |bounds: Aabb| bounds.union_point(point),
                ));
            }
        }
    }

    transformed_bounds.expect("transforming finite bounds should produce bounds")
}

fn transform_point(point: Point, transform: &Matrix) -> Point {
    let transformed =
        transform.transform_homogeneous_point(&[point.x(), point.y(), point.z(), 1.0]);
    let w = transformed[3];
    if w.abs() > f64::EPSILON {
        Point::new(transformed[0] / w, transformed[1] / w, transformed[2] / w)
    } else {
        Point::new(transformed[0], transformed[1], transformed[2])
    }
}

fn transform_vector(vector: Vector, transform: &Matrix) -> Vector {
    let transformed =
        transform.transform_homogeneous_point(&[vector.x(), vector.y(), vector.z(), 0.0]);
    Vector::new(transformed[0], transformed[1], transformed[2])
}

/// A constant-density participating medium bounded by another hittable object.
pub struct ConstantMedium {
    boundary: Box<dyn Hittable>,
    neg_inv_density: f64,
    phase_function: MaterialRef,
    bounds: Option<Aabb>,
}

impl fmt::Debug for ConstantMedium {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConstantMedium")
            .field("neg_inv_density", &self.neg_inv_density)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl ConstantMedium {
    /// Creates a constant-density medium with constant particle color.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn new(boundary: impl Hittable + 'static, density: f64, albedo: LinearColor) -> Self {
        Self::with_phase_function(boundary, density, Arc::new(Isotropic::new(albedo)))
    }

    /// Creates a constant-density medium with a texture-backed phase function.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn from_texture(
        boundary: impl Hittable + 'static,
        density: f64,
        texture: impl RayTexture + 'static,
    ) -> Self {
        Self::with_phase_function(
            boundary,
            density,
            Arc::new(Isotropic::from_texture(texture)),
        )
    }

    /// Creates a constant-density medium from a boxed boundary and constant particle color.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn from_box(boundary: Box<dyn Hittable>, density: f64, albedo: LinearColor) -> Self {
        Self::from_box_with_phase_function(boundary, density, Arc::new(Isotropic::new(albedo)))
    }

    /// Creates a constant-density medium with an explicit phase-function material.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn with_phase_function(
        boundary: impl Hittable + 'static,
        density: f64,
        phase_function: MaterialRef,
    ) -> Self {
        Self::from_box_with_phase_function(Box::new(boundary), density, phase_function)
    }

    /// Creates a constant-density medium from boxed parts.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn from_box_with_phase_function(
        boundary: Box<dyn Hittable>,
        density: f64,
        phase_function: MaterialRef,
    ) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "medium density must be positive and finite"
        );
        let bounds = boundary.bounding_box();
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_function,
            bounds,
        }
    }

    /// Returns the medium boundary bounds.
    #[must_use]
    pub const fn bounds(&self) -> Option<Aabb> {
        self.bounds
    }
}

impl Hittable for ConstantMedium {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let entry_t = self
            .boundary
            .hit_with_rng(ray, Interval::UNIVERSE, rng)?
            .t
            .max(ray_t.min);
        let exit_t = self
            .boundary
            .hit_with_rng(ray, Interval::new(entry_t + 0.0001, INFINITY), rng)?
            .t
            .min(ray_t.max);

        if entry_t >= exit_t {
            return None;
        }

        let entry_t = entry_t.max(0.0);
        let ray_length = ray.direction().length();
        if ray_length <= f64::EPSILON {
            return None;
        }

        let distance_inside_boundary = (exit_t - entry_t) * ray_length;
        let sample = rng.random_double().max(f64::MIN_POSITIVE);
        let hit_distance = self.neg_inv_density * sample.ln();
        if hit_distance > distance_inside_boundary {
            return None;
        }

        let t = entry_t + hit_distance / ray_length;
        Some(HitRecord {
            point: ray.at(t),
            normal: Vector::new(1.0, 0.0, 0.0),
            t,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: self.phase_function.as_ref(),
        })
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

fn rotate_y_point(point: Point, sin_theta: f64, cos_theta: f64) -> Point {
    Point::new(
        cos_theta * point.x() + sin_theta * point.z(),
        point.y(),
        -sin_theta * point.x() + cos_theta * point.z(),
    )
}

fn rotate_y_point_inverse(point: Point, sin_theta: f64, cos_theta: f64) -> Point {
    Point::new(
        cos_theta * point.x() - sin_theta * point.z(),
        point.y(),
        sin_theta * point.x() + cos_theta * point.z(),
    )
}

fn rotate_y_vector(vector: Vector, sin_theta: f64, cos_theta: f64) -> Vector {
    Vector::new(
        cos_theta * vector.x() + sin_theta * vector.z(),
        vector.y(),
        -sin_theta * vector.x() + cos_theta * vector.z(),
    )
}

fn rotate_y_vector_inverse(vector: Vector, sin_theta: f64, cos_theta: f64) -> Vector {
    Vector::new(
        cos_theta * vector.x() - sin_theta * vector.z(),
        vector.y(),
        sin_theta * vector.x() + cos_theta * vector.z(),
    )
}

fn rotate_y_bounds(bounds: Aabb, sin_theta: f64, cos_theta: f64) -> Aabb {
    let mut rotated_bounds = None;

    for x in [bounds.min.0, bounds.max.0] {
        for y in [bounds.min.1, bounds.max.1] {
            for z in [bounds.min.2, bounds.max.2] {
                let point = rotate_y_point(Point::new(x, y, z), sin_theta, cos_theta);
                rotated_bounds = Some(rotated_bounds.map_or_else(
                    || Aabb::from_points(point, point),
                    |bounds: Aabb| bounds.union_point(point),
                ));
            }
        }
    }

    rotated_bounds.expect("rotating finite bounds should produce bounds")
}

/// A collection of hittable scene objects.
#[derive(Default)]
pub struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
    bounds: Option<Aabb>,
    has_unbounded: bool,
}

impl fmt::Debug for HittableList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HittableList")
            .field("len", &self.objects.len())
            .field("bounds", &self.bounds)
            .field("has_unbounded", &self.has_unbounded)
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
            bounds: None,
            has_unbounded: false,
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
        self.bounds = None;
        self.has_unbounded = false;
    }

    /// Adds an object to the scene.
    pub fn add(&mut self, object: impl Hittable + 'static) {
        self.add_box(Box::new(object));
    }

    /// Adds a boxed hittable object to the scene.
    pub fn add_box(&mut self, object: Box<dyn Hittable>) {
        if !self.has_unbounded {
            if let Some(object_bounds) = object.bounding_box() {
                self.bounds = Some(
                    self.bounds
                        .map_or(object_bounds, |bounds| bounds.surrounding(object_bounds)),
                );
            } else {
                self.bounds = None;
                self.has_unbounded = true;
            }
        }
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
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for object in &self.objects {
            if let Some(record) =
                object.hit_with_rng(ray, Interval::new(ray_t.min, closest_so_far), rng)
            {
                closest_so_far = record.t;
                closest_hit = Some(record);
            }
        }

        closest_hit
    }

    fn bounding_box(&self) -> Option<Aabb> {
        if self.has_unbounded {
            None
        } else {
            self.bounds
        }
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
        let mut rng = SampleRng::default();
        hit_object_indices(&self.objects, 0..self.objects.len(), ray, ray_t, &mut rng)
    }
}

impl Hittable for BvhNode {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.bvh.hit(&self.objects, ray, ray_t, rng)
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
                object
                    .bounding_box()
                    .map(|bounds| BvhPrimitiveInfo::new(index, bounds))
            })
            .collect::<Option<Vec<_>>>()?;
        let mut bvh = Self {
            nodes: Vec::with_capacity(objects.len().saturating_mul(2).saturating_sub(1)),
            indices: (0..objects.len()).collect(),
        };
        bvh.build_range(&primitive_info, 0, objects.len());
        Some(bvh)
    }

    fn build_ray_primitives(primitives: &[RayPrimitive]) -> Option<Self> {
        if primitives.is_empty() {
            return None;
        }

        let primitive_info = primitives
            .iter()
            .map(|primitive| primitive.geometry.bounding_box())
            .enumerate()
            .map(|(index, bounds)| bounds.map(|bounds| BvhPrimitiveInfo::new(index, bounds)))
            .collect::<Option<Vec<_>>>()?;
        let mut bvh = Self {
            nodes: Vec::with_capacity(primitives.len().saturating_mul(2).saturating_sub(1)),
            indices: (0..primitives.len()).collect(),
        };
        bvh.build_range(&primitive_info, 0, primitives.len());
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
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'a>> {
        self.hit_node(0, objects, ray, ray_t, rng)
    }

    fn hit_node<'a>(
        &'a self,
        node_index: usize,
        objects: &'a [Box<dyn Hittable>],
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
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
                rng,
            ),
            ObjectBvhNodeKind::Internal { left, right } => {
                let left_hit = self.hit_node(left, objects, ray, ray_t, rng);
                let closest = left_hit.as_ref().map_or(ray_t.max, |hit| hit.t);
                let right_hit =
                    self.hit_node(right, objects, ray, Interval::new(ray_t.min, closest), rng);
                right_hit.or(left_hit)
            }
        }
    }

    fn hit_ray_scene<'a>(
        &'a self,
        primitives: &'a [RayPrimitive],
        materials: &'a [RayMaterial],
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        self.hit_ray_scene_node(0, primitives, materials, ray, ray_t)
    }

    fn hit_ray_scene_node<'a>(
        &'a self,
        node_index: usize,
        primitives: &'a [RayPrimitive],
        materials: &'a [RayMaterial],
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        let node = self.nodes[node_index];
        if !node.bounds.hit_ray(ray, ray_t.min, ray_t.max) {
            return None;
        }

        match node.kind {
            ObjectBvhNodeKind::Leaf { first, count } => hit_ray_scene_indices(
                primitives,
                materials,
                self.indices[first..first + count].iter().copied(),
                ray,
                ray_t,
            ),
            ObjectBvhNodeKind::Internal { left, right } => {
                let left_hit = self.hit_ray_scene_node(left, primitives, materials, ray, ray_t);
                let closest = left_hit.as_ref().map_or(ray_t.max, |hit| hit.t);
                let right_hit = self.hit_ray_scene_node(
                    right,
                    primitives,
                    materials,
                    ray,
                    Interval::new(ray_t.min, closest),
                );
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
    rng: &mut SampleRng,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        if let Some(record) =
            objects[index].hit_with_rng(ray, Interval::new(ray_t.min, closest_so_far), rng)
        {
            closest_so_far = record.t;
            closest_hit = Some(record);
        }
    }

    closest_hit
}

fn hit_ray_scene_indices<'a>(
    primitives: &'a [RayPrimitive],
    materials: &'a [RayMaterial],
    indices: impl IntoIterator<Item = usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        let primitive = primitives[index];
        if let Some(surface) = primitive
            .geometry
            .intersect(ray, Interval::new(ray_t.min, closest_so_far))
        {
            closest_so_far = surface.t;
            closest_hit = Some(HitRecord::from_surface(
                surface,
                &materials[primitive.material],
            ));
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
    bvh: Option<ObjectBvh>,
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
            bvh: None,
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
        self.rebuild_bvh();
    }

    /// Adds a sphere using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_sphere(&mut self, center: Point, radius: f64, material: MaterialId) {
        self.add_primitive(RayGeometry::sphere(center, radius), material);
    }

    /// Adds a moving sphere using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_moving_sphere(
        &mut self,
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: MaterialId,
    ) {
        self.add_primitive(
            RayGeometry::moving_sphere(center_start, center_end, radius),
            material,
        );
    }

    /// Adds a quad using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_quad(&mut self, corner: Point, u: Vector, v: Vector, material: MaterialId) {
        self.add_primitive(RayGeometry::quad(corner, u, v), material);
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

    /// Adds a material and a moving sphere that references it.
    pub fn add_moving_sphere_with_material(
        &mut self,
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: impl Into<RayMaterial>,
    ) -> MaterialId {
        let material = self.add_material(material);
        self.add_moving_sphere(center_start, center_end, radius, material);
        material
    }

    /// Adds a material and a quad that references it.
    pub fn add_quad_with_material(
        &mut self,
        corner: Point,
        u: Vector,
        v: Vector,
        material: impl Into<RayMaterial>,
    ) -> MaterialId {
        let material = self.add_material(material);
        self.add_quad(corner, u, v, material);
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

    /// Returns true when this scene has a built primitive BVH.
    #[must_use]
    pub fn has_bvh(&self) -> bool {
        self.bvh.is_some()
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

    fn rebuild_bvh(&mut self) {
        self.bvh = ObjectBvh::build_ray_primitives(&self.primitives);
    }
}

impl Hittable for RayScene {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.bvh.as_ref().map_or_else(
            || {
                hit_ray_scene_indices(
                    &self.primitives,
                    &self.materials,
                    0..self.primitives.len(),
                    ray,
                    ray_t,
                )
            },
            |bvh| bvh.hit_ray_scene(&self.primitives, &self.materials, ray, ray_t),
        )
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
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
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
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
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

fn sphere_uv(point_on_unit_sphere: Vector) -> (f64, f64) {
    let theta = (-point_on_unit_sphere.y()).acos();
    let phi = (-point_on_unit_sphere.z()).atan2(point_on_unit_sphere.x()) + PI;
    (phi / (2.0 * PI), theta / PI)
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
        let canvas = Canvas::from_fn(3, 3, |x, y| {
            Rgb::from_raw_linear_color(LinearColor::new(
                f64::from(x) / 2.0,
                f64::from(y) / 2.0,
                0.0,
            ))
        });
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
        let ray = Ray::with_time(
            Point::new(0.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, -1.0),
            0.375,
        );
        let color = sky_gradient(&ray);

        assert_close(color.x(), 0.75);
        assert_close(color.y(), 0.85);
        assert_close(color.z(), 1.0);
    }

    #[test]
    fn first_sphere_render_has_red_center() {
        let canvas = RayCamera::new(40, WIDESCREEN_ASPECT_RATIO).render(first_sphere_color);
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
    fn solid_color_texture_ignores_coordinates() {
        let texture = SolidColor::new(LinearColor::new(0.2, 0.4, 0.6));

        assert_eq!(
            texture.value(0.75, 0.25, Point::new(10.0, -4.0, 2.0)),
            LinearColor::new(0.2, 0.4, 0.6)
        );
    }

    #[test]
    fn checker_texture_alternates_in_world_space() {
        let texture = CheckerTexture::from_colors(
            1.0,
            LinearColor::new(0.1, 0.2, 0.3),
            LinearColor::new(0.8, 0.7, 0.6),
        );

        assert_eq!(
            texture.value(0.0, 0.0, Point::new(0.1, 0.1, 0.1)),
            LinearColor::new(0.1, 0.2, 0.3)
        );
        assert_eq!(
            texture.value(0.0, 0.0, Point::new(1.1, 0.1, 0.1)),
            LinearColor::new(0.8, 0.7, 0.6)
        );
    }

    #[test]
    fn image_texture_samples_existing_texture_sampler() {
        let texture =
            ImageTexture::from_canvas(Canvas::from_pixels(2, 1, vec![Rgb::RED, Rgb::GREEN]));

        assert_eq!(
            texture.value(0.0, 0.5, Point::new(0.0, 0.0, 0.0)),
            LinearColor::new(1.0, 0.0, 0.0)
        );
        assert_eq!(
            texture.value(1.0, 0.5, Point::new(0.0, 0.0, 0.0)),
            LinearColor::new(0.0, 1.0, 0.0)
        );
    }

    #[test]
    fn lambertian_scatter_samples_texture_for_attenuation() {
        let material = Lambertian::checker(
            1.0,
            LinearColor::new(0.1, 0.2, 0.3),
            LinearColor::new(0.8, 0.7, 0.6),
        );
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(1.1, 0.1, 0.1),
            normal: Vector::new(0.0, 1.0, 0.0),
            t: 1.0,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(19);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("lambertian should scatter");

        assert_eq!(scatter.attenuation, LinearColor::new(0.8, 0.7, 0.6));
    }

    #[test]
    fn diffuse_light_emits_and_does_not_scatter() {
        let material = DiffuseLight::new(LinearColor::new(4.0, 3.0, 2.0));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, -1.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(41);

        assert_eq!(
            material.emitted(hit.u, hit.v, hit.point),
            LinearColor::new(4.0, 3.0, 2.0)
        );
        assert!(material.scatter(&ray, &hit, &mut rng).is_none());
    }

    #[test]
    fn isotropic_scatter_uses_random_direction_and_texture_attenuation() {
        let material = Isotropic::new(LinearColor::new(0.25, 0.5, 0.75));
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.5);
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, -1.0),
            normal: Vector::new(1.0, 0.0, 0.0),
            t: 1.0,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(43);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("isotropic medium should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_close(scatter.ray.direction().length(), 1.0);
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(0.25, 0.5, 0.75));
    }

    #[test]
    fn sphere_uv_matches_book_reference_points() {
        let left = sphere_uv(Vector::new(-1.0, 0.0, 0.0));
        let right = sphere_uv(Vector::new(1.0, 0.0, 0.0));
        let up = sphere_uv(Vector::new(0.0, 1.0, 0.0));
        let down = sphere_uv(Vector::new(0.0, -1.0, 0.0));
        let front = sphere_uv(Vector::new(0.0, 0.0, 1.0));
        let back = sphere_uv(Vector::new(0.0, 0.0, -1.0));

        assert_close(left.0, 0.0);
        assert_close(left.1, 0.5);
        assert_close(right.0, 0.5);
        assert_close(right.1, 0.5);
        assert_close(up.0, 0.5);
        assert_close(up.1, 1.0);
        assert_close(down.0, 0.5);
        assert_close(down.1, 0.0);
        assert_close(front.0, 0.25);
        assert_close(front.1, 0.5);
        assert_close(back.0, 0.75);
        assert_close(back.1, 0.5);
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
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.5);

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
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
    }

    #[test]
    fn moving_sphere_uses_ray_time_for_hits() {
        let sphere = MovingSphere::with_material(
            Point::new(0.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
            0.5,
            Lambertian::new(LinearColor::new(0.2, 0.2, 0.2)),
        );
        let early = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.0);
        let late = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, -1.0), 1.0);

        let early_hit = sphere
            .hit(&early, Interval::new(0.0, INFINITY))
            .expect("early ray should hit start position");
        let late_hit = sphere
            .hit(&late, Interval::new(0.0, INFINITY))
            .expect("late ray should hit end position");

        assert_close(early_hit.t, 0.5);
        assert!(late_hit.point.y() > early_hit.point.y());
        assert!(sphere.bounding_box().is_some());
    }

    #[test]
    fn quad_hit_records_texture_coordinates_and_material() {
        let quad = Quad::with_material(
            Point::new(-2.0, -2.0, -1.0),
            Vector::new(4.0, 0.0, 0.0),
            Vector::new(0.0, 4.0, 0.0),
            Lambertian::new(LinearColor::new(0.2, 0.4, 0.6)),
        );
        let ray = Ray::new(Point::new(0.0, 1.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(37);

        let record = quad
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("quad should be hit");
        let scatter = record
            .material
            .scatter(&ray, &record, &mut rng)
            .expect("quad material should scatter");

        assert!(record.front_face);
        assert_close(record.t, 1.0);
        assert_close(record.u, 0.5);
        assert_close(record.v, 0.75);
        assert_eq!(record.normal, Vector::new(0.0, 0.0, 1.0));
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
        assert!(quad.bounding_box().is_some());
    }

    #[test]
    fn box_object_builds_six_bounded_quad_sides() {
        let material: MaterialRef = Arc::new(Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)));
        let object = box_object(
            Point::new(1.0, 2.0, 3.0),
            Point::new(-1.0, -2.0, -3.0),
            material,
        );
        let bounds = object.bounding_box().expect("box should be bounded");

        assert_eq!(object.len(), 6);
        assert!(bounds.min.0 <= -1.0);
        assert!(bounds.min.1 <= -2.0);
        assert!(bounds.min.2 <= -3.0);
        assert!(bounds.max.0 >= 1.0);
        assert!(bounds.max.1 >= 2.0);
        assert!(bounds.max.2 >= 3.0);
    }

    #[test]
    fn translated_instance_moves_ray_hits_and_bounds() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let translated = Translate::new(sphere, Vector::new(2.0, 0.0, 0.0));
        let ray = Ray::new(Point::new(2.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = translated
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("translated sphere should be hit");
        let bounds = translated
            .bounding_box()
            .expect("translated sphere should be bounded");

        assert_close(record.t, 0.5);
        assert_close(record.point.x(), 2.0);
        assert_close(bounds.min.0, 1.5);
        assert_close(bounds.max.0, 2.5);
    }

    #[test]
    fn rotate_y_instance_rotates_ray_hits_normals_and_bounds() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let rotated = RotateY::new(sphere, 90.0);
        let ray = Ray::new(Point::new(-1.0, 0.0, -2.0), Vector::new(0.0, 0.0, 1.0));

        let record = rotated
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("rotated sphere should be hit");
        let bounds = rotated
            .bounding_box()
            .expect("rotated sphere should be bounded");

        assert_close(record.t, 1.5);
        assert_close(record.point.x(), -1.0);
        assert_close(record.point.z(), -0.5);
        assert_close(record.normal.z(), -1.0);
        assert!(bounds.min.0 < -1.4);
        assert!(bounds.max.0 < -0.4);
    }

    #[test]
    fn matrix_instance_transforms_ray_hits_normals_and_bounds() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let transform = Matrix::translate(2.0, 0.0, 0.0) * Matrix::scale(2.0, 1.0, 1.0);
        let instance = MatrixInstance::new(sphere, transform).expect("transform should invert");
        let ray = Ray::new(Point::new(2.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = instance
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("matrix instance should be hit");
        let bounds = instance
            .bounding_box()
            .expect("matrix instance should be bounded");

        assert_close(record.t, 0.5);
        assert_close(record.point.x(), 2.0);
        assert_close(record.point.z(), -0.5);
        assert_close(record.normal.z(), 1.0);
        assert_close(bounds.min.0, 1.0);
        assert_close(bounds.max.0, 3.0);
    }

    #[test]
    fn constant_medium_samples_hit_inside_boundary() {
        let boundary = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let medium = ConstantMedium::new(boundary, 1.0e9, LinearColor::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(47);

        let record = medium
            .hit_with_rng(&ray, Interval::new(0.0, INFINITY), &mut rng)
            .expect("dense medium should scatter inside boundary");

        assert!(record.t > 0.5);
        assert!(record.t < 1.5);
        assert_eq!(record.normal, Vector::new(1.0, 0.0, 0.0));
        assert!(record.front_face);
        assert!(medium.bounding_box().is_some());
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
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(0.8, 0.8, 0.8));
    }

    #[test]
    fn dielectric_scatter_refracts_perpendicular_ray() {
        let material = Dielectric::new(RefractiveIndex::GLASS);
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::with_time(
            Point::new(0.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, -1.0),
            0.625,
        );
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
        assert_close(scatter.ray.time(), ray.time());
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
    fn quads_scene_contains_five_bounded_objects() {
        let world = quads_world();

        assert_eq!(world.len(), 5);
        assert!(world.bounding_box().is_some());
    }

    #[test]
    fn light_scenes_are_bounded() {
        let simple = simple_light_world();
        let cornell = cornell_box_world();

        assert_eq!(simple.len(), 4);
        assert_eq!(cornell.len(), 8);
        assert!(simple.bounding_box().is_some());
        assert!(cornell.bounding_box().is_some());
        assert_eq!(cornell_smoke_world().len(), 8);
        assert!(cornell_smoke_world().bounding_box().is_some());
    }

    #[test]
    fn next_week_final_scene_is_bounded() {
        let world = next_week_final_scene_world(SolidColor::new(LinearColor::new(0.1, 0.2, 0.3)));

        assert_eq!(world.len(), 11);
        assert!(world.bounding_box().is_some());
    }

    #[test]
    fn final_scene_world_contains_many_random_spheres() {
        let world = final_scene_world();

        assert!(world.len() > 470);
        assert!(world.bounding_box().is_some());
        assert!(final_scene_bvh_world().bounding_box().is_some());
    }

    #[test]
    fn motion_blur_scene_contains_many_random_spheres() {
        let world = motion_blur_scene_world();
        let ray_scene = motion_blur_ray_scene();

        assert!(world.len() > 470);
        assert_eq!(ray_scene.len(), world.len());
        assert!(world.bounding_box().is_some());
        assert!(ray_scene.bounding_box().is_some());
        assert!(motion_blur_bvh_world().bounding_box().is_some());
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
        assert!(scene.has_bvh());
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
    }

    #[test]
    fn ray_scene_bvh_matches_linear_hit_path() {
        let mut scene = RayScene::new();
        let red = scene.add_material(RayMaterial::lambertian(LinearColor::new(1.0, 0.0, 0.0)));
        let green = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.0, 1.0, 0.0)));
        scene.add_sphere(Point::new(0.0, 0.0, -3.0), 0.5, red);
        scene.add_sphere(Point::new(0.0, 0.0, -1.0), 0.5, green);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let bvh_hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("BVH path should hit");
        let linear_hit = hit_ray_scene_indices(
            scene.primitives(),
            scene.materials(),
            0..scene.len(),
            &ray,
            Interval::new(0.0, INFINITY),
        )
        .expect("linear path should hit");

        assert!(scene.has_bvh());
        assert_close(bvh_hit.t, linear_hit.t);
        assert_eq!(bvh_hit.point, linear_hit.point);
    }

    #[test]
    fn ray_scene_supports_moving_spheres() {
        let mut scene = RayScene::new();
        let material = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.2, 0.4, 0.6)));
        scene.add_moving_sphere(
            Point::new(0.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
            0.5,
            material,
        );
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, -1.0), 1.0);

        let hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("moving sphere should be hit at end position");

        assert!(hit.point.y() > 0.0);
        assert!(scene.bounding_box().is_some());
    }

    #[test]
    fn ray_scene_supports_quads() {
        let mut scene = RayScene::new();
        scene.add_quad_with_material(
            Point::new(-1.0, -1.0, -1.0),
            Vector::new(2.0, 0.0, 0.0),
            Vector::new(0.0, 2.0, 0.0),
            RayMaterial::lambertian(LinearColor::new(0.7, 0.2, 0.1)),
        );
        let ray = Ray::new(Point::new(0.5, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("quad should be hit");

        assert_close(hit.t, 1.0);
        assert_close(hit.u, 0.75);
        assert_close(hit.v, 0.5);
        assert!(scene.bounding_box().is_some());
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
    fn final_scene_can_render_through_path_tracer() {
        let world = final_scene_bvh_world();
        let canvas = PathTracer::new(
            RayCamera::new(1, WIDESCREEN_ASPECT_RATIO)
                .with_samples_per_pixel(1)
                .with_max_depth(50)
                .with_vertical_fov(20.0)
                .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
                .with_view_up(Vector::new(0.0, 1.0, 0.0))
                .with_defocus_angle(0.6)
                .with_focus_distance(10.0),
        )
        .render(&world);

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 1);
    }

    #[test]
    fn motion_blur_scene_can_render_through_path_tracer() {
        let world = motion_blur_bvh_world();
        let canvas = PathTracer::new(
            RayCamera::new(1, WIDESCREEN_ASPECT_RATIO)
                .with_samples_per_pixel(1)
                .with_max_depth(50)
                .with_vertical_fov(20.0)
                .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
                .with_view_up(Vector::new(0.0, 1.0, 0.0))
                .with_defocus_angle(0.6)
                .with_focus_distance(10.0)
                .with_shutter_interval(0.0, 1.0),
        )
        .render(&world);

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
    fn hittable_list_caches_bounds_as_objects_are_added() {
        let mut world = HittableList::new();
        assert!(world.bounding_box().is_none());

        world.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        assert_eq!(
            world.bounding_box(),
            Some(Aabb::new((-0.5, -0.5, -2.5), (0.5, 0.5, -1.5)))
        );

        world.add(Sphere::new(Point::new(2.0, 1.0, -1.0), 0.25));
        assert_eq!(
            world.bounding_box(),
            Some(Aabb::new((-0.5, -0.5, -2.5), (2.25, 1.25, -0.75)))
        );

        world.clear();
        assert!(world.bounding_box().is_none());
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
        let world = normal_sphere_world();
        let canvas = RayCamera::new(40, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(100)
            .render_world_normals(&world);
        let center = canvas
            .get_pixel(20, 11)
            .expect("center pixel should be inside the canvas");

        assert_ne!(*center, Rgb::RED);
        assert!(center.blue > center.red);
    }

    #[test]
    fn diffuse_scene_render_is_gamma_corrected() {
        let world = normal_sphere_world();
        let canvas = RayCamera::new(20, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(100)
            .with_max_depth(50)
            .render_world(&world);
        let center = canvas
            .get_pixel(10, 5)
            .expect("center pixel should be inside the canvas");

        assert!(center.red > 0 || center.green > 0 || center.blue > 0);
    }
}

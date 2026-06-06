//! Ray-tracing BSDF, emitter, and phase-function materials.

use super::{
    HitRecord, LinearColor, rgb_to_linear_color,
    texture::{CheckerTexture, NoiseTexture, RayTexture, SolidColor, TextureRef},
};
use crate::{
    gmath::{random::SampleRng, ray::Ray, vector::Point},
    graphics::{
        colors::Rgb,
        lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex},
        material::SurfaceMaterial,
    },
};
use std::{fmt, sync::Arc};

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

pub(crate) fn default_material() -> MaterialRef {
    Arc::new(Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)))
}

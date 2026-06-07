//! Ray-tracing BSDF, emitter, and phase-function materials.

use super::{
    HitRecord, LinearColor, MaterialPdf, PI, SpherePdf,
    pdf::CosinePdf,
    rgb_to_linear_color,
    texture::{CheckerTexture, NoiseTexture, SolidColor, TextureRef},
};
use crate::{
    gmath::{random::SampleRng, ray::Ray, vector::Point},
    graphics::{
        colors::Rgb,
        lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex},
        material::SurfaceMaterial,
        texture::{SurfaceTexture, TextureSample},
    },
};
use std::{fmt, sync::Arc};

/// Material scattering result with explicit specular and PDF-sampled cases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScatterRecord {
    /// Deterministic or implicitly sampled specular ray that bypasses explicit PDF weighting.
    Specular {
        /// Scattered specular ray.
        ray: Ray,
        /// Per-channel color attenuation.
        attenuation: LinearColor,
    },
    /// PDF-sampled scattering event for diffuse surfaces and volumes.
    Scattering {
        /// Per-channel color attenuation.
        attenuation: LinearColor,
        /// PDF used by material scattering.
        pdf: MaterialPdf,
    },
}

/// A surface material that can scatter rays.
pub trait Material: Send + Sync {
    /// Returns emitted light for this material at a surface point.
    fn emitted(
        &self,
        _ray_in: &Ray,
        _hit: &HitRecord<'_>,
        _u: f64,
        _v: f64,
        _point: Point,
    ) -> LinearColor {
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

    /// Returns this material's scattering PDF for a proposed scattered ray.
    fn scattering_pdf(&self, _ray_in: &Ray, _hit: &HitRecord<'_>, _scattered: &Ray) -> f64 {
        0.0
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
    ///
    /// # Panics
    ///
    /// Panics if any albedo channel is not finite.
    #[must_use]
    pub fn new(albedo: LinearColor) -> Self {
        assert!(albedo.is_finite(), "Lambertian albedo must be finite");
        Self {
            albedo,
            texture: Arc::new(SolidColor::new(albedo)),
        }
    }

    /// Creates a Lambertian material only when `albedo` is finite.
    #[must_use]
    pub fn try_new(albedo: LinearColor) -> Option<Self> {
        albedo.is_finite().then(|| Self {
            albedo,
            texture: Arc::new(SolidColor::new(albedo)),
        })
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
    pub fn from_texture(texture: impl SurfaceTexture + 'static) -> Self {
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
    pub fn texture(&self) -> &dyn SurfaceTexture {
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
        _ray_in: &Ray,
        hit: &HitRecord<'_>,
        _rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let pdf = CosinePdf::new(hit.shading_normal)?;

        Some(ScatterRecord::Scattering {
            attenuation: self
                .texture
                .sample_linear(TextureSample::new(hit.u, hit.v, hit.point)),
            pdf: MaterialPdf::Cosine(pdf),
        })
    }

    fn scattering_pdf(&self, _ray_in: &Ray, hit: &HitRecord<'_>, scattered: &Ray) -> f64 {
        let cosine_theta = hit.shading_normal.dot(scattered.direction().normalized());
        if cosine_theta <= 0.0 {
            0.0
        } else {
            cosine_theta / PI
        }
    }
}

/// Diffuse light-emitting material.
#[derive(Clone, Debug)]
pub struct DiffuseLight {
    texture: TextureRef,
}

impl DiffuseLight {
    /// Creates a light material with a constant emitted color.
    ///
    /// # Panics
    ///
    /// Panics if any emitted color channel is not finite.
    #[must_use]
    pub fn new(emit: LinearColor) -> Self {
        assert!(emit.is_finite(), "diffuse light color must be finite");
        Self::from_texture(SolidColor::new(emit))
    }

    /// Creates a light material only when the emitted color is finite.
    #[must_use]
    pub fn try_new(emit: LinearColor) -> Option<Self> {
        emit.is_finite()
            .then(|| Self::from_texture(SolidColor::new(emit)))
    }

    /// Creates a light material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self { texture }
    }

    /// Creates a light material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl SurfaceTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Returns the texture sampled for emitted radiance.
    #[must_use]
    pub fn texture(&self) -> &dyn SurfaceTexture {
        self.texture.as_ref()
    }
}

impl Material for DiffuseLight {
    fn emitted(
        &self,
        _ray_in: &Ray,
        hit: &HitRecord<'_>,
        u: f64,
        v: f64,
        point: Point,
    ) -> LinearColor {
        if hit.front_face {
            self.texture.sample_linear(TextureSample::new(u, v, point))
        } else {
            LinearColor::default()
        }
    }
}

/// Isotropic phase-function material for constant-density volumes.
#[derive(Clone, Debug)]
pub struct Isotropic {
    texture: TextureRef,
}

impl Isotropic {
    /// Creates an isotropic material with constant attenuation.
    ///
    /// # Panics
    ///
    /// Panics if any albedo channel is not finite.
    #[must_use]
    pub fn new(albedo: LinearColor) -> Self {
        assert!(albedo.is_finite(), "isotropic albedo must be finite");
        Self::from_texture(SolidColor::new(albedo))
    }

    /// Creates an isotropic material only when `albedo` is finite.
    #[must_use]
    pub fn try_new(albedo: LinearColor) -> Option<Self> {
        albedo
            .is_finite()
            .then(|| Self::from_texture(SolidColor::new(albedo)))
    }

    /// Creates an isotropic material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self { texture }
    }

    /// Creates an isotropic material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl SurfaceTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Returns the texture sampled for medium attenuation.
    #[must_use]
    pub fn texture(&self) -> &dyn SurfaceTexture {
        self.texture.as_ref()
    }
}

impl Material for Isotropic {
    fn scatter(
        &self,
        _ray_in: &Ray,
        hit: &HitRecord<'_>,
        _rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        Some(ScatterRecord::Scattering {
            attenuation: self
                .texture
                .sample_linear(TextureSample::new(hit.u, hit.v, hit.point)),
            pdf: MaterialPdf::Sphere(SpherePdf),
        })
    }

    fn scattering_pdf(&self, _ray_in: &Ray, _hit: &HitRecord<'_>, _scattered: &Ray) -> f64 {
        1.0 / (4.0 * PI)
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
    ///
    /// # Panics
    ///
    /// Panics if any albedo channel or `fuzz` is not finite.
    #[must_use]
    pub fn new(albedo: LinearColor, fuzz: f64) -> Self {
        assert!(
            albedo.is_finite() && fuzz.is_finite(),
            "metal material values must be finite"
        );
        Self {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
        }
    }

    /// Creates a metal material only when `albedo` and `fuzz` are finite.
    #[must_use]
    pub fn try_new(albedo: LinearColor, fuzz: f64) -> Option<Self> {
        (albedo.is_finite() && fuzz.is_finite()).then_some(Self {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
        })
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
        let reflected = ray_in
            .direction()
            .normalized()
            .reflected(hit.shading_normal);
        let scattered_direction = reflected + self.fuzz * rng.random_unit_vector_spherical();
        if scattered_direction.dot(hit.normal) <= 0.0 {
            return None;
        }

        Some(ScatterRecord::Specular {
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

        Some(ScatterRecord::Specular {
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
    pub fn textured_lambertian(texture: impl SurfaceTexture + 'static) -> Self {
        Self::Lambertian(Lambertian::from_texture(texture))
    }

    /// Creates a diffuse light material variant.
    #[must_use]
    pub fn diffuse_light(emit: LinearColor) -> Self {
        Self::DiffuseLight(DiffuseLight::new(emit))
    }

    /// Creates a textured diffuse light material variant.
    #[must_use]
    pub fn textured_diffuse_light(texture: impl SurfaceTexture + 'static) -> Self {
        Self::DiffuseLight(DiffuseLight::from_texture(texture))
    }

    /// Creates an isotropic phase-function material variant.
    #[must_use]
    pub fn isotropic(albedo: LinearColor) -> Self {
        Self::Isotropic(Isotropic::new(albedo))
    }

    /// Creates a textured isotropic phase-function material variant.
    #[must_use]
    pub fn textured_isotropic(texture: impl SurfaceTexture + 'static) -> Self {
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

    /// Returns true for light-emitting material variants.
    #[must_use]
    pub const fn is_emissive(&self) -> bool {
        matches!(self, Self::DiffuseLight(_))
    }

    /// Returns true for materials that scatter through a deterministic specular ray.
    #[must_use]
    pub const fn is_delta(&self) -> bool {
        matches!(self, Self::Metal(_) | Self::Dielectric(_))
    }

    /// Returns true for volume phase-function material variants.
    #[must_use]
    pub const fn is_volume_phase(&self) -> bool {
        matches!(self, Self::Isotropic(_))
    }

    /// Returns true for materials that provide explicit PDF-sampled scattering.
    #[must_use]
    pub const fn has_pdf_scatter(&self) -> bool {
        matches!(self, Self::Lambertian(_) | Self::Isotropic(_))
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
    fn emitted(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        u: f64,
        v: f64,
        point: Point,
    ) -> LinearColor {
        match self {
            Self::Lambertian(material) => material.emitted(ray_in, hit, u, v, point),
            Self::DiffuseLight(material) => material.emitted(ray_in, hit, u, v, point),
            Self::Isotropic(material) => material.emitted(ray_in, hit, u, v, point),
            Self::Metal(material) => material.emitted(ray_in, hit, u, v, point),
            Self::Dielectric(material) => material.emitted(ray_in, hit, u, v, point),
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

    fn scattering_pdf(&self, ray_in: &Ray, hit: &HitRecord<'_>, scattered: &Ray) -> f64 {
        match self {
            Self::Lambertian(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::DiffuseLight(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::Isotropic(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::Metal(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::Dielectric(material) => material.scattering_pdf(ray_in, hit, scattered),
        }
    }
}

/// Shared material handle used by hittable objects.
pub type MaterialRef = Arc<dyn Material>;

pub(crate) fn default_material() -> MaterialRef {
    Arc::new(Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_material_constructors_reject_non_finite_values() {
        let finite = LinearColor::new(0.2, 0.3, 0.4);
        let invalid = LinearColor::new(0.2, f64::NAN, 0.4);

        assert!(Lambertian::try_new(finite).is_some());
        assert!(DiffuseLight::try_new(finite).is_some());
        assert!(Isotropic::try_new(finite).is_some());
        assert!(Lambertian::try_new(invalid).is_none());
        assert!(DiffuseLight::try_new(invalid).is_none());
        assert!(Isotropic::try_new(invalid).is_none());
        assert!(Metal::try_new(finite, f64::NAN).is_none());
        assert!(Metal::try_new(invalid, 0.2).is_none());
    }

    #[test]
    #[should_panic(expected = "Lambertian albedo must be finite")]
    fn lambertian_constructor_rejects_non_finite_albedo() {
        let _ = Lambertian::new(LinearColor::new(0.2, f64::NAN, 0.4));
    }

    #[test]
    #[should_panic(expected = "metal material values must be finite")]
    fn metal_constructor_rejects_non_finite_fuzz() {
        let _ = Metal::new(LinearColor::new(0.2, 0.3, 0.4), f64::NAN);
    }

    #[test]
    fn ray_material_flags_describe_scattering_behavior() {
        let lambertian = RayMaterial::lambertian(LinearColor::new(0.2, 0.3, 0.4));
        let light = RayMaterial::diffuse_light(LinearColor::new(3.0, 2.0, 1.0));
        let isotropic = RayMaterial::isotropic(LinearColor::new(0.5, 0.5, 0.5));
        let metal = RayMaterial::metal(LinearColor::new(0.7, 0.6, 0.5), 0.1);
        let dielectric = RayMaterial::dielectric(RefractiveIndex::GLASS);

        assert!(lambertian.has_pdf_scatter());
        assert!(!lambertian.is_delta());
        assert!(!lambertian.is_emissive());
        assert!(light.is_emissive());
        assert!(!light.has_pdf_scatter());
        assert!(isotropic.has_pdf_scatter());
        assert!(isotropic.is_volume_phase());
        assert!(metal.is_delta());
        assert!(dielectric.is_delta());
        assert!(!metal.has_pdf_scatter());
        assert!(!dielectric.has_pdf_scatter());
    }
}

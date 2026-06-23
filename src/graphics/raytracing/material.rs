//! Ray-tracing BSDF, emitter, and phase-function materials.

use super::{
    HitRecord, LinearColor, MaterialPdf, PI, SpherePdf,
    pdf::{CosinePdf, GgxReflectionPdf, HenyeyGreensteinPdf, Pdf},
    rgb_to_linear_color,
    texture::{CheckerTexture, NoiseTexture, NormalMapRef, SolidColor, TextureRef},
};
use crate::{
    gmath::{
        random::SampleRng,
        ray::Ray,
        vector::{Point, Vector},
    },
    graphics::{
        colors::Rgb,
        lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex},
        material::SurfaceMaterial,
        texture::{SurfaceTexture, TextureSample},
    },
};
use std::{fmt, sync::Arc};

#[cfg(feature = "spectral")]
use super::spectrum::{
    ConductorOpticalConstants, MeasuredSpectrum, MuellerMatrix, PolarizationFrame,
    SampledWavelength, Spectrum, conductor_fresnel, dielectric_fresnel,
};

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

    /// Produces a scattered ray for sampled-wavelength spectral transport.
    ///
    /// The default keeps RGB materials compatible by delegating to [`Self::scatter`]. Materials
    /// with wavelength-dependent transport, such as dispersive dielectrics, override this method.
    #[cfg(feature = "spectral")]
    fn scatter_spectral(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        _wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        self.scatter(ray_in, hit, rng)
    }

    /// Returns this material's scattering PDF for a proposed scattered ray.
    fn scattering_pdf(&self, _ray_in: &Ray, _hit: &HitRecord<'_>, _scattered: &Ray) -> f64 {
        0.0
    }

    /// Returns a first-hit albedo suitable for denoising auxiliary output.
    fn denoise_albedo(&self, _hit: &HitRecord<'_>) -> LinearColor {
        LinearColor::new(0.5, 0.5, 0.5)
    }

    /// Returns a normal-map-perturbed world-space shading normal, if this material has one.
    fn normal_map_shading_normal(&self, _hit: &HitRecord<'_>) -> Option<Vector> {
        None
    }

    /// Evaluates this material's sampled spectral attenuation for an existing RGB scatter record.
    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        _hit: &HitRecord<'_>,
        attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        Spectrum::from_linear_rgb(attenuation).sample(wavelength)
    }

    /// Evaluates emitted radiance at one sampled wavelength.
    #[cfg(feature = "spectral")]
    fn spectral_emitted(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        u: f64,
        v: f64,
        point: Point,
        wavelength: SampledWavelength,
    ) -> f64 {
        Spectrum::from_linear_rgb(self.emitted(ray_in, hit, u, v, point)).sample(wavelength)
    }

    /// Returns a Mueller matrix for sampled-wavelength polarized transport.
    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        _ray_in: &Ray,
        _hit: &HitRecord<'_>,
        _scattered: &Ray,
        _incoming_frame: PolarizationFrame,
        _outgoing_frame: PolarizationFrame,
        _wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        MuellerMatrix::depolarizer()
    }
}

/// A texture-like spectral source with an RGB fallback for the non-spectral renderer.
#[cfg(feature = "spectral")]
pub trait SpectralTexture: SurfaceTexture {
    /// Samples scalar spectral reflectance/radiance at one wavelength.
    fn sample_spectral(&self, sample: TextureSample, wavelength: SampledWavelength) -> f64;
}

/// Shared spectral texture handle.
#[cfg(feature = "spectral")]
pub type SpectralTextureRef = Arc<dyn SpectralTexture>;

/// Constant spectral source with a reconstructed RGB fallback.
#[cfg(feature = "spectral")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidSpectrumTexture {
    spectrum: Spectrum,
    rgb_fallback: LinearColor,
}

#[cfg(feature = "spectral")]
impl SolidSpectrumTexture {
    /// Creates a constant spectral source.
    #[must_use]
    pub fn new(spectrum: Spectrum) -> Self {
        Self {
            spectrum,
            rgb_fallback: spectrum.to_linear_rgb(),
        }
    }

    /// Returns the stored spectrum.
    #[must_use]
    pub const fn spectrum(self) -> Spectrum {
        self.spectrum
    }

    /// Returns the reconstructed RGB fallback.
    #[must_use]
    pub const fn rgb_fallback(self) -> LinearColor {
        self.rgb_fallback
    }
}

#[cfg(feature = "spectral")]
impl SpectralTexture for SolidSpectrumTexture {
    fn sample_spectral(&self, _sample: TextureSample, wavelength: SampledWavelength) -> f64 {
        self.spectrum.sample(wavelength)
    }
}

#[cfg(feature = "spectral")]
impl SurfaceTexture for SolidSpectrumTexture {
    fn sample_linear(&self, _sample: TextureSample) -> LinearColor {
        self.rgb_fallback
    }
}

/// Measured spectral source with a reconstructed RGB fallback.
#[cfg(feature = "spectral")]
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredSpectrumTexture {
    spectrum: MeasuredSpectrum,
    rgb_fallback: LinearColor,
}

#[cfg(feature = "spectral")]
impl MeasuredSpectrumTexture {
    /// Creates a measured spectral source.
    #[must_use]
    pub fn new(spectrum: MeasuredSpectrum) -> Self {
        let rgb_fallback = spectrum.to_linear_rgb();
        Self {
            spectrum,
            rgb_fallback,
        }
    }

    /// Returns the measured spectrum.
    #[must_use]
    pub const fn spectrum(&self) -> &MeasuredSpectrum {
        &self.spectrum
    }

    /// Returns the reconstructed RGB fallback.
    #[must_use]
    pub const fn rgb_fallback(&self) -> LinearColor {
        self.rgb_fallback
    }
}

#[cfg(feature = "spectral")]
impl SpectralTexture for MeasuredSpectrumTexture {
    fn sample_spectral(&self, _sample: TextureSample, wavelength: SampledWavelength) -> f64 {
        self.spectrum.sample(wavelength)
    }
}

#[cfg(feature = "spectral")]
impl SurfaceTexture for MeasuredSpectrumTexture {
    fn sample_linear(&self, _sample: TextureSample) -> LinearColor {
        self.rgb_fallback
    }
}

/// Measured eta/k spectra for a conductive interface.
#[cfg(feature = "spectral")]
#[derive(Clone, Debug, PartialEq)]
struct ConductorOpticalConstantsSpectra {
    eta: MeasuredSpectrum,
    k: MeasuredSpectrum,
}

#[cfg(feature = "spectral")]
impl ConductorOpticalConstantsSpectra {
    fn new(eta: MeasuredSpectrum, k: MeasuredSpectrum) -> Self {
        Self { eta, k }
    }

    fn at_wavelength_nm(&self, wavelength_nm: f64) -> ConductorOpticalConstants {
        ConductorOpticalConstants::new(
            self.eta.sample_wavelength_nm(wavelength_nm),
            self.k.sample_wavelength_nm(wavelength_nm),
        )
    }

    fn at(&self, wavelength: SampledWavelength) -> ConductorOpticalConstants {
        self.at_wavelength_nm(wavelength.wavelength_nm())
    }
}

#[derive(Clone, Debug)]
enum MaterialColorSource {
    Constant(SolidColor),
    Texture(TextureRef),
    #[cfg(feature = "spectral")]
    Spectrum(SpectralTextureRef),
}

impl MaterialColorSource {
    fn constant(color: LinearColor) -> Self {
        Self::Constant(SolidColor::new(color))
    }

    fn texture(texture: TextureRef) -> Self {
        Self::Texture(texture)
    }

    #[cfg(feature = "spectral")]
    fn spectrum(spectrum: Spectrum) -> Self {
        Self::spectral_texture(Arc::new(SolidSpectrumTexture::new(spectrum)))
    }

    #[cfg(feature = "spectral")]
    fn measured_spectrum(spectrum: MeasuredSpectrum) -> Self {
        Self::spectral_texture(Arc::new(MeasuredSpectrumTexture::new(spectrum)))
    }

    #[cfg(feature = "spectral")]
    fn spectral_texture(texture: SpectralTextureRef) -> Self {
        Self::Spectrum(texture)
    }

    fn sample(&self, sample: TextureSample) -> LinearColor {
        match self {
            Self::Constant(color) => color.color,
            Self::Texture(texture) => texture.sample_linear(sample),
            #[cfg(feature = "spectral")]
            Self::Spectrum(texture) => texture.sample_linear(sample),
        }
    }

    #[cfg(feature = "spectral")]
    fn sample_spectrum(&self, sample: TextureSample, wavelength: SampledWavelength) -> f64 {
        match self {
            Self::Constant(color) => Spectrum::from_linear_rgb(color.color).sample(wavelength),
            Self::Texture(texture) => {
                Spectrum::from_linear_rgb(texture.sample_linear(sample)).sample(wavelength)
            }
            Self::Spectrum(texture) => texture.sample_spectral(sample, wavelength),
        }
    }

    fn surface_texture(&self) -> &dyn SurfaceTexture {
        match self {
            Self::Constant(color) => color,
            Self::Texture(texture) => texture.as_ref(),
            #[cfg(feature = "spectral")]
            Self::Spectrum(texture) => texture.as_ref(),
        }
    }
}

/// Lambertian diffuse material.
#[derive(Clone)]
pub struct Lambertian {
    /// Representative diffuse reflectance for constant-color compatibility.
    pub albedo: LinearColor,
    color: MaterialColorSource,
    normal_map: Option<NormalMapRef>,
}

impl fmt::Debug for Lambertian {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Lambertian")
            .field("albedo", &self.albedo)
            .field("color", &self.color)
            .field("normal_map", &self.normal_map.is_some())
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
            color: MaterialColorSource::constant(albedo),
            normal_map: None,
        }
    }

    /// Creates a Lambertian material only when `albedo` is finite.
    #[must_use]
    pub fn try_new(albedo: LinearColor) -> Option<Self> {
        albedo.is_finite().then(|| Self {
            albedo,
            color: MaterialColorSource::constant(albedo),
            normal_map: None,
        })
    }

    /// Creates a Lambertian material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self {
            albedo: LinearColor::new(0.5, 0.5, 0.5),
            color: MaterialColorSource::texture(texture),
            normal_map: None,
        }
    }

    /// Creates a Lambertian material from a constant spectral reflectance.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_spectrum(spectrum: Spectrum) -> Self {
        Self {
            albedo: spectrum.to_linear_rgb(),
            color: MaterialColorSource::spectrum(spectrum),
            normal_map: None,
        }
    }

    /// Creates a Lambertian material from measured spectral diffuse reflectance.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectrum(spectrum: MeasuredSpectrum) -> Self {
        Self {
            albedo: spectrum.to_linear_rgb(),
            color: MaterialColorSource::measured_spectrum(spectrum),
            normal_map: None,
        }
    }

    /// Creates a Lambertian material from a shared spectral texture.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_shared_spectral_texture(texture: SpectralTextureRef) -> Self {
        Self {
            albedo: LinearColor::new(0.5, 0.5, 0.5),
            color: MaterialColorSource::spectral_texture(texture),
            normal_map: None,
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
        self.color.surface_texture()
    }

    /// Adds a shared tangent-space normal map.
    #[must_use]
    pub fn with_shared_normal_map(mut self, normal_map: NormalMapRef) -> Self {
        self.normal_map = Some(normal_map);
        self
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
                .color
                .sample(TextureSample::new(hit.u, hit.v, hit.point)),
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

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        self.color
            .sample(TextureSample::new(hit.u, hit.v, hit.point))
    }

    fn normal_map_shading_normal(&self, hit: &HitRecord<'_>) -> Option<Vector> {
        self.normal_map
            .as_deref()
            .and_then(|normal_map| normal_map_world_normal(hit, normal_map))
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        hit: &HitRecord<'_>,
        _attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        self.color
            .sample_spectrum(TextureSample::new(hit.u, hit.v, hit.point), wavelength)
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        _ray_in: &Ray,
        _hit: &HitRecord<'_>,
        _scattered: &Ray,
        _incoming_frame: PolarizationFrame,
        _outgoing_frame: PolarizationFrame,
        _wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        MuellerMatrix::depolarizer()
    }
}

/// Layered diffuse plus GGX glossy material.
#[derive(Clone)]
pub struct LayeredDiffuseGgx {
    /// Representative diffuse reflectance for constant-color compatibility.
    pub diffuse_color: LinearColor,
    /// Representative specular reflectance for constant-color compatibility.
    pub specular_color: LinearColor,
    /// Perceptual GGX roughness in `0.0..=1.0`.
    pub roughness: f64,
    diffuse: MaterialColorSource,
    specular: MaterialColorSource,
    normal_map: Option<NormalMapRef>,
    #[cfg(feature = "spectral")]
    conductor: Option<ConductorOpticalConstants>,
    #[cfg(feature = "spectral")]
    conductor_spectra: Option<ConductorOpticalConstantsSpectra>,
}

impl fmt::Debug for LayeredDiffuseGgx {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("LayeredDiffuseGgx");
        debug
            .field("diffuse_color", &self.diffuse_color)
            .field("specular_color", &self.specular_color)
            .field("roughness", &self.roughness)
            .field("diffuse", &self.diffuse)
            .field("specular", &self.specular)
            .field("normal_map", &self.normal_map.is_some());
        #[cfg(feature = "spectral")]
        debug.field("conductor", &self.conductor);
        #[cfg(feature = "spectral")]
        debug.field("conductor_spectra", &self.conductor_spectra.is_some());
        debug.finish()
    }
}

impl LayeredDiffuseGgx {
    /// Creates a layered diffuse plus GGX material from constant colors.
    ///
    /// # Panics
    ///
    /// Panics if any color channel or `roughness` is not finite.
    #[must_use]
    pub fn new(diffuse_color: LinearColor, specular_color: LinearColor, roughness: f64) -> Self {
        assert!(
            diffuse_color.is_finite() && specular_color.is_finite() && roughness.is_finite(),
            "layered diffuse/GGX material values must be finite"
        );
        Self {
            diffuse_color,
            specular_color,
            roughness: roughness.clamp(0.0, 1.0),
            diffuse: MaterialColorSource::constant(diffuse_color),
            specular: MaterialColorSource::constant(specular_color),
            normal_map: None,
            #[cfg(feature = "spectral")]
            conductor: None,
            #[cfg(feature = "spectral")]
            conductor_spectra: None,
        }
    }

    /// Creates a layered material with a shared diffuse texture and constant specular color.
    ///
    /// # Panics
    ///
    /// Panics if any specular channel or `roughness` is not finite.
    #[must_use]
    pub fn from_shared_diffuse_texture(
        diffuse: TextureRef,
        specular_color: LinearColor,
        roughness: f64,
    ) -> Self {
        assert!(
            specular_color.is_finite() && roughness.is_finite(),
            "layered diffuse/GGX material values must be finite"
        );
        Self {
            diffuse_color: LinearColor::new(0.5, 0.5, 0.5),
            specular_color,
            roughness: roughness.clamp(0.0, 1.0),
            diffuse: MaterialColorSource::texture(diffuse),
            specular: MaterialColorSource::constant(specular_color),
            normal_map: None,
            #[cfg(feature = "spectral")]
            conductor: None,
            #[cfg(feature = "spectral")]
            conductor_spectra: None,
        }
    }

    /// Creates a layered material from a diffuse texture object and constant specular color.
    ///
    /// # Panics
    ///
    /// Panics if any specular channel or `roughness` is not finite.
    #[must_use]
    pub fn from_diffuse_texture(
        diffuse: impl SurfaceTexture + 'static,
        specular_color: LinearColor,
        roughness: f64,
    ) -> Self {
        Self::from_shared_diffuse_texture(Arc::new(diffuse), specular_color, roughness)
    }

    /// Creates a layered material from constant spectral diffuse and specular reflectance.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_spectra(diffuse: Spectrum, specular: Spectrum, roughness: f64) -> Self {
        assert!(
            roughness.is_finite(),
            "layered diffuse/GGX material roughness must be finite"
        );
        Self {
            diffuse_color: diffuse.to_linear_rgb(),
            specular_color: specular.to_linear_rgb(),
            roughness: roughness.clamp(0.0, 1.0),
            diffuse: MaterialColorSource::spectrum(diffuse),
            specular: MaterialColorSource::spectrum(specular),
            normal_map: None,
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Creates a layered material from measured spectral diffuse and specular reflectance.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectra(
        diffuse: MeasuredSpectrum,
        specular: MeasuredSpectrum,
        roughness: f64,
    ) -> Self {
        assert!(
            roughness.is_finite(),
            "layered diffuse/GGX material roughness must be finite"
        );
        Self {
            diffuse_color: diffuse.to_linear_rgb(),
            specular_color: specular.to_linear_rgb(),
            roughness: roughness.clamp(0.0, 1.0),
            diffuse: MaterialColorSource::measured_spectrum(diffuse),
            specular: MaterialColorSource::measured_spectrum(specular),
            normal_map: None,
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Creates a layered material from shared spectral diffuse and specular textures.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_shared_spectral_textures(
        diffuse: SpectralTextureRef,
        specular: SpectralTextureRef,
        roughness: f64,
    ) -> Self {
        assert!(
            roughness.is_finite(),
            "layered diffuse/GGX material roughness must be finite"
        );
        Self {
            diffuse_color: LinearColor::new(0.5, 0.5, 0.5),
            specular_color: LinearColor::new(0.5, 0.5, 0.5),
            roughness: roughness.clamp(0.0, 1.0),
            diffuse: MaterialColorSource::spectral_texture(diffuse),
            specular: MaterialColorSource::spectral_texture(specular),
            normal_map: None,
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Uses conductor eta/k Fresnel for the GGX specular layer in polarized spectral transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_conductor_optical_constants(mut self, eta: f64, k: f64) -> Self {
        self.conductor = Some(ConductorOpticalConstants::new(eta, k));
        self.conductor_spectra = None;
        self
    }

    /// Uses measured conductor eta/k spectra for the GGX specular layer in spectral transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_conductor_optical_constants_spectra(
        mut self,
        eta: MeasuredSpectrum,
        k: MeasuredSpectrum,
    ) -> Self {
        let spectra = ConductorOpticalConstantsSpectra::new(eta, k);
        self.conductor = Some(spectra.at_wavelength_nm(550.0));
        self.conductor_spectra = Some(spectra);
        self
    }

    /// Adds a shared tangent-space normal map.
    #[must_use]
    pub fn with_shared_normal_map(mut self, normal_map: NormalMapRef) -> Self {
        self.normal_map = Some(normal_map);
        self
    }

    fn sampled_colors(&self, hit: &HitRecord<'_>) -> (LinearColor, LinearColor) {
        let sample = TextureSample::new(hit.u, hit.v, hit.point);
        (self.diffuse.sample(sample), self.specular.sample(sample))
    }

    #[cfg(feature = "spectral")]
    fn conductor_constants_at(
        &self,
        wavelength: SampledWavelength,
    ) -> Option<ConductorOpticalConstants> {
        self.conductor_spectra
            .as_ref()
            .map_or(self.conductor, |spectra| Some(spectra.at(wavelength)))
    }
}

impl Material for LayeredDiffuseGgx {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        _rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let outgoing = -ray_in.direction().normalized();
        let diffuse_pdf = CosinePdf::new(hit.shading_normal)?;
        let specular_pdf = GgxReflectionPdf::new(hit.shading_normal, outgoing, self.roughness)?;
        let (diffuse_color, specular_color) = self.sampled_colors(hit);
        let specular_weight = specular_sampling_weight(diffuse_color, specular_color);

        Some(ScatterRecord::Scattering {
            attenuation: reflectance_sum(diffuse_color, specular_color),
            pdf: MaterialPdf::DiffuseGgx {
                diffuse: diffuse_pdf,
                specular: specular_pdf,
                specular_weight,
            },
        })
    }

    fn scattering_pdf(&self, ray_in: &Ray, hit: &HitRecord<'_>, scattered: &Ray) -> f64 {
        let outgoing = -ray_in.direction().normalized();
        let incoming = scattered.direction().normalized();
        if incoming.dot(hit.normal) <= 0.0 || outgoing.dot(hit.normal) <= 0.0 {
            return 0.0;
        }

        let (diffuse_color, specular_color) = self.sampled_colors(hit);
        let specular_weight = specular_sampling_weight(diffuse_color, specular_color);
        let diffuse_pdf = hit
            .shading_normal
            .dot(scattered.direction().normalized())
            .max(0.0)
            / PI;
        let specular_pdf = ggx_reflection_scattering_weight(
            hit.shading_normal,
            outgoing,
            incoming,
            self.roughness,
            specular_luminance(specular_color),
        );
        (1.0 - specular_weight) * diffuse_pdf + specular_weight * specular_pdf
    }

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        let (diffuse_color, specular_color) = self.sampled_colors(hit);
        reflectance_sum(diffuse_color, specular_color)
    }

    fn normal_map_shading_normal(&self, hit: &HitRecord<'_>) -> Option<Vector> {
        self.normal_map
            .as_deref()
            .and_then(|normal_map| normal_map_world_normal(hit, normal_map))
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        hit: &HitRecord<'_>,
        _attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        let sample = TextureSample::new(hit.u, hit.v, hit.point);
        (self.diffuse.sample_spectrum(sample, wavelength)
            + self.specular.sample_spectrum(sample, wavelength))
        .clamp(0.0, 1.0)
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        let outgoing = -ray_in.direction().normalized();
        let incoming = scattered.direction().normalized();
        if incoming.dot(hit.normal) <= 0.0 || outgoing.dot(hit.normal) <= 0.0 {
            return MuellerMatrix::depolarizing_attenuation(0.0);
        }

        let sample = TextureSample::new(hit.u, hit.v, hit.point);
        let (diffuse_color, specular_color) = self.sampled_colors(hit);
        let diffuse_strength = self.diffuse.sample_spectrum(sample, wavelength).max(0.0);
        let specular_strength = self.specular.sample_spectrum(sample, wavelength).max(0.0);
        let specular_weight = specular_sampling_weight(diffuse_color, specular_color);
        let diffuse_pdf = hit.shading_normal.dot(incoming).max(0.0) / PI;
        let specular_pdf = ggx_reflection_scattering_weight(
            hit.shading_normal,
            outgoing,
            incoming,
            self.roughness,
            specular_strength,
        );
        let diffuse_contribution = (1.0 - specular_weight) * diffuse_pdf * diffuse_strength;
        let specular_contribution = specular_weight * specular_pdf * specular_strength;
        let total_contribution = diffuse_contribution + specular_contribution;
        if total_contribution <= f64::EPSILON {
            return MuellerMatrix::depolarizing_attenuation(0.0);
        }

        let diffuse_response = MuellerMatrix::depolarizer();
        if specular_contribution <= f64::EPSILON {
            return diffuse_response;
        }

        let specular_response = ggx_reflection_mueller(
            specular_strength,
            self.conductor_constants_at(wavelength),
            ray_in,
            hit,
            scattered,
            incoming_frame,
            outgoing_frame,
        );
        if diffuse_contribution <= f64::EPSILON {
            return specular_response;
        }

        let diffuse_fraction = diffuse_contribution / total_contribution;
        let specular_fraction = specular_contribution / total_contribution;
        diffuse_response * diffuse_fraction + specular_response * specular_fraction
    }
}

/// Diffuse light-emitting material.
#[derive(Clone, Debug)]
pub struct DiffuseLight {
    color: MaterialColorSource,
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
        Self {
            color: MaterialColorSource::constant(emit),
        }
    }

    /// Creates a light material only when the emitted color is finite.
    #[must_use]
    pub fn try_new(emit: LinearColor) -> Option<Self> {
        emit.is_finite().then(|| Self {
            color: MaterialColorSource::constant(emit),
        })
    }

    /// Creates a light material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self {
            color: MaterialColorSource::texture(texture),
        }
    }

    /// Creates a light material from a constant spectrum.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_spectrum(spectrum: Spectrum) -> Self {
        Self {
            color: MaterialColorSource::spectrum(spectrum),
        }
    }

    /// Creates a light material from measured spectral emission.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectrum(spectrum: MeasuredSpectrum) -> Self {
        Self {
            color: MaterialColorSource::measured_spectrum(spectrum),
        }
    }

    /// Creates a light material from a shared spectral texture.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_shared_spectral_texture(texture: SpectralTextureRef) -> Self {
        Self {
            color: MaterialColorSource::spectral_texture(texture),
        }
    }

    /// Creates a light material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl SurfaceTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Returns the texture sampled for emitted radiance.
    #[must_use]
    pub fn texture(&self) -> &dyn SurfaceTexture {
        self.color.surface_texture()
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
            self.color.sample(TextureSample::new(u, v, point))
        } else {
            LinearColor::default()
        }
    }

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        self.color
            .sample(TextureSample::new(hit.u, hit.v, hit.point))
    }

    #[cfg(feature = "spectral")]
    fn spectral_emitted(
        &self,
        _ray_in: &Ray,
        hit: &HitRecord<'_>,
        u: f64,
        v: f64,
        point: Point,
        wavelength: SampledWavelength,
    ) -> f64 {
        if hit.front_face {
            self.color
                .sample_spectrum(TextureSample::new(u, v, point), wavelength)
        } else {
            0.0
        }
    }
}

/// Isotropic phase-function material for constant-density volumes.
#[derive(Clone, Debug)]
pub struct Isotropic {
    color: MaterialColorSource,
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
        Self {
            color: MaterialColorSource::constant(albedo),
        }
    }

    /// Creates an isotropic material only when `albedo` is finite.
    #[must_use]
    pub fn try_new(albedo: LinearColor) -> Option<Self> {
        albedo.is_finite().then(|| Self {
            color: MaterialColorSource::constant(albedo),
        })
    }

    /// Creates an isotropic material from a shared texture.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef) -> Self {
        Self {
            color: MaterialColorSource::texture(texture),
        }
    }

    /// Creates an isotropic material from a constant spectral attenuation.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_spectrum(spectrum: Spectrum) -> Self {
        Self {
            color: MaterialColorSource::spectrum(spectrum),
        }
    }

    /// Creates an isotropic material from measured spectral attenuation.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectrum(spectrum: MeasuredSpectrum) -> Self {
        Self {
            color: MaterialColorSource::measured_spectrum(spectrum),
        }
    }

    /// Creates an isotropic material from a shared spectral texture.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_shared_spectral_texture(texture: SpectralTextureRef) -> Self {
        Self {
            color: MaterialColorSource::spectral_texture(texture),
        }
    }

    /// Creates an isotropic material from a texture object.
    #[must_use]
    pub fn from_texture(texture: impl SurfaceTexture + 'static) -> Self {
        Self::from_shared_texture(Arc::new(texture))
    }

    /// Returns the texture sampled for medium attenuation.
    #[must_use]
    pub fn texture(&self) -> &dyn SurfaceTexture {
        self.color.surface_texture()
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
                .color
                .sample(TextureSample::new(hit.u, hit.v, hit.point)),
            pdf: MaterialPdf::Sphere(SpherePdf),
        })
    }

    fn scattering_pdf(&self, _ray_in: &Ray, _hit: &HitRecord<'_>, _scattered: &Ray) -> f64 {
        1.0 / (4.0 * PI)
    }

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        self.color
            .sample(TextureSample::new(hit.u, hit.v, hit.point))
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        hit: &HitRecord<'_>,
        _attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        self.color
            .sample_spectrum(TextureSample::new(hit.u, hit.v, hit.point), wavelength)
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        _hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        _wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        scalar_phase_mueller(ray_in, scattered, incoming_frame, outgoing_frame)
    }
}

/// Henyey-Greenstein anisotropic phase-function material for volumes.
#[derive(Clone, Debug)]
pub struct HenyeyGreenstein {
    color: MaterialColorSource,
    g: f64,
}

impl HenyeyGreenstein {
    /// Creates an anisotropic phase-function material with constant attenuation.
    ///
    /// Positive `g` values favor forward scattering, negative values favor back scattering, and
    /// zero matches isotropic scattering.
    ///
    /// # Panics
    ///
    /// Panics if any albedo channel is not finite or if `g` is outside `(-1.0, 1.0)`.
    #[must_use]
    pub fn new(albedo: LinearColor, g: f64) -> Self {
        assert!(
            albedo.is_finite(),
            "Henyey-Greenstein albedo must be finite"
        );
        validate_henyey_greenstein_g(g);
        Self {
            color: MaterialColorSource::constant(albedo),
            g,
        }
    }

    /// Creates an anisotropic phase-function material only when its inputs are valid.
    #[must_use]
    pub fn try_new(albedo: LinearColor, g: f64) -> Option<Self> {
        (albedo.is_finite() && is_valid_henyey_greenstein_g(g)).then(|| Self {
            color: MaterialColorSource::constant(albedo),
            g,
        })
    }

    /// Creates an anisotropic phase-function material from a shared texture.
    ///
    /// # Panics
    ///
    /// Panics if `g` is outside `(-1.0, 1.0)`.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef, g: f64) -> Self {
        validate_henyey_greenstein_g(g);
        Self {
            color: MaterialColorSource::texture(texture),
            g,
        }
    }

    /// Creates an anisotropic phase-function material from a constant spectral attenuation.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_spectrum(spectrum: Spectrum, g: f64) -> Self {
        validate_henyey_greenstein_g(g);
        Self {
            color: MaterialColorSource::spectrum(spectrum),
            g,
        }
    }

    /// Creates an anisotropic phase-function material from measured spectral attenuation.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectrum(spectrum: MeasuredSpectrum, g: f64) -> Self {
        validate_henyey_greenstein_g(g);
        Self {
            color: MaterialColorSource::measured_spectrum(spectrum),
            g,
        }
    }

    /// Creates an anisotropic phase-function material from a shared spectral texture.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_shared_spectral_texture(texture: SpectralTextureRef, g: f64) -> Self {
        validate_henyey_greenstein_g(g);
        Self {
            color: MaterialColorSource::spectral_texture(texture),
            g,
        }
    }

    /// Creates an anisotropic phase-function material from a texture object.
    ///
    /// # Panics
    ///
    /// Panics if `g` is outside `(-1.0, 1.0)`.
    #[must_use]
    pub fn from_texture(texture: impl SurfaceTexture + 'static, g: f64) -> Self {
        Self::from_shared_texture(Arc::new(texture), g)
    }

    /// Returns the texture sampled for medium attenuation.
    #[must_use]
    pub fn texture(&self) -> &dyn SurfaceTexture {
        self.color.surface_texture()
    }

    /// Returns the anisotropy parameter.
    #[must_use]
    pub const fn anisotropy(&self) -> f64 {
        self.g
    }
}

impl Material for HenyeyGreenstein {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        _rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let pdf = HenyeyGreensteinPdf::new(*ray_in.direction(), self.g)?;
        Some(ScatterRecord::Scattering {
            attenuation: self
                .color
                .sample(TextureSample::new(hit.u, hit.v, hit.point)),
            pdf: MaterialPdf::HenyeyGreenstein(pdf),
        })
    }

    fn scattering_pdf(&self, ray_in: &Ray, _hit: &HitRecord<'_>, scattered: &Ray) -> f64 {
        let Some(pdf) = HenyeyGreensteinPdf::new(*ray_in.direction(), self.g) else {
            return 0.0;
        };
        pdf.value(*scattered.direction())
    }

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        self.color
            .sample(TextureSample::new(hit.u, hit.v, hit.point))
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        hit: &HitRecord<'_>,
        _attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        self.color
            .sample_spectrum(TextureSample::new(hit.u, hit.v, hit.point), wavelength)
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        _hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        _wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        scalar_phase_mueller(ray_in, scattered, incoming_frame, outgoing_frame)
    }
}

fn is_valid_henyey_greenstein_g(g: f64) -> bool {
    g.is_finite() && g.abs() < 1.0
}

fn validate_henyey_greenstein_g(g: f64) {
    assert!(
        is_valid_henyey_greenstein_g(g),
        "Henyey-Greenstein anisotropy must be finite and in (-1, 1)"
    );
}

/// GGX/Trowbridge-Reitz microfacet glossy reflection material.
#[derive(Clone)]
pub struct GgxMicrofacet {
    /// Representative normal-incidence specular reflectance for constant-color compatibility.
    pub specular_color: LinearColor,
    /// Perceptual roughness in `0.0..=1.0`.
    pub roughness: f64,
    color: MaterialColorSource,
    normal_map: Option<NormalMapRef>,
    #[cfg(feature = "spectral")]
    conductor: Option<ConductorOpticalConstants>,
    #[cfg(feature = "spectral")]
    conductor_spectra: Option<ConductorOpticalConstantsSpectra>,
}

impl fmt::Debug for GgxMicrofacet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("GgxMicrofacet");
        debug
            .field("specular_color", &self.specular_color)
            .field("roughness", &self.roughness)
            .field("color", &self.color)
            .field("normal_map", &self.normal_map.is_some());
        #[cfg(feature = "spectral")]
        debug.field("conductor", &self.conductor);
        #[cfg(feature = "spectral")]
        debug.field("conductor_spectra", &self.conductor_spectra.is_some());
        debug.finish()
    }
}

impl GgxMicrofacet {
    /// Creates a glossy microfacet material with constant specular color.
    ///
    /// `roughness` is clamped to `0.0..=1.0` after finite validation. A roughness of zero is kept
    /// numerically stable by the underlying GGX sampler.
    ///
    /// # Panics
    ///
    /// Panics if any specular color channel or `roughness` is not finite.
    #[must_use]
    pub fn new(specular_color: LinearColor, roughness: f64) -> Self {
        assert!(
            specular_color.is_finite() && roughness.is_finite(),
            "GGX microfacet material values must be finite"
        );
        Self {
            specular_color,
            roughness: roughness.clamp(0.0, 1.0),
            color: MaterialColorSource::constant(specular_color),
            normal_map: None,
            #[cfg(feature = "spectral")]
            conductor: None,
            #[cfg(feature = "spectral")]
            conductor_spectra: None,
        }
    }

    /// Creates a glossy microfacet material only when its inputs are finite.
    #[must_use]
    pub fn try_new(specular_color: LinearColor, roughness: f64) -> Option<Self> {
        (specular_color.is_finite() && roughness.is_finite())
            .then(|| Self::new(specular_color, roughness))
    }

    /// Creates a glossy microfacet material from a shared texture.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[must_use]
    pub fn from_shared_texture(texture: TextureRef, roughness: f64) -> Self {
        assert!(
            roughness.is_finite(),
            "GGX microfacet material roughness must be finite"
        );
        Self {
            specular_color: LinearColor::new(0.5, 0.5, 0.5),
            roughness: roughness.clamp(0.0, 1.0),
            color: MaterialColorSource::texture(texture),
            normal_map: None,
            #[cfg(feature = "spectral")]
            conductor: None,
            #[cfg(feature = "spectral")]
            conductor_spectra: None,
        }
    }

    /// Creates a glossy microfacet material from a constant spectral specular reflectance.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_spectrum(spectrum: Spectrum, roughness: f64) -> Self {
        assert!(
            roughness.is_finite(),
            "GGX microfacet material roughness must be finite"
        );
        Self {
            specular_color: spectrum.to_linear_rgb(),
            roughness: roughness.clamp(0.0, 1.0),
            color: MaterialColorSource::spectrum(spectrum),
            normal_map: None,
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Creates a glossy microfacet material from measured spectral specular reflectance.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectrum(spectrum: MeasuredSpectrum, roughness: f64) -> Self {
        assert!(
            roughness.is_finite(),
            "GGX microfacet material roughness must be finite"
        );
        Self {
            specular_color: spectrum.to_linear_rgb(),
            roughness: roughness.clamp(0.0, 1.0),
            color: MaterialColorSource::measured_spectrum(spectrum),
            normal_map: None,
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Creates a glossy microfacet material from a shared spectral texture.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_shared_spectral_texture(texture: SpectralTextureRef, roughness: f64) -> Self {
        assert!(
            roughness.is_finite(),
            "GGX microfacet material roughness must be finite"
        );
        Self {
            specular_color: LinearColor::new(0.5, 0.5, 0.5),
            roughness: roughness.clamp(0.0, 1.0),
            color: MaterialColorSource::spectral_texture(texture),
            normal_map: None,
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Creates a glossy microfacet material from a texture object.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[must_use]
    pub fn from_texture(texture: impl SurfaceTexture + 'static, roughness: f64) -> Self {
        Self::from_shared_texture(Arc::new(texture), roughness)
    }

    /// Creates a glossy microfacet material from display RGB bytes.
    #[must_use]
    pub fn from_rgb(color: Rgb, roughness: f64) -> Self {
        Self::new(rgb_to_linear_color(color), roughness)
    }

    /// Creates a glossy microfacet material from existing reflection constants.
    #[must_use]
    pub fn from_reflectance(reflectance: ReflectionConstants, roughness: f64) -> Self {
        Self::new(
            LinearColor::new(reflectance.red, reflectance.green, reflectance.blue),
            roughness,
        )
    }

    /// Creates a glossy microfacet material from the specular component of a Phong material.
    #[must_use]
    pub fn from_phong_specular(material: PhongMaterial, roughness: f64) -> Self {
        Self::from_reflectance(material.specular, roughness)
    }

    /// Returns the texture sampled for specular reflectance.
    #[must_use]
    pub fn texture(&self) -> &dyn SurfaceTexture {
        self.color.surface_texture()
    }

    /// Adds a shared tangent-space normal map.
    #[must_use]
    pub fn with_shared_normal_map(mut self, normal_map: NormalMapRef) -> Self {
        self.normal_map = Some(normal_map);
        self
    }

    /// Uses conductor eta/k Fresnel for polarized spectral transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_conductor_optical_constants(mut self, eta: f64, k: f64) -> Self {
        self.conductor = Some(ConductorOpticalConstants::new(eta, k));
        self.conductor_spectra = None;
        self
    }

    /// Uses measured conductor eta/k spectra for polarized spectral transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_conductor_optical_constants_spectra(
        mut self,
        eta: MeasuredSpectrum,
        k: MeasuredSpectrum,
    ) -> Self {
        let spectra = ConductorOpticalConstantsSpectra::new(eta, k);
        self.conductor = Some(spectra.at_wavelength_nm(550.0));
        self.conductor_spectra = Some(spectra);
        self
    }

    #[cfg(feature = "spectral")]
    fn conductor_constants_at(
        &self,
        wavelength: SampledWavelength,
    ) -> Option<ConductorOpticalConstants> {
        self.conductor_spectra
            .as_ref()
            .map_or(self.conductor, |spectra| Some(spectra.at(wavelength)))
    }
}

impl From<SurfaceMaterial> for GgxMicrofacet {
    fn from(material: SurfaceMaterial) -> Self {
        Self::new(material.specular_color, 0.35)
    }
}

impl Material for GgxMicrofacet {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        _rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        let outgoing = -ray_in.direction().normalized();
        let pdf = GgxReflectionPdf::new(hit.shading_normal, outgoing, self.roughness)?;
        Some(ScatterRecord::Scattering {
            attenuation: self
                .color
                .sample(TextureSample::new(hit.u, hit.v, hit.point)),
            pdf: MaterialPdf::GgxReflection(pdf),
        })
    }

    fn scattering_pdf(&self, ray_in: &Ray, hit: &HitRecord<'_>, scattered: &Ray) -> f64 {
        let outgoing = -ray_in.direction().normalized();
        let incoming = scattered.direction().normalized();
        if incoming.dot(hit.normal) <= 0.0 || outgoing.dot(hit.normal) <= 0.0 {
            return 0.0;
        }
        let specular_color = self
            .color
            .sample(TextureSample::new(hit.u, hit.v, hit.point));
        ggx_reflection_scattering_weight(
            hit.shading_normal,
            outgoing,
            incoming,
            self.roughness,
            specular_luminance(specular_color),
        )
    }

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        self.color
            .sample(TextureSample::new(hit.u, hit.v, hit.point))
    }

    fn normal_map_shading_normal(&self, hit: &HitRecord<'_>) -> Option<Vector> {
        self.normal_map
            .as_deref()
            .and_then(|normal_map| normal_map_world_normal(hit, normal_map))
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        hit: &HitRecord<'_>,
        _attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        self.color
            .sample_spectrum(TextureSample::new(hit.u, hit.v, hit.point), wavelength)
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        ggx_reflection_mueller(
            self.color
                .sample_spectrum(TextureSample::new(hit.u, hit.v, hit.point), wavelength),
            self.conductor_constants_at(wavelength),
            ray_in,
            hit,
            scattered,
            incoming_frame,
            outgoing_frame,
        )
    }
}

fn normal_map_world_normal(
    hit: &HitRecord<'_>,
    normal_map: &super::texture::NormalMap,
) -> Option<Vector> {
    let tangent = hit.tangent?;
    let bitangent = hit.bitangent?;
    let base_normal = hit.shading_normal;
    let tangent_normal =
        normal_map.sample_tangent_normal(TextureSample::new(hit.u, hit.v, hit.point));
    let normal = tangent * tangent_normal.x()
        + bitangent * tangent_normal.y()
        + base_normal * tangent_normal.z();
    if normal.length_squared() <= f64::EPSILON {
        return None;
    }
    let normal = normal.normalized();
    Some(if normal.dot(base_normal) < 0.0 {
        -normal
    } else {
        normal
    })
}

fn ggx_reflection_scattering_weight(
    normal: Vector,
    outgoing: Vector,
    incoming: Vector,
    roughness: f64,
    f0: f64,
) -> f64 {
    let normal = normal.normalized();
    let outgoing = outgoing.normalized();
    let incoming = incoming.normalized();
    let normal_dot_incoming = normal.dot(incoming);
    let normal_dot_outgoing = normal.dot(outgoing);
    if normal_dot_incoming <= 0.0 || normal_dot_outgoing <= 0.0 {
        return 0.0;
    }
    let half = (incoming + outgoing).normalized();
    if half.length_squared() <= f64::EPSILON {
        return 0.0;
    }
    let normal_dot_half = normal.dot(half);
    let outgoing_dot_half = outgoing.dot(half);
    if normal_dot_half <= 0.0 || outgoing_dot_half <= 0.0 {
        return 0.0;
    }
    let distribution = GgxReflectionPdf::normal_distribution(normal_dot_half, roughness);
    let geometry = GgxReflectionPdf::smith_masking_shadowing(
        normal_dot_incoming,
        normal_dot_outgoing,
        roughness,
    );
    let f0 = f0.clamp(0.0, 1.0);
    if f0 <= f64::EPSILON {
        return 0.0;
    }
    let fresnel = GgxReflectionPdf::schlick_fresnel(outgoing_dot_half, f0) / f0;
    distribution * geometry * fresnel / (4.0 * normal_dot_outgoing)
}

fn specular_luminance(color: LinearColor) -> f64 {
    (0.2126 * color.red + 0.7152 * color.green + 0.0722 * color.blue).clamp(0.0, 1.0)
}

fn diffuse_luminance(color: LinearColor) -> f64 {
    (0.2126 * color.red + 0.7152 * color.green + 0.0722 * color.blue).max(0.0)
}

fn specular_sampling_weight(diffuse: LinearColor, specular: LinearColor) -> f64 {
    let diffuse = diffuse_luminance(diffuse);
    let specular = specular_luminance(specular);
    let total = diffuse + specular;
    if total <= f64::EPSILON {
        0.5
    } else {
        (specular / total).clamp(0.05, 0.95)
    }
}

fn reflectance_sum(diffuse: LinearColor, specular: LinearColor) -> LinearColor {
    LinearColor::new(
        (diffuse.red + specular.red).clamp(0.0, 1.0),
        (diffuse.green + specular.green).clamp(0.0, 1.0),
        (diffuse.blue + specular.blue).clamp(0.0, 1.0),
    )
}

#[cfg(feature = "spectral")]
fn metal_reflection_mueller(
    conductor: Option<ConductorOpticalConstants>,
    ray_in: &Ray,
    hit: &HitRecord<'_>,
    scattered: &Ray,
    incoming_frame: PolarizationFrame,
    outgoing_frame: PolarizationFrame,
) -> MuellerMatrix {
    let interface_matrix = conductor.map_or_else(MuellerMatrix::perfect_mirror, |constants| {
        let cos_theta = (-ray_in.direction().normalized())
            .dot(hit.geometric_normal.normalized())
            .abs()
            .min(1.0);
        normalized_conductor_reflection_mueller(constants, cos_theta)
    });
    oriented_interface_mueller_with_normal(
        ray_in,
        scattered,
        incoming_frame,
        outgoing_frame,
        hit.geometric_normal,
        interface_matrix,
    )
}

#[cfg(feature = "spectral")]
fn ggx_reflection_mueller(
    f0: f64,
    conductor: Option<ConductorOpticalConstants>,
    ray_in: &Ray,
    hit: &HitRecord<'_>,
    scattered: &Ray,
    incoming_frame: PolarizationFrame,
    outgoing_frame: PolarizationFrame,
) -> MuellerMatrix {
    let outgoing = -ray_in.direction().normalized();
    let incoming = scattered.direction().normalized();
    if incoming.length_squared() <= f64::EPSILON
        || outgoing.length_squared() <= f64::EPSILON
        || incoming.dot(hit.normal) <= 0.0
        || outgoing.dot(hit.normal) <= 0.0
    {
        return MuellerMatrix::depolarizing_attenuation(0.0);
    }

    let half = incoming + outgoing;
    if half.length_squared() <= f64::EPSILON {
        return MuellerMatrix::depolarizing_attenuation(0.0);
    }
    let mut microfacet_normal = half.normalized();
    if microfacet_normal.dot(hit.shading_normal) < 0.0 {
        microfacet_normal = -microfacet_normal;
    }

    let cos_theta = outgoing.dot(microfacet_normal).abs().min(1.0);
    let fresnel = conductor.map_or_else(
        || normalized_dielectric_reflection_mueller(f0, cos_theta),
        |constants| normalized_conductor_reflection_mueller(constants, cos_theta),
    );
    oriented_interface_mueller_with_normal(
        ray_in,
        scattered,
        incoming_frame,
        outgoing_frame,
        microfacet_normal,
        fresnel,
    )
}

#[cfg(feature = "spectral")]
fn normalized_conductor_reflection_mueller(
    constants: ConductorOpticalConstants,
    cos_theta: f64,
) -> MuellerMatrix {
    let fresnel = conductor_fresnel(cos_theta, constants);
    if fresnel.reflectance <= f64::EPSILON {
        MuellerMatrix::perfect_mirror()
    } else {
        fresnel.reflection.divided_by(fresnel.reflectance)
    }
}

#[cfg(feature = "spectral")]
fn normalized_dielectric_reflection_mueller(f0: f64, cos_theta: f64) -> MuellerMatrix {
    let f0 = f0.clamp(0.0, 1.0);
    if f0 >= 1.0 - f64::EPSILON {
        return MuellerMatrix::perfect_mirror();
    }

    let sqrt_f0 = f0.sqrt();
    let eta = if sqrt_f0 <= f64::EPSILON {
        1.0
    } else {
        (1.0 + sqrt_f0) / (1.0 - sqrt_f0)
    };
    let fresnel = dielectric_fresnel(cos_theta, 1.0, eta);
    if fresnel.reflectance <= f64::EPSILON {
        MuellerMatrix::perfect_mirror()
    } else {
        fresnel.reflection.divided_by(fresnel.reflectance)
    }
}

#[cfg(feature = "spectral")]
fn scalar_phase_mueller(
    ray_in: &Ray,
    scattered: &Ray,
    incoming_frame: PolarizationFrame,
    outgoing_frame: PolarizationFrame,
) -> MuellerMatrix {
    let incoming_direction = *ray_in.direction();
    let outgoing_direction = *scattered.direction();
    let plane_normal = incoming_direction.cross(outgoing_direction);
    if plane_normal.length_squared() <= f64::EPSILON {
        return MuellerMatrix::frame_transform(incoming_frame, outgoing_frame);
    }
    let incoming_plane = PolarizationFrame::from_scattering_plane(incoming_direction, plane_normal);
    let outgoing_plane = PolarizationFrame::from_scattering_plane(outgoing_direction, plane_normal);
    MuellerMatrix::frame_transform(incoming_frame, incoming_plane)
        .followed_by(MuellerMatrix::scalar_attenuation(1.0))
        .followed_by(MuellerMatrix::frame_transform(
            outgoing_plane,
            outgoing_frame,
        ))
}

#[cfg(feature = "spectral")]
fn dielectric_mueller(
    refraction_index: f64,
    ray_in: &Ray,
    hit: &HitRecord<'_>,
    scattered: &Ray,
    incoming_frame: PolarizationFrame,
    outgoing_frame: PolarizationFrame,
) -> MuellerMatrix {
    let unit_direction = ray_in.direction().normalized();
    let cos_theta = (-unit_direction).dot(hit.normal).abs().min(1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let (eta_i, eta_t) = if hit.front_face {
        (1.0, refraction_index)
    } else {
        (refraction_index, 1.0)
    };
    let refraction_ratio = eta_i / eta_t;
    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    let sample_reflectance = if cannot_refract {
        1.0
    } else {
        Dielectric::reflectance(cos_theta, refraction_ratio)
    };
    let fresnel = dielectric_fresnel(cos_theta, eta_i, eta_t);
    let reflected = scattered.direction().normalized().dot(hit.normal) > 0.0;
    let event_matrix = if reflected {
        fresnel
            .reflection
            .divided_by(sample_reflectance.max(f64::EPSILON))
    } else {
        fresnel
            .transmission
            .divided_by((1.0 - sample_reflectance).max(f64::EPSILON))
    };
    oriented_interface_mueller_with_normal(
        ray_in,
        scattered,
        incoming_frame,
        outgoing_frame,
        hit.geometric_normal,
        event_matrix,
    )
}

#[cfg(feature = "spectral")]
fn oriented_interface_mueller_with_normal(
    ray_in: &Ray,
    scattered: &Ray,
    incoming_frame: PolarizationFrame,
    outgoing_frame: PolarizationFrame,
    interface_normal: Vector,
    interface_matrix: MuellerMatrix,
) -> MuellerMatrix {
    let incoming_plane =
        PolarizationFrame::from_scattering_plane(*ray_in.direction(), interface_normal);
    let outgoing_plane =
        PolarizationFrame::from_scattering_plane(*scattered.direction(), interface_normal);
    MuellerMatrix::frame_transform(incoming_frame, incoming_plane)
        .followed_by(interface_matrix)
        .followed_by(MuellerMatrix::frame_transform(
            outgoing_plane,
            outgoing_frame,
        ))
}

/// Reflective metal material.
#[derive(Clone, Debug, PartialEq)]
pub struct Metal {
    /// Reflected ray attenuation.
    pub albedo: LinearColor,
    /// Reflection fuzziness in `0.0..=1.0`.
    pub fuzz: f64,
    #[cfg(feature = "spectral")]
    albedo_spectrum: Option<MeasuredSpectrum>,
    /// Optional conductor optical constants for polarized spectral transport.
    #[cfg(feature = "spectral")]
    pub conductor: Option<ConductorOpticalConstants>,
    #[cfg(feature = "spectral")]
    conductor_spectra: Option<ConductorOpticalConstantsSpectra>,
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
            #[cfg(feature = "spectral")]
            albedo_spectrum: None,
            #[cfg(feature = "spectral")]
            conductor: None,
            #[cfg(feature = "spectral")]
            conductor_spectra: None,
        }
    }

    /// Creates a metal material only when `albedo` and `fuzz` are finite.
    #[must_use]
    pub fn try_new(albedo: LinearColor, fuzz: f64) -> Option<Self> {
        (albedo.is_finite() && fuzz.is_finite()).then(|| Self::new(albedo, fuzz))
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

    /// Creates a metal material from measured spectral reflectance.
    ///
    /// # Panics
    ///
    /// Panics if `fuzz` is not finite.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_measured_spectrum(spectrum: MeasuredSpectrum, fuzz: f64) -> Self {
        assert!(fuzz.is_finite(), "metal material values must be finite");
        Self {
            albedo: spectrum.to_linear_rgb(),
            fuzz: fuzz.clamp(0.0, 1.0),
            albedo_spectrum: Some(spectrum),
            conductor: None,
            conductor_spectra: None,
        }
    }

    /// Uses conductor eta/k Fresnel for polarized spectral transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_conductor_optical_constants(mut self, eta: f64, k: f64) -> Self {
        self.conductor = Some(ConductorOpticalConstants::new(eta, k));
        self.conductor_spectra = None;
        self
    }

    /// Uses measured conductor eta/k spectra for polarized spectral transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_conductor_optical_constants_spectra(
        mut self,
        eta: MeasuredSpectrum,
        k: MeasuredSpectrum,
    ) -> Self {
        let spectra = ConductorOpticalConstantsSpectra::new(eta, k);
        self.conductor = Some(spectra.at_wavelength_nm(550.0));
        self.conductor_spectra = Some(spectra);
        self
    }

    #[cfg(feature = "spectral")]
    fn conductor_constants_at(
        &self,
        wavelength: SampledWavelength,
    ) -> Option<ConductorOpticalConstants> {
        self.conductor_spectra
            .as_ref()
            .map_or(self.conductor, |spectra| Some(spectra.at(wavelength)))
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

    fn denoise_albedo(&self, _hit: &HitRecord<'_>) -> LinearColor {
        self.albedo
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        _hit: &HitRecord<'_>,
        attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        self.albedo_spectrum.as_ref().map_or_else(
            || Spectrum::from_linear_rgb(attenuation).sample(wavelength),
            |spectrum| spectrum.sample(wavelength),
        )
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        metal_reflection_mueller(
            self.conductor_constants_at(wavelength),
            ray_in,
            hit,
            scattered,
            incoming_frame,
            outgoing_frame,
        )
    }
}

/// Transparent dielectric material such as glass, water, or diamond.
#[derive(Clone, Debug, PartialEq)]
pub struct Dielectric {
    /// Refractive index in air/vacuum, or relative to the enclosing medium.
    pub refraction_index: RefractiveIndex,
    #[cfg(feature = "spectral")]
    eta_spectrum: Option<MeasuredSpectrum>,
}

impl Dielectric {
    /// Creates a dielectric material.
    #[must_use]
    pub const fn new(refraction_index: RefractiveIndex) -> Self {
        Self {
            refraction_index,
            #[cfg(feature = "spectral")]
            eta_spectrum: None,
        }
    }

    /// Creates a dielectric material from a raw refractive-index ratio.
    #[must_use]
    pub fn from_ratio(refraction_index: f64) -> Self {
        Self::new(RefractiveIndex::new(refraction_index))
    }

    /// Creates a dielectric material with measured refractive-index dispersion.
    ///
    /// The RGB path uses the spectrum sampled at 550 nm as its representative index.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn from_eta_spectrum(eta: MeasuredSpectrum) -> Self {
        let refraction_index = RefractiveIndex::new(eta.sample_wavelength_nm(550.0));
        Self {
            refraction_index,
            eta_spectrum: Some(eta),
        }
    }

    /// Uses measured refractive-index dispersion for sampled-wavelength transport.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_eta_spectrum(mut self, eta: MeasuredSpectrum) -> Self {
        self.eta_spectrum = Some(eta);
        self
    }

    /// Returns the refractive index used at `wavelength`.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn refraction_index_at(&self, wavelength: SampledWavelength) -> f64 {
        self.refraction_index_at_wavelength_nm(wavelength.wavelength_nm())
    }

    /// Returns the refractive index used at `wavelength_nm`.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn refraction_index_at_wavelength_nm(&self, wavelength_nm: f64) -> f64 {
        self.eta_spectrum
            .as_ref()
            .map_or(self.refraction_index.0, |eta| {
                eta.sample_wavelength_nm(wavelength_nm).max(f64::EPSILON)
            })
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

    /// Derives a GGX glossy ray material from this shared surface material.
    #[must_use]
    pub fn as_ggx_microfacet(&self, roughness: f64) -> GgxMicrofacet {
        GgxMicrofacet::new(self.specular_color, roughness)
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
        Some(Self::scatter_with_refraction_index(
            self.refraction_index.0,
            ray_in,
            hit,
            rng,
        ))
    }

    #[cfg(feature = "spectral")]
    fn scatter_spectral(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        Some(Self::scatter_with_refraction_index(
            self.refraction_index_at(wavelength),
            ray_in,
            hit,
            rng,
        ))
    }

    fn denoise_albedo(&self, _hit: &HitRecord<'_>) -> LinearColor {
        LinearColor::new(1.0, 1.0, 1.0)
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        dielectric_mueller(
            self.refraction_index_at(wavelength),
            ray_in,
            hit,
            scattered,
            incoming_frame,
            outgoing_frame,
        )
    }
}

impl Dielectric {
    fn scatter_with_refraction_index(
        refraction_index: f64,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        rng: &mut SampleRng,
    ) -> ScatterRecord {
        let attenuation = LinearColor::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit.front_face {
            1.0 / refraction_index
        } else {
            refraction_index
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

        ScatterRecord::Specular {
            ray: Ray::with_time(hit.point, direction, ray_in.time()),
            attenuation,
        }
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
    /// Henyey-Greenstein anisotropic phase-function material.
    HenyeyGreenstein(HenyeyGreenstein),
    /// GGX/Trowbridge-Reitz glossy reflection material.
    GgxMicrofacet(GgxMicrofacet),
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

    /// Creates a Lambertian material variant from a constant spectrum.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn spectral_lambertian(spectrum: Spectrum) -> Self {
        Self::Lambertian(Lambertian::from_spectrum(spectrum))
    }

    /// Creates a Lambertian material variant from measured spectral reflectance.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_lambertian(spectrum: MeasuredSpectrum) -> Self {
        Self::Lambertian(Lambertian::from_measured_spectrum(spectrum))
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

    /// Creates a diffuse light material variant from a constant spectrum.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn spectral_diffuse_light(spectrum: Spectrum) -> Self {
        Self::DiffuseLight(DiffuseLight::from_spectrum(spectrum))
    }

    /// Creates a diffuse light material variant from measured spectral emission.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_diffuse_light(spectrum: MeasuredSpectrum) -> Self {
        Self::DiffuseLight(DiffuseLight::from_measured_spectrum(spectrum))
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

    /// Creates an isotropic phase-function material variant from a constant spectrum.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn spectral_isotropic(spectrum: Spectrum) -> Self {
        Self::Isotropic(Isotropic::from_spectrum(spectrum))
    }

    /// Creates an isotropic phase-function material variant from measured spectral attenuation.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_isotropic(spectrum: MeasuredSpectrum) -> Self {
        Self::Isotropic(Isotropic::from_measured_spectrum(spectrum))
    }

    /// Creates a textured isotropic phase-function material variant.
    #[must_use]
    pub fn textured_isotropic(texture: impl SurfaceTexture + 'static) -> Self {
        Self::Isotropic(Isotropic::from_texture(texture))
    }

    /// Creates a Henyey-Greenstein phase-function material variant.
    #[must_use]
    pub fn henyey_greenstein(albedo: LinearColor, g: f64) -> Self {
        Self::HenyeyGreenstein(HenyeyGreenstein::new(albedo, g))
    }

    /// Creates a Henyey-Greenstein phase-function material variant from a constant spectrum.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn spectral_henyey_greenstein(spectrum: Spectrum, g: f64) -> Self {
        Self::HenyeyGreenstein(HenyeyGreenstein::from_spectrum(spectrum, g))
    }

    /// Creates a Henyey-Greenstein material variant from measured spectral attenuation.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_henyey_greenstein(spectrum: MeasuredSpectrum, g: f64) -> Self {
        Self::HenyeyGreenstein(HenyeyGreenstein::from_measured_spectrum(spectrum, g))
    }

    /// Creates a textured Henyey-Greenstein phase-function material variant.
    #[must_use]
    pub fn textured_henyey_greenstein(texture: impl SurfaceTexture + 'static, g: f64) -> Self {
        Self::HenyeyGreenstein(HenyeyGreenstein::from_texture(texture, g))
    }

    /// Creates a GGX/Trowbridge-Reitz glossy reflection material variant.
    #[must_use]
    pub fn ggx_microfacet(specular_color: LinearColor, roughness: f64) -> Self {
        Self::GgxMicrofacet(GgxMicrofacet::new(specular_color, roughness))
    }

    /// Creates a GGX/Trowbridge-Reitz glossy material variant from a constant spectrum.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn spectral_ggx_microfacet(spectrum: Spectrum, roughness: f64) -> Self {
        Self::GgxMicrofacet(GgxMicrofacet::from_spectrum(spectrum, roughness))
    }

    /// Creates a GGX/Trowbridge-Reitz glossy material variant from measured spectral reflectance.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_ggx_microfacet(spectrum: MeasuredSpectrum, roughness: f64) -> Self {
        Self::GgxMicrofacet(GgxMicrofacet::from_measured_spectrum(spectrum, roughness))
    }

    /// Creates a textured GGX/Trowbridge-Reitz glossy reflection material variant.
    #[must_use]
    pub fn textured_ggx_microfacet(texture: impl SurfaceTexture + 'static, roughness: f64) -> Self {
        Self::GgxMicrofacet(GgxMicrofacet::from_texture(texture, roughness))
    }

    /// Creates a metal material variant.
    #[must_use]
    pub fn metal(albedo: LinearColor, fuzz: f64) -> Self {
        Self::Metal(Metal::new(albedo, fuzz))
    }

    /// Creates a metal material variant from measured spectral reflectance.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_metal(spectrum: MeasuredSpectrum, fuzz: f64) -> Self {
        Self::Metal(Metal::from_measured_spectrum(spectrum, fuzz))
    }

    /// Creates a dielectric material variant.
    #[must_use]
    pub const fn dielectric(refraction_index: RefractiveIndex) -> Self {
        Self::Dielectric(Dielectric::new(refraction_index))
    }

    /// Creates a dielectric material variant from measured refractive-index dispersion.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn measured_dielectric(eta: MeasuredSpectrum) -> Self {
        Self::Dielectric(Dielectric::from_eta_spectrum(eta))
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

    /// Derives a GGX glossy ray material from shared surface material data.
    #[must_use]
    pub fn from_surface_ggx_microfacet(material: &SurfaceMaterial, roughness: f64) -> Self {
        Self::GgxMicrofacet(material.as_ggx_microfacet(roughness))
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
        matches!(self, Self::Isotropic(_) | Self::HenyeyGreenstein(_))
    }

    /// Returns true for materials that provide explicit PDF-sampled scattering.
    #[must_use]
    pub const fn has_pdf_scatter(&self) -> bool {
        matches!(
            self,
            Self::Lambertian(_)
                | Self::Isotropic(_)
                | Self::HenyeyGreenstein(_)
                | Self::GgxMicrofacet(_)
        )
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

impl From<HenyeyGreenstein> for RayMaterial {
    fn from(material: HenyeyGreenstein) -> Self {
        Self::HenyeyGreenstein(material)
    }
}

impl From<GgxMicrofacet> for RayMaterial {
    fn from(material: GgxMicrofacet) -> Self {
        Self::GgxMicrofacet(material)
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
            Self::HenyeyGreenstein(material) => material.emitted(ray_in, hit, u, v, point),
            Self::GgxMicrofacet(material) => material.emitted(ray_in, hit, u, v, point),
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
            Self::HenyeyGreenstein(material) => material.scatter(ray_in, hit, rng),
            Self::GgxMicrofacet(material) => material.scatter(ray_in, hit, rng),
            Self::Metal(material) => material.scatter(ray_in, hit, rng),
            Self::Dielectric(material) => material.scatter(ray_in, hit, rng),
        }
    }

    #[cfg(feature = "spectral")]
    fn scatter_spectral(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        wavelength: SampledWavelength,
        rng: &mut SampleRng,
    ) -> Option<ScatterRecord> {
        match self {
            Self::Lambertian(material) => material.scatter_spectral(ray_in, hit, wavelength, rng),
            Self::DiffuseLight(material) => material.scatter_spectral(ray_in, hit, wavelength, rng),
            Self::Isotropic(material) => material.scatter_spectral(ray_in, hit, wavelength, rng),
            Self::HenyeyGreenstein(material) => {
                material.scatter_spectral(ray_in, hit, wavelength, rng)
            }
            Self::GgxMicrofacet(material) => {
                material.scatter_spectral(ray_in, hit, wavelength, rng)
            }
            Self::Metal(material) => material.scatter_spectral(ray_in, hit, wavelength, rng),
            Self::Dielectric(material) => material.scatter_spectral(ray_in, hit, wavelength, rng),
        }
    }

    fn scattering_pdf(&self, ray_in: &Ray, hit: &HitRecord<'_>, scattered: &Ray) -> f64 {
        match self {
            Self::Lambertian(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::DiffuseLight(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::Isotropic(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::HenyeyGreenstein(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::GgxMicrofacet(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::Metal(material) => material.scattering_pdf(ray_in, hit, scattered),
            Self::Dielectric(material) => material.scattering_pdf(ray_in, hit, scattered),
        }
    }

    fn denoise_albedo(&self, hit: &HitRecord<'_>) -> LinearColor {
        match self {
            Self::Lambertian(material) => material.denoise_albedo(hit),
            Self::DiffuseLight(material) => material.denoise_albedo(hit),
            Self::Isotropic(material) => material.denoise_albedo(hit),
            Self::HenyeyGreenstein(material) => material.denoise_albedo(hit),
            Self::GgxMicrofacet(material) => material.denoise_albedo(hit),
            Self::Metal(material) => material.denoise_albedo(hit),
            Self::Dielectric(material) => material.denoise_albedo(hit),
        }
    }

    fn normal_map_shading_normal(&self, hit: &HitRecord<'_>) -> Option<Vector> {
        match self {
            Self::Lambertian(material) => material.normal_map_shading_normal(hit),
            Self::DiffuseLight(material) => material.normal_map_shading_normal(hit),
            Self::Isotropic(material) => material.normal_map_shading_normal(hit),
            Self::HenyeyGreenstein(material) => material.normal_map_shading_normal(hit),
            Self::GgxMicrofacet(material) => material.normal_map_shading_normal(hit),
            Self::Metal(material) => material.normal_map_shading_normal(hit),
            Self::Dielectric(material) => material.normal_map_shading_normal(hit),
        }
    }

    #[cfg(feature = "spectral")]
    fn spectral_attenuation(
        &self,
        hit: &HitRecord<'_>,
        attenuation: LinearColor,
        wavelength: SampledWavelength,
    ) -> f64 {
        match self {
            Self::Lambertian(material) => {
                material.spectral_attenuation(hit, attenuation, wavelength)
            }
            Self::DiffuseLight(material) => {
                material.spectral_attenuation(hit, attenuation, wavelength)
            }
            Self::Isotropic(material) => {
                material.spectral_attenuation(hit, attenuation, wavelength)
            }
            Self::HenyeyGreenstein(material) => {
                material.spectral_attenuation(hit, attenuation, wavelength)
            }
            Self::GgxMicrofacet(material) => {
                material.spectral_attenuation(hit, attenuation, wavelength)
            }
            Self::Metal(material) => material.spectral_attenuation(hit, attenuation, wavelength),
            Self::Dielectric(material) => {
                material.spectral_attenuation(hit, attenuation, wavelength)
            }
        }
    }

    #[cfg(feature = "spectral")]
    fn spectral_emitted(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        u: f64,
        v: f64,
        point: Point,
        wavelength: SampledWavelength,
    ) -> f64 {
        match self {
            Self::Lambertian(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
            Self::DiffuseLight(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
            Self::Isotropic(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
            Self::HenyeyGreenstein(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
            Self::GgxMicrofacet(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
            Self::Metal(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
            Self::Dielectric(material) => {
                material.spectral_emitted(ray_in, hit, u, v, point, wavelength)
            }
        }
    }

    #[cfg(feature = "spectral")]
    fn polarized_scatter_mueller(
        &self,
        ray_in: &Ray,
        hit: &HitRecord<'_>,
        scattered: &Ray,
        incoming_frame: PolarizationFrame,
        outgoing_frame: PolarizationFrame,
        wavelength: SampledWavelength,
    ) -> MuellerMatrix {
        match self {
            Self::Lambertian(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
            Self::DiffuseLight(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
            Self::Isotropic(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
            Self::HenyeyGreenstein(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
            Self::GgxMicrofacet(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
            Self::Metal(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
            Self::Dielectric(material) => material.polarized_scatter_mueller(
                ray_in,
                hit,
                scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            ),
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
    use crate::gmath::vector::Vector;
    #[cfg(feature = "spectral")]
    use crate::graphics::raytracing::spectrum::StokesVector;

    #[cfg(feature = "spectral")]
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn checked_material_constructors_reject_non_finite_values() {
        let finite = LinearColor::new(0.2, 0.3, 0.4);
        let invalid = LinearColor::new(0.2, f64::NAN, 0.4);

        assert!(Lambertian::try_new(finite).is_some());
        assert!(DiffuseLight::try_new(finite).is_some());
        assert!(Isotropic::try_new(finite).is_some());
        assert!(HenyeyGreenstein::try_new(finite, 0.5).is_some());
        assert!(GgxMicrofacet::try_new(finite, 0.35).is_some());
        assert!(Lambertian::try_new(invalid).is_none());
        assert!(DiffuseLight::try_new(invalid).is_none());
        assert!(Isotropic::try_new(invalid).is_none());
        assert!(HenyeyGreenstein::try_new(invalid, 0.5).is_none());
        assert!(GgxMicrofacet::try_new(invalid, 0.35).is_none());
        assert!(GgxMicrofacet::try_new(finite, f64::NAN).is_none());
        assert!(HenyeyGreenstein::try_new(finite, 1.0).is_none());
        assert!(Metal::try_new(finite, f64::NAN).is_none());
        assert!(Metal::try_new(invalid, 0.2).is_none());
    }

    #[test]
    fn constant_material_texture_accessors_sample_expected_color() {
        let color = LinearColor::new(0.2, 0.3, 0.4);
        let sample = TextureSample::new(0.25, 0.75, Point::new(1.0, 2.0, 3.0));

        assert_eq!(
            Lambertian::new(color).texture().sample_linear(sample),
            color
        );
        assert_eq!(
            DiffuseLight::new(color).texture().sample_linear(sample),
            color
        );
        assert_eq!(Isotropic::new(color).texture().sample_linear(sample), color);
        assert_eq!(
            HenyeyGreenstein::new(color, 0.4)
                .texture()
                .sample_linear(sample),
            color
        );
        assert_eq!(
            GgxMicrofacet::new(color, 0.45)
                .texture()
                .sample_linear(sample),
            color
        );
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
    #[should_panic(expected = "Henyey-Greenstein anisotropy must be finite and in (-1, 1)")]
    fn henyey_greenstein_constructor_rejects_invalid_anisotropy() {
        let _ = HenyeyGreenstein::new(LinearColor::new(0.2, 0.3, 0.4), 1.0);
    }

    #[test]
    #[should_panic(expected = "GGX microfacet material values must be finite")]
    fn ggx_microfacet_constructor_rejects_non_finite_roughness() {
        let _ = GgxMicrofacet::new(LinearColor::new(0.2, 0.3, 0.4), f64::NAN);
    }

    #[test]
    fn henyey_greenstein_scatter_uses_phase_pdf_and_texture_attenuation() {
        let material = HenyeyGreenstein::new(LinearColor::new(0.25, 0.5, 0.75), 0.65);
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.5);
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, -1.0),
            normal: Vector::new(1.0, 0.0, 0.0),
            geometric_normal: Vector::new(1.0, 0.0, 0.0),
            shading_normal: Vector::new(1.0, 0.0, 0.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(43);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("Henyey-Greenstein medium should scatter");

        assert_eq!(
            scatter,
            ScatterRecord::Scattering {
                attenuation: LinearColor::new(0.25, 0.5, 0.75),
                pdf: MaterialPdf::HenyeyGreenstein(
                    HenyeyGreensteinPdf::new(Vector::new(0.0, 0.0, -1.0), 0.65)
                        .expect("ray should create pdf")
                )
            }
        );
        assert!(
            material.scattering_pdf(
                &ray,
                &hit,
                &Ray::new(hit.point, Vector::new(0.0, 0.0, -1.0))
            ) > material.scattering_pdf(
                &ray,
                &hit,
                &Ray::new(hit.point, Vector::new(0.0, 0.0, 1.0))
            )
        );
    }

    #[test]
    fn ggx_microfacet_scatter_uses_reflection_pdf_and_brdf_weight() {
        let material = GgxMicrofacet::new(LinearColor::new(0.8, 0.7, 0.6), 0.5);
        let ray = Ray::with_time(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0), 0.25);
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(47);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("GGX should scatter above the surface");

        match scatter {
            ScatterRecord::Scattering { attenuation, pdf } => {
                assert_eq!(attenuation, LinearColor::new(0.8, 0.7, 0.6));
                assert!(matches!(pdf, MaterialPdf::GgxReflection(_)));
                assert!(pdf.value(Vector::new(0.0, 0.0, 1.0)) > 0.0);
            }
            ScatterRecord::Specular { .. } => panic!("GGX should use explicit PDF sampling"),
        }

        let normal_reflection = Ray::new(hit.point, Vector::new(0.0, 0.0, 1.0));
        let grazing_reflection = Ray::new(hit.point, Vector::new(0.8, 0.0, 0.6).normalized());
        assert!(material.scattering_pdf(&ray, &hit, &normal_reflection) > 0.0);
        assert!(
            material.scattering_pdf(&ray, &hit, &normal_reflection)
                > material.scattering_pdf(&ray, &hit, &grazing_reflection)
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn spectral_material_sources_override_rgb_scatter_fallback() {
        let material = Lambertian::from_spectrum(Spectrum::constant(0.25));
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let wavelength = SampledWavelength::new(520.0, 1.0 / 320.0);

        assert_close(
            material.spectral_attenuation(&hit, LinearColor::new(0.9, 0.1, 0.1), wavelength),
            0.25,
        );

        let light = DiffuseLight::from_spectrum(Spectrum::constant(3.0));
        let light_hit = HitRecord {
            material: &light,
            ..hit
        };
        assert_close(
            light.spectral_emitted(
                &ray,
                &light_hit,
                light_hit.u,
                light_hit.v,
                light_hit.point,
                wavelength,
            ),
            3.0,
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn measured_spectral_material_inputs_override_rgb_fallback() {
        let reflectance = MeasuredSpectrum::new(vec![(500.0, 0.2), (600.0, 0.8)]);
        let emission = MeasuredSpectrum::new(vec![(500.0, 2.0), (600.0, 4.0)]);
        let material = Lambertian::from_measured_spectrum(reflectance);
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let wavelength = SampledWavelength::new(550.0, 1.0 / 320.0);

        assert_close(
            material.spectral_attenuation(&hit, LinearColor::new(0.9, 0.1, 0.1), wavelength),
            0.5,
        );

        let light = DiffuseLight::from_measured_spectrum(emission);
        let light_hit = HitRecord {
            material: &light,
            ..hit
        };
        assert_close(
            light.spectral_emitted(
                &ray,
                &light_hit,
                light_hit.u,
                light_hit.v,
                light_hit.point,
                wavelength,
            ),
            3.0,
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn rgb_spectral_fallback_still_uses_scatter_attenuation() {
        let material = Metal::new(LinearColor::new(0.9, 0.1, 0.1), 0.0);
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let wavelength = SampledWavelength::new(620.0, 1.0 / 320.0);
        let attenuation = LinearColor::new(0.9, 0.1, 0.1);

        assert_close(
            material.spectral_attenuation(&hit, attenuation, wavelength),
            Spectrum::from_linear_rgb(attenuation).sample(wavelength),
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn dielectric_eta_spectrum_varies_with_wavelength() {
        let eta = MeasuredSpectrum::new(vec![(450.0, 1.33), (650.0, 1.55)]);
        let material = Dielectric::from_eta_spectrum(eta);

        assert_close(material.refraction_index_at_wavelength_nm(450.0), 1.33);
        assert_close(material.refraction_index_at_wavelength_nm(550.0), 1.44);
        assert_close(material.refraction_index_at_wavelength_nm(650.0), 1.55);
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn measured_conductor_fresnel_varies_by_wavelength() {
        let eta = MeasuredSpectrum::new(vec![(450.0, 0.2), (650.0, 1.5)]);
        let k = MeasuredSpectrum::new(vec![(450.0, 3.0), (650.0, 1.0)]);
        let material = GgxMicrofacet::from_spectrum(Spectrum::constant(0.9), 0.2)
            .with_conductor_optical_constants_spectra(eta, k);
        let blue = material
            .conductor_constants_at(SampledWavelength::new(450.0, 1.0 / 320.0))
            .expect("measured conductor should resolve blue constants");
        let red = material
            .conductor_constants_at(SampledWavelength::new(650.0, 1.0 / 320.0))
            .expect("measured conductor should resolve red constants");

        let blue_fresnel = conductor_fresnel(0.35, blue);
        let red_fresnel = conductor_fresnel(0.35, red);

        assert!((blue.eta - red.eta).abs() > 0.1);
        assert!((blue.k - red.k).abs() > 0.1);
        assert!((blue_fresnel.reflectance - red_fresnel.reflectance).abs() > 0.01);
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn lambertian_diffuse_depolarizes_in_polarized_transport() {
        let material = Lambertian::from_spectrum(Spectrum::constant(0.8));
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let scattered = Ray::new(hit.point, Vector::new(0.2, 0.0, 0.98).normalized());
        let input = StokesVector::new(1.0, 0.6, -0.2, 0.1);
        let output = material
            .polarized_scatter_mueller(
                &ray,
                &hit,
                &scattered,
                PolarizationFrame::from_direction(*ray.direction()),
                PolarizationFrame::from_direction(*scattered.direction()),
                SampledWavelength::new(550.0, 1.0 / 320.0),
            )
            .apply(input);

        assert_eq!(output, StokesVector::unpolarized(input.i));
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn rough_ggx_polarized_mueller_uses_microfacet_fresnel() {
        let material = GgxMicrofacet::from_spectrum(Spectrum::constant(0.04), 0.95);
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let scattered = Ray::new(hit.point, Vector::new(0.99, 0.0, 0.141).normalized());
        let incoming_frame = PolarizationFrame::from_direction(*ray.direction());
        let outgoing_frame = PolarizationFrame::from_direction(*scattered.direction());

        let mueller = material.polarized_scatter_mueller(
            &ray,
            &hit,
            &scattered,
            incoming_frame,
            outgoing_frame,
            SampledWavelength::new(550.0, 1.0 / 320.0),
        );
        let output = mueller.apply(StokesVector::unpolarized(1.0));

        assert!((output.i - 1.0).abs() < 1.0e-10, "{output:?}");
        assert!(output.degree_of_polarization() > 1.0e-4, "{output:?}");
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn layered_diffuse_ggx_diffuse_lobe_depolarizes_in_polarized_transport() {
        let material =
            LayeredDiffuseGgx::from_spectra(Spectrum::constant(0.8), Spectrum::constant(0.0), 0.5);
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let scattered = Ray::new(hit.point, Vector::new(0.2, 0.0, 0.98).normalized());
        let incoming_frame = PolarizationFrame::from_direction(*ray.direction());
        let outgoing_frame = PolarizationFrame::from_direction(*scattered.direction());

        let mueller = material.polarized_scatter_mueller(
            &ray,
            &hit,
            &scattered,
            incoming_frame,
            outgoing_frame,
            SampledWavelength::new(550.0, 1.0 / 320.0),
        );
        let output = mueller.apply(StokesVector::new(1.0, 0.6, -0.2, 0.1));

        assert_eq!(output, StokesVector::unpolarized(1.0));
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn layered_diffuse_ggx_zero_diffuse_matches_specular_mueller() {
        let layered =
            LayeredDiffuseGgx::from_spectra(Spectrum::constant(0.0), Spectrum::constant(0.9), 0.2)
                .with_conductor_optical_constants(0.2, 3.0);
        let specular = GgxMicrofacet::from_spectrum(Spectrum::constant(0.9), 0.2)
            .with_conductor_optical_constants(0.2, 3.0);
        let ray = Ray::new(
            Point::new(0.0, 0.0, 1.0),
            Vector::new(0.6, 0.0, -0.8).normalized(),
        );
        let layered_hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &layered,
        };
        let specular_hit = HitRecord {
            material: &specular,
            ..layered_hit
        };
        let scattered = Ray::new(layered_hit.point, Vector::new(0.6, 0.0, 0.8).normalized());
        let incoming_frame = PolarizationFrame::from_direction(*ray.direction());
        let outgoing_frame = PolarizationFrame::from_direction(*scattered.direction());
        let wavelength = SampledWavelength::new(550.0, 1.0 / 320.0);
        let input = StokesVector::unpolarized(1.0);

        let layered_output = layered
            .polarized_scatter_mueller(
                &ray,
                &layered_hit,
                &scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            )
            .apply(input);
        let specular_output = specular
            .polarized_scatter_mueller(
                &ray,
                &specular_hit,
                &scattered,
                incoming_frame,
                outgoing_frame,
                wavelength,
            )
            .apply(input);

        assert_close(layered_output.i, specular_output.i);
        assert_close(layered_output.q, specular_output.q);
        assert_close(layered_output.u, specular_output.u);
        assert_close(layered_output.v, specular_output.v);
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn scalar_volume_phase_preserves_polarization_magnitude() {
        let material = HenyeyGreenstein::from_spectrum(Spectrum::constant(0.6), 0.35);
        let isotropic = Isotropic::from_spectrum(Spectrum::constant(0.6));
        let ray = Ray::new(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let scattered = Ray::new(hit.point, Vector::new(0.8, 0.0, 0.6).normalized());
        let incoming_frame = PolarizationFrame::from_direction(*ray.direction());
        let outgoing_frame = PolarizationFrame::from_direction(*scattered.direction());
        let input = StokesVector::new(1.0, 0.4, 0.2, 0.1);

        let mueller = material.polarized_scatter_mueller(
            &ray,
            &hit,
            &scattered,
            incoming_frame,
            outgoing_frame,
            SampledWavelength::new(550.0, 1.0 / 320.0),
        );
        let output = mueller.apply(input);

        assert_close(output.i, input.i);
        assert_close(
            output.polarization_magnitude(),
            input.polarization_magnitude(),
        );

        let isotropic_hit = HitRecord {
            material: &isotropic,
            ..hit
        };
        let isotropic_output = isotropic
            .polarized_scatter_mueller(
                &ray,
                &isotropic_hit,
                &scattered,
                incoming_frame,
                outgoing_frame,
                SampledWavelength::new(550.0, 1.0 / 320.0),
            )
            .apply(input);

        assert_close(isotropic_output.i, input.i);
        assert_close(
            isotropic_output.polarization_magnitude(),
            input.polarization_magnitude(),
        );
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn ggx_conductor_eta_k_polarizes_oblique_reflection() {
        let material = GgxMicrofacet::from_spectrum(Spectrum::constant(0.9), 0.2)
            .with_conductor_optical_constants(0.2, 3.0);
        let ray = Ray::new(
            Point::new(0.0, 0.0, 1.0),
            Vector::new(0.6, 0.0, -0.8).normalized(),
        );
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let scattered = Ray::new(hit.point, Vector::new(0.6, 0.0, 0.8).normalized());
        let incoming_frame = PolarizationFrame::from_direction(*ray.direction());
        let outgoing_frame = PolarizationFrame::from_direction(*scattered.direction());

        let mueller = material.polarized_scatter_mueller(
            &ray,
            &hit,
            &scattered,
            incoming_frame,
            outgoing_frame,
            SampledWavelength::new(550.0, 1.0 / 320.0),
        );
        let output = mueller.apply(StokesVector::unpolarized(1.0));

        assert!((output.i - 1.0).abs() < 1.0e-10, "{output:?}");
        assert!(output.degree_of_polarization() > 0.01, "{output:?}");
    }

    #[cfg(feature = "spectral")]
    #[test]
    fn metal_conductor_eta_k_polarizes_oblique_reflection() {
        let material = Metal::new(LinearColor::new(0.9, 0.8, 0.7), 0.0)
            .with_conductor_optical_constants(0.2, 3.0);
        let ray = Ray::new(
            Point::new(0.0, 0.0, 1.0),
            Vector::new(0.6, 0.0, -0.8).normalized(),
        );
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let scattered = Ray::new(hit.point, Vector::new(0.6, 0.0, 0.8).normalized());
        let incoming_frame = PolarizationFrame::from_direction(*ray.direction());
        let outgoing_frame = PolarizationFrame::from_direction(*scattered.direction());

        let mueller = material.polarized_scatter_mueller(
            &ray,
            &hit,
            &scattered,
            incoming_frame,
            outgoing_frame,
            SampledWavelength::new(550.0, 1.0 / 320.0),
        );
        let output = mueller.apply(StokesVector::unpolarized(1.0));

        assert!((output.i - 1.0).abs() < 1.0e-10, "{output:?}");
        assert!(output.degree_of_polarization() > 0.01, "{output:?}");
    }

    #[test]
    fn layered_diffuse_ggx_scatter_uses_mixed_pdf_and_albedo() {
        let material = LayeredDiffuseGgx::new(
            LinearColor::new(0.4, 0.2, 0.1),
            LinearColor::new(0.2, 0.2, 0.2),
            0.45,
        );
        let ray = Ray::with_time(Point::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, -1.0), 0.25);
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, 0.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            tangent: None,
            bitangent: None,
            tangent_handedness: 1.0,
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(53);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("layered material should scatter");

        let ScatterRecord::Scattering { attenuation, pdf } = scatter else {
            panic!("layered material should produce a sampled PDF");
        };
        assert!((attenuation.red - 0.6).abs() < 1e-12);
        assert!((attenuation.green - 0.4).abs() < 1e-12);
        assert!((attenuation.blue - 0.3).abs() < 1e-12);
        assert!(matches!(pdf, MaterialPdf::DiffuseGgx { .. }));
        assert_eq!(material.denoise_albedo(&hit), attenuation);
        assert!(material.scattering_pdf(&ray, &hit, &Ray::new(hit.point, hit.normal)) > 0.0);
    }

    #[test]
    fn ray_material_flags_describe_scattering_behavior() {
        let lambertian = RayMaterial::lambertian(LinearColor::new(0.2, 0.3, 0.4));
        let light = RayMaterial::diffuse_light(LinearColor::new(3.0, 2.0, 1.0));
        let isotropic = RayMaterial::isotropic(LinearColor::new(0.5, 0.5, 0.5));
        let henyey_greenstein =
            RayMaterial::henyey_greenstein(LinearColor::new(0.5, 0.5, 0.5), 0.4);
        let ggx = RayMaterial::ggx_microfacet(LinearColor::new(0.7, 0.6, 0.5), 0.35);
        let metal = RayMaterial::metal(LinearColor::new(0.7, 0.6, 0.5), 0.1);
        let dielectric = RayMaterial::dielectric(RefractiveIndex::GLASS);

        assert!(lambertian.has_pdf_scatter());
        assert!(!lambertian.is_delta());
        assert!(!lambertian.is_emissive());
        assert!(light.is_emissive());
        assert!(!light.has_pdf_scatter());
        assert!(isotropic.has_pdf_scatter());
        assert!(isotropic.is_volume_phase());
        assert!(henyey_greenstein.has_pdf_scatter());
        assert!(henyey_greenstein.is_volume_phase());
        assert!(ggx.has_pdf_scatter());
        assert!(!ggx.is_delta());
        assert!(!ggx.is_volume_phase());
        assert!(metal.is_delta());
        assert!(dielectric.is_delta());
        assert!(!metal.has_pdf_scatter());
        assert!(!dielectric.has_pdf_scatter());
    }
}

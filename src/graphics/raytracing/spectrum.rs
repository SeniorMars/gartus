//! Feature-gated spectral rendering prototype.
//!
//! This module intentionally starts small: it provides a wavelength sample type plus compact
//! spectrum adapters that let RGB and measured spectral material sources be evaluated along one
//! sampled wavelength. Use [`MeasuredSpectrum`] for measured reflectance, emission, eta, and
//! extinction-coefficient data; [`Spectrum::Rgb`] remains a compatibility adapter for RGB assets.

use super::LinearColor;
use crate::{
    gmath::random::SampleRng,
    gmath::vector::Vector,
    graphics::{
        colors::LinearRgb,
        display::{Canvas, HdrImage, ToneMap},
    },
};
use num::complex::Complex64;
use std::{
    fs,
    io::{self, ErrorKind},
    ops::{Add, AddAssign, Div, Mul, MulAssign},
    path::Path,
    sync::OnceLock,
};

/// Shortest sampled visible wavelength, in nanometers.
pub const VISIBLE_WAVELENGTH_MIN_NM: f64 = 380.0;
/// Longest sampled visible wavelength, in nanometers.
pub const VISIBLE_WAVELENGTH_MAX_NM: f64 = 700.0;

const VISIBLE_WAVELENGTH_RANGE_NM: f64 = VISIBLE_WAVELENGTH_MAX_NM - VISIBLE_WAVELENGTH_MIN_NM;
const RESPONSE_EPSILON: f64 = 1.0e-12;
const EQUAL_ENERGY_NORMALIZATION_SAMPLES: u32 = 256;
static EQUAL_ENERGY_LINEAR_RGB_MEAN: OnceLock<LinearRgb> = OnceLock::new();

/// Spectral transport variants supported by the renderer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpectralTransportMode {
    /// One scalar radiance value is transported per sampled wavelength.
    Unpolarized,
    /// Full Stokes vectors are transported through Mueller matrices.
    Polarized,
}

/// Stokes polarization state for polarized spectral transport.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StokesVector {
    /// Total intensity.
    pub i: f64,
    /// Linear polarization along horizontal/vertical axes.
    pub q: f64,
    /// Linear polarization along diagonal axes.
    pub u: f64,
    /// Circular polarization.
    pub v: f64,
}

impl StokesVector {
    /// Creates a Stokes vector from raw components.
    #[must_use]
    pub const fn new(i: f64, q: f64, u: f64, v: f64) -> Self {
        Self { i, q, u, v }
    }

    /// Creates an unpolarized Stokes vector with intensity `i`.
    #[must_use]
    pub const fn unpolarized(i: f64) -> Self {
        Self {
            i,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        }
    }

    /// Returns true when all Stokes components are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.i.is_finite() && self.q.is_finite() && self.u.is_finite() && self.v.is_finite()
    }

    /// Returns the total polarization magnitude.
    #[must_use]
    pub fn polarization_magnitude(self) -> f64 {
        self.q.hypot(self.u).hypot(self.v)
    }

    /// Returns the degree of polarization, clamped to `0..=1`.
    #[must_use]
    pub fn degree_of_polarization(self) -> f64 {
        if !self.is_finite() || self.i.abs() <= f64::EPSILON {
            0.0
        } else {
            (self.polarization_magnitude() / self.i.abs()).clamp(0.0, 1.0)
        }
    }

    /// Returns true when this is finite, non-negative, and not over-polarized.
    #[must_use]
    pub fn is_physical(self) -> bool {
        self.is_finite() && self.i >= 0.0 && self.polarization_magnitude() <= self.i
    }
}

impl Add for StokesVector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.i + rhs.i,
            self.q + rhs.q,
            self.u + rhs.u,
            self.v + rhs.v,
        )
    }
}

impl AddAssign for StokesVector {
    fn add_assign(&mut self, rhs: Self) {
        self.i += rhs.i;
        self.q += rhs.q;
        self.u += rhs.u;
        self.v += rhs.v;
    }
}

impl Mul<f64> for StokesVector {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.i * rhs, self.q * rhs, self.u * rhs, self.v * rhs)
    }
}

impl Mul<StokesVector> for f64 {
    type Output = StokesVector;

    fn mul(self, rhs: StokesVector) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<f64> for StokesVector {
    fn mul_assign(&mut self, rhs: f64) {
        self.i *= rhs;
        self.q *= rhs;
        self.u *= rhs;
        self.v *= rhs;
    }
}

impl Div<f64> for StokesVector {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.i / rhs, self.q / rhs, self.u / rhs, self.v / rhs)
    }
}

/// A Mueller matrix for transforming Stokes vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuellerMatrix {
    rows: [[f64; 4]; 4],
}

impl MuellerMatrix {
    /// Creates a matrix from row-major components.
    #[must_use]
    pub const fn new(rows: [[f64; 4]; 4]) -> Self {
        Self { rows }
    }

    /// Returns the row-major components.
    #[must_use]
    pub const fn rows(self) -> [[f64; 4]; 4] {
        self.rows
    }

    /// Identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Scalar attenuation that preserves the current polarization state.
    #[must_use]
    pub const fn scalar_attenuation(value: f64) -> Self {
        Self::new([
            [value, 0.0, 0.0, 0.0],
            [0.0, value, 0.0, 0.0],
            [0.0, 0.0, value, 0.0],
            [0.0, 0.0, 0.0, value],
        ])
    }

    /// Ideal depolarizer with unit throughput.
    #[must_use]
    pub const fn depolarizer() -> Self {
        Self::depolarizing_attenuation(1.0)
    }

    /// Depolarizing attenuation: output intensity is `value * input.i`.
    #[must_use]
    pub const fn depolarizing_attenuation(value: f64) -> Self {
        Self::new([
            [value, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Mueller rotation for changing the Stokes reference axis by `angle_radians`.
    #[must_use]
    pub fn rotation(angle_radians: f64) -> Self {
        let (sin_2, cos_2) = (2.0 * angle_radians).sin_cos();
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_2, sin_2, 0.0],
            [0.0, -sin_2, cos_2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Matrix for expressing a Stokes vector from one frame in another frame.
    #[must_use]
    pub fn frame_transform(from: PolarizationFrame, to: PolarizationFrame) -> Self {
        Self::rotation(from.rotation_angle_to(to))
    }

    /// Ideal mirror reflection in the scattering-plane basis.
    #[must_use]
    pub const fn perfect_mirror() -> Self {
        Self::scalar_attenuation(1.0)
    }

    /// Fresnel reflection matrix in the s/p basis.
    #[must_use]
    pub fn fresnel_reflection(s_reflectance: f64, p_reflectance: f64) -> Self {
        fresnel_intensity_matrix(s_reflectance, p_reflectance)
    }

    /// Fresnel transmission matrix in the s/p basis.
    #[must_use]
    pub fn fresnel_transmission(s_transmittance: f64, p_transmittance: f64) -> Self {
        fresnel_intensity_matrix(s_transmittance, p_transmittance)
    }

    /// Applies this matrix to a Stokes vector.
    #[must_use]
    pub fn apply(self, stokes: StokesVector) -> StokesVector {
        let values = [stokes.i, stokes.q, stokes.u, stokes.v];
        let mut out = [0.0; 4];
        for (row_index, row) in self.rows.iter().enumerate() {
            out[row_index] = row
                .iter()
                .zip(values)
                .map(|(matrix_value, stokes_value)| matrix_value * stokes_value)
                .sum();
        }
        StokesVector::new(out[0], out[1], out[2], out[3])
    }

    /// Matrix composition where `self` is applied first and `next` second.
    #[must_use]
    pub fn followed_by(self, next: Self) -> Self {
        next * self
    }

    /// Divides every matrix component by `value`.
    #[must_use]
    pub fn divided_by(self, value: f64) -> Self {
        self * (1.0 / value)
    }
}

impl Mul<MuellerMatrix> for MuellerMatrix {
    type Output = MuellerMatrix;

    fn mul(self, rhs: MuellerMatrix) -> Self::Output {
        let mut rows = [[0.0; 4]; 4];
        for (row_index, row) in rows.iter_mut().enumerate() {
            for (column_index, value) in row.iter_mut().enumerate() {
                *value = (0..4)
                    .map(|inner| self.rows[row_index][inner] * rhs.rows[inner][column_index])
                    .sum();
            }
        }
        MuellerMatrix::new(rows)
    }
}

impl Add for MuellerMatrix {
    type Output = MuellerMatrix;

    fn add(self, rhs: MuellerMatrix) -> Self::Output {
        let mut rows = self.rows;
        for (row_index, row) in rows.iter_mut().enumerate() {
            for (column_index, value) in row.iter_mut().enumerate() {
                *value += rhs.rows[row_index][column_index];
            }
        }
        MuellerMatrix::new(rows)
    }
}

impl Mul<f64> for MuellerMatrix {
    type Output = MuellerMatrix;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut rows = self.rows;
        for row in &mut rows {
            for value in row {
                *value *= rhs;
            }
        }
        MuellerMatrix::new(rows)
    }
}

impl Mul<MuellerMatrix> for f64 {
    type Output = MuellerMatrix;

    fn mul(self, rhs: MuellerMatrix) -> Self::Output {
        rhs * self
    }
}

/// Polarization reference frame for a ray direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolarizationFrame {
    direction: Vector,
    reference: Vector,
}

impl PolarizationFrame {
    /// Creates a polarization frame by projecting `reference` perpendicular to `direction`.
    #[must_use]
    pub fn new(direction: Vector, reference: Vector) -> Option<Self> {
        let direction = direction.normalized();
        if direction.length_squared() <= f64::EPSILON {
            return None;
        }
        let projected = reference - direction * reference.dot(direction);
        let reference = if projected.length_squared() <= f64::EPSILON {
            fallback_reference(direction)
        } else {
            projected.normalized()
        };
        Some(Self {
            direction,
            reference,
        })
    }

    /// Creates a deterministic frame for `direction`.
    ///
    /// # Panics
    ///
    /// Panics if `direction` is zero length.
    #[must_use]
    pub fn from_direction(direction: Vector) -> Self {
        let direction = direction.normalized();
        Self::new(direction, fallback_reference(direction))
            .expect("fallback reference should be perpendicular to a non-zero direction")
    }

    /// Creates a frame whose reference axis is perpendicular to a scattering plane.
    #[must_use]
    pub fn from_scattering_plane(direction: Vector, plane_normal: Vector) -> Self {
        let direction = direction.normalized();
        let plane_reference = direction.cross(plane_normal.normalized());
        Self::new(direction, plane_reference).unwrap_or_else(|| Self::from_direction(direction))
    }

    /// Returns the ray direction.
    #[must_use]
    pub const fn direction(self) -> Vector {
        self.direction
    }

    /// Returns the reference axis perpendicular to the ray direction.
    #[must_use]
    pub const fn reference(self) -> Vector {
        self.reference
    }

    /// Returns the signed angle needed to express Stokes data from this frame in `target`.
    #[must_use]
    pub fn rotation_angle_to(self, target: Self) -> f64 {
        let cross = self.reference.cross(target.reference);
        let sin_theta = self.direction.dot(cross);
        let cos_theta = self.reference.dot(target.reference);
        sin_theta.atan2(cos_theta)
    }
}

/// Fresnel Mueller matrices for a sampled dielectric interface event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DielectricFresnel {
    /// Reflection Mueller matrix in the s/p basis.
    pub reflection: MuellerMatrix,
    /// Transmission Mueller matrix in the s/p basis.
    pub transmission: MuellerMatrix,
    /// Average unpolarized reflectance.
    pub reflectance: f64,
    /// Average unpolarized transmittance.
    pub transmittance: f64,
    /// True when the interface is totally internally reflecting.
    pub total_internal_reflection: bool,
}

/// Optical constants for a conductive interface, relative to the incident medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConductorOpticalConstants {
    /// Real part of the complex index of refraction.
    pub eta: f64,
    /// Extinction coefficient.
    pub k: f64,
}

impl ConductorOpticalConstants {
    /// Creates conductor optical constants.
    ///
    /// # Panics
    ///
    /// Panics if either value is not finite.
    #[must_use]
    pub fn new(eta: f64, k: f64) -> Self {
        assert!(
            eta.is_finite() && k.is_finite(),
            "conductor optical constants must be finite"
        );
        Self {
            eta: eta.max(0.0),
            k: k.max(0.0),
        }
    }
}

/// Fresnel Mueller matrix for a conductive interface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConductorFresnel {
    /// Reflection Mueller matrix in the s/p basis.
    pub reflection: MuellerMatrix,
    /// Average unpolarized reflectance.
    pub reflectance: f64,
}

/// Computes non-absorbing dielectric Fresnel Mueller matrices in the s/p basis.
#[must_use]
pub fn dielectric_fresnel(cos_theta_i: f64, eta_i: f64, eta_t: f64) -> DielectricFresnel {
    let cos_theta_i = cos_theta_i.abs().clamp(0.0, 1.0);
    let eta_ratio = eta_i / eta_t;
    let incident_sine_squared = (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    let refracted_sine_squared = eta_ratio * eta_ratio * incident_sine_squared;

    if refracted_sine_squared >= 1.0 {
        return DielectricFresnel {
            reflection: MuellerMatrix::fresnel_reflection(1.0, 1.0),
            transmission: MuellerMatrix::depolarizing_attenuation(0.0),
            reflectance: 1.0,
            transmittance: 0.0,
            total_internal_reflection: true,
        };
    }

    let cos_theta_t = (1.0 - refracted_sine_squared).sqrt();
    let s_amplitude =
        (eta_i * cos_theta_i - eta_t * cos_theta_t) / (eta_i * cos_theta_i + eta_t * cos_theta_t);
    let p_amplitude =
        (eta_t * cos_theta_i - eta_i * cos_theta_t) / (eta_t * cos_theta_i + eta_i * cos_theta_t);
    let rs = (s_amplitude * s_amplitude).clamp(0.0, 1.0);
    let rp = (p_amplitude * p_amplitude).clamp(0.0, 1.0);
    let ts = 1.0 - rs;
    let tp = 1.0 - rp;
    let reflectance = 0.5 * (rs + rp);
    let transmittance = 0.5 * (ts + tp);

    DielectricFresnel {
        reflection: MuellerMatrix::fresnel_reflection(rs, rp),
        transmission: MuellerMatrix::fresnel_transmission(ts, tp),
        reflectance,
        transmittance,
        total_internal_reflection: false,
    }
}

/// Computes absorbing-conductor Fresnel reflection in the s/p basis.
#[must_use]
pub fn conductor_fresnel(
    cos_theta_i: f64,
    constants: ConductorOpticalConstants,
) -> ConductorFresnel {
    let cos_theta = cos_theta_i.abs().clamp(0.0, 1.0);
    let conductor_index = Complex64::new(constants.eta.max(0.0), constants.k.max(0.0));
    if conductor_index.norm_sqr() <= f64::EPSILON {
        return ConductorFresnel {
            reflection: MuellerMatrix::perfect_mirror(),
            reflectance: 1.0,
        };
    }
    let incident_cosine = Complex64::new(cos_theta, 0.0);
    let incident_sine_squared = (1.0 - cos_theta * cos_theta).max(0.0);
    let conductor_cosine = (Complex64::new(1.0, 0.0)
        - Complex64::new(incident_sine_squared, 0.0) / (conductor_index * conductor_index))
        .sqrt();
    let s_amplitude = (incident_cosine - conductor_index * conductor_cosine)
        / (incident_cosine + conductor_index * conductor_cosine);
    let p_amplitude = (conductor_index * incident_cosine - conductor_cosine)
        / (conductor_index * incident_cosine + conductor_cosine);
    let rs = s_amplitude.norm_sqr().clamp(0.0, 1.0);
    let rp = p_amplitude.norm_sqr().clamp(0.0, 1.0);
    ConductorFresnel {
        reflection: fresnel_amplitude_reflection_matrix(s_amplitude, p_amplitude),
        reflectance: 0.5 * (rs + rp),
    }
}

/// Linear floating-point output from the sampled-wavelength spectral renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Reconstructed linear RGB samples, row-major.
    pub linear_rgb: Vec<LinearColor>,
    /// Transport mode used to produce this image.
    pub transport_mode: SpectralTransportMode,
    /// Optional visible-band averaged Stokes output, row-major.
    pub polarization: Option<Vec<StokesVector>>,
}

impl SpectralImage {
    /// Creates a spectral render output from reconstructed linear samples.
    ///
    /// # Panics
    ///
    /// Panics if `linear_rgb.len()` does not match `width * height`.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        linear_rgb: Vec<LinearColor>,
        transport_mode: SpectralTransportMode,
    ) -> Self {
        assert_eq!(
            linear_rgb.len(),
            Canvas::pixel_count(width, height),
            "spectral image sample count must match dimensions"
        );
        Self {
            width,
            height,
            linear_rgb,
            transport_mode,
            polarization: None,
        }
    }

    /// Creates a spectral render output with visible-band averaged polarization samples.
    ///
    /// # Panics
    ///
    /// Panics if either sample buffer does not match `width * height`.
    #[must_use]
    pub fn new_with_polarization(
        width: u32,
        height: u32,
        linear_rgb: Vec<LinearColor>,
        polarization: Vec<StokesVector>,
    ) -> Self {
        assert_eq!(
            polarization.len(),
            Canvas::pixel_count(width, height),
            "spectral polarization sample count must match dimensions"
        );
        let mut image = Self::new(width, height, linear_rgb, SpectralTransportMode::Polarized);
        image.polarization = Some(polarization);
        image
    }

    /// Converts the linear output to a gamma-encoded display canvas.
    pub fn to_canvas(&self) -> Canvas {
        self.to_hdr_image().to_canvas()
    }

    /// Converts the linear output to a display canvas with explicit tone mapping.
    pub fn to_canvas_tone_mapped(&self, tone_map: ToneMap) -> Canvas {
        self.to_hdr_image().to_canvas_tone_mapped(tone_map)
    }

    /// Converts the spectral output to a generic HDR image.
    #[must_use]
    pub fn to_hdr_image(&self) -> HdrImage {
        HdrImage::from_pixels(self.width, self.height, self.linear_rgb.clone())
    }

    /// Saves the linear output as portable float-map data (`PF`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying I/O fails.
    pub fn save_pfm(&self, file_name: &str) -> std::io::Result<()> {
        self.to_hdr_image().save_pfm(file_name)
    }

    /// Returns the reconstructed linear sample at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics only if image dimensions cannot fit in `usize`.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<LinearColor> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let width = usize::try_from(self.width).expect("image width should fit usize");
        let x = usize::try_from(x).expect("pixel x should fit usize");
        let y = usize::try_from(y).expect("pixel y should fit usize");
        self.linear_rgb.get(y * width + x).copied()
    }

    /// Consumes the image and returns the row-major linear samples.
    #[must_use]
    pub fn into_linear_rgb(self) -> Vec<LinearColor> {
        self.linear_rgb
    }

    /// Returns the visible-band averaged Stokes sample at `(x, y)` when present.
    ///
    /// # Panics
    ///
    /// Panics only if image dimensions cannot fit in `usize`.
    #[must_use]
    pub fn polarization_pixel(&self, x: u32, y: u32) -> Option<StokesVector> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let width = usize::try_from(self.width).expect("image width should fit usize");
        let x = usize::try_from(x).expect("pixel x should fit usize");
        let y = usize::try_from(y).expect("pixel y should fit usize");
        self.polarization
            .as_ref()
            .and_then(|samples| samples.get(y * width + x).copied())
    }
}

/// One sampled visible wavelength and its sampling PDF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledWavelength {
    wavelength_nm: f64,
    pdf: f64,
}

impl SampledWavelength {
    /// Creates a sampled wavelength in nanometers.
    ///
    /// # Panics
    ///
    /// Panics if `wavelength_nm` is not inside the supported visible interval or if `pdf` is not
    /// positive and finite.
    #[must_use]
    pub fn new(wavelength_nm: f64, pdf: f64) -> Self {
        assert!(
            (VISIBLE_WAVELENGTH_MIN_NM..=VISIBLE_WAVELENGTH_MAX_NM).contains(&wavelength_nm),
            "sampled wavelength must be inside the visible interval"
        );
        assert!(
            pdf.is_finite() && pdf > 0.0,
            "sampled wavelength pdf must be positive and finite"
        );
        Self { wavelength_nm, pdf }
    }

    /// Uniformly samples a visible wavelength.
    #[must_use]
    pub fn sample_visible(rng: &mut SampleRng) -> Self {
        Self {
            wavelength_nm: rng.random_range(VISIBLE_WAVELENGTH_MIN_NM, VISIBLE_WAVELENGTH_MAX_NM),
            pdf: 1.0 / VISIBLE_WAVELENGTH_RANGE_NM,
        }
    }

    /// Returns the wavelength in nanometers.
    #[must_use]
    pub const fn wavelength_nm(self) -> f64 {
        self.wavelength_nm
    }

    /// Returns the probability density used to sample this wavelength.
    #[must_use]
    pub const fn pdf(self) -> f64 {
        self.pdf
    }

    /// Converts scalar radiance carried at this wavelength into a linear RGB contribution.
    ///
    /// The reconstruction basis is normalized so a constant unit spectrum averages to white under
    /// uniform visible-wavelength sampling.
    #[must_use]
    pub fn reconstruct_linear_rgb(self, radiance: f64) -> LinearRgb {
        wavelength_reconstruction_basis(self.wavelength_nm)
            * (radiance / (self.pdf * VISIBLE_WAVELENGTH_RANGE_NM))
    }

    /// Converts a sampled Stokes vector into a visible-band averaged contribution.
    #[must_use]
    pub fn reconstruct_stokes(self, stokes: StokesVector) -> StokesVector {
        stokes / (self.pdf * VISIBLE_WAVELENGTH_RANGE_NM)
    }
}

/// Measured scalar spectral data sampled at wavelengths in nanometers.
///
/// Samples are stored sorted by wavelength and evaluated with linear interpolation. Values outside
/// the measured interval clamp to the nearest endpoint so sparse reflectance, emission, eta, and
/// extinction-coefficient data remains usable across the visible range.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredSpectrum {
    samples: Vec<(f64, f64)>,
}

impl MeasuredSpectrum {
    /// Creates measured spectral data from `(wavelength_nm, value)` samples.
    ///
    /// # Panics
    ///
    /// Panics if no samples are supplied, any component is not finite, or two samples have the
    /// same wavelength.
    #[must_use]
    pub fn new(samples: Vec<(f64, f64)>) -> Self {
        Self::try_new(samples).expect("measured spectrum samples must be finite and unique")
    }

    /// Creates measured spectral data, returning `None` for invalid samples.
    #[must_use]
    pub fn try_new(mut samples: Vec<(f64, f64)>) -> Option<Self> {
        if samples.is_empty()
            || samples
                .iter()
                .any(|(wavelength, value)| !wavelength.is_finite() || !value.is_finite())
        {
            return None;
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        if samples
            .windows(2)
            .any(|pair| (pair[0].0 - pair[1].0).abs() <= f64::EPSILON)
        {
            return None;
        }
        Some(Self { samples })
    }

    /// Parses a comma-separated spectrum string.
    ///
    /// Each non-empty, non-comment line should contain at least wavelength and value columns.
    /// A non-numeric header row is ignored.
    ///
    /// # Errors
    ///
    /// Returns an error when the text has no valid samples or contains malformed rows after data
    /// has started.
    pub fn from_csv_str(text: &str) -> io::Result<Self> {
        parse_measured_spectrum(text)
    }

    /// Parses a whitespace-separated SPD spectrum string.
    ///
    /// A non-numeric header row is ignored. Commas and semicolons are also accepted so common
    /// exported CSV/SPD files can share the same parser.
    ///
    /// # Errors
    ///
    /// Returns an error when the text has no valid samples or contains malformed rows after data
    /// has started.
    pub fn from_spd_str(text: &str) -> io::Result<Self> {
        parse_measured_spectrum(text)
    }

    /// Loads a comma-separated measured spectrum file.
    ///
    /// # Errors
    ///
    /// Returns an error when reading fails or the data has no valid wavelength/value samples.
    pub fn from_csv_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_csv_str(&text)
    }

    /// Loads a whitespace-separated measured spectrum file.
    ///
    /// # Errors
    ///
    /// Returns an error when reading fails or the data has no valid wavelength/value samples.
    pub fn from_spd_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_spd_str(&text)
    }

    /// Returns the sorted measured samples.
    #[must_use]
    pub fn samples(&self) -> &[(f64, f64)] {
        &self.samples
    }

    /// Evaluates this spectrum at a sampled wavelength.
    #[must_use]
    pub fn sample(&self, wavelength: SampledWavelength) -> f64 {
        self.sample_wavelength_nm(wavelength.wavelength_nm())
    }

    /// Evaluates this spectrum at `wavelength_nm` with linear interpolation.
    ///
    /// # Panics
    ///
    /// Panics if `wavelength_nm` is not finite.
    #[must_use]
    pub fn sample_wavelength_nm(&self, wavelength_nm: f64) -> f64 {
        assert!(
            wavelength_nm.is_finite(),
            "sampled wavelength must be finite"
        );
        if self.samples.len() == 1 || wavelength_nm <= self.samples[0].0 {
            return self.samples[0].1;
        }
        let last = self.samples[self.samples.len() - 1];
        if wavelength_nm >= last.0 {
            return last.1;
        }

        let upper = self
            .samples
            .partition_point(|(sample_wavelength, _)| *sample_wavelength < wavelength_nm);
        let (w0, v0) = self.samples[upper - 1];
        let (w1, v1) = self.samples[upper];
        let t = (wavelength_nm - w0) / (w1 - w0);
        v0 + t * (v1 - v0)
    }

    /// Returns the maximum measured value.
    #[must_use]
    pub fn max_value(&self) -> f64 {
        self.samples
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Returns the trapezoidal integral over the measured wavelength domain.
    #[must_use]
    pub fn integral(&self) -> f64 {
        self.samples
            .windows(2)
            .map(|pair| {
                let (w0, v0) = pair[0];
                let (w1, v1) = pair[1];
                0.5 * (v0 + v1) * (w1 - w0)
            })
            .sum()
    }

    /// Returns a copy scaled by `scale`.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not finite.
    #[must_use]
    pub fn scaled(&self, scale: f64) -> Self {
        assert!(scale.is_finite(), "spectrum scale must be finite");
        Self {
            samples: self
                .samples
                .iter()
                .map(|(wavelength, value)| (*wavelength, value * scale))
                .collect(),
        }
    }

    /// Returns a copy whose maximum value is `target_peak`.
    #[must_use]
    pub fn normalized_to_peak(&self, target_peak: f64) -> Option<Self> {
        if !target_peak.is_finite() {
            return None;
        }
        let peak = self.max_value();
        (peak.abs() > f64::EPSILON).then(|| self.scaled(target_peak / peak))
    }

    /// Returns a copy whose trapezoidal integral is `target_area`.
    #[must_use]
    pub fn normalized_to_area(&self, target_area: f64) -> Option<Self> {
        if !target_area.is_finite() {
            return None;
        }
        let area = self.integral();
        (area.abs() > f64::EPSILON).then(|| self.scaled(target_area / area))
    }

    /// Converts this measured spectrum to a non-negative linear RGB fallback by stratified
    /// visible-range integration.
    ///
    /// Narrow or out-of-gamut spectra can reconstruct negative linear sRGB components. Those are
    /// clamped here so RGB compatibility renders never receive negative attenuation; sampled
    /// spectral transport still evaluates the measured data directly.
    #[must_use]
    pub fn to_linear_rgb(&self) -> LinearRgb {
        const SAMPLE_COUNT: u32 = 64;
        let mut color = LinearColor::default();
        for sample in 0..SAMPLE_COUNT {
            let wavelength = VISIBLE_WAVELENGTH_MIN_NM
                + ((f64::from(sample) + 0.5) / f64::from(SAMPLE_COUNT))
                    * VISIBLE_WAVELENGTH_RANGE_NM;
            let wavelength = SampledWavelength::new(wavelength, 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);
            color += wavelength.reconstruct_linear_rgb(self.sample(wavelength));
        }
        nonnegative_finite_rgb(color / f64::from(SAMPLE_COUNT))
    }
}

/// A compact spectral abstraction used by the prototype spectral renderer.
///
/// [`Self::Rgb`] is an adapter for existing RGB material and texture colors. It gives the
/// sampled-wavelength path a usable spectral value, but it is not a substitute for measured
/// reflectance, emission, or index-of-refraction spectra.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spectrum {
    /// Constant spectral value at every supported wavelength.
    Constant(f64),
    /// RGB data interpreted through smooth wavelength response curves.
    ///
    /// This keeps existing RGB materials compatible with spectral transport; it is not measured
    /// spectral asset data.
    Rgb(LinearRgb),
}

impl Spectrum {
    /// Creates a constant spectrum.
    #[must_use]
    pub const fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Creates an RGB-backed spectrum.
    #[must_use]
    pub const fn from_linear_rgb(color: LinearRgb) -> Self {
        Self::Rgb(color)
    }

    /// Evaluates this spectrum at `wavelength`.
    #[must_use]
    pub fn sample(self, wavelength: SampledWavelength) -> f64 {
        match self {
            Self::Constant(value) => value,
            Self::Rgb(color) => sample_linear_rgb(color, wavelength.wavelength_nm),
        }
    }

    /// Converts this spectrum to linear RGB by stratified integration over the visible range.
    #[must_use]
    pub fn to_linear_rgb(self) -> LinearRgb {
        const SAMPLE_COUNT: u32 = 32;
        let mut color = LinearColor::default();
        for sample in 0..SAMPLE_COUNT {
            let wavelength = VISIBLE_WAVELENGTH_MIN_NM
                + ((f64::from(sample) + 0.5) / f64::from(SAMPLE_COUNT))
                    * VISIBLE_WAVELENGTH_RANGE_NM;
            let wavelength = SampledWavelength::new(wavelength, 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);
            color += wavelength.reconstruct_linear_rgb(self.sample(wavelength));
        }
        color / f64::from(SAMPLE_COUNT)
    }
}

impl From<LinearRgb> for Spectrum {
    fn from(color: LinearRgb) -> Self {
        Self::from_linear_rgb(color)
    }
}

fn parse_measured_spectrum(text: &str) -> io::Result<MeasuredSpectrum> {
    let mut samples = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let columns: Vec<_> = line
            .split(|character: char| {
                character == ',' || character == ';' || character.is_whitespace()
            })
            .filter(|column| !column.is_empty())
            .collect();
        if columns.len() < 2 {
            continue;
        }
        let wavelength = columns[0].parse::<f64>();
        let value = columns[1].parse::<f64>();
        match (wavelength, value) {
            (Ok(wavelength), Ok(value)) => samples.push((wavelength, value)),
            _ if samples.is_empty() => {}
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("invalid measured spectrum row {}", line_index + 1),
                ));
            }
        }
    }
    MeasuredSpectrum::try_new(samples).ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "measured spectrum requires finite samples with unique wavelengths",
        )
    })
}

fn sample_linear_rgb(color: LinearRgb, wavelength_nm: f64) -> f64 {
    let response = rgb_spectrum_response(wavelength_nm);
    let response_sum = response.red + response.green + response.blue;
    if response_sum <= RESPONSE_EPSILON {
        return (color.red + color.green + color.blue) / 3.0;
    }

    ((color.red * response.red) + (color.green * response.green) + (color.blue * response.blue))
        / response_sum
}

fn nonnegative_finite_rgb(color: LinearRgb) -> LinearRgb {
    let sanitize = |component: f64| {
        if component.is_finite() {
            component.max(0.0)
        } else {
            0.0
        }
    };
    LinearRgb::new(
        sanitize(color.red),
        sanitize(color.green),
        sanitize(color.blue),
    )
}

fn wavelength_reconstruction_basis(wavelength_nm: f64) -> LinearRgb {
    let response = cie_linear_rgb_response(wavelength_nm);
    let mean = equal_energy_linear_rgb_mean();
    LinearRgb::new(
        response.red / mean.red,
        response.green / mean.green,
        response.blue / mean.blue,
    )
}

fn rgb_spectrum_response(wavelength_nm: f64) -> LinearRgb {
    let response = cie_linear_rgb_response(wavelength_nm);
    LinearRgb::new(
        response.red.max(0.0),
        response.green.max(0.0),
        response.blue.max(0.0),
    )
}

fn equal_energy_linear_rgb_mean() -> LinearRgb {
    *EQUAL_ENERGY_LINEAR_RGB_MEAN.get_or_init(|| {
        let mut sum = LinearRgb::default();
        for sample in 0..EQUAL_ENERGY_NORMALIZATION_SAMPLES {
            let wavelength = VISIBLE_WAVELENGTH_MIN_NM
                + ((f64::from(sample) + 0.5) / f64::from(EQUAL_ENERGY_NORMALIZATION_SAMPLES))
                    * VISIBLE_WAVELENGTH_RANGE_NM;
            sum += cie_linear_rgb_response(wavelength);
        }
        sum / f64::from(EQUAL_ENERGY_NORMALIZATION_SAMPLES)
    })
}

fn cie_linear_rgb_response(wavelength_nm: f64) -> LinearRgb {
    let [x, y, z] = cie_1931_xyz_fit(wavelength_nm);
    xyz_to_linear_srgb(x, y, z)
}

fn cie_1931_xyz_fit(wavelength_nm: f64) -> [f64; 3] {
    let x = 1.056 * asymmetric_gaussian(wavelength_nm, 599.8, 37.9, 31.0)
        + 0.362 * asymmetric_gaussian(wavelength_nm, 442.0, 16.0, 26.7)
        - 0.065 * asymmetric_gaussian(wavelength_nm, 501.1, 20.4, 26.2);
    let y = 0.821 * asymmetric_gaussian(wavelength_nm, 568.8, 46.9, 40.5)
        + 0.286 * asymmetric_gaussian(wavelength_nm, 530.9, 16.3, 31.1);
    let z = 1.217 * asymmetric_gaussian(wavelength_nm, 437.0, 11.8, 36.0)
        + 0.681 * asymmetric_gaussian(wavelength_nm, 459.0, 26.0, 13.8);
    [x.max(0.0), y.max(0.0), z.max(0.0)]
}

fn asymmetric_gaussian(wavelength_nm: f64, center: f64, left_sigma: f64, right_sigma: f64) -> f64 {
    let sigma = if wavelength_nm < center {
        left_sigma
    } else {
        right_sigma
    };
    let scaled = (wavelength_nm - center) / sigma;
    (-0.5 * scaled * scaled).exp()
}

fn xyz_to_linear_srgb(x: f64, y: f64, z: f64) -> LinearRgb {
    LinearRgb::new(
        3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
        -0.969_266_0 * x + 1.876_010_8 * y + 0.041_556_0 * z,
        0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
    )
}

fn fresnel_intensity_matrix(s_intensity: f64, p_intensity: f64) -> MuellerMatrix {
    let s_intensity = s_intensity.max(0.0);
    let p_intensity = p_intensity.max(0.0);
    let average = 0.5 * (s_intensity + p_intensity);
    let difference = 0.5 * (s_intensity - p_intensity);
    let cross = (s_intensity * p_intensity).sqrt();
    MuellerMatrix::new([
        [average, difference, 0.0, 0.0],
        [difference, average, 0.0, 0.0],
        [0.0, 0.0, cross, 0.0],
        [0.0, 0.0, 0.0, cross],
    ])
}

fn fresnel_amplitude_reflection_matrix(
    s_amplitude: Complex64,
    p_amplitude: Complex64,
) -> MuellerMatrix {
    let s_intensity = s_amplitude.norm_sqr().max(0.0);
    let p_intensity = p_amplitude.norm_sqr().max(0.0);
    let average = 0.5 * (s_intensity + p_intensity);
    let difference = 0.5 * (s_intensity - p_intensity);
    let cross = s_amplitude * p_amplitude.conj();
    MuellerMatrix::new([
        [average, difference, 0.0, 0.0],
        [difference, average, 0.0, 0.0],
        [0.0, 0.0, cross.re, -cross.im],
        [0.0, 0.0, cross.im, cross.re],
    ])
}

fn fallback_reference(direction: Vector) -> Vector {
    let axis = if direction.x().abs() < 0.9 {
        Vector::new(1.0, 0.0, 0.0)
    } else {
        Vector::new(0.0, 1.0, 0.0)
    };
    (axis - direction * axis.dot(direction)).normalized()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::colors::Rgb;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn sampled_wavelength_uses_uniform_visible_pdf() {
        let mut rng = SampleRng::new(7);
        let sample = SampledWavelength::sample_visible(&mut rng);

        assert!(
            (VISIBLE_WAVELENGTH_MIN_NM..VISIBLE_WAVELENGTH_MAX_NM)
                .contains(&sample.wavelength_nm())
        );
        assert_close(sample.pdf(), 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);
    }

    #[test]
    fn neutral_rgb_spectrum_samples_as_neutral() {
        let spectrum = Spectrum::from_linear_rgb(LinearRgb::new(0.4, 0.4, 0.4));

        for wavelength_nm in [400.0, 500.0, 600.0] {
            let sample = SampledWavelength::new(wavelength_nm, 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);
            assert_close(spectrum.sample(sample), 0.4);
        }
    }

    #[test]
    fn constant_spectrum_integrates_to_neutral_rgb() {
        let color = Spectrum::constant(1.0).to_linear_rgb();

        assert!((color.red - 1.0).abs() < 0.05, "{color:?}");
        assert!((color.green - 1.0).abs() < 0.05, "{color:?}");
        assert!((color.blue - 1.0).abs() < 0.05, "{color:?}");
    }

    #[test]
    fn reconstruction_uses_sample_pdf() {
        let uniform = SampledWavelength::new(620.0, 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);
        let rarer = SampledWavelength::new(620.0, 0.5 / VISIBLE_WAVELENGTH_RANGE_NM);

        assert_close(
            rarer.reconstruct_linear_rgb(1.0).red,
            2.0 * uniform.reconstruct_linear_rgb(1.0).red,
        );
    }

    #[test]
    fn rgb_spectrum_keeps_channel_preference() {
        let red = Spectrum::from_linear_rgb(LinearRgb::new(1.0, 0.0, 0.0));
        let red_wavelength = SampledWavelength::new(620.0, 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);
        let blue_wavelength = SampledWavelength::new(460.0, 1.0 / VISIBLE_WAVELENGTH_RANGE_NM);

        assert!(red.sample(red_wavelength) > red.sample(blue_wavelength));
    }

    #[test]
    fn measured_spectrum_interpolates_clamps_and_normalizes() {
        let spectrum = MeasuredSpectrum::new(vec![(600.0, 1.0), (400.0, 0.0)]);

        assert_eq!(spectrum.samples(), &[(400.0, 0.0), (600.0, 1.0)]);
        assert_close(spectrum.sample_wavelength_nm(300.0), 0.0);
        assert_close(spectrum.sample_wavelength_nm(500.0), 0.5);
        assert_close(spectrum.sample_wavelength_nm(700.0), 1.0);
        assert_close(spectrum.integral(), 100.0);

        let peak_normalized = spectrum
            .normalized_to_peak(2.0)
            .expect("positive peak should normalize");
        let area_normalized = spectrum
            .normalized_to_area(20.0)
            .expect("positive area should normalize");

        assert_close(peak_normalized.max_value(), 2.0);
        assert_close(area_normalized.integral(), 20.0);
    }

    #[test]
    fn measured_spectrum_loads_csv_and_spd_data() {
        let csv = "wavelength,value\n400,0.25\n500,0.75\n";
        let spd = "# wavelength value\n400 1.0\n500 3.0\n";

        let csv_spectrum = MeasuredSpectrum::from_csv_str(csv).expect("csv spectrum should parse");
        let spd_spectrum = MeasuredSpectrum::from_spd_str(spd).expect("spd spectrum should parse");

        assert_close(csv_spectrum.sample_wavelength_nm(450.0), 0.5);
        assert_close(spd_spectrum.sample_wavelength_nm(450.0), 2.0);

        let path = std::env::temp_dir().join(format!(
            "gartus_test_spectrum_{}_{}.spd",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, spd).expect("temp spd write should work");
        let loaded = MeasuredSpectrum::from_spd_file(&path).expect("spd file should load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.samples(), spd_spectrum.samples());
    }

    #[test]
    fn measured_equal_energy_spectrum_integrates_to_neutral_rgb() {
        let spectrum = MeasuredSpectrum::new(vec![
            (VISIBLE_WAVELENGTH_MIN_NM, 1.0),
            (VISIBLE_WAVELENGTH_MAX_NM, 1.0),
        ]);
        let color = spectrum.to_linear_rgb();

        assert!((color.red - 1.0).abs() < 0.05, "{color:?}");
        assert!((color.green - 1.0).abs() < 0.05, "{color:?}");
        assert!((color.blue - 1.0).abs() < 0.05, "{color:?}");
    }

    #[test]
    fn measured_spectrum_rgb_fallback_clamps_out_of_gamut_components() {
        let spectrum = MeasuredSpectrum::new(vec![
            (VISIBLE_WAVELENGTH_MIN_NM, 0.0),
            (440.0, 0.0),
            (442.5, 10.0),
            (445.0, 0.0),
            (VISIBLE_WAVELENGTH_MAX_NM, 0.0),
        ]);
        let color = spectrum.to_linear_rgb();

        assert!(color.is_finite(), "{color:?}");
        assert!(color.red >= 0.0, "{color:?}");
        assert!(color.green >= 0.0, "{color:?}");
        assert!(color.blue >= 0.0, "{color:?}");
        assert!(color.max_component() > 0.0, "{color:?}");
    }

    #[test]
    fn cie_fit_tracks_photopic_green_sensitivity() {
        let violet = cie_1931_xyz_fit(420.0);
        let green = cie_1931_xyz_fit(555.0);
        let red = cie_1931_xyz_fit(650.0);

        assert!(green[1] > violet[1]);
        assert!(green[1] > red[1]);
    }

    #[test]
    fn stokes_vector_tracks_polarization_degree() {
        let unpolarized = StokesVector::unpolarized(2.0);
        let partially_polarized = StokesVector::new(2.0, 1.0, 0.0, 0.0);

        assert_close(unpolarized.degree_of_polarization(), 0.0);
        assert_close(partially_polarized.degree_of_polarization(), 0.5);
        assert!(partially_polarized.is_finite());
        assert!(partially_polarized.is_physical());
    }

    #[test]
    fn mueller_depolarizer_removes_polarized_components() {
        let input = StokesVector::new(2.0, 1.0, 0.5, 0.25);
        let output = MuellerMatrix::depolarizer().apply(input);

        assert_eq!(output, StokesVector::unpolarized(2.0));
    }

    #[test]
    fn dielectric_fresnel_suppresses_p_polarized_brewster_reflection() {
        let eta_i: f64 = 1.0;
        let eta_t: f64 = 1.5;
        let brewster_theta = (eta_t / eta_i).atan();
        let fresnel = dielectric_fresnel(brewster_theta.cos(), eta_i, eta_t);
        let p_polarized = StokesVector::new(1.0, -1.0, 0.0, 0.0);

        assert!(fresnel.reflection.apply(p_polarized).i < 1.0e-12);
        assert!(fresnel.reflectance > 0.0);
        assert!(!fresnel.total_internal_reflection);
    }

    #[test]
    fn conductor_fresnel_matches_normal_incidence_eta_k_reflectance() {
        let constants = ConductorOpticalConstants::new(0.2, 3.0);
        let fresnel = conductor_fresnel(1.0, constants);
        let expected = ((constants.eta - 1.0).powi(2) + constants.k * constants.k)
            / ((constants.eta + 1.0).powi(2) + constants.k * constants.k);
        let reflected = fresnel.reflection.apply(StokesVector::unpolarized(1.0));

        assert_close(fresnel.reflectance, expected);
        assert_close(reflected.i, expected);
        assert_close(reflected.degree_of_polarization(), 0.0);
    }

    #[test]
    fn conductor_fresnel_degenerate_constants_remain_finite() {
        let fresnel = conductor_fresnel(0.5, ConductorOpticalConstants::new(0.0, 0.0));
        let reflected = fresnel.reflection.apply(StokesVector::unpolarized(1.0));

        assert_close(fresnel.reflectance, 1.0);
        assert!(reflected.is_finite());
        assert_close(reflected.i, 1.0);
    }

    #[test]
    fn conductor_fresnel_polarizes_oblique_unpolarized_light() {
        let constants = ConductorOpticalConstants::new(0.2, 3.0);
        let fresnel = conductor_fresnel(0.35, constants);
        let reflected = fresnel.reflection.apply(StokesVector::unpolarized(1.0));

        assert!(fresnel.reflectance > 0.0);
        assert!(reflected.degree_of_polarization() > 0.01, "{reflected:?}");
    }

    #[test]
    fn conductor_fresnel_retardance_couples_linear_and_circular_polarization() {
        let constants = ConductorOpticalConstants::new(0.2, 3.0);
        let fresnel = conductor_fresnel(0.35, constants);
        let reflected = fresnel
            .reflection
            .apply(StokesVector::new(1.0, 0.0, 1.0, 0.0));

        assert!(reflected.v.abs() > 0.01, "{reflected:?}");
        assert!(reflected.u.abs() > 0.01, "{reflected:?}");
    }

    #[test]
    fn polarization_frame_transform_rotates_linear_components() {
        let direction = Vector::new(0.0, 0.0, -1.0);
        let x_frame = PolarizationFrame::new(direction, Vector::new(1.0, 0.0, 0.0)).unwrap();
        let y_frame = PolarizationFrame::new(direction, Vector::new(0.0, 1.0, 0.0)).unwrap();
        let transformed = MuellerMatrix::frame_transform(x_frame, y_frame)
            .apply(StokesVector::new(1.0, 1.0, 0.0, 0.0));

        assert_close(transformed.i, 1.0);
        assert_close(transformed.q, -1.0);
        assert_close(transformed.u.abs(), 0.0);
    }

    #[test]
    fn spectral_image_keeps_linear_samples_and_converts_to_canvas() {
        let image = SpectralImage::new(
            2,
            1,
            vec![
                LinearColor::new(0.25, 0.0, 0.0),
                LinearColor::new(0.0, 0.25, 0.0),
            ],
            SpectralTransportMode::Unpolarized,
        );

        assert_eq!(image.pixel(0, 0), Some(LinearColor::new(0.25, 0.0, 0.0)));
        assert_eq!(image.pixel(2, 0), None);
        assert_eq!(image.to_canvas().pixels()[0], Rgb::new(128, 0, 0));
        assert_eq!(image.transport_mode, SpectralTransportMode::Unpolarized);
        assert_eq!(image.polarization, None);
    }

    #[test]
    fn spectral_image_can_store_polarization_samples() {
        let image = SpectralImage::new_with_polarization(
            1,
            1,
            vec![LinearColor::new(0.25, 0.25, 0.25)],
            vec![StokesVector::new(1.0, 0.25, 0.0, 0.0)],
        );

        assert_eq!(image.transport_mode, SpectralTransportMode::Polarized);
        assert_eq!(
            image.polarization_pixel(0, 0),
            Some(StokesVector::new(1.0, 0.25, 0.0, 0.0))
        );
    }
}

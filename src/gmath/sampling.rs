//! Generic probability density functions and sampling distributions.
//!
//! These PDFs operate on directions over solid angle, not on scene intersections. The path tracer
//! composes them with ray-specific PDFs in [`crate::graphics::raytracing::pdf`]. Keeping the
//! sphere, cosine, and mixture distributions here makes the sampling math reusable outside the
//! renderer.

use super::{geometry::OrthonormalBasis, random::SampleRng, vector::Vector};

const SPHERE_AREA: f64 = 4.0 * std::f64::consts::PI;
const HG_ISOTROPIC_EPSILON: f64 = 1e-3;
const GGX_MIN_ALPHA: f64 = 1.0e-4;

/// Direction-sampling probability density function over solid angle.
///
/// `value(direction)` must describe the probability density of directions returned by
/// `generate(rng)`. Callers can then use the Monte Carlo estimator `f(direction) / value(direction)`
/// without bias.
pub trait Pdf {
    /// Returns the PDF value for `direction`.
    fn value(&self, direction: Vector) -> f64;

    /// Generates a direction distributed according to this PDF.
    fn generate(&self, rng: &mut SampleRng) -> Vector;
}

/// Uniform distribution over the unit sphere.
///
/// This is useful for isotropic volume scattering and as a simple baseline distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpherePdf;

impl Pdf for SpherePdf {
    fn value(&self, _direction: Vector) -> f64 {
        1.0 / SPHERE_AREA
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        rng.random_unit_vector_spherical()
    }
}

/// Cosine-weighted hemisphere distribution around a surface normal.
///
/// This matches ideal Lambertian diffuse scattering. Directions are generated relative to an
/// [`OrthonormalBasis`] built from the supplied normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosinePdf {
    basis: OrthonormalBasis,
}

impl CosinePdf {
    /// Creates a cosine PDF around `normal`.
    #[must_use]
    pub fn new(normal: Vector) -> Option<Self> {
        OrthonormalBasis::from_w(normal).map(|basis| Self { basis })
    }

    /// Returns the basis used by this PDF.
    #[must_use]
    pub const fn basis(self) -> OrthonormalBasis {
        self.basis
    }
}

impl Pdf for CosinePdf {
    fn value(&self, direction: Vector) -> f64 {
        let direction_length = direction.length();
        if direction_length <= f64::EPSILON {
            return 0.0;
        }

        let cosine_theta = direction.dot(self.basis.w()) / direction_length;
        if cosine_theta <= 0.0 {
            0.0
        } else {
            cosine_theta / std::f64::consts::PI
        }
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        self.basis.local(rng.random_cosine_direction())
    }
}

/// Henyey-Greenstein phase-function distribution around a forward direction.
///
/// Positive `g` values favor forward scattering, negative values favor back scattering, and zero
/// matches a uniform sphere distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HenyeyGreensteinPdf {
    basis: OrthonormalBasis,
    g: f64,
}

impl HenyeyGreensteinPdf {
    /// Creates a Henyey-Greenstein PDF around `forward`.
    ///
    /// Returns `None` if `forward` cannot form a basis.
    ///
    /// # Panics
    ///
    /// Panics if `g` is not finite or is outside `(-1.0, 1.0)`.
    #[must_use]
    pub fn new(forward: Vector, g: f64) -> Option<Self> {
        assert!(
            g.is_finite() && g.abs() < 1.0,
            "Henyey-Greenstein anisotropy must be finite and in (-1, 1)"
        );
        OrthonormalBasis::from_w(forward).map(|basis| Self { basis, g })
    }

    /// Returns the anisotropy parameter.
    #[must_use]
    pub const fn anisotropy(self) -> f64 {
        self.g
    }

    /// Returns the basis used by this PDF.
    #[must_use]
    pub const fn basis(self) -> OrthonormalBasis {
        self.basis
    }

    /// Evaluates the Henyey-Greenstein phase function for a scattering cosine.
    #[must_use]
    pub fn phase_value(cosine_theta: f64, g: f64) -> f64 {
        if !cosine_theta.is_finite() || !g.is_finite() || g.abs() >= 1.0 {
            return 0.0;
        }
        let denominator = 1.0 + g * g - 2.0 * g * cosine_theta.clamp(-1.0, 1.0);
        if denominator <= f64::EPSILON {
            return 0.0;
        }
        (1.0 - g * g) / (SPHERE_AREA * denominator.powf(1.5))
    }
}

impl Pdf for HenyeyGreensteinPdf {
    fn value(&self, direction: Vector) -> f64 {
        let direction_length = direction.length();
        if direction_length <= f64::EPSILON {
            return 0.0;
        }
        let cosine_theta = direction.dot(self.basis.w()) / direction_length;
        Self::phase_value(cosine_theta, self.g)
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        let r1 = rng.random_double();
        let r2 = rng.random_double();
        let cosine_theta = if self.g.abs() < HG_ISOTROPIC_EPSILON {
            1.0 - 2.0 * r1
        } else {
            let term = (1.0 - self.g * self.g) / (1.0 - self.g + 2.0 * self.g * r1);
            ((1.0 + self.g * self.g - term * term) / (2.0 * self.g)).clamp(-1.0, 1.0)
        };
        let sine_theta = (1.0 - cosine_theta * cosine_theta).max(0.0).sqrt();
        let phi = 2.0 * std::f64::consts::PI * r2;
        self.basis.local(Vector::new(
            phi.cos() * sine_theta,
            phi.sin() * sine_theta,
            cosine_theta,
        ))
    }
}

/// GGX/Trowbridge-Reitz microfacet reflection distribution for one outgoing direction.
///
/// This samples half-vectors from the GGX normal distribution function and reflects the outgoing
/// direction about the sampled half-vector. The resulting PDF is over reflected directions in
/// solid angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxReflectionPdf {
    basis: OrthonormalBasis,
    outgoing: Vector,
    roughness: f64,
    alpha: f64,
}

impl GgxReflectionPdf {
    /// Creates a GGX reflection PDF around `normal` for a unit outgoing direction.
    ///
    /// Returns `None` when the normal cannot form a basis or when `outgoing` is below the surface.
    ///
    /// # Panics
    ///
    /// Panics if `roughness` is not finite.
    #[must_use]
    pub fn new(normal: Vector, outgoing: Vector, roughness: f64) -> Option<Self> {
        assert!(roughness.is_finite(), "GGX roughness must be finite");
        let basis = OrthonormalBasis::from_w(normal)?;
        let outgoing = outgoing.normalized();
        if outgoing.length_squared() <= f64::EPSILON || outgoing.dot(basis.w()) <= 0.0 {
            return None;
        }
        let roughness = roughness.clamp(0.0, 1.0);
        let alpha = ggx_alpha_from_roughness(roughness);
        Some(Self {
            basis,
            outgoing,
            roughness,
            alpha,
        })
    }

    /// Returns the basis used by this PDF.
    #[must_use]
    pub const fn basis(self) -> OrthonormalBasis {
        self.basis
    }

    /// Returns the unit outgoing direction.
    #[must_use]
    pub const fn outgoing(self) -> Vector {
        self.outgoing
    }

    /// Returns the perceptual roughness in `0.0..=1.0`.
    #[must_use]
    pub const fn roughness(self) -> f64 {
        self.roughness
    }

    /// Returns the squared roughness used by the GGX distribution.
    #[must_use]
    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    /// Evaluates the GGX/Trowbridge-Reitz normal distribution function.
    #[must_use]
    pub fn normal_distribution(normal_dot_half: f64, roughness: f64) -> f64 {
        if !normal_dot_half.is_finite() || normal_dot_half <= 0.0 || !roughness.is_finite() {
            return 0.0;
        }
        let alpha = ggx_alpha_from_roughness(roughness);
        let alpha2 = alpha * alpha;
        let cos2 = normal_dot_half * normal_dot_half;
        let denominator = cos2 * (alpha2 - 1.0) + 1.0;
        if denominator <= f64::EPSILON {
            return 0.0;
        }
        alpha2 / (std::f64::consts::PI * denominator * denominator)
    }

    /// Evaluates the Smith masking-shadowing factor for GGX.
    #[must_use]
    pub fn smith_masking_shadowing(
        normal_dot_incoming: f64,
        normal_dot_outgoing: f64,
        roughness: f64,
    ) -> f64 {
        smith_g1(normal_dot_incoming, roughness) * smith_g1(normal_dot_outgoing, roughness)
    }

    /// Returns Schlick's Fresnel approximation as a scalar reflectance.
    #[must_use]
    pub fn schlick_fresnel(cosine: f64, f0: f64) -> f64 {
        if !cosine.is_finite() || !f0.is_finite() {
            return 0.0;
        }
        let f0 = f0.clamp(0.0, 1.0);
        f0 + (1.0 - f0) * (1.0 - cosine.clamp(0.0, 1.0)).powi(5)
    }

    fn half_vector_pdf(&self, half: Vector) -> f64 {
        let normal_dot_half = half.dot(self.basis.w());
        Self::normal_distribution(normal_dot_half, self.roughness) * normal_dot_half.max(0.0)
    }
}

impl Pdf for GgxReflectionPdf {
    fn value(&self, direction: Vector) -> f64 {
        let incoming = direction.normalized();
        if incoming.length_squared() <= f64::EPSILON || incoming.dot(self.basis.w()) <= 0.0 {
            return 0.0;
        }
        let half = (incoming + self.outgoing).normalized();
        let outgoing_dot_half = self.outgoing.dot(half);
        if half.length_squared() <= f64::EPSILON || outgoing_dot_half <= f64::EPSILON {
            return 0.0;
        }
        self.half_vector_pdf(half) / (4.0 * outgoing_dot_half.abs())
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        let r1 = rng.random_double();
        let r2 = rng.random_double();
        let phi = 2.0 * std::f64::consts::PI * r1;
        let alpha2 = self.alpha * self.alpha;
        let clamped_r2 = r2.min(1.0 - f64::EPSILON);
        let tan2_theta = alpha2 * clamped_r2 / (1.0 - clamped_r2);
        let cos_theta = 1.0 / (1.0 + tan2_theta).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let half = self.basis.local(Vector::new(
            phi.cos() * sin_theta,
            phi.sin() * sin_theta,
            cos_theta,
        ));
        let reflected = (-self.outgoing).reflected(half.normalized());
        if reflected.dot(self.basis.w()) > 0.0 {
            reflected
        } else {
            self.outgoing
        }
    }
}

fn ggx_alpha_from_roughness(roughness: f64) -> f64 {
    roughness.clamp(0.0, 1.0).powi(2).max(GGX_MIN_ALPHA)
}

fn smith_g1(normal_dot_direction: f64, roughness: f64) -> f64 {
    if !normal_dot_direction.is_finite() || normal_dot_direction <= 0.0 || !roughness.is_finite() {
        return 0.0;
    }
    let alpha = ggx_alpha_from_roughness(roughness);
    let alpha2 = alpha * alpha;
    let cos2 = normal_dot_direction * normal_dot_direction;
    2.0 * normal_dot_direction / (normal_dot_direction + (alpha2 + (1.0 - alpha2) * cos2).sqrt())
}

/// Weighted mixture of two PDFs.
///
/// Mixtures are useful for combining a material PDF with a light/target PDF. `generate` chooses an
/// input distribution using the configured first-PDF weight, while `value` evaluates both inputs
/// and blends their densities for the generated direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixturePdf<P0, P1> {
    first: P0,
    second: P1,
    first_weight: f64,
}

impl<P0, P1> MixturePdf<P0, P1> {
    /// Creates a 50/50 mixture of `first` and `second`.
    #[must_use]
    pub const fn new(first: P0, second: P1) -> Self {
        Self {
            first,
            second,
            first_weight: 0.5,
        }
    }

    /// Creates a weighted mixture of `first` and `second`.
    ///
    /// `first_weight` is clamped to `0.0..=1.0`; the second PDF receives the remaining weight.
    ///
    /// # Panics
    ///
    /// Panics if `first_weight` is not finite.
    #[must_use]
    pub fn weighted(first: P0, second: P1, first_weight: f64) -> Self {
        assert!(
            first_weight.is_finite(),
            "mixture PDF weight must be finite"
        );
        Self {
            first,
            second,
            first_weight: first_weight.clamp(0.0, 1.0),
        }
    }

    /// Returns the first PDF in the mixture.
    #[must_use]
    pub const fn first(&self) -> &P0 {
        &self.first
    }

    /// Returns the second PDF in the mixture.
    #[must_use]
    pub const fn second(&self) -> &P1 {
        &self.second
    }

    /// Returns the probability weight assigned to the first PDF.
    #[must_use]
    pub const fn first_weight(&self) -> f64 {
        self.first_weight
    }

    /// Returns the probability weight assigned to the second PDF.
    #[must_use]
    pub fn second_weight(&self) -> f64 {
        1.0 - self.first_weight
    }
}

impl<P0: Pdf, P1: Pdf> Pdf for MixturePdf<P0, P1> {
    fn value(&self, direction: Vector) -> f64 {
        self.first_weight * self.first.value(direction)
            + self.second_weight() * self.second.value(direction)
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        if rng.random_double() < self.first_weight {
            self.first.generate(rng)
        } else {
            self.second.generate(rng)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    fn assert_within(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    #[test]
    fn sphere_pdf_is_uniform_over_solid_angle() {
        let pdf = SpherePdf;
        assert_close(
            pdf.value(Vector::new(1.0, 0.0, 0.0)),
            1.0 / (4.0 * std::f64::consts::PI),
        );

        let mut rng = SampleRng::new(11);
        assert_close(pdf.generate(&mut rng).length(), 1.0);
    }

    #[test]
    fn sphere_pdf_samples_average_near_zero() {
        let pdf = SpherePdf;
        let mut rng = SampleRng::new(101);
        let mut sum = Vector::default();
        let samples = 4096;

        for _ in 0..samples {
            let sample = pdf.generate(&mut rng);
            assert_within(sample.length(), 1.0, 1e-12);
            sum += sample;
        }

        let mean = sum / f64::from(samples);
        assert_within(mean.x(), 0.0, 0.025);
        assert_within(mean.y(), 0.0, 0.025);
        assert_within(mean.z(), 0.0, 0.025);
    }

    #[test]
    fn cosine_pdf_weights_directions_by_normal_alignment() {
        let pdf = CosinePdf::new(Vector::new(0.0, 1.0, 0.0)).expect("normal should create basis");

        assert_close(
            pdf.value(Vector::new(0.0, 1.0, 0.0)),
            1.0 / std::f64::consts::PI,
        );
        assert_close(pdf.value(Vector::new(1.0, 0.0, 0.0)), 0.0);

        let mut rng = SampleRng::new(13);
        assert!(pdf.generate(&mut rng).dot(Vector::new(0.0, 1.0, 0.0)) >= 0.0);
    }

    #[test]
    fn cosine_pdf_samples_positive_hemisphere_with_expected_mean_cosine() {
        let normal = Vector::new(0.0, 1.0, 0.0);
        let pdf = CosinePdf::new(normal).expect("normal should create basis");
        let mut rng = SampleRng::new(103);
        let mut cosine_sum = 0.0;
        let samples = 4096;

        for _ in 0..samples {
            let sample = pdf.generate(&mut rng);
            let cosine = sample.dot(normal);
            assert!(cosine >= -1e-12);
            cosine_sum += cosine;
        }

        assert_within(cosine_sum / f64::from(samples), 2.0 / 3.0, 0.025);
    }

    #[test]
    fn henyey_greenstein_pdf_biases_forward_or_backward() {
        let forward = Vector::new(0.0, 0.0, 1.0);
        let forward_pdf = HenyeyGreensteinPdf::new(forward, 0.6).expect("basis should be valid");
        let backward_pdf = HenyeyGreensteinPdf::new(forward, -0.6).expect("basis should be valid");

        assert_close(
            HenyeyGreensteinPdf::phase_value(0.25, 0.0),
            1.0 / (4.0 * std::f64::consts::PI),
        );
        assert!(forward_pdf.value(forward) > forward_pdf.value(-forward));
        assert!(backward_pdf.value(-forward) > backward_pdf.value(forward));
    }

    #[test]
    fn henyey_greenstein_pdf_samples_mean_cosine_near_anisotropy() {
        let forward = Vector::new(0.0, 0.0, 1.0);
        let g = 0.55;
        let pdf = HenyeyGreensteinPdf::new(forward, g).expect("basis should be valid");
        let mut rng = SampleRng::new(107);
        let samples = 8192;
        let mut cosine_sum = 0.0;

        for _ in 0..samples {
            let sample = pdf.generate(&mut rng);
            assert_within(sample.length(), 1.0, 1e-12);
            cosine_sum += sample.dot(forward);
        }

        assert_within(cosine_sum / f64::from(samples), g, 0.035);
    }

    #[test]
    fn ggx_reflection_pdf_evaluates_normal_reflection() {
        let normal = Vector::new(0.0, 0.0, 1.0);
        let pdf = GgxReflectionPdf::new(normal, normal, 1.0).expect("basis should be valid");

        assert_close(pdf.value(normal), 1.0 / (4.0 * std::f64::consts::PI));
        assert_close(pdf.value(Vector::new(0.0, 0.0, -1.0)), 0.0);
    }

    #[test]
    fn ggx_distribution_smith_and_schlick_are_well_behaved() {
        assert!(
            GgxReflectionPdf::normal_distribution(1.0, 0.2)
                > GgxReflectionPdf::normal_distribution(1.0, 0.8)
        );
        assert_close(
            GgxReflectionPdf::smith_masking_shadowing(1.0, 1.0, 0.5),
            1.0,
        );
        assert_close(GgxReflectionPdf::schlick_fresnel(1.0, 0.04), 0.04);
        assert_close(GgxReflectionPdf::schlick_fresnel(0.0, 0.04), 1.0);
    }

    #[test]
    fn ggx_reflection_pdf_samples_above_surface() {
        let normal = Vector::new(0.0, 0.0, 1.0);
        let pdf = GgxReflectionPdf::new(normal, normal, 0.45).expect("basis should be valid");
        let mut rng = SampleRng::new(109);

        for _ in 0..64 {
            let sample = pdf.generate(&mut rng);
            assert_within(sample.length(), 1.0, 1e-12);
            assert!(sample.dot(normal) > 0.0);
            assert!(pdf.value(sample).is_finite());
            assert!(pdf.value(sample) > 0.0);
        }
    }

    #[test]
    fn mixture_pdf_averages_values_and_generates_from_inputs() {
        let sphere = SpherePdf;
        let cosine =
            CosinePdf::new(Vector::new(0.0, 1.0, 0.0)).expect("normal should create basis");
        let pdf = MixturePdf::new(sphere, cosine);

        assert_close(
            pdf.value(Vector::new(0.0, 1.0, 0.0)),
            0.5 * (1.0 / (4.0 * std::f64::consts::PI)) + 0.5 * (1.0 / std::f64::consts::PI),
        );

        let mut rng = SampleRng::new(17);
        assert!(pdf.generate(&mut rng).length_squared() > 0.0);
    }

    #[test]
    fn mixture_pdf_supports_weighted_inputs() {
        let sphere = SpherePdf;
        let cosine =
            CosinePdf::new(Vector::new(0.0, 1.0, 0.0)).expect("normal should create basis");
        let pdf = MixturePdf::weighted(sphere, cosine, 0.25);

        assert_close(pdf.first_weight(), 0.25);
        assert_close(pdf.second_weight(), 0.75);
        assert_close(
            pdf.value(Vector::new(0.0, 1.0, 0.0)),
            0.25 * (1.0 / (4.0 * std::f64::consts::PI)) + 0.75 * (1.0 / std::f64::consts::PI),
        );
    }

    #[test]
    #[should_panic(expected = "mixture PDF weight must be finite")]
    fn mixture_pdf_rejects_non_finite_weight() {
        let _ = MixturePdf::weighted(SpherePdf, SpherePdf, f64::NAN);
    }
}

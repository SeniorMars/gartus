//! Generic probability density functions and sampling distributions.
//!
//! These PDFs operate on directions over solid angle, not on scene intersections. The path tracer
//! composes them with ray-specific PDFs in [`crate::graphics::raytracing::pdf`]. Keeping the
//! sphere, cosine, and mixture distributions here makes the sampling math reusable outside the
//! renderer.

use super::{geometry::OrthonormalBasis, random::SampleRng, vector::Vector};

const SPHERE_AREA: f64 = 4.0 * std::f64::consts::PI;

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

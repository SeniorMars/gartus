//! Generic probability density functions and sampling distributions.

use super::{geometry::OrthonormalBasis, random::SampleRng, vector::Vector};

const SPHERE_AREA: f64 = 4.0 * std::f64::consts::PI;

/// Direction-sampling probability density function over solid angle.
pub trait Pdf {
    /// Returns the PDF value for `direction`.
    fn value(&self, direction: Vector) -> f64;

    /// Generates a direction distributed according to this PDF.
    fn generate(&self, rng: &mut SampleRng) -> Vector;
}

/// Uniform distribution over the unit sphere.
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

/// Equal-weight mixture of two PDFs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixturePdf<P0, P1> {
    first: P0,
    second: P1,
}

impl<P0, P1> MixturePdf<P0, P1> {
    /// Creates a 50/50 mixture of `first` and `second`.
    #[must_use]
    pub const fn new(first: P0, second: P1) -> Self {
        Self { first, second }
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
}

impl<P0: Pdf, P1: Pdf> Pdf for MixturePdf<P0, P1> {
    fn value(&self, direction: Vector) -> f64 {
        0.5 * self.first.value(direction) + 0.5 * self.second.value(direction)
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        if rng.random_double() < 0.5 {
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
}

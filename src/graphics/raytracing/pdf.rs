//! Probability density functions for importance-sampled path tracing.

use super::{Hittable, PI};
use crate::gmath::{geometry::OrthonormalBasis, random::SampleRng, vector::Point, vector::Vector};
use std::fmt;

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
        1.0 / (4.0 * PI)
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
        let cosine_theta = direction.normalized().dot(self.basis.w());
        if cosine_theta <= 0.0 {
            0.0
        } else {
            cosine_theta / PI
        }
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        self.basis.local(rng.random_cosine_direction())
    }
}

/// PDF that samples directions toward a hittable object from an origin.
#[derive(Clone, Copy)]
pub struct HittablePdf<'a> {
    object: &'a dyn Hittable,
    origin: Point,
}

impl fmt::Debug for HittablePdf<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HittablePdf")
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

impl<'a> HittablePdf<'a> {
    /// Creates a PDF over directions from `origin` toward `object`.
    #[must_use]
    pub const fn new(object: &'a dyn Hittable, origin: Point) -> Self {
        Self { object, origin }
    }
}

impl Pdf for HittablePdf<'_> {
    fn value(&self, direction: Vector) -> f64 {
        self.object.pdf_value(self.origin, direction)
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        self.object.random_direction(self.origin, rng)
    }
}

/// Equal-weight mixture of two PDFs.
#[derive(Clone, Copy)]
pub struct MixturePdf<'a> {
    pdfs: [&'a dyn Pdf; 2],
}

impl fmt::Debug for MixturePdf<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("MixturePdf").finish_non_exhaustive()
    }
}

impl<'a> MixturePdf<'a> {
    /// Creates a 50/50 mixture of `first` and `second`.
    #[must_use]
    pub const fn new(first: &'a dyn Pdf, second: &'a dyn Pdf) -> Self {
        Self {
            pdfs: [first, second],
        }
    }
}

impl Pdf for MixturePdf<'_> {
    fn value(&self, direction: Vector) -> f64 {
        0.5 * self.pdfs[0].value(direction) + 0.5 * self.pdfs[1].value(direction)
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        if rng.random_double() < 0.5 {
            self.pdfs[0].generate(rng)
        } else {
            self.pdfs[1].generate(rng)
        }
    }
}

/// Material-side PDFs stored in scatter records without heap allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialPdf {
    /// Uniform sphere sampling.
    Sphere(SpherePdf),
    /// Cosine-weighted hemisphere sampling.
    Cosine(CosinePdf),
}

impl Pdf for MaterialPdf {
    fn value(&self, direction: Vector) -> f64 {
        match self {
            Self::Sphere(pdf) => pdf.value(direction),
            Self::Cosine(pdf) => pdf.value(direction),
        }
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        match self {
            Self::Sphere(pdf) => pdf.generate(rng),
            Self::Cosine(pdf) => pdf.generate(rng),
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
        assert_close(pdf.value(Vector::new(1.0, 0.0, 0.0)), 1.0 / (4.0 * PI));

        let mut rng = SampleRng::new(11);
        assert_close(pdf.generate(&mut rng).length(), 1.0);
    }

    #[test]
    fn cosine_pdf_weights_directions_by_normal_alignment() {
        let pdf = CosinePdf::new(Vector::new(0.0, 1.0, 0.0)).expect("normal should create basis");

        assert_close(pdf.value(Vector::new(0.0, 1.0, 0.0)), 1.0 / PI);
        assert_close(pdf.value(Vector::new(1.0, 0.0, 0.0)), 0.0);

        let mut rng = SampleRng::new(13);
        assert!(pdf.generate(&mut rng).dot(Vector::new(0.0, 1.0, 0.0)) >= 0.0);
    }
}

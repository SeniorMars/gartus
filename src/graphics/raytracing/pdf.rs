//! Probability density functions for importance-sampled path tracing.

use super::{Hittable, PdfContext};
pub use crate::gmath::sampling::{CosinePdf, MixturePdf, Pdf, SpherePdf};
use crate::gmath::{random::SampleRng, vector::Vector};
use std::fmt;

/// PDF that samples directions toward a hittable object from an origin.
#[derive(Clone, Copy)]
pub struct HittablePdf<'a> {
    object: &'a dyn Hittable,
    context: PdfContext,
}

impl fmt::Debug for HittablePdf<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HittablePdf")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

impl<'a> HittablePdf<'a> {
    /// Creates a PDF over directions from the context origin toward `object`.
    #[must_use]
    pub const fn new(object: &'a dyn Hittable, context: PdfContext) -> Self {
        Self { object, context }
    }
}

impl Pdf for HittablePdf<'_> {
    fn value(&self, direction: Vector) -> f64 {
        self.object.pdf_value(self.context, direction)
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        self.object.random_direction(self.context, rng)
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

//! Probability density functions for importance-sampled path tracing.

use super::{Hittable, PdfContext};
pub use crate::gmath::sampling::{
    CosinePdf, GgxReflectionPdf, HenyeyGreensteinPdf, MixturePdf, Pdf, SpherePdf,
};
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
    /// Henyey-Greenstein volume phase-function sampling.
    HenyeyGreenstein(HenyeyGreensteinPdf),
    /// GGX/Trowbridge-Reitz microfacet reflection sampling.
    GgxReflection(GgxReflectionPdf),
    /// Mixture of diffuse cosine and GGX reflection sampling.
    DiffuseGgx {
        /// Diffuse cosine-weighted PDF.
        diffuse: CosinePdf,
        /// GGX reflection PDF.
        specular: GgxReflectionPdf,
        /// Probability of sampling the specular lobe.
        specular_weight: f64,
    },
}

impl Pdf for MaterialPdf {
    fn value(&self, direction: Vector) -> f64 {
        match self {
            Self::Sphere(pdf) => pdf.value(direction),
            Self::Cosine(pdf) => pdf.value(direction),
            Self::HenyeyGreenstein(pdf) => pdf.value(direction),
            Self::GgxReflection(pdf) => pdf.value(direction),
            Self::DiffuseGgx {
                diffuse,
                specular,
                specular_weight,
            } => {
                let specular_weight = (*specular_weight).clamp(0.0, 1.0);
                (1.0 - specular_weight) * diffuse.value(direction)
                    + specular_weight * specular.value(direction)
            }
        }
    }

    fn generate(&self, rng: &mut SampleRng) -> Vector {
        match self {
            Self::Sphere(pdf) => pdf.generate(rng),
            Self::Cosine(pdf) => pdf.generate(rng),
            Self::HenyeyGreenstein(pdf) => pdf.generate(rng),
            Self::GgxReflection(pdf) => pdf.generate(rng),
            Self::DiffuseGgx {
                diffuse,
                specular,
                specular_weight,
            } => {
                if rng.random_double() < (*specular_weight).clamp(0.0, 1.0) {
                    specular.generate(rng)
                } else {
                    diffuse.generate(rng)
                }
            }
        }
    }
}

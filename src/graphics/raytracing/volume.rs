//! Participating media for path tracing.

use super::{
    Aabb, HitRecord, Hittable, INFINITY, Interval, LinearColor, MaterialRef, material::Isotropic,
};
use crate::{
    gmath::{random::SampleRng, ray::Ray, vector::Vector},
    graphics::texture::SurfaceTexture,
};
use std::{fmt, sync::Arc};

/// A constant-density participating medium bounded by another hittable object.
pub struct ConstantMedium {
    boundary: Box<dyn Hittable>,
    neg_inv_density: f64,
    phase_function: MaterialRef,
    bounds: Option<Aabb>,
}

impl fmt::Debug for ConstantMedium {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConstantMedium")
            .field("neg_inv_density", &self.neg_inv_density)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl ConstantMedium {
    /// Creates a constant-density medium with constant particle color.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn new(boundary: impl Hittable + 'static, density: f64, albedo: LinearColor) -> Self {
        Self::with_phase_function(boundary, density, Arc::new(Isotropic::new(albedo)))
    }

    /// Creates a constant-density medium with a texture-backed phase function.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn from_texture(
        boundary: impl Hittable + 'static,
        density: f64,
        texture: impl SurfaceTexture + 'static,
    ) -> Self {
        Self::with_phase_function(
            boundary,
            density,
            Arc::new(Isotropic::from_texture(texture)),
        )
    }

    /// Creates a constant-density medium from a boxed boundary and constant particle color.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn from_box(boundary: Box<dyn Hittable>, density: f64, albedo: LinearColor) -> Self {
        Self::from_box_with_phase_function(boundary, density, Arc::new(Isotropic::new(albedo)))
    }

    /// Creates a constant-density medium with an explicit phase-function material.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn with_phase_function(
        boundary: impl Hittable + 'static,
        density: f64,
        phase_function: MaterialRef,
    ) -> Self {
        Self::from_box_with_phase_function(Box::new(boundary), density, phase_function)
    }

    /// Creates a constant-density medium from boxed parts.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn from_box_with_phase_function(
        boundary: Box<dyn Hittable>,
        density: f64,
        phase_function: MaterialRef,
    ) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "medium density must be positive and finite"
        );
        let bounds = boundary.bounding_box();
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_function,
            bounds,
        }
    }

    /// Returns the medium boundary bounds.
    #[must_use]
    pub const fn bounds(&self) -> Option<Aabb> {
        self.bounds
    }
}

impl Hittable for ConstantMedium {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let entry_t = self
            .boundary
            .hit_with_rng(ray, Interval::UNIVERSE, rng)?
            .t
            .max(ray_t.min);
        let exit_t = self
            .boundary
            .hit_with_rng(ray, Interval::new(entry_t + 0.0001, INFINITY), rng)?
            .t
            .min(ray_t.max);

        if entry_t >= exit_t {
            return None;
        }

        let entry_t = entry_t.max(0.0);
        let ray_length = ray.direction().length();
        if ray_length <= f64::EPSILON {
            return None;
        }

        let distance_inside_boundary = (exit_t - entry_t) * ray_length;
        let sample = rng.random_double().max(f64::MIN_POSITIVE);
        let hit_distance = self.neg_inv_density * sample.ln();
        if hit_distance > distance_inside_boundary {
            return None;
        }

        let t = entry_t + hit_distance / ray_length;
        let normal = Vector::new(1.0, 0.0, 0.0);
        Some(HitRecord {
            point: ray.at(t),
            normal,
            geometric_normal: normal,
            shading_normal: normal,
            t,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: self.phase_function.as_ref(),
        })
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

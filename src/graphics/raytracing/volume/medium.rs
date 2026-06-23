use super::super::{
    Aabb, HitRecord, Hittable, INFINITY, Interval, LinearColor, Material, MaterialRef,
    material::Isotropic,
};
use super::field::{DensityField, DensityFieldRef};
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

struct MediumInterval {
    entry_t: f64,
    ray_length: f64,
    distance_inside_boundary: f64,
}

fn boundary_interval(
    boundary: &dyn Hittable,
    ray: &Ray,
    ray_t: Interval,
    rng: &mut SampleRng,
) -> Option<MediumInterval> {
    let entry_t = boundary
        .hit_with_rng(ray, Interval::UNIVERSE, rng)?
        .t
        .max(ray_t.min);
    let exit_t = boundary
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

    Some(MediumInterval {
        entry_t,
        ray_length,
        distance_inside_boundary: (exit_t - entry_t) * ray_length,
    })
}

fn medium_hit_record<'a>(ray: &Ray, t: f64, material: &'a dyn Material) -> HitRecord<'a> {
    let normal = Vector::new(1.0, 0.0, 0.0);
    HitRecord {
        point: ray.at(t),
        normal,
        geometric_normal: normal,
        shading_normal: normal,
        tangent: None,
        bitangent: None,
        tangent_handedness: 1.0,
        t,
        u: 0.0,
        v: 0.0,
        front_face: true,
        material,
    }
}

impl Hittable for ConstantMedium {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let interval = boundary_interval(self.boundary.as_ref(), ray, ray_t, rng)?;
        let sample = rng.random_double().max(f64::MIN_POSITIVE);
        let hit_distance = self.neg_inv_density * sample.ln();
        if hit_distance > interval.distance_inside_boundary {
            return None;
        }

        let t = interval.entry_t + hit_distance / interval.ray_length;
        Some(medium_hit_record(ray, t, self.phase_function.as_ref()))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

/// A spatially varying participating medium bounded by another hittable object.
///
/// Scattering is sampled with Woodcock tracking against the density field's maximum density. This
/// supports procedural fog, smoke, cloud, and nebula volumes without tessellating the interior.
pub struct NonUniformMedium<D = DensityFieldRef> {
    boundary: Box<dyn Hittable>,
    density_field: D,
    max_density: f64,
    phase_function: MaterialRef,
    bounds: Option<Aabb>,
}

impl<D> fmt::Debug for NonUniformMedium<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NonUniformMedium")
            .field("max_density", &self.max_density)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl<D: DensityField> NonUniformMedium<D> {
    /// Creates a non-uniform medium with constant particle color.
    ///
    /// # Panics
    ///
    /// Panics if `density_field.max_density()` is not positive and finite.
    #[must_use]
    pub fn new(boundary: impl Hittable + 'static, density_field: D, albedo: LinearColor) -> Self {
        Self::with_phase_function(boundary, density_field, Arc::new(Isotropic::new(albedo)))
    }

    /// Creates a non-uniform medium with a texture-backed phase function.
    ///
    /// # Panics
    ///
    /// Panics if `density_field.max_density()` is not positive and finite.
    #[must_use]
    pub fn from_texture(
        boundary: impl Hittable + 'static,
        density_field: D,
        texture: impl SurfaceTexture + 'static,
    ) -> Self {
        Self::with_phase_function(
            boundary,
            density_field,
            Arc::new(Isotropic::from_texture(texture)),
        )
    }

    /// Creates a non-uniform medium from a boxed boundary and constant particle color.
    ///
    /// # Panics
    ///
    /// Panics if `density_field.max_density()` is not positive and finite.
    #[must_use]
    pub fn from_box(boundary: Box<dyn Hittable>, density_field: D, albedo: LinearColor) -> Self {
        Self::from_box_with_phase_function(
            boundary,
            density_field,
            Arc::new(Isotropic::new(albedo)),
        )
    }

    /// Creates a non-uniform medium with an explicit phase-function material.
    ///
    /// # Panics
    ///
    /// Panics if `density_field.max_density()` is not positive and finite.
    #[must_use]
    pub fn with_phase_function(
        boundary: impl Hittable + 'static,
        density_field: D,
        phase_function: MaterialRef,
    ) -> Self {
        Self::from_box_with_phase_function(Box::new(boundary), density_field, phase_function)
    }

    /// Creates a non-uniform medium from boxed parts.
    ///
    /// # Panics
    ///
    /// Panics if `density_field.max_density()` is not positive and finite.
    #[must_use]
    pub fn from_box_with_phase_function(
        boundary: Box<dyn Hittable>,
        density_field: D,
        phase_function: MaterialRef,
    ) -> Self {
        let max_density = density_field.max_density();
        assert!(
            max_density.is_finite() && max_density > 0.0,
            "density field maximum must be positive and finite"
        );
        let bounds = boundary.bounding_box();
        Self {
            boundary,
            density_field,
            max_density,
            phase_function,
            bounds,
        }
    }

    /// Returns the medium boundary bounds.
    #[must_use]
    pub const fn bounds(&self) -> Option<Aabb> {
        self.bounds
    }

    /// Returns the density field.
    #[must_use]
    pub const fn density_field(&self) -> &D {
        &self.density_field
    }

    /// Returns the density majorant used for Woodcock tracking.
    #[must_use]
    pub const fn max_density(&self) -> f64 {
        self.max_density
    }

    fn clamped_density(&self, point: crate::gmath::vector::Point, time: f64) -> f64 {
        let density = self.density_field.density(point, time);
        if density.is_finite() {
            density.clamp(0.0, self.max_density)
        } else {
            0.0
        }
    }
}

impl NonUniformMedium<DensityFieldRef> {
    /// Creates a non-uniform medium from shared boxed density field data.
    ///
    /// # Panics
    ///
    /// Panics if `density_field.max_density()` is not positive and finite.
    #[must_use]
    pub fn from_shared_density_field(
        boundary: impl Hittable + 'static,
        density_field: DensityFieldRef,
        albedo: LinearColor,
    ) -> Self {
        Self::new(boundary, density_field, albedo)
    }
}

impl<D: DensityField> Hittable for NonUniformMedium<D> {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let interval = boundary_interval(self.boundary.as_ref(), ray, ray_t, rng)?;
        let mut distance = 0.0;

        loop {
            let sample = rng.random_double().max(f64::MIN_POSITIVE);
            distance += -sample.ln() / self.max_density;
            if distance > interval.distance_inside_boundary {
                return None;
            }

            let t = interval.entry_t + distance / interval.ray_length;
            let density = self.clamped_density(ray.at(t), ray.time());
            let accept_probability = density / self.max_density;
            if accept_probability.total_cmp(&rng.random_double()).is_gt() {
                return Some(medium_hit_record(ray, t, self.phase_function.as_ref()));
            }
        }
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

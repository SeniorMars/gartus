//! Participating media for path tracing.

use super::{
    Aabb, HitRecord, Hittable, INFINITY, Interval, LinearColor, MaterialRef, material::Isotropic,
};
use crate::{
    gmath::{
        random::SampleRng,
        ray::Ray,
        vector::{Point, Vector},
    },
    graphics::texture::SurfaceTexture,
};
use std::{fmt, sync::Arc};

/// Spatially varying density used by non-uniform participating media.
///
/// `density` returns the local extinction density at a world-space point and ray time. Values at
/// or below zero are treated as empty space by [`NonUniformMedium`]. `max_density` is the majorant
/// used for Woodcock tracking, so it must be greater than or equal to the maximum density the field
/// can return over the medium bounds.
pub trait DensityField: Send + Sync {
    /// Returns the local density at `point` for `time`.
    fn density(&self, point: Point, time: f64) -> f64;

    /// Returns a positive finite upper bound for [`Self::density`].
    fn max_density(&self) -> f64;
}

impl<T: DensityField + ?Sized> DensityField for Arc<T> {
    fn density(&self, point: Point, time: f64) -> f64 {
        (**self).density(point, time)
    }

    fn max_density(&self) -> f64 {
        (**self).max_density()
    }
}

/// Shared density-field handle.
pub type DensityFieldRef = Arc<dyn DensityField>;

/// Constant density field usable with [`NonUniformMedium`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantDensity {
    density: f64,
}

impl ConstantDensity {
    /// Creates a constant density field.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn new(density: f64) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "density field maximum must be positive and finite"
        );
        Self { density }
    }

    /// Returns the stored density.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.density
    }
}

impl DensityField for ConstantDensity {
    fn density(&self, _point: Point, _time: f64) -> f64 {
        self.density
    }

    fn max_density(&self) -> f64 {
        self.density
    }
}

/// Closure-backed density field with an explicit majorant.
pub struct FnDensityField<F> {
    density_fn: F,
    max_density: f64,
}

impl<F> fmt::Debug for FnDensityField<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnDensityField")
            .field("max_density", &self.max_density)
            .finish_non_exhaustive()
    }
}

impl<F> FnDensityField<F> {
    /// Creates a density field from a closure and explicit maximum density.
    ///
    /// # Panics
    ///
    /// Panics if `max_density` is not positive and finite.
    #[must_use]
    pub fn new(max_density: f64, density_fn: F) -> Self {
        assert!(
            max_density.is_finite() && max_density > 0.0,
            "density field maximum must be positive and finite"
        );
        Self {
            density_fn,
            max_density,
        }
    }

    /// Returns the explicit maximum density.
    #[must_use]
    pub const fn maximum_density(&self) -> f64 {
        self.max_density
    }
}

impl<F> DensityField for FnDensityField<F>
where
    F: Fn(Point, f64) -> f64 + Send + Sync,
{
    fn density(&self, point: Point, time: f64) -> f64 {
        (self.density_fn)(point, time)
    }

    fn max_density(&self) -> f64 {
        self.max_density
    }
}

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

fn medium_hit_record<'a>(ray: &Ray, t: f64, material: &'a dyn super::Material) -> HitRecord<'a> {
    let normal = Vector::new(1.0, 0.0, 0.0);
    HitRecord {
        point: ray.at(t),
        normal,
        geometric_normal: normal,
        shading_normal: normal,
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

    fn clamped_density(&self, point: Point, time: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::raytracing::Sphere;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn function_density_field_reports_explicit_majorant() {
        let field = FnDensityField::new(3.5, |point: Point, time| point.x() + time);

        assert_close(field.maximum_density(), 3.5);
        assert_close(field.max_density(), 3.5);
        assert_close(field.density(Point::new(2.0, 0.0, 0.0), 0.25), 2.25);
    }

    #[test]
    fn non_uniform_medium_rejects_empty_density() {
        let boundary = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let field = FnDensityField::new(1.0, |_point: Point, _time| 0.0);
        let medium = NonUniformMedium::new(boundary, field, LinearColor::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(47);

        assert!(
            medium
                .hit_with_rng(&ray, Interval::new(0.0, INFINITY), &mut rng)
                .is_none()
        );
        assert!(medium.bounding_box().is_some());
    }

    #[test]
    fn non_uniform_medium_samples_dense_region() {
        let boundary = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let field = FnDensityField::new(16.0, |point: Point, time| {
            if point.z() < -0.75 && time > 0.5 {
                16.0
            } else {
                0.0
            }
        });
        let medium = NonUniformMedium::new(boundary, field, LinearColor::new(1.0, 1.0, 1.0));
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.75);
        let mut rng = SampleRng::new(47);

        let record = medium
            .hit_with_rng(&ray, Interval::new(0.0, INFINITY), &mut rng)
            .expect("dense region should scatter");

        assert!(record.t > 0.75);
        assert!(record.t < 1.5);
        assert_eq!(record.normal, Vector::new(1.0, 0.0, 0.0));
        assert!(record.front_face);
    }
}

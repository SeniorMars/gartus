//! Signed-distance-field ray marching for path-traced scenes.

use super::{Aabb, HitRecord, Hittable, Interval, Material, MaterialRef};
use crate::gmath::{
    ray::Ray,
    vector::{Point, Vector},
};
use std::{fmt, sync::Arc};

/// Signed distance field that can be sphere-traced as a ray-tracing object.
///
/// `distance` should return an approximate signed distance in world units, where negative values
/// mean "inside" and positive values mean "outside". `bounds` must enclose the visible field.
pub trait DistanceField: Send + Sync {
    /// Returns the signed distance from `point` to the field surface.
    fn distance(&self, point: Point) -> f64;

    /// Returns world-space bounds enclosing the field.
    fn bounds(&self) -> Aabb;
}

impl<T: DistanceField + ?Sized> DistanceField for Arc<T> {
    fn distance(&self, point: Point) -> f64 {
        (**self).distance(point)
    }

    fn bounds(&self) -> Aabb {
        (**self).bounds()
    }
}

/// Shared distance-field handle.
pub type DistanceFieldRef = Arc<dyn DistanceField>;

/// Closure-backed distance field with explicit bounds.
pub struct FnDistanceField<F> {
    distance_fn: F,
    bounds: Aabb,
}

impl<F> fmt::Debug for FnDistanceField<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnDistanceField")
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl<F> FnDistanceField<F> {
    /// Creates a distance field from a closure and explicit world-space bounds.
    #[must_use]
    pub const fn new(bounds: Aabb, distance_fn: F) -> Self {
        Self {
            distance_fn,
            bounds,
        }
    }
}

impl<F> DistanceField for FnDistanceField<F>
where
    F: Fn(Point) -> f64 + Send + Sync,
{
    fn distance(&self, point: Point) -> f64 {
        (self.distance_fn)(point)
    }

    fn bounds(&self) -> Aabb {
        self.bounds
    }
}

/// Ray-marched signed distance field with an existing ray-tracing material.
pub struct SdfObject<D = DistanceFieldRef> {
    field: D,
    material: MaterialRef,
    bounds: Aabb,
    epsilon: f64,
    normal_epsilon: f64,
    max_steps: usize,
}

impl<D> fmt::Debug for SdfObject<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SdfObject")
            .field("bounds", &self.bounds)
            .field("epsilon", &self.epsilon)
            .field("normal_epsilon", &self.normal_epsilon)
            .field("max_steps", &self.max_steps)
            .finish_non_exhaustive()
    }
}

impl<D: DistanceField> SdfObject<D> {
    /// Default surface hit tolerance in world units.
    pub const DEFAULT_EPSILON: f64 = 0.001;

    /// Default normal-estimation delta in world units.
    pub const DEFAULT_NORMAL_EPSILON: f64 = 0.0005;

    /// Default maximum number of ray-marching steps.
    pub const DEFAULT_MAX_STEPS: usize = 192;

    /// Creates an SDF object with a concrete material.
    #[must_use]
    pub fn new(field: D, material: impl Material + 'static) -> Self {
        Self::with_shared_material(field, Arc::new(material))
    }

    /// Creates an SDF object with a shared material handle.
    #[must_use]
    pub fn with_shared_material(field: D, material: MaterialRef) -> Self {
        let bounds = field.bounds();
        Self {
            field,
            material,
            bounds,
            epsilon: Self::DEFAULT_EPSILON,
            normal_epsilon: Self::DEFAULT_NORMAL_EPSILON,
            max_steps: Self::DEFAULT_MAX_STEPS,
        }
    }

    /// Returns the distance field.
    #[must_use]
    pub const fn field(&self) -> &D {
        &self.field
    }

    /// Returns the object material.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }

    /// Returns the ray-marching surface hit tolerance.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the finite-difference normal-estimation delta.
    #[must_use]
    pub const fn normal_epsilon(&self) -> f64 {
        self.normal_epsilon
    }

    /// Returns the maximum number of ray-marching steps.
    #[must_use]
    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Sets the ray-marching surface hit tolerance.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is not positive and finite.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0,
            "SDF epsilon must be positive and finite"
        );
        self.epsilon = epsilon;
        self
    }

    /// Sets the finite-difference normal-estimation delta.
    ///
    /// # Panics
    ///
    /// Panics if `normal_epsilon` is not positive and finite.
    #[must_use]
    pub fn with_normal_epsilon(mut self, normal_epsilon: f64) -> Self {
        assert!(
            normal_epsilon.is_finite() && normal_epsilon > 0.0,
            "SDF normal epsilon must be positive and finite"
        );
        self.normal_epsilon = normal_epsilon;
        self
    }

    /// Sets the maximum number of ray-marching steps.
    ///
    /// # Panics
    ///
    /// Panics if `max_steps` is zero.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        assert!(max_steps > 0, "SDF max steps must be greater than zero");
        self.max_steps = max_steps;
        self
    }

    fn estimate_normal(&self, point: Point, fallback: Vector) -> Vector {
        let e = self.normal_epsilon;
        let dx = Vector::new(e, 0.0, 0.0);
        let dy = Vector::new(0.0, e, 0.0);
        let dz = Vector::new(0.0, 0.0, e);
        let normal = Vector::new(
            self.field.distance(point + dx) - self.field.distance(point - dx),
            self.field.distance(point + dy) - self.field.distance(point - dy),
            self.field.distance(point + dz) - self.field.distance(point - dz),
        )
        .normalized();

        if normal.length_squared() <= f64::EPSILON {
            fallback
        } else {
            normal
        }
    }
}

impl SdfObject<DistanceFieldRef> {
    /// Creates an SDF object from shared boxed distance field data.
    #[must_use]
    pub fn from_shared_distance_field(
        field: DistanceFieldRef,
        material: impl Material + 'static,
    ) -> Self {
        Self::new(field, material)
    }
}

impl<D: DistanceField> Hittable for SdfObject<D> {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut crate::gmath::random::SampleRng,
    ) -> Option<HitRecord<'_>> {
        let interval = ray_bounds_interval(ray, self.bounds, ray_t)?;
        let mut t = interval.min.max(0.0);
        let exit_t = interval.max;
        let fallback_normal = (-*ray.direction()).normalized();

        for _ in 0..self.max_steps {
            if t > exit_t {
                return None;
            }

            let point = ray.at(t);
            let distance = self.field.distance(point);
            if !distance.is_finite() {
                return None;
            }

            if distance.abs() <= self.epsilon {
                let outward_normal = self.estimate_normal(point, fallback_normal);
                return Some(HitRecord::new(
                    ray,
                    point,
                    outward_normal,
                    t,
                    self.material.as_ref(),
                ));
            }

            t += distance.abs().max(0.5 * self.epsilon);
        }

        None
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(self.bounds)
    }
}

fn ray_bounds_interval(ray: &Ray, bounds: Aabb, ray_t: Interval) -> Option<Interval> {
    let origin = [ray.origin().x(), ray.origin().y(), ray.origin().z()];
    let direction = [
        ray.direction().x(),
        ray.direction().y(),
        ray.direction().z(),
    ];
    let min = [bounds.min.0, bounds.min.1, bounds.min.2];
    let max = [bounds.max.0, bounds.max.1, bounds.max.2];
    let mut t_min = ray_t.min;
    let mut t_max = ray_t.max;

    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return None;
            }
            continue;
        }

        let inv_direction = 1.0 / direction[axis];
        let mut t0 = (min[axis] - origin[axis]) * inv_direction;
        let mut t1 = (max[axis] - origin[axis]) * inv_direction;
        if inv_direction < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);
        if t_max < t_min {
            return None;
        }
    }

    Some(Interval::new(t_min, t_max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gmath::random::SampleRng,
        graphics::raytracing::{INFINITY, Lambertian, LinearColor},
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-8);
    }

    fn unit_sphere_field() -> FnDistanceField<impl Fn(Point) -> f64> {
        FnDistanceField::new(
            Aabb::new((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            |point: Point| (point - Point::new(0.0, 0.0, 0.0)).length() - 1.0,
        )
    }

    #[test]
    fn sdf_object_hits_distance_field_surface() {
        let material = Lambertian::new(LinearColor::new(0.4, 0.5, 0.6));
        let object = SdfObject::new(unit_sphere_field(), material).with_epsilon(1e-5);
        let ray = Ray::new(Point::new(0.0, 0.0, -3.0), Vector::new(0.0, 0.0, 1.0));
        let mut rng = SampleRng::new(7);

        let hit = object
            .hit_with_rng(&ray, Interval::new(0.001, INFINITY), &mut rng)
            .expect("ray should hit SDF sphere");

        assert_close(hit.t, 2.0);
        assert_close(hit.point.z(), -1.0);
        assert_close(hit.normal.z(), -1.0);
        assert!(hit.front_face);
    }

    #[test]
    fn sdf_object_misses_when_bounds_are_missed() {
        let material = Lambertian::new(LinearColor::new(0.4, 0.5, 0.6));
        let object = SdfObject::new(unit_sphere_field(), material);
        let ray = Ray::new(Point::new(3.0, 0.0, -3.0), Vector::new(0.0, 0.0, 1.0));
        let mut rng = SampleRng::new(7);

        assert!(
            object
                .hit_with_rng(&ray, Interval::new(0.001, INFINITY), &mut rng)
                .is_none()
        );
    }

    #[test]
    fn sdf_object_reports_bounds() {
        let material = Lambertian::new(LinearColor::new(0.4, 0.5, 0.6));
        let object = SdfObject::new(unit_sphere_field(), material);
        let bounds = object.bounding_box().expect("SDF should expose bounds");

        assert_close(bounds.min.0, -1.0);
        assert_close(bounds.max.2, 1.0);
    }
}

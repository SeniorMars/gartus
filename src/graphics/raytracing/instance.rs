//! Hittable instance transforms.

use super::{Aabb, HitRecord, Hittable, Interval, degrees_to_radians};
use crate::gmath::{
    matrix::Matrix,
    random::SampleRng,
    ray::Ray,
    vector::{Point, Vector},
};
use std::fmt;

/// A translated instance of a hittable object.
pub struct Translate {
    object: Box<dyn Hittable>,
    offset: Vector,
    bounds: Option<Aabb>,
}

impl fmt::Debug for Translate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Translate")
            .field("offset", &self.offset)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl Translate {
    /// Creates a translated instance.
    #[must_use]
    pub fn new(object: impl Hittable + 'static, offset: Vector) -> Self {
        let bounds = object
            .bounding_box()
            .map(|bounds| bounds.translated(offset));
        Self {
            object: Box::new(object),
            offset,
            bounds,
        }
    }

    /// Creates a translated instance from a boxed hittable.
    #[must_use]
    pub fn from_box(object: Box<dyn Hittable>, offset: Vector) -> Self {
        let bounds = object
            .bounding_box()
            .map(|bounds| bounds.translated(offset));
        Self {
            object,
            offset,
            bounds,
        }
    }

    /// Returns the translation offset.
    #[must_use]
    pub const fn offset(&self) -> Vector {
        self.offset
    }
}

impl Hittable for Translate {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let offset_ray = Ray::with_time(*ray.origin() - self.offset, *ray.direction(), ray.time());
        let mut record = self.object.hit_with_rng(&offset_ray, ray_t, rng)?;
        record.point = record.point + self.offset;
        Some(record)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }

    fn pdf_value(&self, origin: Point, direction: Vector) -> f64 {
        self.object.pdf_value(origin - self.offset, direction)
    }

    fn random_direction(&self, origin: Point, rng: &mut SampleRng) -> Vector {
        self.object.random_direction(origin - self.offset, rng)
    }
}

/// A Y-axis rotated instance of a hittable object.
pub struct RotateY {
    object: Box<dyn Hittable>,
    sin_theta: f64,
    cos_theta: f64,
    bounds: Option<Aabb>,
}

impl fmt::Debug for RotateY {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RotateY")
            .field("sin_theta", &self.sin_theta)
            .field("cos_theta", &self.cos_theta)
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl RotateY {
    /// Creates a Y-axis rotated instance.
    #[must_use]
    pub fn new(object: impl Hittable + 'static, angle_degrees: f64) -> Self {
        Self::from_box(Box::new(object), angle_degrees)
    }

    /// Creates a Y-axis rotated instance from a boxed hittable.
    #[must_use]
    pub fn from_box(object: Box<dyn Hittable>, angle_degrees: f64) -> Self {
        let radians = degrees_to_radians(angle_degrees);
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let bounds = object
            .bounding_box()
            .map(|bounds| rotate_y_bounds(bounds, sin_theta, cos_theta));
        Self {
            object,
            sin_theta,
            cos_theta,
            bounds,
        }
    }

    /// Returns the sine of this instance rotation.
    #[must_use]
    pub const fn sin_theta(&self) -> f64 {
        self.sin_theta
    }

    /// Returns the cosine of this instance rotation.
    #[must_use]
    pub const fn cos_theta(&self) -> f64 {
        self.cos_theta
    }
}

impl Hittable for RotateY {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let rotated_ray = Ray::with_time(
            rotate_y_point_inverse(*ray.origin(), self.sin_theta, self.cos_theta),
            rotate_y_vector_inverse(*ray.direction(), self.sin_theta, self.cos_theta),
            ray.time(),
        );
        let mut record = self.object.hit_with_rng(&rotated_ray, ray_t, rng)?;
        record.point = rotate_y_point(record.point, self.sin_theta, self.cos_theta);
        record.normal = rotate_y_vector(record.normal, self.sin_theta, self.cos_theta);
        Some(record)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }

    fn pdf_value(&self, origin: Point, direction: Vector) -> f64 {
        self.object.pdf_value(
            rotate_y_point_inverse(origin, self.sin_theta, self.cos_theta),
            rotate_y_vector_inverse(direction, self.sin_theta, self.cos_theta),
        )
    }

    fn random_direction(&self, origin: Point, rng: &mut SampleRng) -> Vector {
        rotate_y_vector(
            self.object.random_direction(
                rotate_y_point_inverse(origin, self.sin_theta, self.cos_theta),
                rng,
            ),
            self.sin_theta,
            self.cos_theta,
        )
    }
}

/// A generic matrix-transformed instance of a hittable object.
///
/// The transform maps object space into world space. Rays are transformed by the cached inverse
/// before hitting the child object, then hit points and normals are transformed back to world
/// space. Normals use the inverse-transpose transform, which keeps them correct for non-uniform
/// scales as well as rotations and translations.
pub struct MatrixInstance {
    object: Box<dyn Hittable>,
    transform: Matrix,
    inverse: Matrix,
    normal_transform: Matrix,
    bounds: Option<Aabb>,
}

impl fmt::Debug for MatrixInstance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MatrixInstance")
            .field("bounds", &self.bounds)
            .finish_non_exhaustive()
    }
}

impl MatrixInstance {
    /// Creates a transformed instance.
    ///
    /// Returns `None` if `transform` is not an invertible 4x4 matrix.
    #[must_use]
    pub fn new(object: impl Hittable + 'static, transform: Matrix) -> Option<Self> {
        Self::from_box(Box::new(object), transform)
    }

    /// Creates a transformed instance from a boxed hittable.
    ///
    /// Returns `None` if `transform` is not an invertible 4x4 matrix.
    #[must_use]
    pub fn from_box(object: Box<dyn Hittable>, transform: Matrix) -> Option<Self> {
        if transform.rows() != 4 || transform.cols() != 4 {
            return None;
        }
        let inverse = transform.inverse()?;
        let normal_transform = inverse.transpose();
        let bounds = object
            .bounding_box()
            .map(|bounds| transform_bounds(bounds, &transform));
        Some(Self {
            object,
            transform,
            inverse,
            normal_transform,
            bounds,
        })
    }

    /// Returns the object-to-world transform.
    pub const fn transform(&self) -> &Matrix {
        &self.transform
    }

    /// Returns the world-to-object transform.
    pub const fn inverse(&self) -> &Matrix {
        &self.inverse
    }
}

impl Hittable for MatrixInstance {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let object_ray = Ray::with_time(
            transform_point(*ray.origin(), &self.inverse),
            transform_vector(*ray.direction(), &self.inverse),
            ray.time(),
        );
        let mut record = self.object.hit_with_rng(&object_ray, ray_t, rng)?;
        let object_outward_normal = if record.front_face {
            record.normal
        } else {
            -record.normal
        };
        record.point = transform_point(record.point, &self.transform);
        let outward_normal =
            transform_vector(object_outward_normal, &self.normal_transform).normalized();
        record.set_face_normal(ray, outward_normal);
        Some(record)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }

    fn pdf_value(&self, origin: Point, direction: Vector) -> f64 {
        self.object.pdf_value(
            transform_point(origin, &self.inverse),
            transform_vector(direction, &self.inverse),
        )
    }

    fn random_direction(&self, origin: Point, rng: &mut SampleRng) -> Vector {
        transform_vector(
            self.object
                .random_direction(transform_point(origin, &self.inverse), rng),
            &self.transform,
        )
    }
}

fn transform_bounds(bounds: Aabb, transform: &Matrix) -> Aabb {
    let mut transformed_bounds = None;

    for x in [bounds.min.0, bounds.max.0] {
        for y in [bounds.min.1, bounds.max.1] {
            for z in [bounds.min.2, bounds.max.2] {
                let point = transform_point(Point::new(x, y, z), transform);
                transformed_bounds = Some(transformed_bounds.map_or_else(
                    || Aabb::from_points(point, point),
                    |bounds: Aabb| bounds.union_point(point),
                ));
            }
        }
    }

    transformed_bounds.expect("transforming finite bounds should produce bounds")
}

fn transform_point(point: Point, transform: &Matrix) -> Point {
    let transformed =
        transform.transform_homogeneous_point(&[point.x(), point.y(), point.z(), 1.0]);
    let w = transformed[3];
    if w.abs() > f64::EPSILON {
        Point::new(transformed[0] / w, transformed[1] / w, transformed[2] / w)
    } else {
        Point::new(transformed[0], transformed[1], transformed[2])
    }
}

fn transform_vector(vector: Vector, transform: &Matrix) -> Vector {
    let transformed =
        transform.transform_homogeneous_point(&[vector.x(), vector.y(), vector.z(), 0.0]);
    Vector::new(transformed[0], transformed[1], transformed[2])
}

fn rotate_y_point(point: Point, sin_theta: f64, cos_theta: f64) -> Point {
    Point::new(
        cos_theta * point.x() + sin_theta * point.z(),
        point.y(),
        -sin_theta * point.x() + cos_theta * point.z(),
    )
}

fn rotate_y_point_inverse(point: Point, sin_theta: f64, cos_theta: f64) -> Point {
    Point::new(
        cos_theta * point.x() - sin_theta * point.z(),
        point.y(),
        sin_theta * point.x() + cos_theta * point.z(),
    )
}

fn rotate_y_vector(vector: Vector, sin_theta: f64, cos_theta: f64) -> Vector {
    Vector::new(
        cos_theta * vector.x() + sin_theta * vector.z(),
        vector.y(),
        -sin_theta * vector.x() + cos_theta * vector.z(),
    )
}

fn rotate_y_vector_inverse(vector: Vector, sin_theta: f64, cos_theta: f64) -> Vector {
    Vector::new(
        cos_theta * vector.x() - sin_theta * vector.z(),
        vector.y(),
        sin_theta * vector.x() + cos_theta * vector.z(),
    )
}

fn rotate_y_bounds(bounds: Aabb, sin_theta: f64, cos_theta: f64) -> Aabb {
    let mut rotated_bounds = None;

    for x in [bounds.min.0, bounds.max.0] {
        for y in [bounds.min.1, bounds.max.1] {
            for z in [bounds.min.2, bounds.max.2] {
                let point = rotate_y_point(Point::new(x, y, z), sin_theta, cos_theta);
                rotated_bounds = Some(rotated_bounds.map_or_else(
                    || Aabb::from_points(point, point),
                    |bounds: Aabb| bounds.union_point(point),
                ));
            }
        }
    }

    rotated_bounds.expect("rotating finite bounds should produce bounds")
}

//! Minimal ray-tracing helpers following the early "Ray Tracing in One Weekend" steps.

pub use crate::gmath::random::SampleRng;
mod bvh;
pub mod instance;
pub mod material;
pub mod mesh;
pub mod object;
pub mod renderer;
pub mod scene;
pub mod scenes;
pub mod texture;
pub mod volume;
pub mod weekend;

use crate::{
    gmath::polygon_matrix::Bounds3,
    graphics::colors::{LinearRgb, Rgb},
};
pub use instance::{MatrixInstance, RotateY, Translate};
pub use material::{
    Dielectric, DiffuseLight, Isotropic, Lambertian, Material, MaterialRef, Metal, RayMaterial,
    ScatterRecord,
};
pub use mesh::{MeshTriangle, TriangleMesh};
pub use object::{
    HitRecord, Hittable, Intersect, Interval, MovingSphere, Quad, RayGeometry, SceneObject, Sphere,
    SurfaceHit, box_object, hit_sphere, hit_sphere_in_interval, hit_triangle,
};
pub use renderer::PathTracer;
pub use scene::{BvhNode, HittableList, MaterialId, RayPrimitive, RayScene, SphereList};
pub use texture::{CheckerTexture, ImageTexture, NoiseTexture, RayTexture, SolidColor, TextureRef};
pub use volume::ConstantMedium;

/// Floating-point infinity for ray intervals.
pub const INFINITY: f64 = f64::INFINITY;

/// Pi, provided with the book's common ray-tracing constants.
pub const PI: f64 = std::f64::consts::PI;

/// The 16:9 aspect ratio used by the first weekend camera setup.
pub const WIDESCREEN_ASPECT_RATIO: f64 = 16.0 / 9.0;

/// Compatibility alias for [`LinearRgb`]. Prefer [`LinearRgb`] in new shared APIs.
pub type LinearColor = LinearRgb;

/// Axis-aligned bounding box used by ray-tracing acceleration structures.
pub type Aabb = Bounds3;

/// Minimum ray parameter accepted for secondary rays to avoid self-intersections.
pub const SHADOW_ACNE_EPSILON: f64 = 0.001;

/// Converts degrees to radians.
#[must_use]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub(crate) fn component_mul(lhs: LinearColor, rhs: LinearColor) -> LinearColor {
    lhs.component_mul(rhs)
}

/// Converts a linear color component to gamma space using gamma 2.
#[must_use]
pub fn linear_to_gamma(linear_component: f64) -> f64 {
    Rgb::linear_to_gamma_component(linear_component)
}

/// Converts a linear ray-traced color to display RGB with gamma correction.
#[must_use]
pub fn linear_color_to_rgb(color: LinearColor) -> Rgb {
    Rgb::from_linear_color(color)
}

/// Converts display RGB bytes to linear RGB using the library's gamma-2 approximation.
#[must_use]
pub fn rgb_to_linear_color(color: Rgb) -> LinearColor {
    LinearColor::from_rgb_srgb(color)
}

/// Converts RGB bytes to unit channel values without gamma decoding.
#[must_use]
pub fn rgb_bytes_to_unit_color(color: Rgb) -> LinearColor {
    LinearColor::from_rgb_linear_units(color)
}

#[cfg(test)]
mod tests {
    use super::scenes::*;
    use super::*;
    use crate::gmath::{
        geometry::{SphereGeometry, TriangleGeometry},
        matrix::Matrix,
        polygon_matrix::PolygonMatrix,
        ray::Ray,
        vector::{Point, Vector},
    };
    use crate::graphics::lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex};
    use crate::graphics::raytracing::object::sphere_uv;
    use crate::graphics::{camera::RayCamera, display::Canvas};
    use std::sync::Arc;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn unit_gradient_matches_first_ppm_image_corners() {
        let canvas = Canvas::from_fn(3, 3, |x, y| {
            Rgb::from_raw_linear_color(LinearColor::new(
                f64::from(x) / 2.0,
                f64::from(y) / 2.0,
                0.0,
            ))
        });
        assert_eq!(canvas.pixels()[0], Rgb::BLACK);
        assert_eq!(canvas.pixels()[2], Rgb::new(255, 0, 0));
        assert_eq!(canvas.pixels()[8], Rgb::YELLOW);
    }

    #[test]
    fn sphere_intersection_detects_direct_hit_and_miss() {
        let origin = Point::new(0.0, 0.0, 0.0);
        let center = Point::new(0.0, 0.0, -1.0);

        let hit = Ray::new(origin, Vector::new(0.0, 0.0, -1.0));
        assert_close(hit_sphere(center, 0.5, &hit).expect("hit"), 0.5);
        assert_close(
            hit_sphere_in_interval(center, 0.5, &hit, Interval::new(0.6, INFINITY))
                .expect("far hit"),
            1.5,
        );

        let miss = Ray::new(origin, Vector::new(0.0, 1.0, -1.0));
        assert!(hit_sphere(center, 0.5, &miss).is_none());
        let zero_direction = Ray::new(origin, Vector::default());
        assert!(hit_sphere(center, 0.5, &zero_direction).is_none());
    }

    #[test]
    fn sky_gradient_blends_white_to_blue() {
        let ray = Ray::with_time(
            Point::new(0.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, -1.0),
            0.375,
        );
        let color = sky_gradient(&ray);

        assert_close(color.x(), 0.75);
        assert_close(color.y(), 0.85);
        assert_close(color.z(), 1.0);
    }

    #[test]
    fn first_sphere_render_has_red_center() {
        let canvas = RayCamera::new(40, WIDESCREEN_ASPECT_RATIO).render(first_sphere_color);
        let center = canvas
            .get_pixel(20, 11)
            .expect("center pixel should be inside the canvas");

        assert_eq!(*center, Rgb::RED);
    }

    #[test]
    fn interval_contains_and_surrounds_values() {
        let interval = Interval::new(1.0, 2.0);

        assert!(interval.contains(1.0));
        assert!(interval.contains(2.0));
        assert!(!interval.surrounds(1.0));
        assert!(interval.surrounds(1.5));
        assert_close(interval.clamp(3.0), 2.0);
    }

    #[test]
    fn sample_rng_returns_values_in_half_open_range() {
        let mut rng = SampleRng::new(7);

        for _ in 0..100 {
            let value = rng.random_double();
            assert!((0.0..1.0).contains(&value));
            let ranged = rng.random_range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&ranged));
        }
    }

    #[test]
    fn random_unit_vector_is_unit_length() {
        let mut rng = SampleRng::new(11);

        for _ in 0..20 {
            let vector = rng.random_unit_vector();
            assert!((vector.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_on_hemisphere_matches_normal_side() {
        let mut rng = SampleRng::new(13);
        let normal = Vector::new(0.0, 1.0, 0.0);

        for _ in 0..20 {
            assert!(rng.random_on_hemisphere(normal).dot(normal) > 0.0);
        }
    }

    #[test]
    fn random_in_unit_disk_stays_in_xy_unit_disk() {
        let mut rng = SampleRng::new(17);

        for _ in 0..20 {
            let point = rng.random_in_unit_disk();
            assert!(point.length_squared() < 1.0);
            assert_close(point.z(), 0.0);
        }
    }

    #[test]
    fn linear_color_to_rgb_applies_gamma_two() {
        let rgb = linear_color_to_rgb(LinearColor::new(0.25, 0.0, 1.0));

        assert_eq!(rgb, Rgb::new(128, 0, 255));
    }

    #[test]
    fn lambertian_reuses_existing_color_and_material_types() {
        let red = Lambertian::from(Rgb::RED);
        assert_eq!(red.albedo, LinearColor::new(1.0, 0.0, 0.0));

        let reflectance = ReflectionConstants::new(0.2, 0.4, 0.6);
        let from_reflectance = Lambertian::from(reflectance);
        assert_eq!(from_reflectance.albedo, LinearColor::new(0.2, 0.4, 0.6));

        let from_phong = Lambertian::from(PhongMaterial::SILVER);
        assert_eq!(
            from_phong.albedo,
            LinearColor::new(
                PhongMaterial::SILVER.diffuse.red,
                PhongMaterial::SILVER.diffuse.green,
                PhongMaterial::SILVER.diffuse.blue,
            )
        );
    }

    #[test]
    fn solid_color_texture_ignores_coordinates() {
        let texture = SolidColor::new(LinearColor::new(0.2, 0.4, 0.6));

        assert_eq!(
            texture.value(0.75, 0.25, Point::new(10.0, -4.0, 2.0)),
            LinearColor::new(0.2, 0.4, 0.6)
        );
    }

    #[test]
    fn checker_texture_alternates_in_world_space() {
        let texture = CheckerTexture::from_colors(
            1.0,
            LinearColor::new(0.1, 0.2, 0.3),
            LinearColor::new(0.8, 0.7, 0.6),
        );

        assert_eq!(
            texture.value(0.0, 0.0, Point::new(0.1, 0.1, 0.1)),
            LinearColor::new(0.1, 0.2, 0.3)
        );
        assert_eq!(
            texture.value(0.0, 0.0, Point::new(1.1, 0.1, 0.1)),
            LinearColor::new(0.8, 0.7, 0.6)
        );
    }

    #[test]
    fn image_texture_samples_existing_texture_sampler() {
        let texture =
            ImageTexture::from_canvas(Canvas::from_pixels(2, 1, vec![Rgb::RED, Rgb::GREEN]));

        assert_eq!(
            texture.value(0.0, 0.5, Point::new(0.0, 0.0, 0.0)),
            LinearColor::new(1.0, 0.0, 0.0)
        );
        assert_eq!(
            texture.value(1.0, 0.5, Point::new(0.0, 0.0, 0.0)),
            LinearColor::new(0.0, 1.0, 0.0)
        );
    }

    #[test]
    fn lambertian_scatter_samples_texture_for_attenuation() {
        let material = Lambertian::checker(
            1.0,
            LinearColor::new(0.1, 0.2, 0.3),
            LinearColor::new(0.8, 0.7, 0.6),
        );
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(1.1, 0.1, 0.1),
            normal: Vector::new(0.0, 1.0, 0.0),
            geometric_normal: Vector::new(0.0, 1.0, 0.0),
            shading_normal: Vector::new(0.0, 1.0, 0.0),
            t: 1.0,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(19);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("lambertian should scatter");

        assert_eq!(scatter.attenuation, LinearColor::new(0.8, 0.7, 0.6));
    }

    #[test]
    fn diffuse_light_emits_and_does_not_scatter() {
        let material = DiffuseLight::new(LinearColor::new(4.0, 3.0, 2.0));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, -1.0),
            normal: Vector::new(0.0, 0.0, 1.0),
            geometric_normal: Vector::new(0.0, 0.0, 1.0),
            shading_normal: Vector::new(0.0, 0.0, 1.0),
            t: 1.0,
            u: 0.25,
            v: 0.75,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(41);

        assert_eq!(
            material.emitted(hit.u, hit.v, hit.point),
            LinearColor::new(4.0, 3.0, 2.0)
        );
        assert!(material.scatter(&ray, &hit, &mut rng).is_none());
    }

    #[test]
    fn isotropic_scatter_uses_random_direction_and_texture_attenuation() {
        let material = Isotropic::new(LinearColor::new(0.25, 0.5, 0.75));
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.5);
        let hit = HitRecord {
            point: Point::new(0.0, 0.0, -1.0),
            normal: Vector::new(1.0, 0.0, 0.0),
            geometric_normal: Vector::new(1.0, 0.0, 0.0),
            shading_normal: Vector::new(1.0, 0.0, 0.0),
            t: 1.0,
            u: 0.0,
            v: 0.0,
            front_face: true,
            material: &material,
        };
        let mut rng = SampleRng::new(43);

        let scatter = material
            .scatter(&ray, &hit, &mut rng)
            .expect("isotropic medium should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_close(scatter.ray.direction().length(), 1.0);
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(0.25, 0.5, 0.75));
    }

    #[test]
    fn sphere_uv_matches_book_reference_points() {
        let left = sphere_uv(Vector::new(-1.0, 0.0, 0.0));
        let right = sphere_uv(Vector::new(1.0, 0.0, 0.0));
        let up = sphere_uv(Vector::new(0.0, 1.0, 0.0));
        let down = sphere_uv(Vector::new(0.0, -1.0, 0.0));
        let front = sphere_uv(Vector::new(0.0, 0.0, 1.0));
        let back = sphere_uv(Vector::new(0.0, 0.0, -1.0));

        assert_close(left.0, 0.0);
        assert_close(left.1, 0.5);
        assert_close(right.0, 0.5);
        assert_close(right.1, 0.5);
        assert_close(up.0, 0.5);
        assert_close(up.1, 1.0);
        assert_close(down.0, 0.5);
        assert_close(down.1, 0.0);
        assert_close(front.0, 0.25);
        assert_close(front.1, 0.5);
        assert_close(back.0, 0.75);
        assert_close(back.1, 0.5);
    }

    #[test]
    fn metal_reuses_existing_material_types_and_clamps_fuzz() {
        let silver = Metal::from(PhongMaterial::SILVER);
        assert_eq!(
            silver.albedo,
            LinearColor::new(
                PhongMaterial::SILVER.specular.red,
                PhongMaterial::SILVER.specular.green,
                PhongMaterial::SILVER.specular.blue,
            )
        );
        assert_close(silver.fuzz, 0.0);

        let fuzzy = Metal::from_reflectance(ReflectionConstants::new(0.8, 0.6, 0.2), 2.0);
        assert_close(fuzzy.fuzz, 1.0);
    }

    #[test]
    fn sphere_hit_records_front_face_and_unit_normal() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.5);

        let record = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");

        assert!(record.front_face);
        assert_close(record.t, 0.5);
        assert_close(record.normal.length(), 1.0);
        assert_eq!(record.normal, Vector::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn ray_sphere_can_share_existing_sphere_geometry() {
        let geometry = SphereGeometry::new(Point::new(1.0, 2.0, 3.0), 4.0);
        let sphere =
            Sphere::from_geometry(geometry, Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)));

        assert_eq!(sphere.geometry(), geometry);
        assert_eq!(sphere.center(), Point::new(1.0, 2.0, 3.0));
        assert_close(sphere.radius(), 4.0);
    }

    #[test]
    fn lambertian_scatter_returns_attenuated_ray_from_hit_point() {
        let material = Lambertian::new(LinearColor::new(0.2, 0.4, 0.6));
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");
        let mut rng = SampleRng::new(17);

        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("lambertian should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
    }

    #[test]
    fn moving_sphere_uses_ray_time_for_hits() {
        let sphere = MovingSphere::with_material(
            Point::new(0.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
            0.5,
            Lambertian::new(LinearColor::new(0.2, 0.2, 0.2)),
        );
        let early = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.0);
        let late = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, -1.0), 1.0);

        let early_hit = sphere
            .hit(&early, Interval::new(0.0, INFINITY))
            .expect("early ray should hit start position");
        let late_hit = sphere
            .hit(&late, Interval::new(0.0, INFINITY))
            .expect("late ray should hit end position");

        assert_close(early_hit.t, 0.5);
        assert!(late_hit.point.y() > early_hit.point.y());
        assert!(sphere.bounding_box().is_some());
    }

    #[test]
    fn quad_hit_records_texture_coordinates_and_material() {
        let quad = Quad::with_material(
            Point::new(-2.0, -2.0, -1.0),
            Vector::new(4.0, 0.0, 0.0),
            Vector::new(0.0, 4.0, 0.0),
            Lambertian::new(LinearColor::new(0.2, 0.4, 0.6)),
        );
        let ray = Ray::new(Point::new(0.0, 1.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(37);

        let record = quad
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("quad should be hit");
        let scatter = record
            .material
            .scatter(&ray, &record, &mut rng)
            .expect("quad material should scatter");

        assert!(record.front_face);
        assert_close(record.t, 1.0);
        assert_close(record.u, 0.5);
        assert_close(record.v, 0.75);
        assert_eq!(record.normal, Vector::new(0.0, 0.0, 1.0));
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
        assert!(quad.bounding_box().is_some());
    }

    #[test]
    fn box_object_builds_six_bounded_quad_sides() {
        let material: MaterialRef = Arc::new(Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)));
        let object = box_object(
            Point::new(1.0, 2.0, 3.0),
            Point::new(-1.0, -2.0, -3.0),
            material,
        );
        let bounds = object.bounding_box().expect("box should be bounded");

        assert_eq!(object.len(), 6);
        assert!(bounds.min.0 <= -1.0);
        assert!(bounds.min.1 <= -2.0);
        assert!(bounds.min.2 <= -3.0);
        assert!(bounds.max.0 >= 1.0);
        assert!(bounds.max.1 >= 2.0);
        assert!(bounds.max.2 >= 3.0);
    }

    #[test]
    fn translated_instance_moves_ray_hits_and_bounds() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let translated = Translate::new(sphere, Vector::new(2.0, 0.0, 0.0));
        let ray = Ray::new(Point::new(2.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = translated
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("translated sphere should be hit");
        let bounds = translated
            .bounding_box()
            .expect("translated sphere should be bounded");

        assert_close(record.t, 0.5);
        assert_close(record.point.x(), 2.0);
        assert_close(bounds.min.0, 1.5);
        assert_close(bounds.max.0, 2.5);
    }

    #[test]
    fn rotate_y_instance_rotates_ray_hits_normals_and_bounds() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let rotated = RotateY::new(sphere, 90.0);
        let ray = Ray::new(Point::new(-1.0, 0.0, -2.0), Vector::new(0.0, 0.0, 1.0));

        let record = rotated
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("rotated sphere should be hit");
        let bounds = rotated
            .bounding_box()
            .expect("rotated sphere should be bounded");

        assert_close(record.t, 1.5);
        assert_close(record.point.x(), -1.0);
        assert_close(record.point.z(), -0.5);
        assert_close(record.normal.z(), -1.0);
        assert!(bounds.min.0 < -1.4);
        assert!(bounds.max.0 < -0.4);
    }

    #[test]
    fn matrix_instance_transforms_ray_hits_normals_and_bounds() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let transform = Matrix::translate(2.0, 0.0, 0.0) * Matrix::scale(2.0, 1.0, 1.0);
        let instance = MatrixInstance::new(sphere, transform).expect("transform should invert");
        let ray = Ray::new(Point::new(2.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = instance
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("matrix instance should be hit");
        let bounds = instance
            .bounding_box()
            .expect("matrix instance should be bounded");

        assert_close(record.t, 0.5);
        assert_close(record.point.x(), 2.0);
        assert_close(record.point.z(), -0.5);
        assert_close(record.normal.z(), 1.0);
        assert_close(bounds.min.0, 1.0);
        assert_close(bounds.max.0, 3.0);
    }

    #[test]
    fn constant_medium_samples_hit_inside_boundary() {
        let boundary = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let medium = ConstantMedium::new(boundary, 1.0e9, LinearColor::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(47);

        let record = medium
            .hit_with_rng(&ray, Interval::new(0.0, INFINITY), &mut rng)
            .expect("dense medium should scatter inside boundary");

        assert!(record.t > 0.5);
        assert!(record.t < 1.5);
        assert_eq!(record.normal, Vector::new(1.0, 0.0, 0.0));
        assert!(record.front_face);
        assert!(medium.bounding_box().is_some());
    }

    #[test]
    fn metal_scatter_reflects_incoming_ray() {
        let material = Metal::new(LinearColor::new(0.8, 0.8, 0.8), 0.0);
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");
        let mut rng = SampleRng::new(19);

        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("front-face metal hit should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_eq!(scatter.ray.direction(), &Vector::new(0.0, 0.0, 1.0));
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(0.8, 0.8, 0.8));
    }

    #[test]
    fn dielectric_scatter_refracts_perpendicular_ray() {
        let material = Dielectric::new(RefractiveIndex::GLASS);
        let sphere = Sphere::with_material(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::with_time(
            Point::new(0.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, -1.0),
            0.625,
        );
        let hit = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("sphere should be hit");
        let mut rng = SampleRng::new(23);

        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("dielectric should scatter");

        assert_eq!(scatter.ray.origin(), &hit.point);
        assert_eq!(scatter.ray.direction(), &Vector::new(0.0, 0.0, -1.0));
        assert_close(scatter.ray.time(), ray.time());
        assert_eq!(scatter.attenuation, LinearColor::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn dielectric_reflectance_increases_at_grazing_angles() {
        let straight = Dielectric::reflectance(1.0, RefractiveIndex::GLASS.0);
        let grazing = Dielectric::reflectance(0.1, RefractiveIndex::GLASS.0);

        assert!(grazing > straight);
    }

    #[test]
    fn metal_sphere_scene_contains_four_objects() {
        let world = metal_sphere_world();

        assert_eq!(world.len(), 4);
    }

    #[test]
    fn wide_angle_sphere_scene_contains_two_objects() {
        let world = wide_angle_sphere_world();

        assert_eq!(world.len(), 2);
    }

    #[test]
    fn dielectric_sphere_scene_contains_hollow_glass_setup() {
        let world = dielectric_sphere_world();

        assert_eq!(world.len(), 5);
    }

    #[test]
    fn quads_scene_contains_five_bounded_objects() {
        let world = quads_world();

        assert_eq!(world.len(), 5);
        assert!(world.bounding_box().is_some());
    }

    #[test]
    fn light_scenes_are_bounded() {
        let simple = simple_light_world();
        let cornell = cornell_box_world();

        assert_eq!(simple.len(), 4);
        assert_eq!(cornell.len(), 8);
        assert!(simple.bounding_box().is_some());
        assert!(cornell.bounding_box().is_some());
        assert_eq!(cornell_smoke_world().len(), 8);
        assert!(cornell_smoke_world().bounding_box().is_some());
    }

    #[test]
    fn next_week_final_scene_is_bounded() {
        let world = next_week_final_scene_world(SolidColor::new(LinearColor::new(0.1, 0.2, 0.3)));

        assert_eq!(world.len(), 11);
        assert!(world.bounding_box().is_some());
    }

    #[test]
    fn final_scene_world_contains_many_random_spheres() {
        let world = final_scene_world();

        assert!(world.len() > 470);
        assert!(world.bounding_box().is_some());
        assert!(final_scene_bvh_world().bounding_box().is_some());
    }

    #[test]
    fn motion_blur_scene_contains_many_random_spheres() {
        let world = motion_blur_scene_world();
        let ray_scene = motion_blur_ray_scene();

        assert!(world.len() > 470);
        assert_eq!(ray_scene.len(), world.len());
        assert!(world.bounding_box().is_some());
        assert!(ray_scene.bounding_box().is_some());
        assert!(motion_blur_bvh_world().bounding_box().is_some());
    }

    #[test]
    fn ray_scene_uses_material_table_for_hits() {
        let mut scene = RayScene::new();
        let material = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.2, 0.4, 0.6)));
        scene.add_sphere(Point::new(0.0, 0.0, -1.0), 0.5, material);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(31);

        let hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("scene should be hit");
        let scatter = hit
            .material
            .scatter(&ray, &hit, &mut rng)
            .expect("lambertian should scatter");

        assert_eq!(scene.len(), 1);
        assert_eq!(scene.material_count(), 1);
        assert!(scene.has_bvh());
        assert_eq!(scatter.attenuation, LinearColor::new(0.2, 0.4, 0.6));
    }

    #[test]
    fn ray_scene_bvh_matches_linear_hit_path() {
        let mut scene = RayScene::new();
        let red = scene.add_material(RayMaterial::lambertian(LinearColor::new(1.0, 0.0, 0.0)));
        let green = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.0, 1.0, 0.0)));
        scene.add_sphere(Point::new(0.0, 0.0, -3.0), 0.5, red);
        scene.add_sphere(Point::new(0.0, 0.0, -1.0), 0.5, green);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let bvh_hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("BVH path should hit");
        let linear_hit = scene
            .hit_bruteforce(&ray, Interval::new(0.0, INFINITY))
            .expect("linear path should hit");

        assert!(scene.has_bvh());
        assert_close(bvh_hit.t, linear_hit.t);
        assert_eq!(bvh_hit.point, linear_hit.point);
    }

    #[test]
    fn ray_scene_supports_moving_spheres() {
        let mut scene = RayScene::new();
        let material = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.2, 0.4, 0.6)));
        scene.add_moving_sphere(
            Point::new(0.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
            0.5,
            material,
        );
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 1.0, -1.0), 1.0);

        let hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("moving sphere should be hit at end position");

        assert!(hit.point.y() > 0.0);
        assert!(scene.bounding_box().is_some());
    }

    #[test]
    fn ray_scene_supports_quads() {
        let mut scene = RayScene::new();
        scene.add_quad_with_material(
            Point::new(-1.0, -1.0, -1.0),
            Vector::new(2.0, 0.0, 0.0),
            Vector::new(0.0, 2.0, 0.0),
            RayMaterial::lambertian(LinearColor::new(0.7, 0.2, 0.1)),
        );
        let ray = Ray::new(Point::new(0.5, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let hit = scene
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("quad should be hit");

        assert_close(hit.t, 1.0);
        assert_close(hit.u, 0.75);
        assert_close(hit.v, 0.5);
        assert!(scene.bounding_box().is_some());
    }

    #[test]
    fn final_scene_ray_scene_matches_compatibility_scene_size() {
        let compatibility_world = final_scene_world();
        let ray_scene = final_scene_ray_scene();

        assert_eq!(ray_scene.len(), compatibility_world.len());
        assert_eq!(ray_scene.material_count(), ray_scene.len());
        assert!(ray_scene.bounding_box().is_some());
    }

    #[test]
    fn final_scene_can_render_through_path_tracer() {
        let world = final_scene_bvh_world();
        let canvas = PathTracer::new(
            RayCamera::new(1, WIDESCREEN_ASPECT_RATIO)
                .with_samples_per_pixel(1)
                .with_max_depth(50)
                .with_vertical_fov(20.0)
                .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
                .with_view_up(Vector::new(0.0, 1.0, 0.0))
                .with_defocus_angle(0.6)
                .with_focus_distance(10.0),
        )
        .render(&world);

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 1);
    }

    #[test]
    fn motion_blur_scene_can_render_through_path_tracer() {
        let world = motion_blur_bvh_world();
        let canvas = PathTracer::new(
            RayCamera::new(1, WIDESCREEN_ASPECT_RATIO)
                .with_samples_per_pixel(1)
                .with_max_depth(50)
                .with_vertical_fov(20.0)
                .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
                .with_view_up(Vector::new(0.0, 1.0, 0.0))
                .with_defocus_angle(0.6)
                .with_focus_distance(10.0)
                .with_shutter_interval(0.0, 1.0),
        )
        .render(&world);

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 1);
    }

    #[test]
    fn path_tracer_wraps_camera_render_entrypoint() {
        let world = normal_sphere_world();
        let tracer = PathTracer::new(RayCamera::new(4, 1.0).with_samples_per_pixel(1));

        let canvas = tracer.render(&world);

        assert_eq!(tracer.camera().image_width(), 4);
        assert_eq!(canvas.width(), 4);
        assert!(canvas.upper_left_origin);
    }

    #[test]
    fn sphere_hit_flips_normal_for_inside_ray() {
        let sphere = Sphere::new(Point::new(0.0, 0.0, 0.0), 1.0);
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, 1.0));

        let record = sphere
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("inside ray should exit sphere");

        assert!(!record.front_face);
        assert_eq!(record.normal, Vector::new(-0.0, -0.0, -1.0));
    }

    #[test]
    fn triangle_hit_reports_barycentrics() {
        let p0 = Point::new(0.0, 0.0, -1.0);
        let p1 = Point::new(1.0, 0.0, -1.0);
        let p2 = Point::new(0.0, 1.0, -1.0);
        let ray = Ray::new(Point::new(0.25, 0.25, 0.0), Vector::new(0.0, 0.0, -1.0));

        let (t, u, v) = hit_triangle(p0, p1, p2, &ray, Interval::new(0.0, INFINITY))
            .expect("triangle should be hit");

        assert_close(t, 1.0);
        assert_close(u, 0.25);
        assert_close(v, 0.25);
        assert_eq!(ray.at(t), Point::new(0.25, 0.25, -1.0));
    }

    #[test]
    fn triangle_hit_rejects_edge_parallel_behind_and_degenerate_cases() {
        let p0 = Point::new(0.0, 0.0, -1.0);
        let p1 = Point::new(1.0, 0.0, -1.0);
        let p2 = Point::new(0.0, 1.0, -1.0);

        let outside = Ray::new(Point::new(1.1, 0.1, 0.0), Vector::new(0.0, 0.0, -1.0));
        assert!(hit_triangle(p0, p1, p2, &outside, Interval::new(0.0, INFINITY)).is_none());

        let parallel = Ray::new(Point::new(0.25, 0.25, -1.0), Vector::new(1.0, 0.0, 0.0));
        assert!(hit_triangle(p0, p1, p2, &parallel, Interval::new(0.0, INFINITY)).is_none());

        let behind = Ray::new(Point::new(0.25, 0.25, -2.0), Vector::new(0.0, 0.0, -1.0));
        assert!(hit_triangle(p0, p1, p2, &behind, Interval::new(0.0, INFINITY)).is_none());

        let degenerate = Ray::new(Point::new(0.25, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        assert!(
            hit_triangle(
                p0,
                Point::new(0.5, 0.0, -1.0),
                p1,
                &degenerate,
                Interval::new(0.0, INFINITY)
            )
            .is_none()
        );
    }

    #[test]
    fn triangle_scene_object_is_two_sided_and_flips_backface_normal() {
        let triangle = SceneObject::new(
            TriangleGeometry::new(
                Point::new(0.0, 0.0, -1.0),
                Point::new(1.0, 0.0, -1.0),
                Point::new(0.0, 1.0, -1.0),
            ),
            Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)),
        );
        let ray = Ray::new(Point::new(0.25, 0.25, -2.0), Vector::new(0.0, 0.0, 1.0));

        let record = triangle
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("backface should still be hit");

        assert!(!record.front_face);
        assert_eq!(record.normal, Vector::new(-0.0, -0.0, -1.0));
    }

    #[test]
    fn triangle_mesh_bvh_matches_bruteforce_hits() {
        let mut polygons = PolygonMatrix::new();
        polygons.push_polygons(&[
            [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (0.0, 1.0, -1.0)],
            [(-1.0, 0.0, -2.0), (0.0, 0.0, -2.0), (-1.0, 1.0, -2.0)],
            [(0.0, -1.0, -3.0), (1.0, -1.0, -3.0), (0.0, 0.0, -3.0)],
            [(-1.0, -1.0, -4.0), (0.0, -1.0, -4.0), (-1.0, 0.0, -4.0)],
            [(0.25, 0.25, -5.0), (1.25, 0.25, -5.0), (0.25, 1.25, -5.0)],
        ]);
        let mesh = TriangleMesh::from_polygon_matrix(
            &polygons,
            Lambertian::new(LinearColor::new(0.2, 0.2, 0.2)),
        );

        for x in [-0.75, -0.25, 0.25, 0.75, 1.5] {
            for y in [-0.75, -0.25, 0.25, 0.75, 1.5] {
                let ray = Ray::new(Point::new(x, y, 0.0), Vector::new(0.0, 0.0, -1.0));
                let bvh_hit = mesh
                    .hit(&ray, Interval::new(0.0, INFINITY))
                    .map(|hit| hit.t);
                let brute_hit = mesh
                    .hit_bruteforce(&ray, Interval::new(0.0, INFINITY))
                    .map(|hit| hit.t);
                assert_eq!(bvh_hit, brute_hit);
            }
        }
    }

    #[test]
    fn triangle_mesh_preserves_texture_coordinates_and_shading_normals() {
        let triangle = MeshTriangle::new(TriangleGeometry::new(
            Point::new(0.0, 0.0, -1.0),
            Point::new(1.0, 0.0, -1.0),
            Point::new(0.0, 1.0, -1.0),
        ))
        .with_texcoords([(0.2, 0.4), (0.8, 0.4), (0.2, 0.9)])
        .with_vertex_normals([
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
        ]);
        let mesh = TriangleMesh::with_mesh_triangles_and_shared_material(
            vec![triangle],
            Arc::new(Lambertian::new(LinearColor::new(0.2, 0.2, 0.2))),
        );
        let ray = Ray::new(Point::new(0.25, 0.25, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = mesh
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("triangle mesh should be hit");

        assert_close(record.t, 1.0);
        assert_close(record.u, 0.35);
        assert_close(record.v, 0.525);
        assert_eq!(record.normal, Vector::new(0.0, 0.0, 1.0));
        assert_eq!(record.geometric_normal, Vector::new(0.0, 0.0, 1.0));
        assert_eq!(record.shading_normal, Vector::new(0.0, 1.0, 0.0));
    }

    #[cfg(feature = "external")]
    #[test]
    fn material_mesh_converts_to_triangle_mesh_groups() {
        let mesh = crate::external::meshify_with_materials("examples/data/meshes/teapot.obj")
            .expect("load teapot mesh");
        let triangle_meshes = TriangleMesh::from_material_mesh_lambertian(&mesh);

        assert!(!triangle_meshes.is_empty());
        assert_eq!(
            triangle_meshes.iter().map(TriangleMesh::len).sum::<usize>(),
            mesh.triangle_count()
        );
        assert!(
            triangle_meshes
                .iter()
                .all(|triangle_mesh| triangle_mesh.bounding_box().is_some())
        );
    }

    #[test]
    fn hittable_list_returns_closest_hit() {
        let mut world = HittableList::new();
        world.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        world.add(Sphere::new(Point::new(0.0, 0.0, -1.0), 0.25));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = world
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("world should be hit");

        assert_close(record.t, 0.75);
    }

    #[test]
    fn hittable_list_caches_bounds_as_objects_are_added() {
        let mut world = HittableList::new();
        assert!(world.bounding_box().is_none());

        world.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        assert_eq!(
            world.bounding_box(),
            Some(Aabb::new((-0.5, -0.5, -2.5), (0.5, 0.5, -1.5)))
        );

        world.add(Sphere::new(Point::new(2.0, 1.0, -1.0), 0.25));
        assert_eq!(
            world.bounding_box(),
            Some(Aabb::new((-0.5, -0.5, -2.5), (2.25, 1.25, -0.75)))
        );

        world.clear();
        assert!(world.bounding_box().is_none());
    }

    #[test]
    fn object_bvh_returns_closest_hit() {
        let mut world = HittableList::new();
        world.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        world.add(Sphere::new(Point::new(0.0, 0.0, -1.0), 0.25));
        let bvh = world.into_bvh().expect("bounded world should build bvh");
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));

        let record = bvh
            .hit(&ray, Interval::new(0.0, INFINITY))
            .expect("BVH world should be hit");

        assert_close(record.t, 0.75);
    }

    #[test]
    fn object_bvh_matches_bruteforce_hits() {
        let mut world = HittableList::new();
        for z in 1..8 {
            let x = if z % 2 == 0 { -0.5 } else { 0.5 };
            world.add(Sphere::new(Point::new(x, 0.0, -f64::from(z)), 0.35));
        }
        let bvh = world.into_bvh().expect("bounded world should build bvh");

        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for y in [-0.5, 0.0, 0.5] {
                let ray = Ray::new(Point::new(x, y, 0.0), Vector::new(0.0, 0.0, -1.0));
                let bvh_hit = bvh.hit(&ray, Interval::new(0.0, INFINITY)).map(|hit| hit.t);
                let brute_hit = bvh
                    .hit_bruteforce(&ray, Interval::new(0.0, INFINITY))
                    .map(|hit| hit.t);
                assert_eq!(bvh_hit, brute_hit);
            }
        }
    }

    #[test]
    fn normal_scene_render_colors_sphere_by_normal() {
        let world = normal_sphere_world();
        let canvas = RayCamera::new(40, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(100)
            .render_world_normals(&world);
        let center = canvas
            .get_pixel(20, 11)
            .expect("center pixel should be inside the canvas");

        assert_ne!(*center, Rgb::RED);
        assert!(center.blue > center.red);
    }

    #[test]
    fn diffuse_scene_render_is_gamma_corrected() {
        let world = normal_sphere_world();
        let canvas = RayCamera::new(20, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(100)
            .with_max_depth(50)
            .render_world(&world);
        let center = canvas
            .get_pixel(10, 5)
            .expect("center pixel should be inside the canvas");

        assert!(center.red > 0 || center.green > 0 || center.blue > 0);
    }
}

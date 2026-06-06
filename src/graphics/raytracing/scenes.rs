//! Built-in ray-tracing scene builders and educational color helpers.

use super::{
    BvhNode, ConstantMedium, Dielectric, DiffuseLight, Hittable, HittableList, INFINITY, Interval,
    Lambertian, LinearColor, MaterialRef, Metal, MovingSphere, PI, Quad, RayMaterial, RayScene,
    RayTexture, RotateY, SampleRng, Sphere, SphereList, Translate, box_object, component_mul,
    hit_sphere,
};
use crate::gmath::{ray::Ray, vector::Point, vector::Vector};
use crate::graphics::lighting::{PhongMaterial, ReflectionConstants, RefractiveIndex};
use std::sync::Arc;

/// Computes the blue-to-white background gradient for a ray.
#[must_use]
pub fn sky_gradient(ray: &Ray) -> LinearColor {
    let unit_direction = ray.direction().normalized();
    let a = 0.5 * (unit_direction.y() + 1.0);
    (1.0 - a) * LinearColor::new(1.0, 1.0, 1.0) + a * LinearColor::new(0.5, 0.7, 1.0)
}

/// Computes the first sphere scene from the book: a red sphere over a blue sky.
#[must_use]
pub fn first_sphere_color(ray: &Ray) -> LinearColor {
    if hit_sphere(Point::new(0.0, 0.0, -1.0), 0.5, ray).is_some() {
        LinearColor::new(1.0, 0.0, 0.0)
    } else {
        sky_gradient(ray)
    }
}

/// Computes a normal-visualization color for a ray cast into `world`.
#[must_use]
pub fn normal_scene_color(ray: &Ray, world: &dyn Hittable) -> LinearColor {
    if let Some(record) = world.hit(ray, Interval::new(0.0, INFINITY)) {
        0.5 * (LinearColor::from(record.normal) + LinearColor::new(1.0, 1.0, 1.0))
    } else {
        sky_gradient(ray)
    }
}

/// Returns the book's first multi-object world: a sphere over a large ground sphere.
#[must_use]
pub fn normal_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    let material = Lambertian::new(LinearColor::new(0.5, 0.5, 0.5));
    world.add(Sphere::with_material(
        Point::new(0.0, 0.0, -1.0),
        0.5,
        material.clone(),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        material,
    ));
    world
}

/// Returns the book's first mixed diffuse/metal sphere world.
#[must_use]
pub fn metal_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        Lambertian::new(LinearColor::new(0.8, 0.8, 0.0)),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 0.0, -1.2),
        0.5,
        Lambertian::new(LinearColor::new(0.1, 0.2, 0.5)),
    ));
    world.add(Sphere::with_material(
        Point::new(-1.0, 0.0, -1.0),
        0.5,
        Metal::from_phong_specular(PhongMaterial::SILVER, 0.0),
    ));
    world.add(Sphere::with_material(
        Point::new(1.0, 0.0, -1.0),
        0.5,
        Metal::from_reflectance(ReflectionConstants::new(0.8, 0.6, 0.2), 0.3),
    ));
    world
}

/// Returns the book's two-sphere scene for checking camera field of view.
#[must_use]
pub fn wide_angle_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    let radius = (PI / 4.0).cos();

    world.add(Sphere::with_material(
        Point::new(-radius, 0.0, -1.0),
        radius,
        Lambertian::new(LinearColor::new(0.0, 0.0, 1.0)),
    ));
    world.add(Sphere::with_material(
        Point::new(radius, 0.0, -1.0),
        radius,
        Lambertian::new(LinearColor::new(1.0, 0.0, 0.0)),
    ));
    world
}

/// Returns the book's two large checkered spheres scene.
#[must_use]
pub fn checkered_spheres_world() -> HittableList {
    let checker = Lambertian::checker(
        0.32,
        LinearColor::new(0.2, 0.3, 0.1),
        LinearColor::new(0.9, 0.9, 0.9),
    );
    let mut world = HittableList::new();
    world.add(Sphere::with_material(
        Point::new(0.0, -10.0, 0.0),
        10.0,
        checker.clone(),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 10.0, 0.0),
        10.0,
        checker,
    ));
    world
}

/// Returns the book's two Perlin-textured spheres scene.
#[must_use]
pub fn perlin_spheres_world() -> HittableList {
    let material = Lambertian::marble(4.0, 29);
    let mut world = HittableList::new();
    world.add(Sphere::with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        material.clone(),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 2.0, 0.0),
        2.0,
        material,
    ));
    world
}

/// Returns the book's five-quads scene.
#[must_use]
pub fn quads_world() -> HittableList {
    let mut world = HittableList::with_capacity(5);

    let left_red = Lambertian::new(LinearColor::new(1.0, 0.2, 0.2));
    let back_green = Lambertian::new(LinearColor::new(0.2, 1.0, 0.2));
    let right_blue = Lambertian::new(LinearColor::new(0.2, 0.2, 1.0));
    let upper_orange = Lambertian::new(LinearColor::new(1.0, 0.5, 0.0));
    let lower_teal = Lambertian::new(LinearColor::new(0.2, 0.8, 0.8));

    world.add(Quad::with_material(
        Point::new(-3.0, -2.0, 5.0),
        Vector::new(0.0, 0.0, -4.0),
        Vector::new(0.0, 4.0, 0.0),
        left_red,
    ));
    world.add(Quad::with_material(
        Point::new(-2.0, -2.0, 0.0),
        Vector::new(4.0, 0.0, 0.0),
        Vector::new(0.0, 4.0, 0.0),
        back_green,
    ));
    world.add(Quad::with_material(
        Point::new(3.0, -2.0, 1.0),
        Vector::new(0.0, 0.0, 4.0),
        Vector::new(0.0, 4.0, 0.0),
        right_blue,
    ));
    world.add(Quad::with_material(
        Point::new(-2.0, 3.0, 1.0),
        Vector::new(4.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 4.0),
        upper_orange,
    ));
    world.add(Quad::with_material(
        Point::new(-2.0, -3.0, 5.0),
        Vector::new(4.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, -4.0),
        lower_teal,
    ));

    world
}

/// Returns the book's simple light scene with Perlin spheres and emissive objects.
#[must_use]
pub fn simple_light_world() -> HittableList {
    let perlin = Lambertian::marble(4.0, 29);
    let light = DiffuseLight::new(LinearColor::new(4.0, 4.0, 4.0));
    let mut world = HittableList::with_capacity(4);

    world.add(Sphere::with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        perlin.clone(),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 2.0, 0.0),
        2.0,
        perlin,
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 7.0, 0.0),
        2.0,
        light.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(3.0, 1.0, -2.0),
        Vector::new(2.0, 0.0, 0.0),
        Vector::new(0.0, 2.0, 0.0),
        light,
    ));

    world
}

/// Returns the book's Cornell box scene with two rotated block instances.
#[must_use]
pub fn cornell_box_world() -> HittableList {
    let red = Lambertian::new(LinearColor::new(0.65, 0.05, 0.05));
    let white = Lambertian::new(LinearColor::new(0.73, 0.73, 0.73));
    let white_shared: MaterialRef = Arc::new(white.clone());
    let green = Lambertian::new(LinearColor::new(0.12, 0.45, 0.15));
    let light = DiffuseLight::new(LinearColor::new(15.0, 15.0, 15.0));
    let mut world = HittableList::with_capacity(8);

    world.add(Quad::with_material(
        Point::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        green,
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        red,
    ));
    world.add(Quad::with_material(
        Point::new(343.0, 554.0, 332.0),
        Vector::new(-130.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, -105.0),
        light,
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 0.0, 0.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(555.0, 555.0, 555.0),
        Vector::new(-555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, -555.0),
        white.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 0.0, 555.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        white,
    ));
    world.add(Translate::new(
        RotateY::new(
            box_object(
                Point::new(0.0, 0.0, 0.0),
                Point::new(165.0, 330.0, 165.0),
                white_shared.clone(),
            ),
            15.0,
        ),
        Vector::new(265.0, 0.0, 295.0),
    ));
    world.add(Translate::new(
        RotateY::new(
            box_object(
                Point::new(0.0, 0.0, 0.0),
                Point::new(165.0, 165.0, 165.0),
                white_shared,
            ),
            -18.0,
        ),
        Vector::new(130.0, 0.0, 65.0),
    ));

    world
}

/// Returns the book's Cornell box scene with two constant-density smoke volumes.
#[must_use]
pub fn cornell_smoke_world() -> HittableList {
    let red = Lambertian::new(LinearColor::new(0.65, 0.05, 0.05));
    let white = Lambertian::new(LinearColor::new(0.73, 0.73, 0.73));
    let white_shared: MaterialRef = Arc::new(white.clone());
    let green = Lambertian::new(LinearColor::new(0.12, 0.45, 0.15));
    let light = DiffuseLight::new(LinearColor::new(7.0, 7.0, 7.0));
    let mut world = HittableList::with_capacity(8);

    world.add(Quad::with_material(
        Point::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        green,
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        red,
    ));
    world.add(Quad::with_material(
        Point::new(113.0, 554.0, 127.0),
        Vector::new(330.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 305.0),
        light,
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 555.0, 0.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 0.0, 0.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(0.0, 0.0, 555.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        white,
    ));

    let box1 = Translate::new(
        RotateY::new(
            box_object(
                Point::new(0.0, 0.0, 0.0),
                Point::new(165.0, 330.0, 165.0),
                white_shared.clone(),
            ),
            15.0,
        ),
        Vector::new(265.0, 0.0, 295.0),
    );
    let box2 = Translate::new(
        RotateY::new(
            box_object(
                Point::new(0.0, 0.0, 0.0),
                Point::new(165.0, 165.0, 165.0),
                white_shared,
            ),
            -18.0,
        ),
        Vector::new(130.0, 0.0, 65.0),
    );

    world.add(ConstantMedium::new(
        box1,
        0.01,
        LinearColor::new(0.0, 0.0, 0.0),
    ));
    world.add(ConstantMedium::new(
        box2,
        0.01,
        LinearColor::new(1.0, 1.0, 1.0),
    ));

    world
}

/// Returns the final "Ray Tracing: The Next Week" scene.
///
/// `earth_texture` is supplied by the caller so the library scene can be tested without requiring
/// external image loading, while examples can pass an [`super::ImageTexture`].
///
/// # Panics
///
/// Panics if one of the built-in final-scene ground boxes or clustered spheres does not provide a
/// bounding box.
#[must_use]
pub fn next_week_final_scene_world(earth_texture: impl RayTexture + 'static) -> HittableList {
    let mut rng = SampleRng::new(61);
    let ground: MaterialRef = Arc::new(Lambertian::new(LinearColor::new(0.48, 0.83, 0.53)));
    let mut boxes1 = HittableList::with_capacity(20 * 20);

    for i in 0..20 {
        for j in 0..20 {
            let width = 100.0;
            let x0 = -1000.0 + f64::from(i) * width;
            let z0 = -1000.0 + f64::from(j) * width;
            let y0 = 0.0;
            let x1 = x0 + width;
            let y1 = rng.random_range(1.0, 101.0);
            let z1 = z0 + width;
            boxes1.add(box_object(
                Point::new(x0, y0, z0),
                Point::new(x1, y1, z1),
                ground.clone(),
            ));
        }
    }

    let mut world = HittableList::with_capacity(11);
    world.add(
        boxes1
            .into_bvh()
            .expect("final scene ground boxes are bounded"),
    );
    world.add(Quad::with_material(
        Point::new(123.0, 554.0, 147.0),
        Vector::new(300.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 265.0),
        DiffuseLight::new(LinearColor::new(7.0, 7.0, 7.0)),
    ));
    world.add(MovingSphere::with_material(
        Point::new(400.0, 400.0, 200.0),
        Point::new(430.0, 400.0, 200.0),
        50.0,
        Lambertian::new(LinearColor::new(0.7, 0.3, 0.1)),
    ));
    world.add(Sphere::with_material(
        Point::new(260.0, 150.0, 45.0),
        50.0,
        Dielectric::new(RefractiveIndex::GLASS),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 150.0, 145.0),
        50.0,
        Metal::new(LinearColor::new(0.8, 0.8, 0.9), 1.0),
    ));

    let blue_boundary = Sphere::with_material(
        Point::new(360.0, 150.0, 145.0),
        70.0,
        Dielectric::new(RefractiveIndex::GLASS),
    );
    world.add(blue_boundary.clone());
    world.add(ConstantMedium::new(
        blue_boundary,
        0.2,
        LinearColor::new(0.2, 0.4, 0.9),
    ));
    world.add(ConstantMedium::new(
        Sphere::with_material(
            Point::new(0.0, 0.0, 0.0),
            5000.0,
            Dielectric::new(RefractiveIndex::GLASS),
        ),
        0.0001,
        LinearColor::new(1.0, 1.0, 1.0),
    ));
    world.add(Sphere::with_material(
        Point::new(400.0, 200.0, 400.0),
        100.0,
        Lambertian::from_texture(earth_texture),
    ));
    world.add(Sphere::with_material(
        Point::new(220.0, 280.0, 300.0),
        80.0,
        Lambertian::noise(0.2, 29),
    ));

    let white: MaterialRef = Arc::new(Lambertian::new(LinearColor::new(0.73, 0.73, 0.73)));
    let mut boxes2 = HittableList::with_capacity(1000);
    for _ in 0..1000 {
        let center = Point::new(
            rng.random_range(0.0, 165.0),
            rng.random_range(0.0, 165.0),
            rng.random_range(0.0, 165.0),
        );
        boxes2.add(Sphere::with_shared_material(center, 10.0, white.clone()));
    }
    world.add(Translate::new(
        RotateY::new(
            boxes2
                .into_bvh()
                .expect("final scene sphere cluster is bounded"),
            15.0,
        ),
        Vector::new(-100.0, 270.0, 395.0),
    ));

    world
}

/// Returns the book's diffuse/metal scene with a hollow glass sphere.
#[must_use]
pub fn dielectric_sphere_world() -> HittableList {
    let mut world = HittableList::new();
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        Lambertian::new(LinearColor::new(0.8, 0.8, 0.0)),
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, 0.0, -1.2),
        0.5,
        Lambertian::new(LinearColor::new(0.1, 0.2, 0.5)),
    ));
    world.add(Sphere::with_material(
        Point::new(-1.0, 0.0, -1.0),
        0.5,
        Dielectric::new(RefractiveIndex::GLASS),
    ));
    world.add(Sphere::with_material(
        Point::new(-1.0, 0.0, -1.0),
        0.4,
        Dielectric::new(RefractiveIndex::AIR.relative_to(RefractiveIndex::GLASS)),
    ));
    world.add(Sphere::with_material(
        Point::new(1.0, 0.0, -1.0),
        0.5,
        Metal::from_reflectance(ReflectionConstants::new(0.8, 0.6, 0.2), 0.0),
    ));
    world
}

/// Returns the final random-spheres scene from the book.
#[must_use]
pub fn final_scene_world() -> HittableList {
    let spheres = final_scene_spheres();
    let mut world = HittableList::with_capacity(spheres.len());
    for sphere in spheres {
        world.add(sphere);
    }
    world
}

/// Returns the final random-spheres scene wrapped in an object BVH.
///
/// # Panics
///
/// Panics if one of the built-in scene objects does not provide a bounding box.
#[must_use]
pub fn final_scene_bvh_world() -> BvhNode {
    final_scene_world()
        .into_bvh()
        .expect("final random-spheres scene objects are bounded")
}

/// Returns the final random-spheres scene in a sphere-specialized container.
#[must_use]
pub fn final_scene_sphere_list() -> SphereList {
    let spheres = final_scene_spheres();
    let mut world = SphereList::with_capacity(spheres.len());
    for sphere in spheres {
        world.add(sphere);
    }
    world
}

/// Returns the final random-spheres scene in a data-oriented ray scene.
#[must_use]
pub fn final_scene_ray_scene() -> RayScene {
    let mut rng = SampleRng::new(61);
    let mut world = RayScene::with_capacity(22 * 22 + 4, 22 * 22 + 4);

    world.add_sphere_with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        RayMaterial::lambertian(LinearColor::new(0.5, 0.5, 0.5)),
    );

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random_double();
            let center = Point::new(
                f64::from(a) + 0.9 * rng.random_double(),
                0.2,
                f64::from(b) + 0.9 * rng.random_double(),
            );

            if (center - Point::new(4.0, 0.2, 0.0)).length() <= 0.9 {
                continue;
            }

            if choose_material < 0.8 {
                let albedo = component_mul(
                    LinearColor::from(rng.random_vector()),
                    LinearColor::from(rng.random_vector()),
                );
                world.add_sphere_with_material(center, 0.2, RayMaterial::lambertian(albedo));
            } else if choose_material < 0.95 {
                let albedo = LinearColor::from(rng.random_vector_range(0.5, 1.0));
                let fuzz = rng.random_range(0.0, 0.5);
                world.add_sphere_with_material(center, 0.2, RayMaterial::metal(albedo, fuzz));
            } else {
                world.add_sphere_with_material(
                    center,
                    0.2,
                    RayMaterial::dielectric(RefractiveIndex::GLASS),
                );
            }
        }
    }

    world.add_sphere_with_material(
        Point::new(0.0, 1.0, 0.0),
        1.0,
        RayMaterial::dielectric(RefractiveIndex::GLASS),
    );
    world.add_sphere_with_material(
        Point::new(-4.0, 1.0, 0.0),
        1.0,
        RayMaterial::lambertian(LinearColor::new(0.4, 0.2, 0.1)),
    );
    world.add_sphere_with_material(
        Point::new(4.0, 1.0, 0.0),
        1.0,
        RayMaterial::metal(LinearColor::new(0.7, 0.6, 0.5), 0.0),
    );

    world.build_bvh();
    world
}

/// Returns the final random-spheres scene with diffuse small spheres moving during the shutter.
#[must_use]
pub fn motion_blur_scene_world() -> HittableList {
    let mut rng = SampleRng::new(61);
    let mut world = HittableList::with_capacity(22 * 22 + 4);

    world.add(Sphere::with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random_double();
            let center = Point::new(
                f64::from(a) + 0.9 * rng.random_double(),
                0.2,
                f64::from(b) + 0.9 * rng.random_double(),
            );

            if (center - Point::new(4.0, 0.2, 0.0)).length() <= 0.9 {
                continue;
            }

            if choose_material < 0.8 {
                let albedo = component_mul(
                    LinearColor::from(rng.random_vector()),
                    LinearColor::from(rng.random_vector()),
                );
                let center_end = center + Vector::new(0.0, rng.random_range(0.0, 0.5), 0.0);
                world.add(MovingSphere::with_material(
                    center,
                    center_end,
                    0.2,
                    Lambertian::new(albedo),
                ));
            } else if choose_material < 0.95 {
                let albedo = LinearColor::from(rng.random_vector_range(0.5, 1.0));
                let fuzz = rng.random_range(0.0, 0.5);
                world.add(Sphere::with_material(center, 0.2, Metal::new(albedo, fuzz)));
            } else {
                world.add(Sphere::with_material(
                    center,
                    0.2,
                    Dielectric::new(RefractiveIndex::GLASS),
                ));
            }
        }
    }

    world.add(Sphere::with_material(
        Point::new(0.0, 1.0, 0.0),
        1.0,
        Dielectric::new(RefractiveIndex::GLASS),
    ));
    world.add(Sphere::with_material(
        Point::new(-4.0, 1.0, 0.0),
        1.0,
        Lambertian::new(LinearColor::new(0.4, 0.2, 0.1)),
    ));
    world.add(Sphere::with_material(
        Point::new(4.0, 1.0, 0.0),
        1.0,
        Metal::new(LinearColor::new(0.7, 0.6, 0.5), 0.0),
    ));

    world
}

/// Returns the motion-blur random-spheres scene wrapped in an object BVH.
///
/// # Panics
///
/// Panics if one of the built-in scene objects does not provide a bounding box.
#[must_use]
pub fn motion_blur_bvh_world() -> BvhNode {
    motion_blur_scene_world()
        .into_bvh()
        .expect("motion-blur scene objects are bounded")
}

/// Returns the motion-blur random-spheres scene in a data-oriented ray scene.
#[must_use]
pub fn motion_blur_ray_scene() -> RayScene {
    let mut rng = SampleRng::new(61);
    let mut world = RayScene::with_capacity(22 * 22 + 4, 22 * 22 + 4);

    world.add_sphere_with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        RayMaterial::lambertian(LinearColor::new(0.5, 0.5, 0.5)),
    );

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random_double();
            let center = Point::new(
                f64::from(a) + 0.9 * rng.random_double(),
                0.2,
                f64::from(b) + 0.9 * rng.random_double(),
            );

            if (center - Point::new(4.0, 0.2, 0.0)).length() <= 0.9 {
                continue;
            }

            if choose_material < 0.8 {
                let albedo = component_mul(
                    LinearColor::from(rng.random_vector()),
                    LinearColor::from(rng.random_vector()),
                );
                let center_end = center + Vector::new(0.0, rng.random_range(0.0, 0.5), 0.0);
                world.add_moving_sphere_with_material(
                    center,
                    center_end,
                    0.2,
                    RayMaterial::lambertian(albedo),
                );
            } else if choose_material < 0.95 {
                let albedo = LinearColor::from(rng.random_vector_range(0.5, 1.0));
                let fuzz = rng.random_range(0.0, 0.5);
                world.add_sphere_with_material(center, 0.2, RayMaterial::metal(albedo, fuzz));
            } else {
                world.add_sphere_with_material(
                    center,
                    0.2,
                    RayMaterial::dielectric(RefractiveIndex::GLASS),
                );
            }
        }
    }

    world.add_sphere_with_material(
        Point::new(0.0, 1.0, 0.0),
        1.0,
        RayMaterial::dielectric(RefractiveIndex::GLASS),
    );
    world.add_sphere_with_material(
        Point::new(-4.0, 1.0, 0.0),
        1.0,
        RayMaterial::lambertian(LinearColor::new(0.4, 0.2, 0.1)),
    );
    world.add_sphere_with_material(
        Point::new(4.0, 1.0, 0.0),
        1.0,
        RayMaterial::metal(LinearColor::new(0.7, 0.6, 0.5), 0.0),
    );

    world.build_bvh();
    world
}

fn final_scene_spheres() -> Vec<Sphere> {
    let mut rng = SampleRng::new(61);
    let mut spheres = Vec::with_capacity(22 * 22 + 4);

    spheres.push(Sphere::with_material(
        Point::new(0.0, -1000.0, 0.0),
        1000.0,
        Lambertian::new(LinearColor::new(0.5, 0.5, 0.5)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random_double();
            let center = Point::new(
                f64::from(a) + 0.9 * rng.random_double(),
                0.2,
                f64::from(b) + 0.9 * rng.random_double(),
            );

            if (center - Point::new(4.0, 0.2, 0.0)).length() <= 0.9 {
                continue;
            }

            if choose_material < 0.8 {
                let albedo = component_mul(
                    LinearColor::from(rng.random_vector()),
                    LinearColor::from(rng.random_vector()),
                );
                spheres.push(Sphere::with_material(center, 0.2, Lambertian::new(albedo)));
            } else if choose_material < 0.95 {
                let albedo = LinearColor::from(rng.random_vector_range(0.5, 1.0));
                let fuzz = rng.random_range(0.0, 0.5);
                spheres.push(Sphere::with_material(center, 0.2, Metal::new(albedo, fuzz)));
            } else {
                spheres.push(Sphere::with_material(
                    center,
                    0.2,
                    Dielectric::new(RefractiveIndex::GLASS),
                ));
            }
        }
    }

    spheres.push(Sphere::with_material(
        Point::new(0.0, 1.0, 0.0),
        1.0,
        Dielectric::new(RefractiveIndex::GLASS),
    ));
    spheres.push(Sphere::with_material(
        Point::new(-4.0, 1.0, 0.0),
        1.0,
        Lambertian::new(LinearColor::new(0.4, 0.2, 0.1)),
    ));
    spheres.push(Sphere::with_material(
        Point::new(4.0, 1.0, 0.0),
        1.0,
        Metal::new(LinearColor::new(0.7, 0.6, 0.5), 0.0),
    ));

    spheres
}

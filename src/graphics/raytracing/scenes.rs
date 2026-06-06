//! Built-in ray-tracing scenes and educational render helpers.

use super::{
    Canvas, Dielectric, Hittable, HittableList, INFINITY, Interval, Lambertian, LinearColor, Metal,
    PI, PathTracer, PhongMaterial, Point, Ray, RayCamera, RayMaterial, RayScene,
    ReflectionConstants, RefractiveIndex, Rgb, SampleRng, Sphere, SphereList, Vector,
    WIDESCREEN_ASPECT_RATIO, component_mul, hit_sphere,
};

/// Computes the blue-to-white background gradient for a ray.
#[must_use]
pub fn sky_gradient(ray: &Ray) -> LinearColor {
    let unit_direction = ray.direction().normalized();
    let a = 0.5 * (unit_direction.y() + 1.0);
    (1.0 - a) * LinearColor::new(1.0, 1.0, 1.0) + a * LinearColor::new(0.5, 0.7, 1.0)
}

/// Renders the book's first red/green PPM gradient into a [`Canvas`].
pub fn render_unit_gradient(width: u32, height: u32) -> Canvas {
    let denom_x = f64::from(width.saturating_sub(1)).max(1.0);
    let denom_y = f64::from(height.saturating_sub(1)).max(1.0);
    Canvas::from_fn(width, height, |x, y| {
        Rgb::from_raw_linear_color(LinearColor::new(
            f64::from(x) / denom_x,
            f64::from(y) / denom_y,
            0.0,
        ))
    })
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

/// Renders the first sphere scene from the book with a 16:9 camera.
pub fn render_first_sphere(image_width: u32) -> Canvas {
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO).render(first_sphere_color)
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
        material,
    ));
    world.add(Sphere::with_material(
        Point::new(0.0, -100.5, -1.0),
        100.0,
        material,
    ));
    world
}

/// Renders the normals-colored sphere and ground scene from the book.
pub fn render_normal_sphere_scene(image_width: u32) -> Canvas {
    let world = normal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .render_world_normals(&world)
}

/// Renders the diffuse sphere and ground scene from the book.
pub fn render_diffuse_sphere_scene(image_width: u32) -> Canvas {
    let world = normal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .render_world(&world)
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

/// Renders the mixed diffuse/metal sphere scene from the book.
pub fn render_metal_sphere_scene(image_width: u32) -> Canvas {
    let world = metal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .render_world(&world)
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

/// Renders the wide-angle camera test scene from the book.
pub fn render_wide_angle_sphere_scene(image_width: u32) -> Canvas {
    let world = wide_angle_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .with_vertical_fov(90.0)
        .render_world(&world)
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

/// Renders the hollow-glass dielectric sphere scene from the book.
pub fn render_dielectric_sphere_scene(image_width: u32) -> Canvas {
    let world = dielectric_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .with_vertical_fov(20.0)
        .with_look_at(Point::new(-2.0, 2.0, 1.0), Point::new(0.0, 0.0, -1.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .render_world(&world)
}

/// Renders the dielectric scene with defocus blur enabled.
pub fn render_defocus_sphere_scene(image_width: u32) -> Canvas {
    let world = dielectric_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(100)
        .with_max_depth(50)
        .with_vertical_fov(20.0)
        .with_look_at(Point::new(-2.0, 2.0, 1.0), Point::new(0.0, 0.0, -1.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_defocus_angle(10.0)
        .with_focus_distance(3.4)
        .render_world(&world)
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

/// Renders the final random-spheres scene from the book.
pub fn render_final_scene(image_width: u32) -> Canvas {
    render_final_scene_with_samples(image_width, 10)
}

/// Renders the final random-spheres scene with a caller-selected sample count.
///
/// The book uses 500 samples per pixel for the cover-quality image. The shorter
/// [`render_final_scene`] helper uses 10 samples so examples complete quickly.
pub fn render_final_scene_with_samples(image_width: u32, samples_per_pixel: u32) -> Canvas {
    let world = final_scene_ray_scene();
    PathTracer::new(
        RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(samples_per_pixel)
            .with_max_depth(50)
            .with_vertical_fov(20.0)
            .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
            .with_view_up(Vector::new(0.0, 1.0, 0.0))
            .with_defocus_angle(0.6)
            .with_focus_distance(10.0),
    )
    .render(&world)
}

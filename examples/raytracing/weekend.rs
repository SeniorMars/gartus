use gartus::{
    gmath::vector::{Point, Vector},
    graphics::{
        camera::RayCamera,
        colors::Rgb,
        display::Canvas,
        raytracing::{
            LinearColor, PathTracer, WIDESCREEN_ASPECT_RATIO,
            scenes::{
                dielectric_sphere_world, final_scene_bvh_world, first_sphere_color,
                metal_sphere_world, normal_sphere_world, wide_angle_sphere_world,
            },
        },
    },
};
use std::{error::Error, fs};

const WIDTH: u32 = 400;
const FINAL_WIDTH: u32 = 600;
const SAMPLES: u32 = 1_000;
const MAX_DEPTH: u32 = 50;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let renders = [
        ("unit_gradient", render_unit_gradient(WIDTH, WIDTH / 2)),
        ("first_sphere", render_first_sphere(WIDTH)),
        ("normal_sphere", render_normal_sphere_scene(WIDTH)),
        ("diffuse_sphere", render_diffuse_sphere_scene(WIDTH)),
        ("metal_spheres", render_metal_sphere_scene(WIDTH)),
        ("wide_angle_spheres", render_wide_angle_sphere_scene(WIDTH)),
        ("dielectric_spheres", render_dielectric_sphere_scene(WIDTH)),
        ("defocus_spheres", render_defocus_sphere_scene(WIDTH)),
        ("final_scene", render_final_scene(FINAL_WIDTH)),
    ];

    for (name, canvas) in renders {
        let path = format!("final/raytracing/{name}.ppm");
        canvas.save_binary(&path)?;
        println!("saved {path}");
    }

    Ok(())
}

fn render_unit_gradient(width: u32, height: u32) -> Canvas {
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

fn render_first_sphere(image_width: u32) -> Canvas {
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO).render(first_sphere_color)
}

fn render_normal_sphere_scene(image_width: u32) -> Canvas {
    let world = normal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(SAMPLES)
        .render_world_normals(&world)
}

fn render_diffuse_sphere_scene(image_width: u32) -> Canvas {
    let world = normal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(SAMPLES)
        .with_max_depth(MAX_DEPTH)
        .render_world(&world)
}

fn render_metal_sphere_scene(image_width: u32) -> Canvas {
    let world = metal_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(SAMPLES)
        .with_max_depth(MAX_DEPTH)
        .render_world(&world)
}

fn render_wide_angle_sphere_scene(image_width: u32) -> Canvas {
    let world = wide_angle_sphere_world();
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(SAMPLES)
        .with_max_depth(MAX_DEPTH)
        .with_vertical_fov(90.0)
        .render_world(&world)
}

fn render_dielectric_sphere_scene(image_width: u32) -> Canvas {
    let world = dielectric_sphere_world();
    viewpoint_camera(image_width, 20.0).render_world(&world)
}

fn render_defocus_sphere_scene(image_width: u32) -> Canvas {
    let world = dielectric_sphere_world();
    viewpoint_camera(image_width, 20.0)
        .with_defocus_angle(10.0)
        .with_focus_distance(3.4)
        .render_world(&world)
}

fn render_final_scene(image_width: u32) -> Canvas {
    let world = final_scene_bvh_world();
    PathTracer::new(
        RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(SAMPLES)
            .with_max_depth(MAX_DEPTH)
            .with_vertical_fov(20.0)
            .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
            .with_view_up(Vector::new(0.0, 1.0, 0.0))
            .with_defocus_angle(0.6)
            .with_focus_distance(10.0),
    )
    .render(&world)
}

fn viewpoint_camera(image_width: u32, vertical_fov: f64) -> RayCamera {
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(SAMPLES)
        .with_max_depth(MAX_DEPTH)
        .with_vertical_fov(vertical_fov)
        .with_look_at(Point::new(-2.0, 2.0, 1.0), Point::new(0.0, 0.0, -1.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
}

//! Recreates the major renders from *Ray Tracing: The Next Week*.
//!
//! Outputs PNG files to `final/raytracing/`. Use `cargo run --profile render --example
//! next_weekend` for full renders; lower `WIDTH`, `SAMPLES`, or `FINAL_SAMPLES` while iterating.

use gartus::{
    gmath::vector::{Point, Vector},
    graphics::{
        camera::RayCamera,
        colors::LinearRgb,
        display::Canvas,
        raytracing::{
            ImageTexture, PathTracer, WIDESCREEN_ASPECT_RATIO,
            scenes::{
                checkered_spheres_world, cornell_box_world, cornell_smoke_world,
                motion_blur_bvh_world, next_week_final_scene_world, perlin_spheres_world,
                quads_world, simple_light_world,
            },
        },
    },
};
use std::{error::Error, fs};

const WIDTH: u32 = 400;
const SAMPLES: u32 = 1_000;
const MAX_DEPTH: u32 = 50;
const FINAL_SAMPLES: u32 = 1_000;
const FINAL_MAX_DEPTH: u32 = 40;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;
    let earth_texture = ImageTexture::from_file("examples/data/images/earthmap.jpg")?;

    render_and_save("checkered_spheres", render_checkered_spheres)?;
    render_and_save("perlin_spheres", render_perlin_spheres)?;
    render_and_save("motion_blur_spheres", render_motion_blur_spheres)?;
    render_and_save("quads", render_quads)?;
    render_and_save("simple_light", render_simple_light)?;
    render_and_save("cornell_box", render_cornell_box)?;
    render_and_save("cornell_smoke", render_cornell_smoke)?;
    render_and_save("next_week_final", || {
        render_next_week_final_scene(earth_texture)
    })?;

    Ok(())
}

fn render_and_save(name: &str, render: impl FnOnce() -> Canvas) -> Result<(), Box<dyn Error>> {
    let canvas = render();
    let path = format!("final/raytracing/{name}.png");
    canvas.save_extension(&path)?;
    println!("saved {path}");
    Ok(())
}

fn render_checkered_spheres() -> Canvas {
    let world = checkered_spheres_world();
    render_two_sphere_texture_scene(&world)
}

fn render_perlin_spheres() -> Canvas {
    let world = perlin_spheres_world();
    render_two_sphere_texture_scene(&world)
}

fn render_two_sphere_texture_scene(world: &dyn gartus::graphics::raytracing::Hittable) -> Canvas {
    RayCamera::new(WIDTH, WIDESCREEN_ASPECT_RATIO)
        .with_samples_per_pixel(SAMPLES)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearRgb::new(0.70, 0.80, 1.00))
        .with_vertical_fov(20.0)
        .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .render_world(world)
}

fn render_motion_blur_spheres() -> Canvas {
    let world = motion_blur_bvh_world();
    PathTracer::new(
        RayCamera::new(WIDTH, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(SAMPLES)
            .with_max_depth(MAX_DEPTH)
            .with_background(LinearRgb::new(0.70, 0.80, 1.00))
            .with_vertical_fov(20.0)
            .with_look_at(Point::new(13.0, 2.0, 3.0), Point::new(0.0, 0.0, 0.0))
            .with_view_up(Vector::new(0.0, 1.0, 0.0))
            .with_defocus_angle(0.6)
            .with_focus_distance(10.0)
            .with_shutter_interval(0.0, 1.0),
    )
    .render(&world)
}

fn render_quads() -> Canvas {
    let world = quads_world();
    PathTracer::new(
        RayCamera::new(WIDTH, 1.0)
            .with_samples_per_pixel(SAMPLES)
            .with_max_depth(MAX_DEPTH)
            .with_background(LinearRgb::new(0.70, 0.80, 1.00))
            .with_vertical_fov(80.0)
            .with_look_at(Point::new(0.0, 0.0, 9.0), Point::new(0.0, 0.0, 0.0))
            .with_view_up(Vector::new(0.0, 1.0, 0.0)),
    )
    .render(&world)
}

fn render_simple_light() -> Canvas {
    let world = simple_light_world();
    PathTracer::new(
        RayCamera::new(WIDTH, WIDESCREEN_ASPECT_RATIO)
            .with_samples_per_pixel(SAMPLES)
            .with_max_depth(MAX_DEPTH)
            .with_background(LinearRgb::default())
            .with_vertical_fov(20.0)
            .with_look_at(Point::new(26.0, 3.0, 6.0), Point::new(0.0, 2.0, 0.0))
            .with_view_up(Vector::new(0.0, 1.0, 0.0)),
    )
    .render(&world)
}

fn render_cornell_box() -> Canvas {
    let world = cornell_box_world();
    render_cornell_scene(&world)
}

fn render_cornell_smoke() -> Canvas {
    let world = cornell_smoke_world();
    render_cornell_scene(&world)
}

fn render_cornell_scene(world: &dyn gartus::graphics::raytracing::Hittable) -> Canvas {
    PathTracer::new(
        RayCamera::new(WIDTH, 1.0)
            .with_samples_per_pixel(SAMPLES)
            .with_max_depth(MAX_DEPTH)
            .with_background(LinearRgb::default())
            .with_vertical_fov(40.0)
            .with_look_at(
                Point::new(278.0, 278.0, -800.0),
                Point::new(278.0, 278.0, 0.0),
            )
            .with_view_up(Vector::new(0.0, 1.0, 0.0)),
    )
    .render(world)
}

fn render_next_week_final_scene(earth_texture: ImageTexture) -> Canvas {
    let world = next_week_final_scene_world(earth_texture);
    PathTracer::new(
        RayCamera::new(WIDTH, 1.0)
            .with_samples_per_pixel(FINAL_SAMPLES)
            .with_max_depth(FINAL_MAX_DEPTH)
            .with_background(LinearRgb::default())
            .with_vertical_fov(40.0)
            .with_look_at(
                Point::new(478.0, 278.0, -600.0),
                Point::new(278.0, 278.0, 0.0),
            )
            .with_view_up(Vector::new(0.0, 1.0, 0.0)),
    )
    .render(&world)
}

use gartus::graphics::raytracing::{INFINITY, Interval};
use gartus::prelude::*;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

const ASPECT_RATIO: f64 = 16.0 / 9.0;

fn build_benchmark_scene() -> RayScene {
    let mut scene = RayScene::with_capacity(3, 130);
    let ground = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.45, 0.47, 0.43)));
    let warm = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.75, 0.45, 0.32)));
    let cool = scene.add_material(RayMaterial::lambertian(LinearColor::new(0.25, 0.45, 0.78)));

    scene.add_sphere(Point::new(0.0, -1000.65, -4.0), 1000.0, ground);

    for row in 0..8 {
        for column in 0..16 {
            let x = (f64::from(column) - 7.5) * 0.45;
            let z = -1.4 - f64::from(row) * 0.55;
            let y = -0.35 + 0.04 * f64::from((row + column) % 3);
            let material = if (row + column) % 2 == 0 { warm } else { cool };
            scene.add_sphere(Point::new(x, y, z), 0.16, material);
        }
    }

    scene.build_bvh_with_options(BvhBuildOptions::new().with_leaf_size(4));
    scene
}

fn benchmark_camera() -> RayCamera {
    RayCamera::new(96, ASPECT_RATIO)
        .with_samples_per_pixel(2)
        .with_max_depth(5)
        .with_vertical_fov(35.0)
        .with_look_at(Point::new(0.0, 1.1, 2.6), Point::new(0.0, -0.25, -3.4))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_rng_seed(0x5eed_2026)
}

fn primary_probe_rays(camera: RayCamera) -> Vec<Ray> {
    let mut rays = Vec::new();
    for y in (0..camera.image_height()).step_by(4) {
        for x in (0..camera.image_width()).step_by(4) {
            rays.push(camera.ray_for_pixel(x, y));
        }
    }
    rays
}

fn time_render(label: &str, tracer: PathTracer, scene: &RayScene) -> Duration {
    let start = Instant::now();
    let image = tracer.render_ray_scene(scene);
    let elapsed = start.elapsed();
    black_box(image.pixels());
    println!("{label:<28} {:>8.3} ms", elapsed.as_secs_f64() * 1_000.0);
    elapsed
}

fn time_progressive_render(label: &str, tracer: PathTracer, scene: &RayScene) -> Duration {
    let start = Instant::now();
    let mut updates = 0_usize;
    let image = tracer
        .render_ray_scene_progressive(scene, |update| {
            updates += 1;
            black_box(update.progress());
            Ok::<_, std::convert::Infallible>(())
        })
        .expect("progress callback is infallible");
    let elapsed = start.elapsed();
    black_box(image.pixels());
    println!(
        "{label:<28} {:>8.3} ms ({updates} tile updates)",
        elapsed.as_secs_f64() * 1_000.0
    );
    elapsed
}

fn print_bvh_stats(scene: &RayScene, camera: RayCamera) {
    let rays = primary_probe_rays(camera);
    let stats = scene.bvh_traversal_stats_for_rays(&rays, Interval::new(0.001, INFINITY));
    println!(
        "bvh stats: rays={} nodes={} node_hits={} leaves={} candidates={} leaf_hits={} max_stack={}",
        stats.rays,
        stats.node_bounds_tests,
        stats.node_bounds_hits,
        stats.leaf_visits,
        stats.primitive_candidates,
        stats.leaf_hit_results,
        stats.max_stack_depth
    );
}

fn main() {
    let scene = build_benchmark_scene();
    let camera = benchmark_camera();
    println!(
        "render benchmark scene: primitives={} bvh_nodes={:?}",
        scene.len(),
        scene.bvh_node_count()
    );
    print_bvh_stats(&scene, camera);

    for tile_size in [8, 16, 32] {
        let tracer =
            PathTracer::new(camera).with_options(RenderOptions::new().tile_size(tile_size));
        time_render(&format!("tile_size_{tile_size}"), tracer, &scene);
    }

    let tracer = PathTracer::new(camera).with_options(RenderOptions::new().tile_size(16));
    time_progressive_render("progressive_tile_16", tracer, &scene);
}

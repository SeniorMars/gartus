//! Prism Rain Conservatory.
//!
//! A looping path-traced animation of refractive glass rain, spectral light knots, a black
//! mirror-water floor, custom triangular prisms, a rotating mirrored lotus basin, and a pulsing
//! glass lens oracle.

use gartus::graphics::raytracing::BvhNode;
use gartus::prelude::*;
use std::{error::Error, f64::consts::PI, fs, sync::Arc};

const IMAGE_WIDTH: u32 = 480;
const ADAPTIVE_MIN_SAMPLES: u32 = 24;
const ADAPTIVE_MAX_SAMPLES: u32 = 224;
const ADAPTIVE_ERROR_THRESHOLD: f64 = 0.030;
const MAX_DEPTH: u32 = 14;
const FRAMES: usize = 36;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;
    let static_scene = Arc::new(build_static_scene());

    let options = AnimationRenderOptions::new(
        "anim",
        "prism-rain-conservatory-",
        FRAMES,
        "final/raytracing/prism_rain_conservatory.gif",
    )
    .delay_cs(4)
    .preview(12, "final/raytracing/prism_rain_conservatory.png")
    .unique_frame_dir(true)
    .show_progress(true);

    FrameRecorder::render_gif_auto(options, move |frame| Ok(render_frame(frame, &static_scene)))?;

    println!(
        "saved final/raytracing/prism_rain_conservatory.png and final/raytracing/prism_rain_conservatory.gif"
    );

    Ok(())
}

struct StaticScene {
    world: BvhNode,
    lights: WeightedSamplingTargetList,
}

struct DynamicScene {
    world: BvhNode,
    lights: WeightedSamplingTargetList,
}

fn render_frame(frame: usize, static_scene: &StaticScene) -> Canvas {
    let t = frame as f64 / FRAMES as f64;
    let dynamic_scene = build_dynamic_scene(t);
    let mut world = HittableLayers::with_capacity(2);
    world.add(&static_scene.world);
    world.add(&dynamic_scene.world);

    let mut lights = HittableLayers::with_capacity(2);
    lights.add(&static_scene.lights);
    lights.add(&dynamic_scene.lights);

    render_scene(&world, &lights)
}

fn build_static_scene() -> StaticScene {
    let mut world = HittableList::with_capacity(82);
    let mut lights = WeightedSamplingTargetList::with_capacity(1);

    let obsidian_water = Metal::new(color(0.018, 0.024, 0.030), 0.035);
    let floor_edge = Lambertian::new(color(0.04, 0.07, 0.07));
    let wall = Lambertian::marble(0.075, 419);
    let moss = Arc::new(Lambertian::new(color(0.03, 0.11, 0.08))) as MaterialRef;
    let brass = Arc::new(Metal::new(color(0.78, 0.55, 0.23), 0.18)) as MaterialRef;

    add_room(&mut world, obsidian_water, floor_edge, wall);
    add_skylight(&mut world, &mut lights);
    add_shelves_and_plants(&mut world, moss, brass);
    add_haze(&mut world);

    let world = world
        .into_bvh()
        .expect("static prism rain scene should contain only bounded objects");

    StaticScene { world, lights }
}

fn build_dynamic_scene(t: f64) -> DynamicScene {
    let mut world = HittableList::with_capacity(220);
    let mut lights = WeightedSamplingTargetList::with_capacity(36);

    let glass = Arc::new(Dielectric::new(RefractiveIndex::GLASS)) as MaterialRef;
    let diamond = Arc::new(Dielectric::new(RefractiveIndex::DIAMOND)) as MaterialRef;

    add_emissive_panes(&mut world, &mut lights, t);
    add_spectral_knots(&mut world, &mut lights, t);
    add_mirror_lotus(&mut world, t);
    add_prism_rain(&mut world, glass.clone(), diamond, t);
    add_lens_oracle(&mut world, &mut lights, glass, t);

    let world = world
        .into_bvh()
        .expect("dynamic prism rain scene should contain only bounded objects");

    DynamicScene { world, lights }
}

fn add_room(
    world: &mut HittableList,
    obsidian_water: Metal,
    floor_edge: Lambertian,
    wall: Lambertian,
) {
    world.add(Quad::with_material(
        Point::new(-7.5, 0.0, -7.5),
        Vector::new(15.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 15.0),
        obsidian_water,
    ));

    world.add(box_object(
        Point::new(-7.6, -0.38, -7.6),
        Point::new(7.6, 0.02, 7.6),
        Arc::new(floor_edge),
    ));

    world.add(Quad::with_material(
        Point::new(-7.5, 0.0, 7.5),
        Vector::new(15.0, 0.0, 0.0),
        Vector::new(0.0, 6.4, 0.0),
        wall.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(-7.5, 0.0, -7.5),
        Vector::new(0.0, 0.0, 15.0),
        Vector::new(0.0, 6.4, 0.0),
        wall.clone(),
    ));
    world.add(Quad::with_material(
        Point::new(7.5, 0.0, -7.5),
        Vector::new(0.0, 0.0, 15.0),
        Vector::new(0.0, 6.4, 0.0),
        wall,
    ));
}

fn add_emissive_panes(world: &mut HittableList, lights: &mut WeightedSamplingTargetList, t: f64) {
    let pane_specs = [
        (-5.8, 0.85, 1.10, 3.7, color(11.0, 1.20, 2.20)),
        (-4.0, 1.45, 0.82, 2.9, color(0.75, 8.6, 3.0)),
        (-2.55, 0.55, 0.72, 4.4, color(11.5, 6.4, 0.95)),
        (-1.08, 1.95, 0.56, 2.15, color(0.60, 5.8, 11.0)),
        (0.42, 0.70, 0.92, 3.8, color(12.5, 1.7, 6.3)),
        (2.05, 1.35, 0.76, 3.15, color(1.2, 9.6, 8.6)),
        (3.72, 0.78, 1.15, 4.05, color(11.8, 8.8, 1.2)),
        (5.6, 1.82, 0.64, 2.45, color(4.8, 1.6, 12.0)),
    ];

    for (index, (x, y, width, height, emit)) in pane_specs.into_iter().enumerate() {
        let pulse = 0.78 + 0.38 * (t * PI * 2.0 + index as f64 * 0.91).sin().max(0.0);
        let emit = emit * pulse;
        let corner = Point::new(x, y + height, 7.485);
        let u = Vector::new(width, 0.0, 0.0);
        let v = Vector::new(0.0, -height, 0.0);
        world.add(Quad::with_material(corner, u, v, DiffuseLight::new(emit)));
        lights.add_quad_weighted(corner, u, v, width * height * pulse);

        world.add(box_object(
            Point::new(x - 0.06, y - 0.06, 7.42),
            Point::new(x + width + 0.06, y + height + 0.06, 7.47),
            Arc::new(Metal::new(color(0.40, 0.28, 0.10), 0.25)),
        ));
    }
}

fn add_skylight(world: &mut HittableList, lights: &mut WeightedSamplingTargetList) {
    let skylight = Point::new(-2.4, 6.32, -0.8);
    let u = Vector::new(4.8, 0.0, 0.0);
    let v = Vector::new(0.0, 0.0, 3.0);
    world.add(Quad::with_material(
        skylight,
        u,
        v,
        DiffuseLight::new(color(1.3, 3.6, 5.4)),
    ));
    lights.add_quad_weighted(skylight, u, v, 8.0);
}

fn add_spectral_knots(world: &mut HittableList, lights: &mut WeightedSamplingTargetList, t: f64) {
    for strand in 0..3 {
        let phase = strand as f64 / 3.0 * PI * 2.0 + t * PI * 2.0 * (0.45 + strand as f64 * 0.08);
        for bead in 0..34 {
            let u = bead as f64 / 33.0;
            let angle = u * PI * 2.0 * 2.35 + phase;
            let envelope = (PI * u).sin().powf(0.42);
            let x = angle.cos() * (0.52 + 2.05 * envelope);
            let y = 0.86 + u * 4.75 + (angle * 1.7 + t * PI * 2.0).sin() * 0.18;
            let z = angle.sin() * (0.42 + 1.22 * envelope) - 0.55;
            let radius = 0.045 + 0.028 * (1.0 - (u - 0.5).abs() * 2.0);
            let hue = u * 300.0 + strand as f64 * 70.0 + t * 360.0;
            let emit = spectral_color(hue, 5.2 + 2.2 * envelope);
            let center = Point::new(x, y, z);
            world.add(Sphere::with_material(
                center,
                radius,
                DiffuseLight::new(emit),
            ));

            if bead % 4 == 0 {
                lights.add_sphere_weighted(center, radius, 0.35 + envelope);
            }
        }
    }
}

fn add_shelves_and_plants(world: &mut HittableList, moss: MaterialRef, brass: MaterialRef) {
    for row in 0..3 {
        let y = 0.72 + row as f64 * 1.36;
        world.add(box_object(
            Point::new(-6.4, y, 6.68),
            Point::new(6.4, y + 0.12, 6.95),
            brass.clone(),
        ));
    }

    for i in 0..34 {
        let x = -6.1 + i as f64 * 0.37;
        let row = i % 3;
        let z = 6.15 + (i % 5) as f64 * 0.07;
        let h = 0.28 + hash01(i, 11) * 0.86;
        let y = 0.84 + row as f64 * 1.36;
        let lean = (hash01(i, 19) - 0.5) * 0.22;
        world.add(box_object(
            Point::new(x - 0.025, y, z - 0.025),
            Point::new(x + 0.025 + lean, y + h, z + 0.025),
            moss.clone(),
        ));

        if i % 4 == 0 {
            world.add(Sphere::with_material(
                Point::new(x + lean, y + h + 0.06, z),
                0.065,
                Lambertian::new(color(0.06, 0.35 + hash01(i, 23) * 0.22, 0.12)),
            ));
        }
    }
}

fn add_mirror_lotus(world: &mut HittableList, t: f64) {
    let petal_material = Arc::new(Metal::new(color(0.86, 0.66, 0.36), 0.04)) as MaterialRef;

    for i in 0..18 {
        let a = i as f64 / 18.0 * PI * 2.0 + t * PI * 2.0 * 0.08;
        let width = 0.20 + 0.05 * (i % 3) as f64;
        let inner = 0.72;
        let outer = 2.42 + 0.18 * (i % 4) as f64;
        let lift = 0.012 * (t * PI * 2.0 + i as f64 * 0.45).sin();
        let p0 = lotus_point(a, inner, 0.032 + lift);
        let p1 = lotus_point(a - width, outer * 0.72, 0.042 + lift);
        let p2 = lotus_point(a, outer, 0.060 + lift);
        let p3 = lotus_point(a + width, outer * 0.72, 0.042 + lift);

        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon(p0, p1, p2);
        mesh.add_polygon(p0, p2, p3);
        mesh.add_polygon(p2, p1, p0);
        mesh.add_polygon(p3, p2, p0);
        world.add(TriangleMesh::from_shared_polygon_matrix(
            &mesh,
            petal_material.clone(),
        ));
    }
}

fn add_prism_rain(world: &mut HittableList, glass: MaterialRef, diamond: MaterialRef, t: f64) {
    for i in 0..64 {
        let seed = i as f64;
        let angle = seed * 2.399_963_229_728_653 + t * PI * 2.0 * 0.11;
        let radius = 0.55 + (i as f64 / 64.0).sqrt() * 5.4;
        let x = angle.cos() * radius + (hash01(i, 37) - 0.5) * 0.42;
        let z = angle.sin() * radius * 0.72 - 0.45 + (hash01(i, 41) - 0.5) * 0.55;
        let fall = (hash01(i, 43) + t * (0.21 + hash01(i, 67) * 0.16)).fract();
        let y = 1.05 + (1.0 - fall) * 4.95;
        let droplet_radius = 0.055 + hash01(i, 47) * 0.095;
        world.add(Sphere::with_material(
            Point::new(x, y, z),
            droplet_radius,
            Dielectric::new(RefractiveIndex::WATER),
        ));

        if i % 8 == 0 {
            let mesh = triangular_prism_mesh(
                Point::new(x * 0.82, y + 0.18, z * 0.82),
                0.18 + hash01(i, 53) * 0.14,
                0.76 + hash01(i, 59) * 0.55,
                angle + t * PI * 2.0 * 0.5,
            );
            world.add(TriangleMesh::from_shared_polygon_matrix(
                &mesh,
                if i % 14 == 0 {
                    diamond.clone()
                } else {
                    glass.clone()
                },
            ));
        }
    }
}

fn add_lens_oracle(
    world: &mut HittableList,
    lights: &mut WeightedSamplingTargetList,
    glass: MaterialRef,
    t: f64,
) {
    let lens_center = Point::new(0.0, 1.45, 0.05);
    world.add(Sphere::with_material(
        lens_center,
        1.12,
        Dielectric::new(RefractiveIndex::GLASS),
    ));

    let core = Point::new(0.0, 1.45, 0.05);
    let pulse = 0.75 + 0.35 * (t * PI * 2.0).sin().max(0.0);
    world.add(Sphere::with_material(
        core,
        0.16,
        DiffuseLight::new(color(5.2, 2.0, 0.85) * pulse),
    ));
    lights.add_sphere_weighted(core, 0.16, 3.0 * pulse);

    for i in 0..10 {
        let a = i as f64 / 10.0 * PI * 2.0 + t * PI * 2.0 * 0.16;
        let x = a.cos() * 1.48;
        let z = a.sin() * 1.08;
        let y = 1.45 + (i as f64 * 0.71).sin() * 0.32;
        let mesh = triangular_prism_mesh(Point::new(x, y, z), 0.12, 0.48, a + PI / 6.0);
        world.add(TriangleMesh::from_shared_polygon_matrix(
            &mesh,
            glass.clone(),
        ));
    }
}

fn add_haze(world: &mut HittableList) {
    let boundary = box_object(
        Point::new(-7.4, 0.02, -7.2),
        Point::new(7.4, 6.0, 7.2),
        Arc::new(Dielectric::new(RefractiveIndex::GLASS)),
    );
    world.add(ConstantMedium::new(
        boundary,
        0.003,
        color(0.45, 0.72, 0.88),
    ));
}

fn render_scene(world: &dyn Hittable, lights: &dyn Hittable) -> Canvas {
    let lookfrom = Point::new(5.4, 2.75, -8.2);
    let lookat = Point::new(0.0, 1.92, 0.72);

    PathTracer::new(
        RayCamera::new(IMAGE_WIDTH, 1.0)
            .with_adaptive_sampling(
                ADAPTIVE_MIN_SAMPLES,
                ADAPTIVE_MAX_SAMPLES,
                ADAPTIVE_ERROR_THRESHOLD,
            )
            .with_max_depth(MAX_DEPTH)
            .with_background(color(0.006, 0.010, 0.018))
            .with_vertical_fov(42.0)
            .with_look_at(lookfrom, lookat)
            .with_view_up(Vector::new(0.0, 1.0, 0.0))
            .with_defocus_angle(0.10)
            .with_focus_distance(8.45),
    )
    .render_with_lights(world, lights)
}

fn triangular_prism_mesh(center: Point, radius: f64, height: f64, yaw: f64) -> PolygonMatrix {
    let mut mesh = PolygonMatrix::new();
    mesh.add_prism_with_yaw((center.x(), center.y(), center.z()), 3, radius, height, yaw);
    mesh
}

fn lotus_point(angle: f64, radius: f64, y: f64) -> (f64, f64, f64) {
    (angle.cos() * radius, y, angle.sin() * radius - 0.16)
}

fn spectral_color(hue: f64, intensity: f64) -> LinearColor {
    let hue = hue.rem_euclid(360.0);
    let x = 1.0 - ((hue / 60.0) % 2.0 - 1.0).abs();
    let (r, g, b) = match hue {
        h if h < 60.0 => (1.0, x, 0.08),
        h if h < 120.0 => (x, 1.0, 0.10),
        h if h < 180.0 => (0.08, 1.0, x),
        h if h < 240.0 => (0.10, x, 1.0),
        h if h < 300.0 => (x, 0.10, 1.0),
        _ => (1.0, 0.08, x),
    };
    color(r * intensity, g * intensity, b * intensity)
}

fn color(red: f64, green: f64, blue: f64) -> LinearColor {
    LinearColor::new(red, green, blue)
}

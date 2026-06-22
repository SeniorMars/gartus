//! Path-traced stable-fluid smoke volume.
//!
//! This runs a small `StableFluidGrid2`, exports each frame as a thick `GridDensityField`, caches
//! the raw density grids, and renders one cached frame through `NonUniformMedium`.
//!
//! Outputs a PNG to `final/raytracing/simulated_smoke_volume.png`.

use gartus::prelude::*;
use std::{error::Error, fs, path::PathBuf, sync::Arc};

const IMAGE_WIDTH: u32 = 320;
const STRATIFIED_GRID_WIDTH: u32 = 6;
const MAX_DEPTH: u32 = 18;
const SIM_WIDTH: usize = 72;
const SIM_HEIGHT: usize = 72;
const SMOKE_DEPTH: usize = 15;
const SMOKE_FALLOFF: f64 = 1.35;
const FRAME: usize = 95;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;
    fs::create_dir_all(cache_dir())?;

    let canvas = render()?;
    let path = "final/raytracing/simulated_smoke_volume.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn render() -> Result<Canvas, Box<dyn Error>> {
    let (world, lights) = build_scene()?;
    let camera = RayCamera::new(IMAGE_WIDTH, 16.0 / 9.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.006, 0.010, 0.018))
        .with_vertical_fov(36.0)
        .with_look_at(Point::new(4.8, 3.0, 5.5), Point::new(0.0, 1.25, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    Ok(PathTracer::new(camera).render_with_lights(&world, &lights))
}

fn build_scene() -> Result<(HittableList, WeightedSamplingTargetList), Box<dyn Error>> {
    let mut world = HittableList::with_capacity(5);
    let mut lights = WeightedSamplingTargetList::with_capacity(2);

    world.add(Quad::with_material(
        Point::new(-7.0, -0.05, -5.0),
        Vector::new(14.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 10.0),
        Lambertian::checker(
            0.7,
            LinearColor::new(0.045, 0.052, 0.060),
            LinearColor::new(0.022, 0.027, 0.035),
        ),
    ));

    let key_light_corner = Point::new(-2.5, 4.5, -0.6);
    let key_light_u = Vector::new(5.0, 0.0, 0.0);
    let key_light_v = Vector::new(0.0, 0.0, 2.8);
    world.add(Quad::with_material(
        key_light_corner,
        key_light_u,
        key_light_v,
        DiffuseLight::new(LinearColor::new(5.2, 6.4, 8.6)),
    ));
    lights.add_quad_weighted(key_light_corner, key_light_u, key_light_v, 10.0);

    let accent_center = Point::new(-2.7, 0.35, 1.4);
    world.add(Sphere::with_material(
        accent_center,
        0.18,
        DiffuseLight::new(LinearColor::new(0.7, 2.0, 7.0)),
    ));
    lights.add_sphere_weighted(accent_center, 0.18, 2.0);

    let smoke_bounds = smoke_bounds();
    let smoke_density = cached_smoke_frame(FRAME)?;
    world.add(NonUniformMedium::with_phase_function(
        box_object(
            smoke_bounds.min,
            smoke_bounds.max,
            Arc::new(Lambertian::new(LinearColor::new(0.0, 0.0, 0.0))),
        ),
        smoke_density,
        Arc::new(HenyeyGreenstein::new(
            LinearColor::new(0.72, 0.84, 1.0),
            0.45,
        )),
    ));

    Ok((world, lights))
}

fn cached_smoke_frame(frame: usize) -> Result<GridDensityField, Box<dyn Error>> {
    let smoke_bounds = smoke_bounds();
    let path = cache_path(frame);
    if path.exists() {
        return Ok(GridDensityField::load_grid(path)?);
    }

    let mut sim = StableFluidGrid2::new([SIM_WIDTH, SIM_HEIGHT])
        .with_dt(1.0 / 30.0)
        .with_diffusion(0.0002)
        .with_viscosity(0.00004)
        .with_solver_iterations(20);

    for step in 0..=frame {
        inject_smoke(&mut sim, step);
        sim.step();
        let grid = sim.to_density_volume(smoke_bounds, SMOKE_DEPTH, SMOKE_FALLOFF);
        grid.save_grid_with_frame(cache_path(step), step)?;
    }

    Ok(GridDensityField::load_grid(path)?)
}

fn inject_smoke(sim: &mut StableFluidGrid2, step: usize) {
    let step_f = f64::from(u32::try_from(step).expect("step fits u32"));
    let sway = (step_f * 0.19).sin();
    let source = [
        f64::from(u32::try_from(SIM_WIDTH).expect("width fits u32")) * 0.5 + 2.4 * sway,
        f64::from(u32::try_from(SIM_HEIGHT).expect("height fits u32")) * 0.18,
    ];
    sim.apply_emitter(
        StableFluidEmitter::new(source, 3.8)
            .with_density(0.36)
            .with_velocity([1.1 * sway, 9.5]),
    );
    sim.add_radial_impulse(source, 6.5, 0.45);
    sim.add_wind_field(|cell| {
        let height = cell[1] / f64::from(u32::try_from(SIM_HEIGHT - 1).expect("height fits u32"));
        [0.004 * (height * 5.0 + step_f * 0.11).sin(), 0.0]
    });
    sim.apply_buoyancy(0.18);
    sim.apply_vorticity_confinement(5.0);
}

fn smoke_bounds() -> GridBounds {
    GridBounds::new(Point::new(-1.9, 0.0, -0.5), Point::new(1.9, 3.4, 0.5))
}

fn cache_dir() -> PathBuf {
    PathBuf::from("final/raytracing/cache/stable_fluid_volume")
}

fn cache_path(frame: usize) -> PathBuf {
    cache_dir().join(format!("smoke_{frame:04}.gdf"))
}

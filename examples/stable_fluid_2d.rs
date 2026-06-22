//! 2D stable-fluid smoke preview.
//!
//! This runs `StableFluidGrid2` directly and writes a quick grayscale/blue preview. Use the
//! raytracing example to render the same simulation as participating media.
//!
//! Outputs a PNG to `final/raytracing/stable_fluid_2d.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const WIDTH: usize = 320;
const HEIGHT: usize = 192;
const STEPS: usize = 150;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let sim = simulated_smoke();
    let canvas = preview_canvas(&sim);
    let path = "final/raytracing/stable_fluid_2d.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn simulated_smoke() -> StableFluidGrid2 {
    let mut sim = StableFluidGrid2::new([WIDTH, HEIGHT])
        .with_dt(1.0 / 24.0)
        .with_diffusion(0.00008)
        .with_viscosity(0.00003)
        .with_solver_iterations(28);
    sim.set_solid_circle(
        [
            f64::from(u32::try_from(WIDTH).expect("width fits u32")) * 0.53,
            f64::from(u32::try_from(HEIGHT).expect("height fits u32")) * 0.45,
        ],
        18.0,
        true,
    );

    for step in 0..STEPS {
        let step_f = f64::from(u32::try_from(step).expect("step fits u32"));
        let sway = (step_f * 0.11).sin();
        let source = [
            f64::from(u32::try_from(WIDTH).expect("width fits u32")) * 0.5 + 8.0 * sway,
            f64::from(u32::try_from(HEIGHT).expect("height fits u32")) * 0.2,
        ];
        sim.apply_emitter(
            StableFluidEmitter::new(source, 9.0)
                .with_density(3.1)
                .with_velocity([7.0 * sway, 30.0]),
        );
        sim.add_radial_impulse(source, 14.0, 1.2);
        sim.add_wind_field(|cell| {
            let height = cell[1] / f64::from(u32::try_from(HEIGHT - 1).expect("height fits u32"));
            [0.006 * (height * 8.0 + step_f * 0.09).sin(), 0.0]
        });
        sim.apply_buoyancy(0.35);
        sim.apply_vorticity_confinement(2.6);
        sim.step();
    }

    sim
}

fn preview_canvas(sim: &StableFluidGrid2) -> Canvas {
    let max_density = sim
        .densities()
        .iter()
        .map(|density| f64::from(*density))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let mut pixels = Vec::with_capacity(WIDTH * HEIGHT);

    for y in 0..HEIGHT {
        let sim_y = HEIGHT - 1 - y;
        for x in 0..WIDTH {
            if sim.is_solid([x, sim_y]) {
                pixels.push(Rgb::new(13, 18, 24));
                continue;
            }
            let density = sim.density_at([x, sim_y]) / max_density;
            let value = density.powf(0.35).clamp(0.0, 1.0);
            pixels.push(Rgb::new(
                to_byte(8.0 + 48.0 * value + 90.0 * value * value),
                to_byte(12.0 + 108.0 * value + 105.0 * value * value),
                to_byte(24.0 + 165.0 * value + 66.0 * value * value),
            ));
        }
    }

    Canvas::from_pixels_rgb_only(
        u32::try_from(WIDTH).expect("preview width fits u32"),
        u32::try_from(HEIGHT).expect("preview height fits u32"),
        pixels,
        true,
        false,
    )
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn to_byte(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

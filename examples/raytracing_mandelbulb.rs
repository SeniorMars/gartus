//! Path-traced Mandelbulb example using the library's generic SDF ray marcher.
//!
//! The Mandelbulb distance estimator is intentionally local to this example. The reusable library
//! piece is `DistanceField` + `SdfObject`, which lets examples and applications bring their own
//! fractal fields without growing the main API around one specific shape.
//!
//! Outputs a PNG to `final/raytracing/mandelbulb_reliquary.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 420;
const STRATIFIED_GRID_WIDTH: u32 = 10;
const MAX_DEPTH: u32 = 8;

#[derive(Clone, Copy, Debug)]
struct Mandelbulb {
    center: Point,
    radius: f64,
    power: f64,
    iterations: usize,
    bailout: f64,
}

impl Mandelbulb {
    const NORMALIZED_RADIUS: f64 = 1.5;

    fn new(center: Point, radius: f64) -> Self {
        Self {
            center,
            radius,
            power: 8.0,
            iterations: 12,
            bailout: 8.0,
        }
    }

    fn scale(self) -> f64 {
        self.radius / Self::NORMALIZED_RADIUS
    }
}

impl DistanceField for Mandelbulb {
    #[allow(clippy::many_single_char_names)]
    fn distance(&self, point: Point) -> f64 {
        let scale = self.scale();
        let seed = (point - self.center) / scale;
        let mut z = seed;
        let mut derivative = 1.0;
        let mut current_radius = 0.0;

        for _ in 0..self.iterations {
            current_radius = z.length();
            if current_radius > self.bailout {
                break;
            }
            if current_radius <= f64::EPSILON {
                return 0.0;
            }

            let theta = (z.z() / current_radius).acos() * self.power;
            let phi = z.y().atan2(z.x()) * self.power;
            let radius_power = current_radius.powf(self.power);
            derivative = current_radius.powf(self.power - 1.0) * self.power * derivative + 1.0;

            let sin_theta = theta.sin();
            z = radius_power
                * Vector::new(sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos())
                + seed;
        }

        if current_radius <= f64::EPSILON || derivative <= f64::EPSILON {
            0.0
        } else {
            0.5 * current_radius.ln() * current_radius / derivative * scale
        }
    }

    fn bounds(&self) -> Bounds3 {
        Bounds3::new(
            (
                self.center.x() - self.radius,
                self.center.y() - self.radius,
                self.center.z() - self.radius,
            ),
            (
                self.center.x() + self.radius,
                self.center.y() + self.radius,
                self.center.z() + self.radius,
            ),
        )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let (world, lights) = build_scene();
    let camera = RayCamera::new(IMAGE_WIDTH, 1.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.006, 0.008, 0.013))
        .with_vertical_fov(34.0)
        .with_look_at(Point::new(4.1, 2.15, 5.0), Point::new(0.0, 0.4, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0));

    let canvas = PathTracer::new(camera).render_with_lights(&world, &lights);
    let path = "final/raytracing/mandelbulb_reliquary.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn build_scene() -> (HittableList, WeightedSamplingTargetList) {
    let mut world = HittableList::with_capacity(8);
    let mut lights = WeightedSamplingTargetList::with_capacity(3);

    let floor = Lambertian::new(LinearColor::new(0.12, 0.115, 0.105));
    let back_wall = Lambertian::new(LinearColor::new(0.035, 0.043, 0.06));
    let fractal_material = Lambertian::new(LinearColor::new(0.58, 0.48, 0.9));
    let left_glow = DiffuseLight::new(LinearColor::new(7.0, 1.1, 5.8));
    let right_glow = DiffuseLight::new(LinearColor::new(0.9, 5.4, 8.0));

    world.add(Quad::with_material(
        Point::new(-4.0, -1.15, -4.0),
        Vector::new(8.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 8.0),
        floor,
    ));
    world.add(Quad::with_material(
        Point::new(-4.0, -1.15, -2.2),
        Vector::new(8.0, 0.0, 0.0),
        Vector::new(0.0, 5.0, 0.0),
        back_wall,
    ));

    let bulb = Mandelbulb::new(Point::new(0.0, 0.0, 0.0), 1.65);
    world.add(
        SdfObject::new(bulb, fractal_material)
            .with_epsilon(0.0007)
            .with_normal_epsilon(0.0015)
            .with_max_steps(320),
    );

    let left_light = Point::new(-1.8, 1.4, 0.9);
    let right_light = Point::new(1.7, 0.2, 1.2);
    world.add(Sphere::with_material(left_light, 0.22, left_glow));
    world.add(Sphere::with_material(right_light, 0.18, right_glow));
    lights.add_sphere_weighted(left_light, 0.22, 5.0);
    lights.add_sphere_weighted(right_light, 0.18, 4.0);

    (world, lights)
}

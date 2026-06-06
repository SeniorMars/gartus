use gartus::{
    gmath::vector::{Point, Vector},
    graphics::{
        camera::RayCamera,
        colors::LinearRgb,
        display::Canvas,
        lighting::RefractiveIndex,
        raytracing::{
            Dielectric, DiffuseLight, HittableList, Lambertian, MaterialRef, PathTracer, Quad,
            RotateY, Sphere, Translate, box_object,
        },
    },
};
use std::{error::Error, fs, sync::Arc};

const IMAGE_WIDTH: u32 = 600;
const SAMPLES_PER_PIXEL: u32 = 1_000;
const MAX_DEPTH: u32 = 50;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let (world, sampling_targets) = life_final_scene();
    let canvas = render_scene(&world, &sampling_targets);
    let path = "final/raytracing/life_final.ppm";
    canvas.save_binary(path)?;
    println!("saved {path}");

    Ok(())
}

fn life_final_scene() -> (HittableList, HittableList) {
    let red = Lambertian::new(LinearRgb::new(0.65, 0.05, 0.05));
    let white = Lambertian::new(LinearRgb::new(0.73, 0.73, 0.73));
    let white_shared: MaterialRef = Arc::new(white.clone());
    let green = Lambertian::new(LinearRgb::new(0.12, 0.45, 0.15));
    let light = DiffuseLight::new(LinearRgb::new(15.0, 15.0, 15.0));
    let glass = Dielectric::new(RefractiveIndex::GLASS);

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
                white_shared,
            ),
            15.0,
        ),
        Vector::new(265.0, 0.0, 295.0),
    ));
    world.add(Sphere::with_material(
        Point::new(190.0, 90.0, 190.0),
        90.0,
        glass,
    ));

    let mut sampling_targets = HittableList::with_capacity(2);
    sampling_targets.add(Quad::new(
        Point::new(343.0, 554.0, 332.0),
        Vector::new(-130.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, -105.0),
    ));
    sampling_targets.add(Sphere::new(Point::new(190.0, 90.0, 190.0), 90.0));

    (world, sampling_targets)
}

fn render_scene(
    world: &dyn gartus::graphics::raytracing::Hittable,
    lights: &HittableList,
) -> Canvas {
    PathTracer::new(
        RayCamera::new(IMAGE_WIDTH, 1.0)
            .with_samples_per_pixel(SAMPLES_PER_PIXEL)
            .with_stratified_sampling()
            .with_max_depth(MAX_DEPTH)
            .with_background(LinearRgb::default())
            .with_vertical_fov(40.0)
            .with_look_at(
                Point::new(278.0, 278.0, -800.0),
                Point::new(278.0, 278.0, 0.0),
            )
            .with_view_up(Vector::new(0.0, 1.0, 0.0)),
    )
    .render_with_lights(world, lights)
}

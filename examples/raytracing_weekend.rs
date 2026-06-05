use gartus::prelude::render_first_sphere;
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let canvas = render_first_sphere(400);
    let output = "final/raytracing_weekend.ppm";
    canvas.save_ascii(output)?;
    println!("saved {output}");

    Ok(())
}

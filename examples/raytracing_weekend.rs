use gartus::prelude::render_final_scene;
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let canvas = render_final_scene(400);
    let output = "final/raytracing_weekend.ppm";
    canvas.save_ascii(output)?;
    println!("saved {output}");

    Ok(())
}

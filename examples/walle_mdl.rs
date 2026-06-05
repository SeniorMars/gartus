use gartus::{
    graphics::colors::Rgb,
    mdl::{RenderConfig, run_file},
};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let output = "pics/walle.png";
    fs::create_dir_all("pics")?;

    run_file(
        "scripts/walle.mdl",
        RenderConfig::new_with_bg(500, 500, Rgb::BLACK, Rgb::WHITE)
            .display_enabled(false)
            .wrapped(false)
            .save_override(output),
    )?;

    println!("rendered scripts/walle.mdl to {output}");
    Ok(())
}

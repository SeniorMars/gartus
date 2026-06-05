use gartus::{
    graphics::{animation::AnimationRenderOptions, colors::Rgb},
    mdl::{RenderConfig, compile_file, executor::execute_compiled_gif_with_options},
};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let compiled = compile_file("scripts/student.mdl").map_err(|errors| {
        errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let options = AnimationRenderOptions::new(
        "final/student_frames",
        "student-",
        compiled.animation().frames(),
        "final/student.gif",
    )
    .delay_cs(3)
    .preview(30, "final/student.png")
    .unique_frame_dir(true);

    execute_compiled_gif_with_options(
        &compiled,
        RenderConfig::new_with_bg(500, 500, Rgb::WHITE, Rgb::BLACK)
            .display_enabled(false)
            .wrapped(false),
        options,
    )?;

    println!("rendered scripts/student.mdl to final/student.gif and final/student.png");
    Ok(())
}

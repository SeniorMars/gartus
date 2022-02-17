use std::process::Command;

use crate::graphics::{
    colors::{ColorSpace, Rgb},
    display::Canvas,
};
/// Returns a new animation given a file name prefix.
/// TODO: update documentation
/// # Arguments
///
/// * `file_name_prefix` - The prefix of the name the animation belongs to
/// * `output` - The final name of the animation
///
/// # Examples
///
/// Basic usage:
/// ```no_run
/// use crate::curves_rs::utils;
/// use crate::curves_rs::prelude::{Canvas, Rgb};
/// use crate::curves_rs::graphics::display::{AnimationConfig, CanvasConfig};
/// let file_prefix = "test";
/// let purplish = Rgb::new(17, 46, 81);
/// let mut canvas = Canvas::new_with_bg(512, 512, 255, purplish);
/// canvas.set_config(CanvasConfig::new(true, false));
/// canvas.set_animation(AnimationConfig::new(file_prefix.to_string()));
/// utils::animation(&canvas, "final.gif");
/// ```
pub fn animation<C>(canvas: &Canvas<C>, output: &str)
where
    C: ColorSpace,
    Rgb: From<C>,
{
    println!("Making a new animation: {}", output);
    Command::new("convert")
        .arg("-delay")
        .arg("1.2")
        .arg(&format!("./anim/{}*", canvas.config().file_prefix()))
        .arg(output)
        .spawn()
        .unwrap()
        .wait()
        .expect("Could not make animation");
}

/// Open's a given animation
///
/// # Arguments
///
/// * `file_name` - The animation to open.
///
/// # Examples
///
/// Basic usage:
/// ```no_run
/// use crate::curves_rs::utils;
/// utils::view_animation("owl.gif");
/// ```
pub fn view_animation(file_name: &str) {
    // animate doesn't play nicely
    println!("Playing animation: {}", &file_name);
    // Command::new("animate")
    Command::new("open")
        .arg(&file_name)
        .spawn()
        .expect("Could not view animation");
}

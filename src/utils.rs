use std::process::{Command, Stdio};
/// Returns a new animation given a file name prefix.
///
/// # Arguments
///
/// * `file_name_prefix` - The prefix of the name the animation belongs to
/// * `output` - The final name of the animation
///
/// # Examples
///
/// Basic usage:
/// ```
/// use crate::curves_rs::utils;
/// utils::animation("cool_picture", "gifs/final.gif");
/// ```
pub fn animation(file_name_prefix: &str, output: &str) {
    println!("Making a new animation: {}", output);
    let mut child = Command::new("convert")
        .arg("-delay")
        .arg("1.2")
        .arg(&format!("anim/{}*", file_name_prefix))
        .arg(output)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.wait().expect("Could not make animation");
}

/// Open's a given animation using imagemagick's `animate`.
///
/// # Arguments
///
/// * `file_name` - The animation to open.
///
/// # Examples
///
/// Basic usage:
/// ```
/// use crate::curves_rs::utils;
/// utils::view_animation("final.gif");
/// ```
pub fn view_animation(file_name: &str) {
    // animate doesn't play nicely
    println!("Playing animation: {}", &file_name);
    // Command::new("animate")
    Command::new("sxiv")
        .arg(&file_name)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Could not view animation");
}

#[allow(dead_code)]
/// Returns and calcuates the new x,y corrdinates from the polar corrdinate system
///
/// # Arguments
///
/// * `magnitude` - A f64 number that represents the magnitude of R in the polar corrdinate system
/// * `angle_degrees` - A f64 number that represents theta in the polar corrdinate system
///
pub(crate) fn polar_to_xy(magnitude: f64, theta: f64) -> (f64, f64) {
    let (dy, dx) = theta.to_radians().sin_cos();
    (dx * magnitude, dy * magnitude)
}

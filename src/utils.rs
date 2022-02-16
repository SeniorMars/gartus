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
    Command::new("convert")
        .arg("-delay")
        .arg("1.2")
        .arg(&format!("./anim/{}*", file_name_prefix))
        .arg(output)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap()
        .wait()
        .expect("Could not make animation");
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

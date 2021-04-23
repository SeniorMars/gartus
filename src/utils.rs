use std::process::{Command, Stdio};

// TODO: make a better version
#[allow(dead_code)]
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
/// use curves_rs::utils;
/// utils::animation("cool_picture", "final.gif");
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
        .spawn().unwrap();
    let _result = child.wait().expect("Could not make animation");
}

#[allow(dead_code)]
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
/// use curves_rs::utils;
/// utils::view_animation("final.gif");
/// ```
pub fn view_animation(file_name: &str) {
    // animate doesn't play nicely
    println!("Playing animation: {}", &file_name);
    Command::new("animate")
        .arg(&file_name)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn().expect("Could not view animation");
}

use std::process::{Command, Stdio};

// TODO: make a better version
#[allow(dead_code)]
pub fn animation(file_name_prefix: &str, output: &str) {
    println!("Making a new animation: {}", output);
    let mut child = Command::new("convert")
        .arg("-delay")
        .arg("2.7")
        .arg(&format!("anim/{}*", file_name_prefix))
        .arg(output)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn().unwrap();
    let _result = child.wait().unwrap();
}

#[allow(dead_code)]
pub fn view_animation(file_name: &str) {
    // animate doesn't play nicely
    let mut child = Command::new("animate")
        .arg(file_name)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn().unwrap();
    let _result = child.wait().unwrap();
}

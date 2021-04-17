use std::io;
use std::process::{Child, Command, Stdio};

#[allow(dead_code)]
pub fn view_animation(file_name: &str) -> io::Result<Child> {
    Command::new("animate")
        .arg(file_name)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
}

// TODO: make a better version
#[allow(dead_code)]
pub fn animation(file_name_prefix: &str, output: &str) -> io::Result<Child> {
    println!("Making a new animation: {}", output);
    Command::new("convert")
        .arg("-delay")
        .arg("2.7")
        .arg(&format!("anim/{}*", file_name_prefix))
        .arg(output)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
}

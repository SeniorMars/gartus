use crate::graphics::animation::FrameRecorder;
use std::io;
use std::process::{Child, Command, Stdio};

#[allow(dead_code)]
pub fn view_animation(file_name: &str) -> io::Result<Child> {
    // animate doesn't play nicely
    Command::new("open")
        .arg(file_name)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
}

#[allow(dead_code)]
pub fn animation(recorder: &FrameRecorder, output: &str) -> io::Result<()> {
    println!("Making a new animation: {}", output);
    recorder.encode_gif(output)
}

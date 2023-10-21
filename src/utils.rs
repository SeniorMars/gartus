use std::{io, path::Path, process::Command};

use crate::graphics::animation::FrameRecorder;
/// Returns a new animation given a file name prefix.
///
/// # Arguments
///
/// * `frame_prefix` - The prefix of frames inside `anim/`
/// * `output` - The final name of the animation
///
/// # Errors
/// Returns an error if `ImageMagick` fails or no matching frames exist.
pub fn animation_from_prefix(frame_prefix: &str, output: &str) -> io::Result<()> {
    println!("Making a new animation: {output}");
    encode_existing_frames(frame_prefix, output, 2)?;
    println!("Animation completed");
    Ok(())
}

/// Encodes an explicit recorder's frames.
///
/// # Errors
/// Returns an error if `ImageMagick` fails.
pub fn animation(recorder: &FrameRecorder, output: &str) -> io::Result<()> {
    println!("Making a new animation: {output}");
    recorder.encode_gif(output)?;
    println!("Animation completed");
    Ok(())
}

fn encode_existing_frames(frame_prefix: &str, output: &str, delay_cs: u16) -> io::Result<()> {
    let mut frames = std::fs::read_dir("anim")?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    frames.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(frame_prefix))
    });
    frames.sort();
    if frames.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no frames found in anim/ with prefix `{frame_prefix}`"),
        ));
    }

    let mut command = Command::new("magick");
    command.arg("-delay").arg(delay_cs.to_string());
    for frame in frames {
        command.arg(frame);
    }
    command.arg(Path::new(output));

    let status = command.status().map_err(|err| {
        io::Error::new(
            err.kind(),
            format!(
                "failed to run ImageMagick `magick`; is ImageMagick installed and in PATH? {err}"
            ),
        )
    })?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "ImageMagick `magick` failed with status {status} while encoding animation"
        )));
    }
    Ok(())
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
/// use crate::gartus::utils;
/// utils::view_animation("owl.gif");
/// ```
/// # Panics
/// Panics if the `open` command cannot be spawned or waited on.
pub fn view_animation(file_name: &str) {
    // animate doesn't play nicely
    println!("Playing animation: {}", &file_name);
    // Command::new("animate")
    Command::new("open")
        .arg(file_name)
        .spawn()
        .expect("Could not view animation")
        .wait()
        .expect("Could not wait for animation process");
}

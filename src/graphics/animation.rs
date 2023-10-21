use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix},
    graphics::display::Canvas,
};
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

/// Explicit frame recorder for animations.
#[derive(Debug, Clone)]
pub struct FrameRecorder {
    dir: PathBuf,
    prefix: String,
    delay_cs: u16,
    frame_index: usize,
}

impl FrameRecorder {
    /// Creates a recorder that writes frames as PPM files into `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>, prefix: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            prefix: prefix.into(),
            delay_cs: 2,
            frame_index: 0,
        }
    }

    /// Sets the GIF delay in centiseconds.
    #[must_use]
    pub fn with_delay(mut self, delay_cs: u16) -> Self {
        self.delay_cs = delay_cs;
        self
    }

    /// Returns the next frame index.
    #[must_use]
    pub fn frame_index(&self) -> usize {
        self.frame_index
    }

    /// Captures the current canvas as the next frame.
    ///
    /// # Errors
    /// Returns `Err` if the directory cannot be created or the PPM file cannot be written.
    pub fn capture(&mut self, canvas: &Canvas) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.dir)?;
        let path = self.frame_path(self.frame_index);
        canvas.save_binary(path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "animation path is not valid UTF-8",
            )
        })?)?;
        self.frame_index += 1;
        Ok(path)
    }

    /// Captures a frame after drawing transformed edges onto a clone of `base`.
    ///
    /// # Errors
    /// Returns `Err` if writing the frame file fails.
    pub fn capture_drawn(
        &mut self,
        base: &Canvas,
        edges: &EdgeMatrix,
        transform: &Matrix,
    ) -> io::Result<PathBuf> {
        let mut frame = base.clone();
        frame.draw_transformed(edges, transform);
        self.capture(&frame)
    }

    /// Encodes captured frames to a GIF using `ImageMagick` `magick`.
    ///
    /// # Errors
    /// Returns `Err` if no frames have been captured, if `magick` cannot be spawned, or if it exits with a failure status.
    pub fn encode_gif(&self, output: impl AsRef<Path>) -> io::Result<()> {
        if self.frame_index == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot encode animation with no captured frames",
            ));
        }

        let mut command = Command::new("magick");
        command.arg("-delay").arg(self.delay_cs.to_string());
        for idx in 0..self.frame_index {
            command.arg(self.frame_path(idx));
        }
        command.arg(output.as_ref());

        let status = command.status().map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("failed to run ImageMagick `magick`; is ImageMagick installed and in PATH? {err}"),
            )
        })?;

        if !status.success() {
            return Err(io::Error::other(format!(
                "ImageMagick `magick` failed with status {status} while encoding animation"
            )));
        }
        Ok(())
    }

    fn frame_path(&self, index: usize) -> PathBuf {
        self.dir.join(format!("{}{:08}.ppm", self.prefix, index))
    }
}

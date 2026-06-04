use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix},
    graphics::display::Canvas,
};
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

/// Explicit frame recorder for animations.
#[derive(Debug, Clone)]
pub struct FrameRecorder {
    dir: PathBuf,
    prefix: String,
    delay_cs: u16,
    frame_index: usize,
}

/// Options for rendering a whole GIF through [`FrameRecorder::render_gif`].
#[derive(Debug, Clone)]
pub struct AnimationRenderOptions {
    dir: PathBuf,
    prefix: String,
    frames: usize,
    output: PathBuf,
    delay_cs: u16,
    preview: Option<(usize, PathBuf)>,
    cleanup_frames: bool,
    clear_existing_frames: bool,
    unique_frame_dir: bool,
}

impl AnimationRenderOptions {
    /// Creates options for a GIF render.
    #[must_use]
    pub fn new(
        dir: impl Into<PathBuf>,
        prefix: impl Into<String>,
        frames: usize,
        output: impl Into<PathBuf>,
    ) -> Self {
        Self {
            dir: dir.into(),
            prefix: prefix.into(),
            frames,
            output: output.into(),
            delay_cs: 2,
            preview: None,
            cleanup_frames: true,
            clear_existing_frames: true,
            unique_frame_dir: false,
        }
    }

    /// Sets the GIF delay in centiseconds.
    #[must_use]
    pub fn delay_cs(mut self, delay_cs: u16) -> Self {
        self.delay_cs = delay_cs;
        self
    }

    /// Saves a preview still from `frame_index` to `output`.
    #[must_use]
    pub fn preview(mut self, frame_index: usize, output: impl Into<PathBuf>) -> Self {
        self.preview = Some((frame_index, output.into()));
        self
    }

    /// Sets whether generated PPM frames are deleted after a successful GIF encode.
    #[must_use]
    pub fn cleanup_frames(mut self, cleanup_frames: bool) -> Self {
        self.cleanup_frames = cleanup_frames;
        self
    }

    /// Sets whether existing matching frame files are removed before rendering.
    #[must_use]
    pub fn clear_existing_frames(mut self, clear_existing_frames: bool) -> Self {
        self.clear_existing_frames = clear_existing_frames;
        self
    }

    /// Uses a unique subdirectory under `dir` for this render's temporary frames.
    #[must_use]
    pub fn unique_frame_dir(mut self, unique_frame_dir: bool) -> Self {
        self.unique_frame_dir = unique_frame_dir;
        self
    }
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

    /// Removes existing PPM frames that match this recorder's prefix.
    ///
    /// # Errors
    /// Returns `Err` if the frame directory cannot be read or a matching frame cannot be deleted.
    pub fn clear_existing_frames(&self) -> io::Result<usize> {
        if !self.dir.exists() {
            return Ok(0);
        }

        let mut removed = 0;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if file_name.starts_with(&self.prefix)
                && path.extension().is_some_and(|ext| ext == "ppm")
            {
                fs::remove_file(path)?;
                removed += 1;
            }
        }
        Ok(removed)
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

    /// Encodes captured frames to a GIF and then removes the generated PPM frames.
    ///
    /// # Errors
    /// Returns `Err` if encoding fails or the frame cleanup fails.
    pub fn encode_gif_and_cleanup(&self, output: impl AsRef<Path>) -> io::Result<()> {
        self.encode_gif(output)?;
        self.clear_existing_frames()?;
        Ok(())
    }

    /// Renders frames with `render`, encodes a GIF, optionally saves a preview, and cleans up frames.
    ///
    /// # Errors
    /// Returns `Err` if frame rendering, frame capture, preview saving, GIF encoding, or cleanup fails.
    pub fn render_gif<F>(options: AnimationRenderOptions, mut render: F) -> io::Result<()>
    where
        F: FnMut(usize) -> io::Result<Canvas>,
    {
        let AnimationRenderOptions {
            dir,
            prefix,
            frames,
            output,
            delay_cs,
            preview,
            cleanup_frames,
            clear_existing_frames,
            unique_frame_dir,
        } = options;

        if frames == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot render animation with zero frames",
            ));
        }

        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        if let Some((_, ref preview_output)) = preview
            && let Some(parent) = preview_output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        let frame_dir = if unique_frame_dir {
            dir.join(format!("{}run-{}", prefix, unique_run_id()))
        } else {
            dir
        };

        let mut recorder = Self::new(frame_dir, prefix).with_delay(delay_cs);
        if clear_existing_frames {
            recorder.clear_existing_frames()?;
        }

        for frame in 0..frames {
            let canvas = render(frame)?;
            if let Some((preview_frame, ref preview_output)) = preview
                && frame == preview_frame
            {
                canvas.save_extension(preview_output.to_str().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "preview path is not valid UTF-8",
                    )
                })?)?;
            }
            recorder.capture(&canvas)?;
        }

        if cleanup_frames {
            recorder.encode_gif_and_cleanup(&output)?;
            if unique_frame_dir {
                fs::remove_dir(&recorder.dir)?;
            }
            Ok(())
        } else {
            recorder.encode_gif(output)
        }
    }

    fn frame_path(&self, index: usize) -> PathBuf {
        self.dir.join(format!("{}{:08}.ppm", self.prefix, index))
    }
}

fn unique_run_id() -> String {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    format!("{}-{since_epoch}", std::process::id())
}

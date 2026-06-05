use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix},
    graphics::display::Canvas,
};
use std::{
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

/// Explicit frame recorder for animations.
#[derive(Debug)]
pub struct FrameRecorder {
    dir: PathBuf,
    prefix: String,
    delay_cs: u16,
    frame_index: usize,
    captured_paths: Vec<PathBuf>,
}

/// Error returned while rendering and encoding an animation.
#[derive(Debug)]
pub enum AnimationError<E> {
    /// Frame rendering failed with the caller's error type.
    Render(E),
    /// Frame writing, preview writing, cleanup, or encoding failed.
    Io(io::Error),
    /// Animation options are inconsistent.
    InvalidOptions(String),
}

impl<E: fmt::Display> fmt::Display for AnimationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Render(error) => write!(f, "animation frame render failed: {error}"),
            Self::Io(error) => write!(f, "animation I/O error: {error}"),
            Self::InvalidOptions(error) => f.write_str(error),
        }
    }
}

impl<E: Error + 'static> Error for AnimationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Render(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::InvalidOptions(_) => None,
        }
    }
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

    /// Returns the requested frame count.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }
}

impl FrameRecorder {
    /// Creates a recorder that writes frames as PPM files into `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>, prefix: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            prefix: sanitize_prefix(&prefix.into()),
            delay_cs: 2,
            frame_index: 0,
            captured_paths: Vec::new(),
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
        self.captured_paths.push(path.clone());
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

    /// Removes existing generated PPM frames that match this recorder's prefix and 8-digit suffix.
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
            if is_generated_frame_name(file_name, &self.prefix) {
                fs::remove_file(path)?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Removes only the frame files captured by this recorder.
    ///
    /// # Errors
    /// Returns `Err` if a captured frame cannot be deleted.
    pub fn clear_captured_frames(&self) -> io::Result<usize> {
        let mut removed = 0;
        for path in &self.captured_paths {
            match fs::remove_file(path) {
                Ok(()) => removed += 1,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
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

        let output = output.as_ref();
        let run_encoder = |binary: &str| {
            let mut command = Command::new(binary);
            command
                .arg("-delay")
                .arg(self.delay_cs.to_string())
                .arg("-loop")
                .arg("0");
            for path in &self.captured_paths {
                command.arg(path);
            }
            command.arg(output);
            command.status()
        };

        let (binary, status) = match run_encoder("magick") {
            Ok(status) => ("magick", status),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                ("convert", run_encoder("convert").map_err(|convert_error| {
                    io::Error::new(
                        convert_error.kind(),
                        format!(
                            "failed to run ImageMagick `magick` or `convert`; is ImageMagick installed and in PATH? {convert_error}"
                        ),
                    )
                })?)
            }
            Err(error) => {
                return Err(io::Error::new(
                    error.kind(),
                    format!("failed to run ImageMagick `magick`: {error}"),
                ));
            }
        };

        if !status.success() {
            return Err(io::Error::other(format!(
                "ImageMagick `{binary}` failed with status {status} while encoding animation"
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
        self.clear_captured_frames()?;
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
        Self::render_gif_with_recorder(options, |frame, preview_output, recorder| {
            let canvas = render(frame).map_err(AnimationError::Render)?;
            if let Some(preview_output) = preview_output {
                save_preview(&canvas, preview_output).map_err(AnimationError::Io)?;
            }
            recorder.capture(&canvas).map_err(AnimationError::Io)?;
            Ok(())
        })
        .map_err(animation_error_into_io)
    }

    /// Renders frames by letting `render` write directly through a recorder.
    ///
    /// # Errors
    /// Returns `Err` if frame rendering, frame capture, preview saving, GIF encoding, or cleanup fails.
    pub fn render_gif_with_recorder<F, E>(
        options: AnimationRenderOptions,
        mut render: F,
    ) -> Result<(), AnimationError<E>>
    where
        F: FnMut(usize, Option<&Path>, &mut FrameRecorder) -> Result<(), AnimationError<E>>,
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
            return Err(AnimationError::InvalidOptions(
                "cannot render animation with zero frames".to_string(),
            ));
        }

        if let Some((preview_frame, _)) = preview
            && preview_frame >= frames
        {
            return Err(AnimationError::InvalidOptions(format!(
                "preview frame {preview_frame} is outside {frames} animation frames"
            )));
        }

        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(AnimationError::Io)?;
        }
        if let Some((_, ref preview_output)) = preview
            && let Some(parent) = preview_output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(AnimationError::Io)?;
        }

        let prefix = sanitize_prefix(&prefix);
        let frame_dir = if unique_frame_dir {
            dir.join(format!("{}run-{}", prefix, unique_run_id()))
        } else {
            dir
        };

        let mut recorder = Self::new(frame_dir, prefix).with_delay(delay_cs);
        if clear_existing_frames {
            recorder
                .clear_existing_frames()
                .map_err(AnimationError::Io)?;
        }

        let result = (|| {
            for frame in 0..frames {
                let preview_output = preview.as_ref().and_then(|(preview_frame, output)| {
                    (*preview_frame == frame).then_some(output.as_path())
                });
                render(frame, preview_output, &mut recorder)?;
            }
            recorder.encode_gif(&output).map_err(AnimationError::Io)
        })();

        if cleanup_frames {
            let cleanup_result = recorder
                .clear_captured_frames()
                .and_then(|_| {
                    if unique_frame_dir {
                        fs::remove_dir_all(&recorder.dir)
                    } else {
                        Ok(())
                    }
                })
                .map_err(AnimationError::Io);
            if result.is_ok() {
                cleanup_result?;
            }
        }

        result
    }

    fn frame_path(&self, index: usize) -> PathBuf {
        self.dir.join(format!("{}{:08}.ppm", self.prefix, index))
    }
}

fn save_preview(canvas: &Canvas, preview_output: &Path) -> io::Result<()> {
    canvas.save_extension(preview_output.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "preview path is not valid UTF-8",
        )
    })?)
}

fn animation_error_into_io(error: AnimationError<io::Error>) -> io::Error {
    match error {
        AnimationError::Render(error) | AnimationError::Io(error) => error,
        AnimationError::InvalidOptions(error) => io::Error::new(io::ErrorKind::InvalidInput, error),
    }
}

fn sanitize_prefix(prefix: &str) -> String {
    let sanitized = prefix
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();

    if sanitized.is_empty() {
        "frame-".to_string()
    } else {
        sanitized
    }
}

fn is_generated_frame_name(file_name: &str, prefix: &str) -> bool {
    let Some(rest) = file_name.strip_prefix(prefix) else {
        return false;
    };
    let Some(number) = rest.strip_suffix(".ppm") else {
        return false;
    };
    number.len() == 8 && number.chars().all(|ch| ch.is_ascii_digit())
}

fn unique_run_id() -> String {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    format!("{}-{since_epoch}", std::process::id())
}

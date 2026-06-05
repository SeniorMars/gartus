//! Animation planning for compiled MDL programs.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

/// Per-frame knob values.
pub type KnobMap = HashMap<String, f64>;

/// Precomputed animation state for an MDL program.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationPlan {
    basename: String,
    frames: usize,
    frame_knobs: Vec<KnobMap>,
}

impl AnimationPlan {
    pub(crate) fn new(basename: String, frames: usize, frame_knobs: Vec<KnobMap>) -> Self {
        assert!(frames > 0, "animation frame count must be positive");
        assert_eq!(
            frame_knobs.len(),
            frames,
            "frame knob table must match frame count"
        );
        Self {
            basename,
            frames,
            frame_knobs,
        }
    }

    /// Returns the frame filename basename.
    #[must_use]
    pub fn basename(&self) -> &str {
        &self.basename
    }

    /// Returns the number of frames.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Returns true when the plan contains more than one frame.
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.frames > 1
    }

    /// Returns all frame knob maps.
    #[must_use]
    pub fn frame_knobs(&self) -> &[KnobMap] {
        &self.frame_knobs
    }

    /// Returns one frame's knob map.
    #[must_use]
    pub fn knobs_for_frame(&self, frame: usize) -> Option<&KnobMap> {
        self.frame_knobs.get(frame)
    }
}

/// File naming options for rendered MDL animation frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameOutputConfig {
    output_dir: PathBuf,
    extension: String,
    padding: usize,
}

impl Default for FrameOutputConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("frames"),
            extension: "ppm".to_string(),
            padding: 8,
        }
    }
}

impl FrameOutputConfig {
    /// Creates frame output options rooted at `output_dir`.
    #[must_use]
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self::default().output_dir(output_dir)
    }

    /// Sets the directory where frames are written.
    #[must_use]
    pub fn output_dir(mut self, output_dir: impl Into<PathBuf>) -> Self {
        self.output_dir = output_dir.into();
        self
    }

    /// Sets the frame file extension. A leading `.` is ignored.
    #[must_use]
    pub fn extension(mut self, extension: impl Into<String>) -> Self {
        let extension = extension.into();
        let extension = extension.trim_start_matches('.');
        self.extension = if extension.is_empty() {
            "ppm".to_string()
        } else {
            extension.to_string()
        };
        self
    }

    /// Sets the zero-padding width for frame numbers.
    #[must_use]
    pub const fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Returns the frame output directory.
    #[must_use]
    pub fn output_dir_path(&self) -> &Path {
        &self.output_dir
    }

    /// Returns the frame file extension without a leading dot.
    #[must_use]
    pub fn extension_str(&self) -> &str {
        &self.extension
    }

    /// Returns the frame number padding width.
    #[must_use]
    pub const fn padding_width(&self) -> usize {
        self.padding
    }

    /// Builds one frame path from the compiled basename and frame index.
    #[must_use]
    pub fn frame_path(&self, basename: &str, frame: usize) -> PathBuf {
        let basename = sanitize_basename(basename);
        self.output_dir.join(format!(
            "{basename}{frame:0width$}.{extension}",
            width = self.padding,
            extension = self.extension
        ))
    }
}

fn sanitize_basename(name: &str) -> String {
    let sanitized = name
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
        "frame".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::FrameOutputConfig;

    #[test]
    fn frame_output_config_builds_padded_paths() {
        let config = FrameOutputConfig::new("out").extension(".png").padding(3);

        assert_eq!(
            config.frame_path("spin", 7),
            std::path::Path::new("out/spin007.png")
        );
    }

    #[test]
    fn frame_output_config_defaults_to_frames_ppm_eight_digits() {
        let config = FrameOutputConfig::default();

        assert_eq!(
            config.frame_path("frame", 12),
            std::path::Path::new("frames/frame00000012.ppm")
        );
    }

    #[test]
    fn frame_output_config_sanitizes_basename() {
        let config = FrameOutputConfig::new("out");

        assert_eq!(
            config.frame_path("../../bad name", 0),
            std::path::Path::new("out/______bad_name00000000.ppm")
        );
    }
}

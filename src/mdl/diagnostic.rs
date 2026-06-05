//! Source diagnostics for MDL parsing.

use std::{fmt, path::PathBuf};

/// A source location and message produced by the MDL front end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Optional source filename.
    pub source_name: Option<PathBuf>,
    /// One-based source line.
    pub line: usize,
    /// One-based starting source column.
    pub col_start: usize,
    /// One-based ending source column.
    pub col_end: usize,
    /// Diagnostic message.
    pub message: String,
    /// Optional guidance for fixing the error.
    pub help: Option<String>,
}

impl Diagnostic {
    /// Creates a diagnostic at a source span.
    #[must_use]
    pub fn new(line: usize, col_start: usize, col_end: usize, message: impl Into<String>) -> Self {
        Self {
            source_name: None,
            line,
            col_start,
            col_end,
            message: message.into(),
            help: None,
        }
    }

    /// Creates a diagnostic for a whole line.
    #[must_use]
    pub fn line(line: usize, message: impl Into<String>) -> Self {
        Self::new(line, 1, 1, message)
    }

    /// Attaches a help message.
    #[must_use]
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Attaches a source filename.
    #[must_use]
    pub fn with_source(mut self, source_name: impl Into<PathBuf>) -> Self {
        self.source_name = Some(source_name.into());
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source_name) = &self.source_name {
            write!(f, "{}:", source_name.display())?;
        }
        write!(
            f,
            "line {}, col {}: {}",
            self.line, self.col_start, self.message
        )?;
        if let Some(help) = &self.help {
            write!(f, "\n  help: {help}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Diagnostic {}

#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]
#![warn(clippy::pedantic)]
//! An amateur computer graphics engine made in Rust.
//!
//! Provides an "art canvas" to work with drawings, a mini matrix library
//! with several 3D transformations, and an MDL compiler front end for scripts.
//! This library is still a work in progress project. Be warn.
//!
//! New script code should use [`mdl`]. The legacy two-line parser is available
//! behind the `old_parser` feature.
//!
//! # The prelude
//!
//! Importing each trait individually can become a chore, so the `prelude`
//! module is provided to allow you to import the main traits all at once.
//! For example:
//!
//! ```rust
//! use gartus::prelude::*;
//! ```

#[cfg(feature = "external")]
/// A module that includes method to read external PPM and allows them to be used with this system.
pub mod external;
/// This module hosts all the math needed for computer graphics
pub mod gmath;
/// This module hosts all the needed struts to playing
/// around with computer graphics.
pub mod graphics;
/// This module hosts the Motion Description Language compiler front end.
pub mod mdl;
#[cfg(feature = "old_parser")]
#[doc = "Legacy two-line script parser implementation. Prefer `mdl` for new MDL scripts."]
pub mod old_parser;
#[cfg(feature = "old_parser")]
#[doc = "Compatibility re-export for the legacy parser API."]
pub mod parser {
    /// Legacy parser implementation.
    pub use crate::old_parser::Parser;
    /// Legacy parser error type.
    pub use crate::old_parser::ParserError;
}
/// prelude
pub mod prelude;
/// This module provides utilities that might be needed to use more
/// advance features that are not fully integrated into parser
/// or graphics modules.
pub mod utils;

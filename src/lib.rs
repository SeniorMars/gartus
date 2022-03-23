#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]
//! An amateur computer graphics engine made in Rust.
//!
//! Provides an "art canvas" to work with drawings, a mini matrix library
//! with several 3D transformations, and a parser to read scripts.
//! This library is still a work in progress project. Be warn.
//!
//! # The prelude
//!
//! Importing each trait individually can become a chore, so `prelude` module is provided
//! to allow you to import the main traits all at once. For
//! example:
//!
//! ```rust
//! use gartus::prelude::*;
//! ```

/// A module that includes method to read external PPM and allows them to be used with this system.
pub mod external;
/// This module hosts all the math needed for computer graphics
pub mod gmath;
/// This module hosts all the needed struts to playing
/// around with computer graphics.
pub mod graphics;
/// This module hosts a [Parser] that allows an image to be created through
/// a detailed specification. More information found in the module.
pub mod parser;
/// prelude with
pub mod prelude;
/// This module provides utilities that might be needed to use more
/// advance features that are not fully integrated into parser
/// or graphics modules.
pub mod utils;

#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]
//! An amateur computer graphics engine made in Rust.
//!
//! Provides an "art canvas" to work with drawings, a mini matrix library
//! with several 3D transformations, and a parser to read scripts.
//! This library is still a work in progress project. Be warn.

/// This module hosts all the needed struts to playing
/// around with computer graphics.
pub mod graphics;
/// This module hosts a [Parser] that allows an image to be created through
/// a detailed specification. More information found in the module.
pub mod parser;
/// This module provides utilities that might be needed to use more
/// advance features that are not fully integrated into parser
/// or graphics modules.
pub mod utils;

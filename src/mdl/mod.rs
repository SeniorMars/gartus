//! Motion Description Language (MDL) compiler front end.
//!
//! This module parses MDL source into typed commands and executes those commands through the
//! runtime pipeline. Raster modes (`wireframe`, `flat`, `gouraud`, `phong`, and `toon`) draw into a
//! [`Canvas`](crate::graphics::display::Canvas). `shading raytrace` keeps the same MDL geometry,
//! material constants, camera, background, and point lights, then routes `save` / `display` through
//! the path tracer.

pub mod animation;
pub mod ast;
pub mod diagnostic;
pub mod executor;
pub mod lexer;
pub mod loader;
pub mod parser;
pub mod runtime;
pub mod semantic;

pub use ast::{Command, Program};
pub use diagnostic::Diagnostic;
pub use loader::{
    MdlError, compile_file, compile_source, parse_file, parse_source, run_file, run_file_streaming,
    run_source, run_source_streaming,
};
pub use runtime::RenderConfig;
pub use semantic::CompiledProgram;

//! Motion Description Language (MDL) compiler front end.
//!
//! This module parses MDL source into typed commands. Rendering and runtime
//! state live in later compiler stages, not in the parser.

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

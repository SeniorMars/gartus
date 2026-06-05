//! Source loading and include expansion for MDL programs.

use super::{
    ast::{Command, Program, Spanned},
    diagnostic::Diagnostic,
    executor::{ExecutionError, execute_compiled_program, for_each_compiled_frame},
    lexer::Span,
    parser::parse_script,
    runtime::{RenderConfig, Runtime},
    semantic::{CompiledProgram, compile},
};
use std::{
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

/// Error produced by a full MDL parse/compile/execute pipeline.
#[derive(Debug)]
pub enum MdlError {
    /// Front-end diagnostics from parsing includes or semantic compilation.
    Diagnostics(Vec<Diagnostic>),
    /// Runtime execution error.
    Execution(ExecutionError),
}

impl fmt::Display for MdlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Diagnostics(errors) => {
                writeln!(f, "{} MDL diagnostic(s)", errors.len())?;
                for error in errors {
                    writeln!(f, "{error}")?;
                }
                Ok(())
            }
            Self::Execution(error) => write!(f, "{error}"),
        }
    }
}

impl Error for MdlError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Diagnostics(_) => None,
            Self::Execution(error) => Some(error),
        }
    }
}

impl From<ExecutionError> for MdlError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}

/// Parses source text and expands any `include` commands against `source_dir`.
///
/// # Errors
/// Returns parse diagnostics, include I/O diagnostics, or include-cycle diagnostics.
pub fn parse_source(source: &str, source_dir: Option<&Path>) -> Result<Program, Vec<Diagnostic>> {
    let mut loader = Loader::default();
    loader.parse_source(source, source_dir)
}

/// Parses an MDL file and expands any `include` commands relative to each source file.
///
/// # Errors
/// Returns parse diagnostics, include I/O diagnostics, or include-cycle diagnostics.
pub fn parse_file(path: impl AsRef<Path>) -> Result<Program, Vec<Diagnostic>> {
    let mut loader = Loader::default();
    loader.parse_file(path.as_ref())
}

/// Parses and semantically compiles source text after include expansion.
///
/// # Errors
/// Returns include/parser diagnostics or semantic diagnostics.
pub fn compile_source(
    source: &str,
    source_dir: Option<&Path>,
) -> Result<CompiledProgram, Vec<Diagnostic>> {
    parse_source(source, source_dir).and_then(compile)
}

/// Parses and semantically compiles a file after include expansion.
///
/// # Errors
/// Returns include/parser diagnostics or semantic diagnostics.
pub fn compile_file(path: impl AsRef<Path>) -> Result<CompiledProgram, Vec<Diagnostic>> {
    parse_file(path).and_then(compile)
}

/// Parses, compiles, and executes source text after include expansion.
///
/// This convenience/debug API returns every rendered frame runtime and therefore
/// retains every frame canvas in memory. Prefer [`run_source_streaming`] for
/// real animations.
///
/// # Errors
/// Returns front-end diagnostics or execution errors.
pub fn run_source(
    source: &str,
    source_dir: Option<&Path>,
    config: RenderConfig,
) -> Result<Vec<Runtime>, MdlError> {
    let compiled = compile_source(source, source_dir).map_err(MdlError::Diagnostics)?;
    let config = if let Some(source_dir) = source_dir {
        config.source_dir(source_dir)
    } else {
        config
    };
    execute_compiled_program(&compiled, &config).map_err(MdlError::Execution)
}

/// Parses, compiles, and streams source frames after include expansion.
///
/// This avoids retaining every rendered frame canvas in memory.
///
/// # Errors
/// Returns front-end diagnostics, execution errors, or callback errors.
pub fn run_source_streaming(
    source: &str,
    source_dir: Option<&Path>,
    config: RenderConfig,
    visit: impl FnMut(usize, &Runtime) -> Result<(), ExecutionError>,
) -> Result<(), MdlError> {
    let compiled = compile_source(source, source_dir).map_err(MdlError::Diagnostics)?;
    let config = if let Some(source_dir) = source_dir {
        config.source_dir(source_dir)
    } else {
        config
    };
    for_each_compiled_frame(&compiled, &config, visit).map_err(MdlError::Execution)
}

/// Parses, compiles, and executes an MDL file after include expansion.
///
/// This convenience/debug API returns every rendered frame runtime and therefore
/// retains every frame canvas in memory. Prefer [`run_file_streaming`] for real
/// animations.
///
/// # Errors
/// Returns front-end diagnostics or execution errors.
pub fn run_file(path: impl AsRef<Path>, config: RenderConfig) -> Result<Vec<Runtime>, MdlError> {
    let path = path.as_ref();
    let compiled = compile_file(path).map_err(MdlError::Diagnostics)?;
    let config = if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        config.source_dir(parent)
    } else {
        config
    };
    execute_compiled_program(&compiled, &config).map_err(MdlError::Execution)
}

/// Parses, compiles, and streams file frames after include expansion.
///
/// This avoids retaining every rendered frame canvas in memory.
///
/// # Errors
/// Returns front-end diagnostics, execution errors, or callback errors.
pub fn run_file_streaming(
    path: impl AsRef<Path>,
    config: RenderConfig,
    visit: impl FnMut(usize, &Runtime) -> Result<(), ExecutionError>,
) -> Result<(), MdlError> {
    let path = path.as_ref();
    let compiled = compile_file(path).map_err(MdlError::Diagnostics)?;
    let config = if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        config.source_dir(parent)
    } else {
        config
    };
    for_each_compiled_frame(&compiled, &config, visit).map_err(MdlError::Execution)
}

#[derive(Debug, Default)]
struct Loader {
    active_files: Vec<PathBuf>,
}

impl Loader {
    fn parse_source(
        &mut self,
        source: &str,
        source_dir: Option<&Path>,
    ) -> Result<Program, Vec<Diagnostic>> {
        let program = parse_script(source)?;
        self.expand_program(program, source_dir)
    }

    fn parse_file(&mut self, path: &Path) -> Result<Program, Vec<Diagnostic>> {
        let path = canonicalize_source(path)?;
        self.check_cycle(&path)?;
        self.parse_canonical_file(&path)
    }

    fn parse_canonical_file(&mut self, path: &Path) -> Result<Program, Vec<Diagnostic>> {
        let source = fs::read_to_string(path).map_err(|error| {
            vec![Diagnostic::line(
                1,
                format!("could not read MDL source `{}`: {error}", path.display()),
            )]
        })?;

        let mut program = parse_script(&source).map_err(|errors| tag_diagnostics(errors, path))?;
        for command in &mut program.commands {
            command.source_name = Some(path.to_path_buf());
        }

        self.active_files.push(path.to_path_buf());
        let source_dir = path.parent();
        let parsed = self.expand_program(program, source_dir);
        self.active_files.pop();
        parsed
    }

    fn parse_include_file(
        &mut self,
        path: &Path,
        filename: &str,
        span: Span,
        source_name: Option<&Path>,
    ) -> Result<Program, Vec<Diagnostic>> {
        let path = fs::canonicalize(path).map_err(|error| {
            vec![diagnostic_at_include(
                span,
                source_name,
                format!(
                    "could not resolve include `{filename}` as `{}`: {error}",
                    path.display()
                ),
            )]
        })?;
        self.check_cycle_at(&path, filename, span, source_name)?;
        self.parse_canonical_file(&path)
    }

    fn expand_program(
        &mut self,
        program: Program,
        source_dir: Option<&Path>,
    ) -> Result<Program, Vec<Diagnostic>> {
        let mut commands = Vec::new();
        let mut errors = Vec::new();

        for command in program.commands {
            let Spanned {
                node,
                span,
                source_name,
            } = command;
            match node {
                Command::Include(filename) => {
                    let path = resolve_include_path(source_dir, &filename);
                    match self.parse_include_file(&path, &filename, span, source_name.as_deref()) {
                        Ok(mut included) => commands.append(&mut included.commands),
                        Err(mut include_errors) => errors.append(&mut include_errors),
                    }
                }
                other => commands.push(Spanned {
                    node: other,
                    span,
                    source_name,
                }),
            }
        }

        if errors.is_empty() {
            Ok(Program { commands })
        } else {
            Err(errors)
        }
    }

    fn check_cycle(&self, path: &Path) -> Result<(), Vec<Diagnostic>> {
        if !self.active_files.iter().any(|active| active == path) {
            return Ok(());
        }

        let mut chain = self
            .active_files
            .iter()
            .map(|active| active.display().to_string())
            .collect::<Vec<_>>();
        chain.push(path.display().to_string());
        Err(vec![
            Diagnostic::line(1, format!("include cycle detected: {}", chain.join(" -> ")))
                .with_source(path),
        ])
    }

    fn check_cycle_at(
        &self,
        path: &Path,
        filename: &str,
        span: Span,
        source_name: Option<&Path>,
    ) -> Result<(), Vec<Diagnostic>> {
        if !self.active_files.iter().any(|active| active == path) {
            return Ok(());
        }

        let mut chain = self
            .active_files
            .iter()
            .map(|active| active.display().to_string())
            .collect::<Vec<_>>();
        chain.push(path.display().to_string());
        Err(vec![diagnostic_at_include(
            span,
            source_name,
            format!(
                "include `{filename}` creates an include cycle: {}",
                chain.join(" -> ")
            ),
        )])
    }
}

fn diagnostic_at_include(
    span: Span,
    source_name: Option<&Path>,
    message: impl Into<String>,
) -> Diagnostic {
    let diagnostic = Diagnostic::new(span.line, span.col_start, span.col_end, message);
    if let Some(source_name) = source_name {
        diagnostic.with_source(source_name)
    } else {
        diagnostic
    }
}

fn tag_diagnostics(errors: Vec<Diagnostic>, source_name: &Path) -> Vec<Diagnostic> {
    errors
        .into_iter()
        .map(|error| error.with_source(source_name))
        .collect()
}

fn resolve_include_path(source_dir: Option<&Path>, filename: &str) -> PathBuf {
    let path = Path::new(filename);
    if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(source_dir) = source_dir {
        source_dir.join(path)
    } else {
        path.to_path_buf()
    }
}

fn canonicalize_source(path: &Path) -> Result<PathBuf, Vec<Diagnostic>> {
    fs::canonicalize(path).map_err(|error| {
        vec![Diagnostic::line(
            1,
            format!("could not resolve MDL source `{}`: {error}", path.display()),
        )]
    })
}

#[cfg(test)]
mod tests {
    use super::{
        MdlError, compile_file, parse_file, run_file, run_file_streaming, run_source,
        run_source_streaming,
    };
    use crate::mdl::{
        Command, RenderConfig,
        ast::{ShapeCommand, TransformCommand},
    };
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gartus-mdl-loader-{}-{name}", std::process::id()))
    }

    fn write(path: &Path, source: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, source).unwrap();
    }

    #[test]
    fn parse_file_splices_relative_includes() {
        let dir = temp_dir("splice");
        let main = dir.join("main.mdl");
        let child = dir.join("child.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&child, "move 2 0 0\n");
        write(&main, "include child.mdl\nmove 5 0 0\n");

        let program = parse_file(&main).unwrap();

        assert_eq!(program.commands.len(), 2);
        assert!(matches!(
            program.commands[0].node,
            Command::Transform(TransformCommand::Move { x: 2.0, .. })
        ));
        assert!(matches!(
            program.commands[1].node,
            Command::Transform(TransformCommand::Move { x: 5.0, .. })
        ));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn included_animation_commands_feed_semantic_compile() {
        let dir = temp_dir("semantic");
        let main = dir.join("main.mdl");
        let child = dir.join("anim.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&child, "frames 3\nvary k 0 2 0 1\n");
        write(&main, "include anim.mdl\nmove 10 0 0 k\n");

        let compiled = compile_file(&main).unwrap();

        assert_eq!(compiled.animation().frames(), 3);
        assert_eq!(
            compiled.animation().knobs_for_frame(2).unwrap().get("k"),
            Some(&1.0)
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn included_semantic_errors_keep_included_source_name() {
        let dir = temp_dir("semantic-source");
        let main = dir.join("main.mdl");
        let child = dir.join("sub").join("child.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&child, "frames 2\nvary k 0 2 0 1\n");
        write(&main, "include sub/child.mdl\n");

        let errors = compile_file(&main).unwrap_err();

        let canonical_child = fs::canonicalize(&child).unwrap();
        assert_eq!(
            errors[0].source_name.as_deref(),
            Some(canonical_child.as_path())
        );
        assert_eq!(errors[0].line, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn included_mesh_paths_resolve_relative_to_included_file() {
        let dir = temp_dir("mesh-paths");
        let main = dir.join("main.mdl");
        let child = dir.join("sub").join("child.mdl");
        let mesh = dir.join("meshes").join("tri.obj");
        let _ = fs::remove_dir_all(&dir);
        write(&mesh, "");
        write(&child, "mesh :../meshes/tri.obj\n");
        write(&main, "include sub/child.mdl\n");

        let program = parse_file(&main).unwrap();
        let Command::Shape(ShapeCommand::Mesh { filename, .. }) = &program.commands[0].node else {
            panic!("expected included mesh command");
        };

        assert_eq!(filename, "../meshes/tri.obj");
        let canonical_child = fs::canonicalize(&child).unwrap();
        assert_eq!(
            program.commands[0].source_name.as_deref(),
            Some(canonical_child.as_path())
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn compile_file_is_entry_api() {
        let dir = temp_dir("compile-api");
        let main = dir.join("main.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&main, "frames 2\n");

        let compiled = compile_file(&main).unwrap();

        assert_eq!(compiled.animation().frames(), 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn run_source_parses_compiles_and_executes() {
        let frames = run_source(
            "frames 2\nvary k 0 1 0 1\nmove 10 0 0 k",
            None,
            RenderConfig::new(10, 10).display_enabled(false),
        )
        .unwrap();

        assert_eq!(frames.len(), 2);
        assert!((frames[1].top_transform().get(0, 3) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn run_source_streaming_visits_frames_without_collecting() {
        let mut visited = Vec::new();
        run_source_streaming(
            "frames 2\nvary k 0 1 0 1\nmove 10 0 0 k",
            None,
            RenderConfig::new(10, 10).display_enabled(false),
            |frame, runtime| {
                visited.push((frame, runtime.top_transform().get(0, 3)));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(visited.len(), 2);
        assert!((visited[1].1 - 10.0).abs() < 1e-9);
    }

    #[test]
    fn run_file_runs_expanded_includes() {
        let dir = temp_dir("execute");
        let main = dir.join("main.mdl");
        let child = dir.join("child.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&child, "move 2 0 0\n");
        write(&main, "include child.mdl\nmove 5 0 0\n");

        let frames = run_file(&main, RenderConfig::new(10, 10).display_enabled(false)).unwrap();

        assert_eq!(frames.len(), 1);
        assert!((frames[0].top_transform().get(0, 3) - 7.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn run_file_streaming_runs_expanded_includes() {
        let dir = temp_dir("execute-streaming");
        let main = dir.join("main.mdl");
        let child = dir.join("child.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&child, "move 2 0 0\n");
        write(&main, "include child.mdl\nmove 5 0 0\n");
        let mut transforms = Vec::new();

        run_file_streaming(
            &main,
            RenderConfig::new(10, 10).display_enabled(false),
            |_frame, runtime| {
                transforms.push(runtime.top_transform().get(0, 3));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(transforms.len(), 1);
        assert!((transforms[0] - 7.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn runtime_errors_from_file_keep_source_location() {
        let dir = temp_dir("runtime-source");
        let main = dir.join("main.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&main, "move 1 0 0\nmove 1 0 0 missing\n");

        let error = run_file(&main, RenderConfig::new(10, 10).display_enabled(false)).unwrap_err();

        let MdlError::Execution(crate::mdl::executor::ExecutionError::Located {
            source_name,
            span,
            ..
        }) = error
        else {
            panic!("expected located runtime error");
        };
        let canonical_main = fs::canonicalize(&main).unwrap();
        assert_eq!(source_name.as_deref(), Some(canonical_main.as_path()));
        assert_eq!(span.line, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn include_cycles_are_rejected() {
        let dir = temp_dir("cycle");
        let a = dir.join("a.mdl");
        let b = dir.join("b.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&a, "include b.mdl\n");
        write(&b, "include a.mdl\n");

        let errors = parse_file(&a).unwrap_err();

        assert!(errors[0].message.contains("include cycle"));
        let canonical_b = fs::canonicalize(&b).unwrap();
        assert_eq!(
            errors[0].source_name.as_deref(),
            Some(canonical_b.as_path())
        );
        assert_eq!(errors[0].line, 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_include_errors_point_to_include_command() {
        let dir = temp_dir("missing-include");
        let main = dir.join("main.mdl");
        let _ = fs::remove_dir_all(&dir);
        write(&main, "move 1 0 0\ninclude missing.mdl\n");

        let errors = parse_file(&main).unwrap_err();

        let canonical_main = fs::canonicalize(&main).unwrap();
        assert_eq!(
            errors[0].source_name.as_deref(),
            Some(canonical_main.as_path())
        );
        assert_eq!(errors[0].line, 2);
        assert!(errors[0].message.contains("could not resolve include"));
        let _ = fs::remove_dir_all(dir);
    }
}

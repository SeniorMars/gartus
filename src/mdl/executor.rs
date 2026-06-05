//! Executor for typed MDL programs.

use super::{
    animation::FrameOutputConfig,
    ast::{
        AnimationCommand, Axis, CameraCommand, ColorSpec, Command, ControlCommand, CurveCommand,
        FilterCommand, Material, OutputCommand, PointRef, Program, RenderCommand, ShadingMode,
        ShapeCommand, Spanned, TransformCommand, Vec2, Vec3,
    },
    lexer::Span,
    runtime::{Light, MaterialConstants, RenderConfig, Runtime, rgb_from_vec3},
    semantic::CompiledProgram,
};
use crate::{
    external::MeshMaterial,
    gmath::{edge_matrix::DEFAULT_CURVE_STEP, matrix::Matrix},
    graphics::{
        animation::{AnimationRenderOptions, FrameRecorder},
        colors::Rgb,
        display::{PolygonColorMode, ShadingMode as CanvasShadingMode},
    },
};
use std::{
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

const DEFAULT_3D_STEPS: usize = 100;

/// Error produced while executing an MDL program.
#[derive(Debug)]
pub enum ExecutionError {
    /// Runtime error annotated with the command that produced it.
    Located {
        /// Optional source filename.
        source_name: Option<PathBuf>,
        /// Command span.
        span: Span,
        /// Underlying execution error.
        error: Box<ExecutionError>,
    },
    /// An I/O error while saving or displaying.
    Io(io::Error),
    /// Attempted to pop the base coordinate-system stack entry.
    StackUnderflow,
    /// A transform referenced an unknown knob.
    UnknownKnob(String),
    /// Geometry referenced unknown material constants.
    UnknownConstants(String),
    /// Geometry referenced an unknown saved coordinate system.
    UnknownCoordSystem(String),
    /// A command is parsed but not part of this executor slice yet.
    UnsupportedCommand(&'static str),
    /// A shading mode has no renderer yet.
    UnsupportedShading(ShadingMode),
    /// A named color was not recognized.
    UnknownColor(String),
    /// A filter name, argument count, or argument value was invalid.
    InvalidFilter {
        /// Filter name.
        name: String,
        /// Optional numeric filter value.
        value: Option<f64>,
        /// Error detail.
        reason: String,
    },
    /// Mesh loading failed.
    Mesh {
        /// Mesh filename.
        filename: String,
        /// Error detail.
        error: String,
    },
    /// Requested an animation frame outside the compiled plan.
    InvalidFrame {
        /// Requested frame index.
        frame: usize,
        /// Number of available frames.
        frames: usize,
    },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Located {
                source_name,
                span,
                error,
            } => {
                if let Some(source_name) = source_name {
                    write!(
                        f,
                        "{}:{}:{}: {error}",
                        source_name.display(),
                        span.line,
                        span.col_start
                    )
                } else {
                    write!(f, "line {}, col {}: {error}", span.line, span.col_start)
                }
            }
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::StackUnderflow => write!(f, "cannot pop the base coordinate-system stack entry"),
            Self::UnknownKnob(name) => write!(f, "unknown knob `{name}`"),
            Self::UnknownConstants(name) => write!(f, "unknown constants `{name}`"),
            Self::UnknownCoordSystem(name) => write!(f, "unknown coordinate system `{name}`"),
            Self::UnsupportedCommand(command) => {
                write!(f, "command `{command}` requires a later MDL compiler stage")
            }
            Self::UnsupportedShading(mode) => write!(f, "unsupported shading mode `{mode:?}`"),
            Self::UnknownColor(name) => write!(f, "unknown color `{name}`"),
            Self::InvalidFilter {
                name,
                value,
                reason,
            } => match value {
                Some(value) => write!(f, "invalid filter `{name}` with value {value}: {reason}"),
                None => write!(f, "invalid filter `{name}`: {reason}"),
            },
            Self::Mesh { filename, error } => {
                write!(f, "could not load mesh `{filename}`: {error}")
            }
            Self::InvalidFrame { frame, frames } => {
                write!(f, "frame {frame} is outside compiled frame count {frames}")
            }
        }
    }
}

impl Error for ExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Located { error, .. } => Some(error.as_ref()),
            Self::Io(error) => Some(error),
            Self::StackUnderflow
            | Self::UnknownKnob(_)
            | Self::UnknownConstants(_)
            | Self::UnknownCoordSystem(_)
            | Self::UnsupportedCommand(_)
            | Self::UnsupportedShading(_)
            | Self::UnknownColor(_)
            | Self::InvalidFilter { .. }
            | Self::Mesh { .. }
            | Self::InvalidFrame { .. } => None,
        }
    }
}

fn with_location(error: ExecutionError, command: &Spanned<Command>) -> ExecutionError {
    if matches!(error, ExecutionError::Located { .. }) {
        return error;
    }

    ExecutionError::Located {
        source_name: command.source_name.clone(),
        span: command.span,
        error: Box::new(error),
    }
}

/// Executes a parsed MDL program with the given render configuration.
///
/// This low-level entry point expects any `include` commands to have already
/// been expanded. Use `run_file`, `compile_file`, or `parse_file` for real
/// MDL files that may include other files.
///
/// # Errors
/// Returns an execution error for runtime failures such as stack underflow,
/// unknown symbols, save/display I/O failures, or unsupported animation commands.
pub fn execute_program(
    program: &Program,
    config: &RenderConfig,
) -> Result<Runtime, ExecutionError> {
    let mut runtime = Runtime::new(config);
    execute_into(&mut runtime, program)?;
    Ok(runtime)
}

/// Executes every frame of a compiled MDL program and returns every frame runtime.
///
/// This is useful for tests and debugging, but it retains every frame canvas in
/// memory. Prefer [`for_each_compiled_frame`] or the file/GIF helpers for large
/// animations.
///
/// # Errors
/// Returns an execution error for the first frame that fails.
pub fn execute_compiled_program(
    compiled: &CompiledProgram,
    config: &RenderConfig,
) -> Result<Vec<Runtime>, ExecutionError> {
    let mut frames = Vec::with_capacity(compiled.animation().frames());
    for frame in 0..compiled.animation().frames() {
        frames.push(execute_compiled_frame_from_config(compiled, config, frame)?);
    }
    Ok(frames)
}

/// Executes compiled frames one at a time and passes each runtime to `visit`.
///
/// This avoids retaining every frame canvas in memory when callers only need to
/// save, inspect, or encode one frame at a time.
///
/// # Errors
/// Returns the first execution error or callback error.
pub fn for_each_compiled_frame(
    compiled: &CompiledProgram,
    config: &RenderConfig,
    mut visit: impl FnMut(usize, &Runtime) -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    let mut runtime = Runtime::new(config);
    for frame in 0..compiled.animation().frames() {
        execute_compiled_frame_into(&mut runtime, compiled, frame)?;
        visit(frame, &runtime)?;
    }
    Ok(())
}

/// Executes every compiled frame and writes `basenameNNNNNNNN.ppm` files into `dir`.
///
/// The generated frame basename comes from the compiled [`AnimationPlan`](super::AnimationPlan).
/// Ordinary MDL `save` commands are disabled while frames render so each frame is
/// written exactly once after command execution.
///
/// # Errors
/// Returns an execution error if frame rendering or file writing fails.
pub fn execute_compiled_frames_to_files(
    compiled: &CompiledProgram,
    config: RenderConfig,
    dir: impl AsRef<Path>,
) -> Result<Vec<PathBuf>, ExecutionError> {
    execute_compiled_frames_with_options(compiled, config, FrameOutputConfig::new(dir.as_ref()))
}

/// Executes every compiled frame and writes files using configurable naming options.
///
/// # Errors
/// Returns an execution error if frame rendering or file writing fails.
#[allow(clippy::needless_pass_by_value)]
pub fn execute_compiled_frames_with_options(
    compiled: &CompiledProgram,
    config: RenderConfig,
    output: FrameOutputConfig,
) -> Result<Vec<PathBuf>, ExecutionError> {
    fs::create_dir_all(output.output_dir_path()).map_err(ExecutionError::Io)?;

    let mut paths = Vec::with_capacity(compiled.animation().frames());
    let config = config.save_enabled(false);
    for_each_compiled_frame(compiled, &config, |frame, runtime| {
        let path = output.frame_path(compiled.animation().basename(), frame);
        runtime.save_to_path(&path)?;
        paths.push(path);
        Ok(())
    })?;
    Ok(paths)
}

/// Executes every compiled frame using the default `frames/` output convention.
///
/// # Errors
/// Returns an execution error if frame rendering or file writing fails.
pub fn execute_compiled_frames_to_default_dir(
    compiled: &CompiledProgram,
    config: RenderConfig,
) -> Result<Vec<PathBuf>, ExecutionError> {
    execute_compiled_frames_with_options(compiled, config, FrameOutputConfig::default())
}

/// Executes a compiled program and encodes its frames as a GIF.
///
/// Frames are rendered through the existing [`FrameRecorder`] GIF pipeline. MDL
/// `save` commands are disabled during GIF rendering so they do not overwrite a
/// static filename while each frame is being generated.
///
/// # Errors
/// Returns an execution error if frame rendering or GIF encoding fails.
pub fn execute_compiled_gif(
    compiled: &CompiledProgram,
    config: RenderConfig,
    output: impl AsRef<Path>,
) -> Result<(), ExecutionError> {
    let prefix = format!("{}-", compiled.animation().basename());
    let options = AnimationRenderOptions::new(
        "anim",
        prefix,
        compiled.animation().frames(),
        output.as_ref(),
    );
    execute_compiled_gif_with_options(compiled, config, options)
}

/// Executes a compiled program and encodes its frames as a GIF with explicit options.
///
/// # Errors
/// Returns an execution error if frame rendering or GIF encoding fails.
#[allow(clippy::needless_pass_by_value)]
pub fn execute_compiled_gif_with_options(
    compiled: &CompiledProgram,
    config: RenderConfig,
    options: AnimationRenderOptions,
) -> Result<(), ExecutionError> {
    let config = config.save_enabled(false);
    let mut runtime = Runtime::new(&config);
    FrameRecorder::render_gif(options, |frame| {
        execute_compiled_frame_into(&mut runtime, compiled, frame)
            .map(|()| runtime.canvas().clone())
            .map_err(|error| io::Error::other(error.to_string()))
    })
    .map_err(ExecutionError::Io)
}

/// Executes one frame of a compiled MDL program.
///
/// # Errors
/// Returns an execution error if `frame` is out of range or runtime execution fails.
pub fn execute_compiled_frame(
    compiled: &CompiledProgram,
    config: &RenderConfig,
    frame: usize,
) -> Result<Runtime, ExecutionError> {
    let mut runtime = Runtime::new(config);
    execute_compiled_frame_into(&mut runtime, compiled, frame)?;
    Ok(runtime)
}

fn execute_compiled_frame_from_config(
    compiled: &CompiledProgram,
    config: &RenderConfig,
    frame: usize,
) -> Result<Runtime, ExecutionError> {
    let mut runtime = Runtime::new(config);
    execute_compiled_frame_into(&mut runtime, compiled, frame)?;
    Ok(runtime)
}

fn execute_compiled_frame_into(
    runtime: &mut Runtime,
    compiled: &CompiledProgram,
    frame: usize,
) -> Result<(), ExecutionError> {
    let frame_knobs =
        compiled
            .animation()
            .knobs_for_frame(frame)
            .ok_or(ExecutionError::InvalidFrame {
                frame,
                frames: compiled.animation().frames(),
            })?;
    runtime.reset_for_frame();
    runtime.set_basename(compiled.animation().basename().to_string());
    runtime.set_frames(compiled.animation().frames());
    runtime.seed_frame_knobs(frame_knobs.iter());
    execute_compiled_into(runtime, compiled)
}

/// Executes a parsed MDL program into an existing runtime.
///
/// # Errors
/// Returns an execution error for runtime failures.
pub fn execute_into(runtime: &mut Runtime, program: &Program) -> Result<(), ExecutionError> {
    for command in &program.commands {
        if command.node.is_quit() {
            break;
        }
        execute_command(runtime, &command.node, command.source_name.as_deref())
            .map_err(|error| with_location(error, command))?;
    }
    Ok(())
}

fn execute_compiled_into(
    runtime: &mut Runtime,
    compiled: &CompiledProgram,
) -> Result<(), ExecutionError> {
    for command in compiled.commands() {
        if command.node.is_quit() {
            break;
        }
        execute_command(runtime, &command.node, command.source_name.as_deref())
            .map_err(|error| with_location(error, command))?;
    }
    Ok(())
}

fn execute_command(
    runtime: &mut Runtime,
    command: &Command,
    source_name: Option<&Path>,
) -> Result<(), ExecutionError> {
    match command {
        Command::Control(command) => execute_control_command(runtime, command),
        Command::Transform(command) => execute_transform_command(runtime, command),
        Command::Curve(command) => execute_curve_command(runtime, command),
        Command::Shape(command) => execute_shape_command(runtime, command, source_name),
        Command::Animation(command) => execute_animation_command(runtime, command),
        Command::Render(command) => execute_render_state_command(runtime, command),
        Command::Camera(command) => {
            execute_camera_command(runtime, command);
            Ok(())
        }
        Command::Output(command) => execute_output_command(runtime, command),
        Command::Include(_) | Command::Filter(_) => execute_misc_command(runtime, command),
    }
}

fn execute_control_command(
    runtime: &mut Runtime,
    command: &ControlCommand,
) -> Result<(), ExecutionError> {
    match command {
        ControlCommand::Apply | ControlCommand::Quit => Ok(()),
        ControlCommand::Push => {
            runtime.push_stack();
            Ok(())
        }
        ControlCommand::Pop => runtime.pop_stack(),
        ControlCommand::Ident => {
            runtime.set_top_identity();
            Ok(())
        }
        ControlCommand::Clear => {
            runtime.clear_canvas();
            Ok(())
        }
        ControlCommand::Reset => {
            runtime.reset();
            Ok(())
        }
    }
}

fn execute_transform_command(
    runtime: &mut Runtime,
    command: &TransformCommand,
) -> Result<(), ExecutionError> {
    match command {
        TransformCommand::Move { x, y, z, knob } => {
            let k = runtime.knob_value(knob.as_deref())?;
            runtime.apply_transform(Matrix::translate(x * k, y * k, z * k));
            Ok(())
        }
        TransformCommand::Scale { x, y, z, knob } => {
            let k = runtime.knob_value(knob.as_deref())?;
            runtime.apply_transform(Matrix::scale(x * k, y * k, z * k));
            Ok(())
        }
        TransformCommand::Rotate {
            axis,
            degrees,
            knob,
        } => {
            let k = runtime.knob_value(knob.as_deref())?;
            let degrees = degrees * k;
            let transform = match axis {
                Axis::X => Matrix::rotate_x(degrees),
                Axis::Y => Matrix::rotate_y(degrees),
                Axis::Z => Matrix::rotate_z(degrees),
            };
            runtime.apply_transform(transform);
            Ok(())
        }
        TransformCommand::Reflect { axis } => {
            let transform = match axis {
                Axis::X => Matrix::reflect_xz(),
                Axis::Y => Matrix::reflect_yz(),
                Axis::Z => Matrix::reflect_xy(),
            };
            runtime.apply_transform(transform);
            Ok(())
        }
        TransformCommand::Shear {
            axis,
            sh0,
            sh1,
            knob,
        } => {
            let k = runtime.knob_value(knob.as_deref())?;
            let transform = match axis {
                Axis::X => Matrix::shearing_x(sh0 * k, sh1 * k),
                Axis::Y => Matrix::shearing_y(sh0 * k, sh1 * k),
                Axis::Z => Matrix::shearing_z(sh0 * k, sh1 * k),
            };
            runtime.apply_transform(transform);
            Ok(())
        }
    }
}

fn execute_curve_command(
    runtime: &mut Runtime,
    command: &CurveCommand,
) -> Result<(), ExecutionError> {
    match command {
        CurveCommand::Circle { center, radius } => {
            let center = *center;
            draw_edges(runtime, |edges| {
                edges.add_circle(center.x, center.y, center.z, *radius, DEFAULT_CURVE_STEP);
            });
            Ok(())
        }
        CurveCommand::Hermite { p0, p1, r0, r1 } => {
            let (p0, p1, r0, r1) = (*p0, *p1, *r0, *r1);
            draw_edges(runtime, |edges| {
                edges.add_hermite(
                    vec2_tuple(p0),
                    vec2_tuple(p1),
                    vec2_tuple(r0),
                    vec2_tuple(r1),
                );
            });
            Ok(())
        }
        CurveCommand::Bezier { p0, p1, p2, p3 } => {
            let (p0, p1, p2, p3) = (*p0, *p1, *p2, *p3);
            draw_edges(runtime, |edges| {
                edges.add_bezier3(
                    vec2_tuple(p0),
                    vec2_tuple(p1),
                    vec2_tuple(p2),
                    vec2_tuple(p3),
                );
            });
            Ok(())
        }
        CurveCommand::BezierN { degree, points } => {
            let x_points: Vec<f64> = points.iter().map(|point| point.x).collect();
            let y_points: Vec<f64> = points.iter().map(|point| point.y).collect();
            draw_edges(runtime, |edges| {
                edges.add_beziern(*degree, &x_points, &y_points);
            });
            Ok(())
        }
        CurveCommand::BezierSurface { steps, controls } => {
            let controls = bezier_surface_controls(controls);
            draw_polygons(runtime, None, None, |polygons| {
                polygons.add_bezier_surface(controls, *steps);
            })
        }
    }
}

fn execute_shape_command(
    runtime: &mut Runtime,
    command: &ShapeCommand,
    source_name: Option<&Path>,
) -> Result<(), ExecutionError> {
    match command {
        ShapeCommand::Texture { .. } => Err(ExecutionError::UnsupportedCommand("texture")),
        ShapeCommand::Line { constants, p0, p1 } => {
            draw_line(runtime, constants.as_deref(), p0, p1)
        }
        ShapeCommand::Mesh { .. } | ShapeCommand::MeshReverse { .. } => {
            execute_mesh_shape(runtime, command, source_name)
        }
        ShapeCommand::Box { .. }
        | ShapeCommand::Sphere { .. }
        | ShapeCommand::Torus { .. }
        | ShapeCommand::Cylinder { .. }
        | ShapeCommand::Cone { .. }
        | ShapeCommand::Pyramid { .. } => execute_solid_shape(runtime, command),
    }
}

fn execute_mesh_shape(
    runtime: &mut Runtime,
    command: &ShapeCommand,
    source_name: Option<&Path>,
) -> Result<(), ExecutionError> {
    match command {
        ShapeCommand::Mesh {
            constants,
            filename,
            coord_system,
        } => draw_mesh(
            runtime,
            constants.as_deref(),
            filename,
            coord_system.as_deref(),
            source_name,
            false,
        ),
        ShapeCommand::MeshReverse {
            constants,
            filename,
            coord_system,
        } => draw_mesh(
            runtime,
            constants.as_deref(),
            filename,
            coord_system.as_deref(),
            source_name,
            true,
        ),
        _ => unreachable!("non-mesh shape dispatched to mesh executor"),
    }
}

fn execute_solid_shape(
    runtime: &mut Runtime,
    command: &ShapeCommand,
) -> Result<(), ExecutionError> {
    match command {
        ShapeCommand::Box {
            constants,
            corner,
            h,
            w,
            d,
            coord_system,
        } => draw_box(
            runtime,
            constants.as_deref(),
            *corner,
            *h,
            *w,
            *d,
            coord_system.as_deref(),
        ),
        ShapeCommand::Sphere {
            constants,
            center,
            radius,
            coord_system,
        } => draw_sphere(
            runtime,
            constants.as_deref(),
            *center,
            *radius,
            coord_system.as_deref(),
        ),
        ShapeCommand::Torus {
            constants,
            center,
            r0,
            r1,
            coord_system,
        } => draw_torus(
            runtime,
            constants.as_deref(),
            *center,
            *r0,
            *r1,
            coord_system.as_deref(),
        ),
        ShapeCommand::Cylinder {
            constants,
            center,
            radius,
            height,
            coord_system,
        } => draw_cylinder(
            runtime,
            constants.as_deref(),
            *center,
            *radius,
            *height,
            coord_system.as_deref(),
        ),
        ShapeCommand::Cone {
            constants,
            center,
            radius,
            height,
            coord_system,
        } => draw_cone(
            runtime,
            constants.as_deref(),
            *center,
            *radius,
            *height,
            coord_system.as_deref(),
        ),
        ShapeCommand::Pyramid {
            constants,
            center,
            base_length,
            height,
            coord_system,
        } => draw_pyramid(
            runtime,
            constants.as_deref(),
            *center,
            *base_length,
            *height,
            coord_system.as_deref(),
        ),
        ShapeCommand::Line { .. }
        | ShapeCommand::Mesh { .. }
        | ShapeCommand::MeshReverse { .. }
        | ShapeCommand::Texture { .. } => {
            unreachable!("non-solid shape dispatched to solid executor")
        }
    }
}

fn draw_box(
    runtime: &mut Runtime,
    constants: Option<&str>,
    corner: Vec3,
    h: f64,
    w: f64,
    d: f64,
    coord_system: Option<&str>,
) -> Result<(), ExecutionError> {
    draw_polygons(runtime, constants, coord_system, |polygons| {
        polygons.add_box((corner.x, corner.y, corner.z), w, h, d);
    })
}

fn draw_sphere(
    runtime: &mut Runtime,
    constants: Option<&str>,
    center: Vec3,
    radius: f64,
    coord_system: Option<&str>,
) -> Result<(), ExecutionError> {
    draw_polygons(runtime, constants, coord_system, |polygons| {
        polygons.add_sphere((center.x, center.y, center.z), radius, DEFAULT_3D_STEPS);
    })
}

fn draw_torus(
    runtime: &mut Runtime,
    constants: Option<&str>,
    center: Vec3,
    r0: f64,
    r1: f64,
    coord_system: Option<&str>,
) -> Result<(), ExecutionError> {
    draw_polygons(runtime, constants, coord_system, |polygons| {
        polygons.add_torus((center.x, center.y, center.z), r0, r1, DEFAULT_3D_STEPS);
    })
}

fn draw_cylinder(
    runtime: &mut Runtime,
    constants: Option<&str>,
    center: Vec3,
    radius: f64,
    height: f64,
    coord_system: Option<&str>,
) -> Result<(), ExecutionError> {
    draw_polygons(runtime, constants, coord_system, |polygons| {
        polygons.add_cylinder((center.x, center.y, center.z), radius, height, 24);
    })
}

fn draw_cone(
    runtime: &mut Runtime,
    constants: Option<&str>,
    center: Vec3,
    radius: f64,
    height: f64,
    coord_system: Option<&str>,
) -> Result<(), ExecutionError> {
    draw_polygons(runtime, constants, coord_system, |polygons| {
        polygons.add_cone((center.x, center.y, center.z), radius, height, 24);
    })
}

fn draw_pyramid(
    runtime: &mut Runtime,
    constants: Option<&str>,
    center: Vec3,
    base_length: f64,
    height: f64,
    coord_system: Option<&str>,
) -> Result<(), ExecutionError> {
    draw_polygons(runtime, constants, coord_system, |polygons| {
        polygons.add_pyramid((center.x, center.y, center.z), base_length, height);
    })
}

fn execute_animation_command(
    runtime: &mut Runtime,
    command: &AnimationCommand,
) -> Result<(), ExecutionError> {
    match command {
        AnimationCommand::Basename(name) => runtime.set_basename(name.clone()),
        AnimationCommand::Frames(frames) => runtime.set_frames(*frames),
        AnimationCommand::Set { knob, value } => runtime.set_knob(knob.clone(), *value),
        AnimationCommand::SetKnobs(value) => runtime.set_all_knobs(*value),
        AnimationCommand::SaveKnobs(name) => runtime.save_knobs(name.clone()),
        AnimationCommand::Tween { .. } => return Err(ExecutionError::UnsupportedCommand("tween")),
        AnimationCommand::Vary { .. } => return Err(ExecutionError::UnsupportedCommand("vary")),
    }
    Ok(())
}

fn execute_render_state_command(
    runtime: &mut Runtime,
    command: &RenderCommand,
) -> Result<(), ExecutionError> {
    match command {
        RenderCommand::Color(color) => set_color(runtime, color)?,
        RenderCommand::Ambient { color } => runtime.set_ambient(*color),
        RenderCommand::Light { color, position } => runtime.add_light(Light {
            color: *color,
            position: *position,
        }),
        RenderCommand::Constants {
            name,
            material,
            color,
        } => runtime.set_constants(name.clone(), *material, *color),
        RenderCommand::Shading(mode) => set_shading(runtime, *mode)?,
        RenderCommand::SaveCoordSystem(name) => runtime.save_coord_system(name.clone()),
    }
    Ok(())
}

fn execute_camera_command(runtime: &mut Runtime, command: &CameraCommand) {
    match command {
        CameraCommand::Camera { eye, aim } => runtime.set_camera(*eye, *aim),
        CameraCommand::Focal(value) => runtime.set_focal(*value),
    }
}

fn execute_output_command(
    runtime: &mut Runtime,
    command: &OutputCommand,
) -> Result<(), ExecutionError> {
    match command {
        OutputCommand::GenerateRayfiles => runtime.set_generate_rayfiles(),
        OutputCommand::Save(filename) => runtime.save(filename)?,
        OutputCommand::Display => runtime.display()?,
    }
    Ok(())
}

fn execute_misc_command(runtime: &mut Runtime, command: &Command) -> Result<(), ExecutionError> {
    match command {
        Command::Include(_) => Err(ExecutionError::UnsupportedCommand("include")),
        Command::Filter(filter) => execute_filter_command(runtime, filter),
        _ => unreachable!("non-misc command dispatched to misc executor"),
    }
}

fn execute_filter_command(
    runtime: &mut Runtime,
    filter: &FilterCommand,
) -> Result<(), ExecutionError> {
    apply_filter(runtime, &filter.name, filter.value)
}

fn draw_edges(
    runtime: &mut Runtime,
    build: impl FnOnce(&mut crate::gmath::edge_matrix::EdgeMatrix),
) {
    let transform = runtime.top_transform().clone();
    runtime.with_tmp_edges(build);
    runtime.transform_tmp_edges(&transform);
    runtime.draw_tmp_edges();
}

fn draw_polygons(
    runtime: &mut Runtime,
    constants: Option<&str>,
    coord_system: Option<&str>,
    build: impl FnOnce(&mut crate::gmath::polygon_matrix::PolygonMatrix),
) -> Result<(), ExecutionError> {
    let transform = runtime.transform_for(coord_system)?;
    let material = runtime.material_for(constants)?;
    let previous = runtime.apply_draw_state(material);

    runtime.with_tmp_polygons(build);
    runtime.transform_tmp_polygons(&transform);
    runtime.draw_tmp_polygons();

    runtime.restore_draw_state(previous);
    Ok(())
}

fn draw_line(
    runtime: &mut Runtime,
    constants: Option<&str>,
    p0: &PointRef,
    p1: &PointRef,
) -> Result<(), ExecutionError> {
    let p0 = transform_point_ref(runtime, p0)?;
    let p1 = transform_point_ref(runtime, p1)?;
    let material = runtime.material_for(constants)?;
    let previous = runtime.apply_draw_state(material);

    runtime.with_tmp_edges(|edges| {
        edges.push_edge(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
    });
    runtime.draw_tmp_edges();

    runtime.restore_draw_state(previous);
    Ok(())
}

fn transform_point_ref(runtime: &Runtime, point: &PointRef) -> Result<Vec3, ExecutionError> {
    let transform = runtime.transform_for(point.coord_system.as_deref())?;
    let point =
        transform.transform_homogeneous_point(&[point.point.x, point.point.y, point.point.z, 1.0]);
    Ok(Vec3::new(point[0], point[1], point[2]))
}

fn set_color(runtime: &mut Runtime, color: &ColorSpec) -> Result<(), ExecutionError> {
    let color = match color {
        ColorSpec::Rgb(color) => rgb_from_vec3(*color),
        ColorSpec::Name(name) => Rgb::name_to_const(&name.to_lowercase())
            .ok_or_else(|| ExecutionError::UnknownColor(name.clone()))?,
    };
    runtime.canvas_mut().set_line_pixel(color);
    Ok(())
}

fn vec2_tuple(point: Vec2) -> (f64, f64) {
    (point.x, point.y)
}

fn bezier_surface_controls(controls: &[Vec3]) -> [[(f64, f64, f64); 4]; 4] {
    let mut result = [[(0.0, 0.0, 0.0); 4]; 4];
    for (index, control) in controls.iter().enumerate().take(16) {
        result[index / 4][index % 4] = (control.x, control.y, control.z);
    }
    result
}

#[cfg(feature = "external")]
fn draw_mesh(
    runtime: &mut Runtime,
    constants: Option<&str>,
    filename: &str,
    coord_system: Option<&str>,
    source_name: Option<&Path>,
    reverse: bool,
) -> Result<(), ExecutionError> {
    let transform = runtime.transform_for(coord_system)?;
    let material = runtime.material_for(constants)?;
    let path = runtime.resolve_mesh_path(filename, source_name);

    let mesh = crate::external::meshify_with_materials(path.to_string_lossy().as_ref()).map_err(
        |error| ExecutionError::Mesh {
            filename: path.display().to_string(),
            error: error.to_string(),
        },
    )?;

    for group in mesh.groups {
        let draw_material =
            material_with_mesh_material(material, group.material, group.diffuse_color);
        let previous = runtime.apply_draw_state(draw_material);

        let mut polygons = group.polygons;
        if reverse {
            polygons.reverse_winding();
        }
        polygons.apply_in_place(&transform);
        runtime.canvas_mut().draw_polygons(&polygons);

        runtime.restore_draw_state(previous);
    }

    Ok(())
}

#[cfg(feature = "external")]
fn material_with_mesh_material(
    material: Option<MaterialConstants>,
    mesh_material: Option<MeshMaterial>,
    diffuse_color: Option<Rgb>,
) -> Option<MaterialConstants> {
    if mesh_material.is_none() && diffuse_color.is_none() {
        return material;
    }

    let mut constants = material.unwrap_or(MaterialConstants {
        material: Material::new(0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.5),
        color: Vec3::new(0.0, 0.0, 0.0),
    });

    if let Some(mesh_material) = mesh_material {
        if let Some([red, green, blue]) = mesh_material.ambient {
            constants.material.kar = red;
            constants.material.kag = green;
            constants.material.kab = blue;
        }
        if let Some([red, green, blue]) = mesh_material.diffuse {
            constants.material.kdr = red;
            constants.material.kdg = green;
            constants.material.kdb = blue;
        }
        if let Some([red, green, blue]) = mesh_material.specular {
            constants.material.ksr = red;
            constants.material.ksg = green;
            constants.material.ksb = blue;
        }
    } else if let Some(color) = diffuse_color {
        let red = f64::from(color.red) / 255.0;
        let green = f64::from(color.green) / 255.0;
        let blue = f64::from(color.blue) / 255.0;

        constants.material.kar *= red;
        constants.material.kdr *= red;
        constants.material.kag *= green;
        constants.material.kdg *= green;
        constants.material.kab *= blue;
        constants.material.kdb *= blue;
    }

    if let Some(color) = diffuse_color {
        constants.color = Vec3::new(
            f64::from(color.red),
            f64::from(color.green),
            f64::from(color.blue),
        );
    }

    Some(constants)
}

#[cfg(not(feature = "external"))]
fn draw_mesh(
    _runtime: &mut Runtime,
    _constants: Option<&str>,
    filename: &str,
    _coord_system: Option<&str>,
    _source_name: Option<&Path>,
    _reverse: bool,
) -> Result<(), ExecutionError> {
    Err(ExecutionError::Mesh {
        filename: filename.to_string(),
        error: "mesh command requires the `external` feature".to_string(),
    })
}
fn set_shading(runtime: &mut Runtime, mode: ShadingMode) -> Result<(), ExecutionError> {
    let (shading, color_mode) = match mode {
        ShadingMode::Wireframe => (CanvasShadingMode::Wireframe, PolygonColorMode::LineColor),
        ShadingMode::Flat => (CanvasShadingMode::Flat, PolygonColorMode::PhongReflection),
        ShadingMode::Gouraud => (
            CanvasShadingMode::Gouraud,
            PolygonColorMode::PhongReflection,
        ),
        ShadingMode::Phong => (CanvasShadingMode::Phong, PolygonColorMode::PhongReflection),
        ShadingMode::Raytrace => return Err(ExecutionError::UnsupportedShading(mode)),
    };
    runtime.canvas_mut().set_shading_mode(shading);
    runtime.canvas_mut().set_polygon_color_mode(color_mode);
    Ok(())
}

#[cfg(feature = "filters")]
fn apply_filter(
    runtime: &mut Runtime,
    name: &str,
    value: Option<f64>,
) -> Result<(), ExecutionError> {
    let canvas = match (name, value) {
        ("solarize", Some(value)) => runtime.canvas().solarize(filter_u8(name, value)?),
        ("black_and_white", Some(value)) => {
            runtime.canvas().black_and_white(filter_u8(name, value)?)
        }
        ("brightness", Some(value)) => runtime.canvas().adjust_brightness(filter_i16(name, value)?),
        ("posterize", Some(value)) => runtime.canvas().posterize(filter_u8(name, value)?),
        ("gaussian", Some(value)) => runtime.canvas().gaussian_blur(filter_f32(name, value)?),
        ("contrast", Some(value)) => runtime.canvas().adjust_contrast(filter_f32(name, value)?),
        ("grayscale", None) => runtime.canvas().grayscale(),
        ("sepia", None) => runtime.canvas().sepia(),
        ("reflect", None) => runtime.canvas().reflect(),
        ("blur", None) => runtime.canvas().blur(),
        ("sobel", None) => runtime.canvas().sobel(),
        ("invert", None) => runtime.canvas().invert(),
        ("edge", None) => runtime.canvas().laplacian_edge_detection(),
        ("emboss", None) => runtime.canvas().emboss(),
        ("oil", None) => runtime.canvas().oil_painting(),
        ("watercolor", None) => runtime.canvas().watercolor(),
        ("bilateral", None) => runtime.canvas().bilateral_filter(2, 3.0, 32.0),
        ("unsharp", None) => runtime.canvas().unsharp_mask(1.0, 1.0),
        ("histogram" | "histogram_equalization", None) => runtime.canvas().histogram_equalization(),
        ("clahe", None) => runtime.canvas().clahe(32, 16),
        ("canny", None) => runtime.canvas().canny(40, 100),
        ("floyd_steinberg" | "floyd", None) => runtime.canvas().floyd_steinberg_dither(),
        _ => {
            return Err(invalid_filter(
                name,
                value,
                "unknown filter or invalid argument count",
            ));
        }
    };
    runtime.replace_canvas(canvas);
    Ok(())
}

#[cfg(not(feature = "filters"))]
fn apply_filter(
    _runtime: &mut Runtime,
    name: &str,
    value: Option<f64>,
) -> Result<(), ExecutionError> {
    Err(invalid_filter(
        name,
        value,
        "filter command requires the `filters` feature",
    ))
}

fn invalid_filter(name: &str, value: Option<f64>, reason: &str) -> ExecutionError {
    ExecutionError::InvalidFilter {
        name: name.to_string(),
        value,
        reason: reason.to_string(),
    }
}

#[cfg(feature = "filters")]
fn filter_u8(name: &str, value: f64) -> Result<u8, ExecutionError> {
    if value.is_finite()
        && value.fract() == 0.0
        && (f64::from(u8::MIN)..=f64::from(u8::MAX)).contains(&value)
    {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        return Ok(value as u8);
    }
    Err(invalid_filter(
        name,
        Some(value),
        "expected an integer from 0 to 255",
    ))
}

#[cfg(feature = "filters")]
fn filter_i16(name: &str, value: f64) -> Result<i16, ExecutionError> {
    if value.is_finite()
        && value.fract() == 0.0
        && (f64::from(i16::MIN)..=f64::from(i16::MAX)).contains(&value)
    {
        #[allow(clippy::cast_possible_truncation)]
        return Ok(value as i16);
    }
    Err(invalid_filter(
        name,
        Some(value),
        "expected an integer in i16 range",
    ))
}

#[allow(clippy::cast_possible_truncation)]
#[cfg(feature = "filters")]
fn filter_f32(name: &str, value: f64) -> Result<f32, ExecutionError> {
    if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX) {
        Ok(value as f32)
    } else {
        Err(invalid_filter(
            name,
            Some(value),
            "expected a finite f32 value",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExecutionError, execute_compiled_frame, execute_compiled_frames_to_files,
        execute_compiled_program, execute_into, execute_program,
    };
    use crate::{
        gmath::matrix::Matrix,
        graphics::colors::Rgb,
        mdl::{
            animation::FrameOutputConfig,
            ast::Vec3,
            parser::parse_script,
            runtime::{RenderConfig, Runtime, Symbol},
            semantic::compile,
        },
    };

    fn execute(src: &str) -> crate::mdl::runtime::Runtime {
        let program = parse_script(src).expect("script parses");
        execute_program(
            &program,
            &RenderConfig::new_with_bg(200, 200, Rgb::WHITE, Rgb::BLACK).display_enabled(false),
        )
        .expect("script executes")
    }

    fn error_kind(error: &ExecutionError) -> &ExecutionError {
        match error {
            ExecutionError::Located { error, .. } => error.as_ref(),
            error => error,
        }
    }

    #[test]
    fn push_pop_restores_previous_transform() {
        let runtime = execute("move 10 0 0\npush\nmove 5 0 0\npop");

        assert!((runtime.top_transform().get(0, 3) - 10.0).abs() < 1e-9);
        assert_eq!(runtime.stack_len(), 1);
    }

    #[test]
    fn pop_on_base_stack_errors() {
        let program = parse_script("pop").unwrap();
        let error = execute_program(&program, &RenderConfig::new(10, 10).display_enabled(false))
            .unwrap_err();

        assert!(matches!(error_kind(&error), ExecutionError::StackUnderflow));
    }

    #[test]
    fn runtime_errors_include_command_location() {
        let program = parse_script("move 1 0 0\nmove 1 0 0 missing").unwrap();
        let error = execute_program(&program, &RenderConfig::new(10, 10).display_enabled(false))
            .unwrap_err();

        assert!(matches!(
            error,
            ExecutionError::Located {
                span,
                ref error,
                ..
            } if span.line == 2 && matches!(error.as_ref(), ExecutionError::UnknownKnob(name) if name == "missing")
        ));
    }

    #[test]
    fn move_then_scale_uses_top_times_transform_order() {
        let runtime = execute("move 10 0 0\nscale 2 2 2");
        let matrix = runtime.top_transform();

        assert!((matrix.get(0, 0) - 2.0).abs() < 1e-9);
        assert!((matrix.get(0, 3) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn transform_knob_scales_transform_parameters() {
        let runtime = execute("set k 0.5\nmove 10 0 0 k");

        assert!((runtime.top_transform().get(0, 3) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn scale_knob_multiplies_scale_factors_per_mdl_rule() {
        let runtime = execute("set k 0\nscale 2 3 4 k");
        let matrix = runtime.top_transform();

        assert!((matrix.get(0, 0) - 0.0).abs() < 1e-9);
        assert!((matrix.get(1, 1) - 0.0).abs() < 1e-9);
        assert!((matrix.get(2, 2) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn compiled_frame_uses_precomputed_vary_knobs() {
        let program = parse_script("frames 3\nvary k 0 2 0 1\nmove 10 0 0 k").unwrap();
        let compiled = compile(program).unwrap();

        let frame0 = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            0,
        )
        .unwrap();
        let frame1 = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            1,
        )
        .unwrap();
        let frame2 = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            2,
        )
        .unwrap();

        assert!((frame0.top_transform().get(0, 3) - 0.0).abs() < 1e-9);
        assert!((frame1.top_transform().get(0, 3) - 5.0).abs() < 1e-9);
        assert!((frame2.top_transform().get(0, 3) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn compiled_execution_preserves_set_source_order_for_non_animation() {
        let program = parse_script("set k 1\nmove 10 0 0 k\nset k 2\nmove 10 0 0 k").unwrap();
        let compiled = compile(program).unwrap();

        let runtime = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            0,
        )
        .unwrap();

        assert!((runtime.top_transform().get(0, 3) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn animation_frame_knobs_override_runtime_set_commands() {
        let program = parse_script("frames 2\nvary k 0 1 0 1\nset k 100\nmove 10 0 0 k").unwrap();
        let compiled = compile(program).unwrap();

        let frame0 = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            0,
        )
        .unwrap();
        let frame1 = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            1,
        )
        .unwrap();

        assert!((frame0.top_transform().get(0, 3) - 0.0).abs() < 1e-9);
        assert!((frame1.top_transform().get(0, 3) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn compiled_frame_reports_animation_metadata() {
        let program = parse_script("frames 4\nbasename spin\nmove 1 0 0").unwrap();
        let compiled = compile(program).unwrap();

        let runtime = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            2,
        )
        .unwrap();

        assert_eq!(runtime.basename(), "spin");
        assert_eq!(runtime.frames(), 4);
    }

    #[test]
    fn compiled_program_executes_all_frames() {
        let program = parse_script("frames 2\nvary k 0 1 0 1\nmove 10 0 0 k").unwrap();
        let compiled = compile(program).unwrap();
        let frames =
            execute_compiled_program(&compiled, &RenderConfig::new(10, 10).display_enabled(false))
                .unwrap();

        assert_eq!(frames.len(), 2);
        assert!((frames[1].top_transform().get(0, 3) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn for_each_compiled_frame_streams_frames_without_collecting() {
        let program = parse_script("frames 2\nvary k 0 1 0 1\nmove 10 0 0 k").unwrap();
        let compiled = compile(program).unwrap();
        let mut translations = Vec::new();

        super::for_each_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            |_, runtime| {
                translations.push(runtime.top_transform().get(0, 3));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(translations, vec![0.0, 10.0]);
    }

    #[test]
    fn for_each_compiled_frame_restores_config_canvas_baseline() {
        let program = parse_script("frames 2").unwrap();
        let compiled = compile(program).unwrap();
        let mut samples = Vec::new();

        super::for_each_compiled_frame(
            &compiled,
            &RenderConfig::new_with_bg(2, 2, Rgb::RED, Rgb::WHITE).display_enabled(false),
            |_, runtime| {
                samples.push((runtime.canvas().pixels()[0], runtime.canvas().line_color()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(samples, vec![(Rgb::WHITE, Rgb::RED); 2]);
    }

    #[test]
    fn compiled_frame_files_use_basename_and_disable_script_save_commands() {
        let dir =
            std::env::temp_dir().join(format!("gartus-mdl-frames-{}-redirect", std::process::id()));
        let static_output = dir.join("static.png");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let program = parse_script(&format!(
            "frames 2\nbasename spin\nvary k 0 1 0 1\nmove 10 0 0 k\nsave {}",
            static_output.to_string_lossy()
        ))
        .unwrap();
        let compiled = compile(program).unwrap();
        let paths = execute_compiled_frames_to_files(
            &compiled,
            RenderConfig::new(20, 20).display_enabled(false),
            &dir,
        )
        .unwrap();

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].file_name().unwrap(), "spin00000000.ppm");
        assert_eq!(paths[1].file_name().unwrap(), "spin00000001.ppm");
        assert!(paths[0].exists());
        assert!(paths[1].exists());
        assert!(
            !static_output.exists(),
            "animated save should be disabled while final frame output is written once"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn compiled_frame_files_save_without_save_command() {
        let dir =
            std::env::temp_dir().join(format!("gartus-mdl-frames-{}-auto", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let program = parse_script("frames 1\nbasename auto\nline 0 0 0 10 10 0").unwrap();
        let compiled = compile(program).unwrap();
        let paths = execute_compiled_frames_to_files(
            &compiled,
            RenderConfig::new(20, 20).display_enabled(false),
            &dir,
        )
        .unwrap();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].file_name().unwrap(), "auto00000000.ppm");
        assert!(paths[0].exists());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn compiled_frame_files_accept_output_options() {
        let dir =
            std::env::temp_dir().join(format!("gartus-mdl-frames-{}-options", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let program = parse_script("frames 1\nbasename opt\nline 0 0 0 10 10 0").unwrap();
        let compiled = compile(program).unwrap();
        let paths = super::execute_compiled_frames_with_options(
            &compiled,
            RenderConfig::new(20, 20).display_enabled(false),
            FrameOutputConfig::new(&dir).extension("png").padding(3),
        )
        .unwrap();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].file_name().unwrap(), "opt000.png");
        assert!(paths[0].exists());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn compiled_frame_rejects_out_of_range_frame() {
        let program = parse_script("frames 1").unwrap();
        let compiled = compile(program).unwrap();
        let error = execute_compiled_frame(
            &compiled,
            &RenderConfig::new(10, 10).display_enabled(false),
            1,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ExecutionError::InvalidFrame {
                frame: 1,
                frames: 1
            }
        ));
    }

    #[test]
    fn shear_and_reflect_execute_like_matrix_transforms() {
        let runtime = execute("set k 0.5\nshear x 2 4 k\nreflect z");
        let expected = Matrix::shearing_x(1.0, 2.0) * Matrix::reflect_xy();

        assert_eq!(runtime.top_transform(), &expected);
    }

    #[test]
    fn ident_apply_and_quit_execute_as_compatibility_commands() {
        let runtime = execute("move 10 0 0\nident\napply\nquit\nmove 90 0 0");

        assert_eq!(runtime.top_transform(), &Matrix::identity_matrix(4));
    }

    #[test]
    fn reset_clears_transform_stack_and_symbols() {
        let runtime = execute("set k 3\nmove 10 0 0\npush\nreset");

        assert_eq!(runtime.stack_len(), 1);
        assert_eq!(runtime.top_transform(), &Matrix::identity_matrix(4));
        assert!(runtime.symbol("k").is_none());
    }

    #[test]
    fn reset_resets_canvas_lighting_too() {
        let runtime = execute("ambient 200 0 0\nlight 255 0 0 0 0 1\nreset");

        assert_eq!(runtime.lights().len(), 0);
        assert_eq!(runtime.ambient(), Vec3::new(50.0, 50.0, 50.0));
        assert_eq!(runtime.canvas().lighting().ambient, Rgb::new(50, 50, 50));
        assert!(runtime.canvas().lighting().point_lights.is_empty());
    }

    #[test]
    fn save_coord_system_copies_current_top() {
        let runtime = execute("move 10 0 0\nsave_coord_system saved\nmove 90 0 0");
        let saved = runtime.coord_system("saved").expect("saved coord system");

        assert!((saved.get(0, 3) - 10.0).abs() < 1e-9);
        assert!((runtime.top_transform().get(0, 3) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn constants_are_stored_and_validated() {
        let runtime = execute("constants metal 1 2 3 4 5 6 7 8 9\nsphere metal 0 0 0 20");

        assert!(matches!(
            runtime.symbol("metal"),
            Some(Symbol::Constants(_))
        ));
    }

    #[test]
    fn constants_without_rgb_default_to_black() {
        let runtime = execute("color red\nconstants mat 1 2 3 4 5 6 7 8 9\nline mat 0 0 0 10 10 0");

        assert!(matches!(
            runtime.symbol("mat"),
            Some(Symbol::Constants(constants)) if constants.color == Vec3::new(0.0, 0.0, 0.0)
        ));
        assert!(!runtime.canvas().pixels().contains(&Rgb::RED));
    }

    #[test]
    fn mdl_lights_accumulate_in_canvas_lighting() {
        let runtime = execute("light 255 0 0 0 0 1\nlight 0 255 0 0 1 1");
        let lighting = runtime.canvas().lighting();

        assert_eq!(runtime.lights().len(), 2);
        assert_eq!(lighting.point_lights.len(), 2);
        assert_eq!(lighting.point_lights[0].color, Rgb::RED);
        assert_eq!(lighting.point_lights[1].color, Rgb::GREEN);
    }

    #[test]
    fn unknown_constants_error_at_execution_time() {
        let program = parse_script("sphere missing 0 0 0 20").unwrap();
        let error = execute_program(
            &program,
            &RenderConfig::new(100, 100).display_enabled(false),
        )
        .unwrap_err();

        assert!(
            matches!(error_kind(&error), ExecutionError::UnknownConstants(name) if name == "missing")
        );
    }

    #[cfg(feature = "external")]
    #[test]
    fn failed_mesh_load_does_not_leave_material_draw_state_applied() {
        let program = parse_script(
            "color red\nconstants mat 1 1 1 1 1 1 1 1 1 0 0 255\nmesh mat :missing.obj",
        )
        .unwrap();
        let config = RenderConfig::new(20, 20).display_enabled(false);
        let mut runtime = Runtime::new(&config);

        let error = execute_into(&mut runtime, &program).unwrap_err();

        assert!(matches!(error_kind(&error), ExecutionError::Mesh { .. }));
        assert_eq!(runtime.canvas().line_color(), Rgb::RED);
    }

    #[test]
    fn line_endpoints_can_use_saved_coordinate_systems() {
        let runtime = execute(
            "move 20 0 0\nsave_coord_system left\nmove 60 0 0\nsave_coord_system right\nline 0 0 0 left 0 10 0 right",
        );

        let drawn = runtime
            .canvas()
            .pixels()
            .iter()
            .any(|pixel| *pixel != Rgb::BLACK);
        assert!(drawn);
    }

    #[test]
    fn sphere_draws_pixels() {
        let runtime = execute("move 100 100 0\nsphere 0 0 0 30");

        let drawn = runtime
            .canvas()
            .pixels()
            .iter()
            .any(|pixel| *pixel != Rgb::BLACK);
        assert!(drawn);
    }

    #[test]
    fn color_and_curve_extensions_draw_pixels() {
        let runtime =
            execute("color red\ncircle 100 100 0 30\nbezier 50 100 75 150 125 150 150 100");

        let drawn_red = runtime.canvas().pixels().contains(&Rgb::RED);
        assert!(drawn_red);
    }

    #[test]
    fn extra_solid_extensions_draw_pixels() {
        let runtime = execute(
            "move 100 100 0\ncylinder 0 0 0 20 50\ncone 40 0 0 20 50\npyramid -40 0 0 40 50",
        );

        let drawn = runtime
            .canvas()
            .pixels()
            .iter()
            .any(|pixel| *pixel != Rgb::BLACK);
        assert!(drawn);
    }

    #[cfg(feature = "filters")]
    #[test]
    fn filter_command_applies_canvas_filter() {
        let runtime = execute("filter invert");

        assert!(
            runtime
                .canvas()
                .pixels()
                .iter()
                .all(|pixel| *pixel == Rgb::WHITE)
        );
    }

    #[cfg(feature = "filters")]
    #[test]
    fn filter_command_validates_numeric_arguments() {
        let program = parse_script("filter solarize 12.5").unwrap();
        let error = execute_program(
            &program,
            &RenderConfig::new_with_bg(2, 2, Rgb::WHITE, Rgb::BLACK).display_enabled(false),
        )
        .unwrap_err();

        assert!(matches!(
            error_kind(&error),
            ExecutionError::InvalidFilter { name, .. } if name == "solarize"
        ));
    }

    #[cfg(not(feature = "filters"))]
    #[test]
    fn filter_command_reports_missing_filters_feature() {
        let program = parse_script("filter invert").unwrap();
        let error = execute_program(
            &program,
            &RenderConfig::new_with_bg(2, 2, Rgb::WHITE, Rgb::BLACK).display_enabled(false),
        )
        .unwrap_err();

        assert!(error.to_string().contains("filters"));
    }

    #[test]
    fn face_mdl_parses_executes_and_saves() {
        let _ = std::fs::remove_file("face.png");
        let source = include_str!("../../tests/face.mdl");
        let program = parse_script(source).expect("face.mdl parses");
        let runtime = execute_program(
            &program,
            &RenderConfig::new(500, 500).display_enabled(false),
        )
        .expect("face.mdl executes");

        assert!(std::path::Path::new("face.png").exists());
        let drawn = runtime
            .canvas()
            .pixels()
            .iter()
            .any(|pixel| *pixel != Rgb::BLACK);
        assert!(drawn);
        let _ = std::fs::remove_file("face.png");
    }
}

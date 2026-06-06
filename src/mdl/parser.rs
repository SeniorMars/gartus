//! Line parser for MDL commands.

use super::{
    ast::{
        AnimationCommand, Axis, CameraCommand, ColorSpec, Command, ControlCommand, CurveCommand,
        FilterCommand, Material, OutputCommand, PointRef, Program, RenderCommand, ShadingMode,
        ShapeCommand, Spanned, TransformCommand, VaryInterpolation, Vec2, Vec3,
    },
    diagnostic::Diagnostic,
    lexer::{Span, Token, TokenKind, lex_line},
};

const MAX_BEZIERN_DEGREE: usize = 1_000;

type NamedGeometryArgs<const N: usize> = (Option<String>, [f64; N], Option<String>);

/// Parses an MDL source string into a typed command list.
///
/// This parser only reads the text it is given. It does not expand `include`
/// commands or resolve paths; use `parse_file`, `compile_file`, or `run_file`
/// for complete MDL file execution.
///
/// # Errors
/// Returns all line-level diagnostics found during lexing or parsing.
pub fn parse_script(src: &str) -> Result<Program, Vec<Diagnostic>> {
    let mut commands = Vec::new();
    let mut errors = Vec::new();

    for (idx, line) in src.lines().enumerate() {
        let line_no = idx + 1;
        let tokens = match lex_line(line_no, line) {
            Ok(tokens) => tokens,
            Err(error) => {
                errors.push(error);
                continue;
            }
        };

        if tokens.is_empty() {
            continue;
        }

        match parse_command(&tokens) {
            Ok(command) => commands.push(Spanned::new(command, tokens[0].span)),
            Err(error) => errors.push(error),
        }
    }

    if errors.is_empty() {
        Ok(Program { commands })
    } else {
        Err(errors)
    }
}

fn control(command: ControlCommand) -> Command {
    Command::Control(command)
}

fn transform(command: TransformCommand) -> Command {
    Command::Transform(command)
}

fn curve(command: CurveCommand) -> Command {
    Command::Curve(command)
}

fn shape(command: ShapeCommand) -> Command {
    Command::Shape(command)
}

fn animation(command: AnimationCommand) -> Command {
    Command::Animation(command)
}

fn render(command: RenderCommand) -> Command {
    Command::Render(command)
}

fn camera(command: CameraCommand) -> Command {
    Command::Camera(command)
}

fn output(command: OutputCommand) -> Command {
    Command::Output(command)
}

fn parse_command(tokens: &[Token]) -> Result<Command, Diagnostic> {
    let command_token = &tokens[0];
    let command = expect_command_name(command_token)?;
    let args = &tokens[1..];

    match command {
        "apply" | "web" => parse_no_args(command_token, args, control(ControlCommand::Apply)),
        "quit" => parse_no_args(command_token, args, control(ControlCommand::Quit)),
        "push" => parse_no_args(command_token, args, control(ControlCommand::Push)),
        "pop" => parse_no_args(command_token, args, control(ControlCommand::Pop)),
        "ident" => parse_no_args(command_token, args, control(ControlCommand::Ident)),
        "clear" => parse_no_args(command_token, args, control(ControlCommand::Clear)),
        "reset" => parse_no_args(command_token, args, control(ControlCommand::Reset)),
        "move" => parse_move(command_token, args),
        "scale" => parse_scale(command_token, args),
        "rotate" => parse_rotate(command_token, args),
        "reflect" => parse_reflect(command_token, args),
        "shear" => parse_shear(command_token, args),
        "color" => parse_color(command_token, args),
        "circle" => parse_circle(command_token, args),
        "hermite" => parse_hermite(command_token, args),
        "bezier" => parse_bezier(command_token, args),
        "beziern" => parse_beziern(command_token, args),
        "bezier_surface" => parse_bezier_surface(command_token, args),
        "sphere" => parse_sphere(command_token, args),
        "torus" => parse_torus(command_token, args),
        "box" => parse_box(command_token, args),
        "line" => parse_line(command_token, args),
        "mesh" => parse_mesh(command_token, args),
        "mesh_reverse" => parse_mesh_reverse(command_token, args),
        "texture" => parse_texture(command_token, args),
        "cylinder" => parse_cylinder(command_token, args),
        "cone" => parse_cone(command_token, args),
        "pyramid" => parse_pyramid(command_token, args),
        "basename" => parse_basename(command_token, args),
        "frames" => parse_frames(command_token, args),
        "set" => parse_set(command_token, args),
        "save_knobs" | "saveknobs" => parse_save_knobs(command_token, args),
        "tween" => parse_tween(command_token, args),
        "vary" => parse_vary(command_token, args),
        "setknobs" => parse_setknobs(command_token, args),
        "light" => parse_light(command_token, args),
        "ambient" => parse_ambient(command_token, args),
        "constants" => parse_constants(command_token, args),
        "shading" => parse_shading(command_token, args),
        "save_coord_system" | "save_coordinate_system" => {
            parse_save_coord_system(command_token, args)
        }
        "camera" => parse_camera(command_token, args),
        "save" => parse_save(command_token, args),
        "include" => parse_include(command_token, args),
        "filter" => parse_filter(command_token, args),
        "display" => parse_no_args(command_token, args, output(OutputCommand::Display)),
        "focal" => parse_focal(command_token, args),
        // `gereate_rayfiles` is the spelling in the original MDL grammar.
        "generate_rayfiles" | "gereate_rayfiles" => {
            parse_no_args(command_token, args, output(OutputCommand::GenerateRayfiles))
        }
        other => Err(diag_at_token(
            command_token,
            format!("unknown command `{other}`"),
        )),
    }
}

fn parse_no_args(command: &Token, args: &[Token], parsed: Command) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[0], "command takes no arguments")?;
    Ok(parsed)
}

fn parse_move(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[3, 4], "move x y z [knob]")?;
    let x = expect_number(command, args, 0)?;
    let y = expect_number(command, args, 1)?;
    let z = expect_number(command, args, 2)?;
    let knob = optional_ident(command, args, 3)?;
    Ok(transform(TransformCommand::Move { x, y, z, knob }))
}

fn parse_scale(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[3, 4], "scale x y z [knob]")?;
    let x = expect_number(command, args, 0)?;
    let y = expect_number(command, args, 1)?;
    let z = expect_number(command, args, 2)?;
    let knob = optional_ident(command, args, 3)?;
    Ok(transform(TransformCommand::Scale { x, y, z, knob }))
}

fn parse_rotate(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[2, 3], "rotate x|y|z degrees [knob]")?;

    let axis = match expect_word_ref(command, args, 0, "rotate axis")? {
        "x" => Axis::X,
        "y" => Axis::Y,
        "z" => Axis::Z,
        other => {
            return Err(
                diag_at_token(&args[0], format!("invalid rotate axis `{other}`"))
                    .with_help("expected exactly one of `x`, `y`, or `z`"),
            );
        }
    };

    let degrees = expect_number(command, args, 1)?;
    let knob = optional_ident(command, args, 2)?;

    Ok(transform(TransformCommand::Rotate {
        axis,
        degrees,
        knob,
    }))
}

fn parse_reflect(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "reflect x|y|z")?;
    Ok(transform(TransformCommand::Reflect {
        axis: parse_axis(command, args, 0, "reflect axis")?,
    }))
}

fn parse_shear(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[3, 4], "shear x|y|z sh0 sh1 [knob]")?;
    Ok(transform(TransformCommand::Shear {
        axis: parse_axis(command, args, 0, "shear axis")?,
        sh0: expect_number(command, args, 1)?,
        sh1: expect_number(command, args, 2)?,
        knob: optional_ident(command, args, 3)?,
    }))
}

fn parse_color(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1, 3], "color name|r g b")?;
    if args.len() == 1 {
        return Ok(render(RenderCommand::Color(ColorSpec::Name(expect_ident(
            command,
            args,
            0,
            "color name",
        )?))));
    }

    Ok(render(RenderCommand::Color(ColorSpec::Rgb(parse_rgb(
        command, args, 0,
    )?))))
}

fn parse_circle(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[4], "circle x y z r")?;
    Ok(curve(CurveCommand::Circle {
        center: parse_vec3(command, args, 0)?,
        radius: expect_number(command, args, 3)?,
    }))
}

fn parse_hermite(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[8], "hermite x0 y0 x1 y1 rx0 ry0 rx1 ry1")?;
    Ok(curve(CurveCommand::Hermite {
        p0: parse_vec2(command, args, 0)?,
        p1: parse_vec2(command, args, 2)?,
        r0: parse_vec2(command, args, 4)?,
        r1: parse_vec2(command, args, 6)?,
    }))
}

fn parse_bezier(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[8], "bezier x0 y0 x1 y1 x2 y2 x3 y3")?;
    Ok(curve(CurveCommand::Bezier {
        p0: parse_vec2(command, args, 0)?,
        p1: parse_vec2(command, args, 2)?,
        p2: parse_vec2(command, args, 4)?,
        p3: parse_vec2(command, args, 6)?,
    }))
}

fn parse_beziern(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    if args.is_empty() {
        return Err(diag_at_token(command, "wrong number of arguments")
            .with_help("beziern degree x0 y0 x1 y1 ..."));
    }

    let degree = expect_usize(command, args, 0)?;
    if degree > MAX_BEZIERN_DEGREE {
        return Err(diag_at_token(
            &args[0],
            format!("beziern degree must be <= {MAX_BEZIERN_DEGREE}"),
        ));
    }
    let point_count = degree
        .checked_add(1)
        .ok_or_else(|| diag_at_token(&args[0], "beziern degree is too large"))?;
    let coord_count = point_count
        .checked_mul(2)
        .ok_or_else(|| diag_at_token(&args[0], "beziern degree is too large"))?;
    let expected = coord_count
        .checked_add(1)
        .ok_or_else(|| diag_at_token(&args[0], "beziern degree is too large"))?;
    if args.len() != expected {
        return Err(diag_at_token(
            args.first().unwrap_or(command),
            format!(
                "beziern degree {degree} expects {} coordinate values, got {}",
                coord_count,
                args.len().saturating_sub(1)
            ),
        )
        .with_help("beziern degree x0 y0 x1 y1 ..."));
    }

    let mut points = Vec::with_capacity(point_count);
    for index in (1..args.len()).step_by(2) {
        points.push(Vec2::new(
            expect_number(command, args, index)?,
            expect_number(command, args, index + 1)?,
        ));
    }

    Ok(curve(CurveCommand::BezierN { degree, points }))
}

fn parse_bezier_surface(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[49],
        "bezier_surface steps followed by 16 x y z control points",
    )?;
    let steps = expect_positive_usize(command, args, 0)?;
    let mut controls = Vec::with_capacity(16);
    for index in (1..args.len()).step_by(3) {
        controls.push(parse_vec3(command, args, index)?);
    }
    Ok(curve(CurveCommand::BezierSurface { steps, controls }))
}

fn parse_sphere(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    let (constants, nums, coord_system) = parse_optional_name_numbers_name::<4>(
        command,
        args,
        "sphere [constants] x y z r [coord_system]",
    )?;
    Ok(shape(ShapeCommand::Sphere {
        constants,
        center: Vec3::new(nums[0], nums[1], nums[2]),
        radius: nums[3],
        coord_system,
    }))
}

fn parse_torus(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    let (constants, nums, coord_system) = parse_optional_name_numbers_name::<5>(
        command,
        args,
        "torus [constants] x y z r0 r1 [coord_system]",
    )?;
    Ok(shape(ShapeCommand::Torus {
        constants,
        center: Vec3::new(nums[0], nums[1], nums[2]),
        r0: nums[3],
        r1: nums[4],
        coord_system,
    }))
}

fn parse_box(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    let (constants, nums, coord_system) = parse_optional_name_numbers_name::<6>(
        command,
        args,
        "box [constants] x0 y0 z0 h w d [coord_system]",
    )?;
    Ok(shape(ShapeCommand::Box {
        constants,
        corner: Vec3::new(nums[0], nums[1], nums[2]),
        h: nums[3],
        w: nums[4],
        d: nums[5],
        coord_system,
    }))
}

fn parse_cylinder(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    let (constants, nums, coord_system) = parse_optional_name_numbers_name::<5>(
        command,
        args,
        "cylinder [constants] x y z r h [coord_system]",
    )?;
    Ok(shape(ShapeCommand::Cylinder {
        constants,
        center: Vec3::new(nums[0], nums[1], nums[2]),
        radius: nums[3],
        height: nums[4],
        coord_system,
    }))
}

fn parse_cone(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    let (constants, nums, coord_system) = parse_optional_name_numbers_name::<5>(
        command,
        args,
        "cone [constants] x y z r h [coord_system]",
    )?;
    Ok(shape(ShapeCommand::Cone {
        constants,
        center: Vec3::new(nums[0], nums[1], nums[2]),
        radius: nums[3],
        height: nums[4],
        coord_system,
    }))
}

fn parse_pyramid(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    let (constants, nums, coord_system) = parse_optional_name_numbers_name::<5>(
        command,
        args,
        "pyramid [constants] x y z base_length h [coord_system]",
    )?;
    Ok(shape(ShapeCommand::Pyramid {
        constants,
        center: Vec3::new(nums[0], nums[1], nums[2]),
        base_length: nums[3],
        height: nums[4],
        coord_system,
    }))
}

fn parse_line(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    if args.first().is_some_and(is_ident_token) {
        let constants = expect_ident(command, args, 0, "constants name")?;
        return parse_line_inner(command, Some(constants), &args[1..]);
    }

    parse_line_inner(command, None, args)
}

fn parse_line_inner(
    command: &Token,
    constants: Option<String>,
    args: &[Token],
) -> Result<Command, Diagnostic> {
    let (p0, used) = parse_point_ref(command, args)?;
    let (p1, used2) = parse_point_ref(command, &args[used..])?;
    if used + used2 != args.len() {
        return Err(diag_at_token(
            &args[used + used2],
            "unexpected extra argument in line command",
        )
        .with_help("line [constants] x0 y0 z0 [coord_system0] x1 y1 z1 [coord_system1]"));
    }
    Ok(shape(ShapeCommand::Line { constants, p0, p1 }))
}

fn parse_mesh(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    parse_mesh_like(command, args, false)
}

fn parse_mesh_reverse(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    parse_mesh_like(command, args, true)
}

fn parse_mesh_like(command: &Token, args: &[Token], reverse: bool) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[1, 2, 3],
        "mesh [constants] :filename [coord_system]",
    )?;

    Ok(match args.len() {
        1 => mesh_command(None, expect_filename(command, args, 0)?, None, reverse),
        2 if matches!(args[0].kind, TokenKind::Filename(_)) => mesh_command(
            None,
            expect_filename(command, args, 0)?,
            Some(expect_ident(command, args, 1, "coord_system name")?),
            reverse,
        ),
        2 => mesh_command(
            Some(expect_ident(command, args, 0, "constants name")?),
            expect_filename(command, args, 1)?,
            None,
            reverse,
        ),
        3 => mesh_command(
            Some(expect_ident(command, args, 0, "constants name")?),
            expect_filename(command, args, 1)?,
            Some(expect_ident(command, args, 2, "coord_system name")?),
            reverse,
        ),
        _ => unreachable!("argument count checked"),
    })
}

fn mesh_command(
    constants: Option<String>,
    filename: String,
    coord_system: Option<String>,
    reverse: bool,
) -> Command {
    if reverse {
        shape(ShapeCommand::MeshReverse {
            constants,
            filename,
            coord_system,
        })
    } else {
        shape(ShapeCommand::Mesh {
            constants,
            filename,
            coord_system,
        })
    }
}

fn parse_texture(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[13],
        "texture filename x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3",
    )?;
    let filename = expect_text(command, args, 0, "texture filename")?;
    let nums = parse_numbers::<12>(command, &args[1..13])?;
    Ok(shape(ShapeCommand::Texture {
        filename,
        points: [
            Vec3::new(nums[0], nums[1], nums[2]),
            Vec3::new(nums[3], nums[4], nums[5]),
            Vec3::new(nums[6], nums[7], nums[8]),
            Vec3::new(nums[9], nums[10], nums[11]),
        ],
    }))
}

fn parse_basename(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "basename name")?;
    Ok(animation(AnimationCommand::Basename(expect_text(
        command, args, 0, "basename",
    )?)))
}

fn parse_frames(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "frames num_frames")?;
    Ok(animation(AnimationCommand::Frames(expect_usize(
        command, args, 0,
    )?)))
}

fn parse_set(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[2], "set knobname value")?;
    Ok(animation(AnimationCommand::Set {
        knob: expect_ident(command, args, 0, "knob name")?,
        value: expect_number(command, args, 1)?,
    }))
}

fn parse_save_knobs(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "save_knobs knoblist")?;
    Ok(animation(AnimationCommand::SaveKnobs(expect_ident(
        command,
        args,
        0,
        "knob list name",
    )?)))
}

fn parse_tween(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[4],
        "tween start_frame end_frame knoblist0 knoblist1",
    )?;
    Ok(animation(AnimationCommand::Tween {
        start_frame: expect_usize(command, args, 0)?,
        end_frame: expect_usize(command, args, 1)?,
        knoblist0: expect_ident(command, args, 2, "start knob list name")?,
        knoblist1: expect_ident(command, args, 3, "end knob list name")?,
    }))
}

fn parse_vary(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[5, 6, 7],
        "vary knob start_frame end_frame start_val end_val [linear|exponential|logarithmic|smoothstep|power exponent]",
    )?;
    let interpolation = parse_vary_interpolation(command, args)?;
    Ok(animation(AnimationCommand::Vary {
        knob: expect_ident(command, args, 0, "knob name")?,
        start_frame: expect_usize(command, args, 1)?,
        end_frame: expect_usize(command, args, 2)?,
        start_val: expect_number(command, args, 3)?,
        end_val: expect_number(command, args, 4)?,
        interpolation,
    }))
}

fn parse_vary_interpolation(
    command: &Token,
    args: &[Token],
) -> Result<VaryInterpolation, Diagnostic> {
    if args.len() == 5 {
        return Ok(VaryInterpolation::Linear);
    }

    let mode = expect_ident_ref(command, args, 5, "vary interpolation mode")?.to_ascii_lowercase();
    match mode.as_str() {
        "linear" => {
            if args.len() == 6 {
                Ok(VaryInterpolation::Linear)
            } else {
                Err(diag_at_token(&args[6], "`linear` does not take an argument"))
            }
        }
        "exponential" | "exp" => {
            if args.len() == 6 {
                Ok(VaryInterpolation::Exponential)
            } else {
                Err(diag_at_token(
                    &args[6],
                    "`exponential` does not take an argument",
                ))
            }
        }
        "logarithmic" | "log" => {
            if args.len() == 6 {
                Ok(VaryInterpolation::Logarithmic)
            } else {
                Err(diag_at_token(
                    &args[6],
                    "`logarithmic` does not take an argument",
                ))
            }
        }
        "smoothstep" => {
            if args.len() == 6 {
                Ok(VaryInterpolation::Smoothstep)
            } else {
                Err(diag_at_token(&args[6], "`smoothstep` does not take an argument"))
            }
        }
        "power" | "pow" => {
            if args.len() != 7 {
                return Err(diag_at_token(command, "`power` requires an exponent"));
            }
            let exponent = expect_number(command, args, 6)?;
            if exponent.is_finite() && exponent > 0.0 {
                Ok(VaryInterpolation::Power(exponent))
            } else {
                Err(diag_at_token(
                    &args[6],
                    "expected positive finite power exponent",
                ))
            }
        }
        other => Err(diag_at_token(&args[5], format!("invalid vary mode `{other}`")).with_help(
            "expected one of `linear`, `exponential`, `logarithmic`, `smoothstep`, or `power exponent`",
        )),
    }
}

fn parse_setknobs(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "setknobs value")?;
    Ok(animation(AnimationCommand::SetKnobs(expect_number(
        command, args, 0,
    )?)))
}

fn parse_light(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[6, 7, 8],
        "light r g b x y z [knob] | light name x y z [knob] r g b",
    )?;
    let (name, color, position, knob) = if args.len() == 8 {
        let name = expect_ident(command, args, 0, "light name")?;
        (
            Some(name),
            parse_rgb(command, args, 5)?,
            parse_vec3(command, args, 1)?,
            Some(expect_ident(command, args, 4, "light knob")?),
        )
    } else if args.len() == 7 && is_ident_token(&args[0]) {
        let name = expect_ident(command, args, 0, "light name")?;
        // The 11_anim C grammar uses: light name x y z r g b.
        (
            Some(name),
            parse_rgb(command, args, 4)?,
            parse_vec3(command, args, 1)?,
            None,
        )
    } else if args.len() == 7 {
        (
            None,
            parse_rgb(command, args, 0)?,
            parse_vec3(command, args, 3)?,
            Some(expect_ident(command, args, 6, "light knob")?),
        )
    } else {
        // The prose MDL spec uses: light r g b x y z.
        (
            None,
            parse_rgb(command, args, 0)?,
            parse_vec3(command, args, 3)?,
            None,
        )
    };
    Ok(render(RenderCommand::Light {
        name,
        color,
        position,
        knob,
    }))
}

fn parse_ambient(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[3], "ambient r g b")?;
    Ok(render(RenderCommand::Ambient {
        color: parse_rgb(command, args, 0)?,
    }))
}

fn parse_constants(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[10, 13],
        "constants name kar kdr ksr kag kdg ksg kab kdb ksb [r] [g] [b]",
    )?;
    let name = expect_ident(command, args, 0, "constants name")?;
    let nums = parse_numbers::<9>(command, &args[1..10])?;
    let material = Material::new(
        nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6], nums[7], nums[8],
    );
    let color = if args.len() == 13 {
        parse_rgb(command, args, 10)?
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };
    Ok(render(RenderCommand::Constants {
        name,
        material,
        color,
    }))
}

fn parse_shading(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[1],
        "shading wireframe|flat|gouraud|phong|toon|raytrace",
    )?;
    let mode = match expect_ident_ref(command, args, 0, "shading mode")? {
        "wireframe" => ShadingMode::Wireframe,
        "flat" => ShadingMode::Flat,
        "gouraud" => ShadingMode::Gouraud,
        "phong" => ShadingMode::Phong,
        "toon" => ShadingMode::Toon,
        "raytrace" => ShadingMode::Raytrace,
        other => {
            return Err(
                diag_at_token(&args[0], format!("invalid shading mode `{other}`")).with_help(
                    "expected one of `wireframe`, `flat`, `gouraud`, `phong`, `toon`, or `raytrace`",
                ),
            );
        }
    };
    Ok(render(RenderCommand::Shading(mode)))
}

fn parse_save_coord_system(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "save_coord_system name")?;
    Ok(render(RenderCommand::SaveCoordSystem(expect_ident(
        command,
        args,
        0,
        "coordinate-system name",
    )?)))
}

fn parse_camera(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(
        command,
        args,
        &[6],
        "camera eye_x eye_y eye_z aim_x aim_y aim_z",
    )?;
    Ok(camera(CameraCommand::Camera {
        eye: parse_vec3(command, args, 0)?,
        aim: parse_vec3(command, args, 3)?,
    }))
}

fn parse_save(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "save filename")?;
    Ok(output(OutputCommand::Save(expect_text(
        command, args, 0, "filename",
    )?)))
}

fn parse_include(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "include filename")?;
    Ok(Command::Include(expect_text(command, args, 0, "filename")?))
}

fn parse_filter(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1, 2], "filter name [value]")?;
    Ok(Command::Filter(FilterCommand {
        name: expect_ident(command, args, 0, "filter name")?,
        value: if args.len() == 2 {
            Some(expect_number(command, args, 1)?)
        } else {
            None
        },
    }))
}

fn parse_focal(command: &Token, args: &[Token]) -> Result<Command, Diagnostic> {
    expect_len(command, args, &[1], "focal value")?;
    Ok(camera(CameraCommand::Focal(expect_number(
        command, args, 0,
    )?)))
}

fn parse_axis(
    command: &Token,
    args: &[Token],
    index: usize,
    role: &str,
) -> Result<Axis, Diagnostic> {
    match expect_word_ref(command, args, index, role)? {
        "x" => Ok(Axis::X),
        "y" => Ok(Axis::Y),
        "z" => Ok(Axis::Z),
        other => Err(
            diag_at_token(&args[index], format!("invalid axis `{other}`"))
                .with_help("expected exactly one of `x`, `y`, or `z`"),
        ),
    }
}

fn parse_optional_name_numbers_name<const N: usize>(
    command: &Token,
    args: &[Token],
    syntax: &str,
) -> Result<NamedGeometryArgs<N>, Diagnostic> {
    match args.len() {
        n if n == N => Ok((None, parse_numbers::<N>(command, args)?, None)),
        n if n == N + 1 => {
            if is_ident_token(&args[0]) {
                let constants = Some(expect_ident(command, args, 0, "constants name")?);
                let nums = parse_numbers::<N>(command, &args[1..])?;
                Ok((constants, nums, None))
            } else if is_ident_token(&args[N]) {
                let nums = parse_numbers::<N>(command, &args[..N])?;
                let coord_system = Some(expect_ident(command, args, N, "coord_system name")?);
                Ok((None, nums, coord_system))
            } else {
                Err(
                    diag_at_token(&args[0], "expected numeric geometry arguments")
                        .with_help(syntax),
                )
            }
        }
        n if n == N + 2 => {
            let constants = Some(expect_ident(command, args, 0, "constants name")?);
            let nums = parse_numbers::<N>(command, &args[1..=N])?;
            let coord_system = Some(expect_ident(command, args, N + 1, "coord_system name")?);
            Ok((constants, nums, coord_system))
        }
        _ => Err(diag_at_token(command, "wrong number of arguments").with_help(syntax)),
    }
}

fn parse_point_ref(command: &Token, args: &[Token]) -> Result<(PointRef, usize), Diagnostic> {
    if args.len() < 3 {
        return Err(diag_at_token(command, "expected point").with_help("x y z [coord_system]"));
    }

    let point = parse_vec3(command, args, 0)?;
    if args.get(3).is_some_and(is_ident_token) {
        Ok((
            PointRef {
                point,
                coord_system: Some(expect_ident(command, args, 3, "coord_system name")?),
            },
            4,
        ))
    } else {
        Ok((
            PointRef {
                point,
                coord_system: None,
            },
            3,
        ))
    }
}

fn parse_vec3(command: &Token, args: &[Token], start: usize) -> Result<Vec3, Diagnostic> {
    Ok(Vec3::new(
        expect_number(command, args, start)?,
        expect_number(command, args, start + 1)?,
        expect_number(command, args, start + 2)?,
    ))
}

fn parse_rgb(command: &Token, args: &[Token], start: usize) -> Result<Vec3, Diagnostic> {
    Ok(Vec3::new(
        f64::from(expect_u8_number(command, args, start)?),
        f64::from(expect_u8_number(command, args, start + 1)?),
        f64::from(expect_u8_number(command, args, start + 2)?),
    ))
}

fn parse_vec2(command: &Token, args: &[Token], start: usize) -> Result<Vec2, Diagnostic> {
    Ok(Vec2::new(
        expect_number(command, args, start)?,
        expect_number(command, args, start + 1)?,
    ))
}

fn parse_numbers<const N: usize>(command: &Token, args: &[Token]) -> Result<[f64; N], Diagnostic> {
    let mut nums = [0.0; N];
    for (index, num) in nums.iter_mut().enumerate() {
        *num = expect_number(command, args, index)?;
    }
    Ok(nums)
}

fn expect_command_name(token: &Token) -> Result<&str, Diagnostic> {
    match &token.kind {
        TokenKind::Word(word) if is_valid_ident(word) => Ok(word),
        TokenKind::Word(word) => Err(diag_at_token(
            token,
            format!("invalid command name `{word}`"),
        )
        .with_help("command names must use letters, digits, and underscores")),
        TokenKind::Number(_) | TokenKind::Filename(_) => Err(diag_at_token(
            token,
            "expected command name at start of line",
        )),
    }
}

fn expect_len(
    command: &Token,
    args: &[Token],
    allowed: &[usize],
    syntax: &str,
) -> Result<(), Diagnostic> {
    if allowed.contains(&args.len()) {
        return Ok(());
    }

    let token = args.first().unwrap_or(command);
    Err(diag_at_token(
        token,
        format!("wrong number of arguments; got {}", args.len()),
    )
    .with_help(syntax))
}

fn optional_ident(
    command: &Token,
    args: &[Token],
    index: usize,
) -> Result<Option<String>, Diagnostic> {
    if args.len() > index {
        Ok(Some(expect_ident(command, args, index, "identifier")?))
    } else {
        Ok(None)
    }
}

fn expect_number(command: &Token, args: &[Token], index: usize) -> Result<f64, Diagnostic> {
    let token = args.get(index).unwrap_or(command);
    match token.kind {
        TokenKind::Number(value) => Ok(value),
        TokenKind::Word(_) | TokenKind::Filename(_) => Err(diag_at_token(token, "expected number")),
    }
}

fn expect_usize(command: &Token, args: &[Token], index: usize) -> Result<usize, Diagnostic> {
    let token = args.get(index).unwrap_or(command);
    let value = expect_number(command, args, index)?;
    if value.fract() != 0.0 || value < 0.0 {
        return Err(diag_at_token(token, "expected non-negative integer"));
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    {
        if value > usize::MAX as f64 {
            return Err(diag_at_token(token, "integer is too large"));
        }
        Ok(value as usize)
    }
}

fn expect_u8_number(command: &Token, args: &[Token], index: usize) -> Result<u8, Diagnostic> {
    let token = args.get(index).unwrap_or(command);
    let value = expect_number(command, args, index)?;
    if value.fract() != 0.0 || !(0.0..=255.0).contains(&value) {
        return Err(diag_at_token(
            token,
            "expected integer color channel from 0 to 255",
        ));
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value as u8)
}

fn expect_positive_usize(
    command: &Token,
    args: &[Token],
    index: usize,
) -> Result<usize, Diagnostic> {
    let token = args.get(index).unwrap_or(command);
    let value = expect_usize(command, args, index)?;
    if value == 0 {
        return Err(diag_at_token(token, "expected positive integer"));
    }
    Ok(value)
}

fn expect_ident(
    command: &Token,
    args: &[Token],
    index: usize,
    role: &str,
) -> Result<String, Diagnostic> {
    Ok(expect_ident_ref(command, args, index, role)?.to_string())
}

fn expect_ident_ref<'a>(
    command: &Token,
    args: &'a [Token],
    index: usize,
    role: &str,
) -> Result<&'a str, Diagnostic> {
    let Some(token) = args.get(index) else {
        return Err(diag_at_token(command, format!("expected {role}")));
    };
    match &token.kind {
        TokenKind::Word(word) if is_valid_ident(word) => Ok(word),
        TokenKind::Word(word) => Err(diag_at_token(
            token,
            format!("expected {role}, got `{word}`"),
        )
        .with_help("identifiers must match [A-Za-z_][A-Za-z0-9_]*")),
        TokenKind::Number(_) => Err(diag_at_token(token, format!("expected {role}, got number"))),
        TokenKind::Filename(_) => Err(diag_at_token(
            token,
            format!("expected {role}, got filename"),
        )),
    }
}

fn expect_word_ref<'a>(
    command: &Token,
    args: &'a [Token],
    index: usize,
    role: &str,
) -> Result<&'a str, Diagnostic> {
    let Some(token) = args.get(index) else {
        return Err(diag_at_token(command, format!("expected {role}")));
    };
    match &token.kind {
        TokenKind::Word(word) => Ok(word),
        TokenKind::Number(_) => Err(diag_at_token(token, format!("expected {role}, got number"))),
        TokenKind::Filename(_) => Err(diag_at_token(
            token,
            format!("expected {role}, got filename"),
        )),
    }
}

fn expect_filename(command: &Token, args: &[Token], index: usize) -> Result<String, Diagnostic> {
    let token = args.get(index).unwrap_or(command);
    match &token.kind {
        TokenKind::Filename(filename) => Ok(filename.clone()),
        TokenKind::Word(word) => Err(diag_at_token(
            token,
            format!("expected mesh filename starting with `:`, got `{word}`"),
        )),
        TokenKind::Number(_) => Err(diag_at_token(
            token,
            "expected mesh filename starting with `:`, got number",
        )),
    }
}

fn expect_text(
    command: &Token,
    args: &[Token],
    index: usize,
    role: &str,
) -> Result<String, Diagnostic> {
    let token = args.get(index).unwrap_or(command);
    match &token.kind {
        TokenKind::Word(text) | TokenKind::Filename(text) => Ok(text.clone()),
        TokenKind::Number(_) => Err(diag_at_token(token, format!("expected {role}, got number"))),
    }
}

fn is_ident_token(token: &Token) -> bool {
    matches!(&token.kind, TokenKind::Word(word) if is_valid_ident(word))
}

fn is_valid_ident(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn diag_at_token(token: &Token, message: impl Into<String>) -> Diagnostic {
    diag_at_span(token.span, message)
}

fn diag_at_span(span: Span, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(span.line, span.col_start, span.col_end, message)
}

#[cfg(test)]
mod tests {
    use super::parse_script;
    use crate::mdl::{
        ast::{
            AnimationCommand, Axis, ColorSpec, Command, ControlCommand, CurveCommand,
            OutputCommand, RenderCommand, ShadingMode, ShapeCommand, TransformCommand,
            VaryInterpolation, Vec3,
        },
        lexer::lex_line,
    };
    use std::fmt::Write as _;

    #[test]
    fn parses_basic_transform_commands() {
        let program =
            parse_script("push\nmove 1 2 3\nscale 2 2 2 grow\nrotate y 23 k1\npop").unwrap();

        assert_eq!(program.commands.len(), 5);
        assert_eq!(
            program.commands[1].node,
            Command::Transform(TransformCommand::Move {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                knob: None,
            })
        );
        assert_eq!(
            program.commands[3].node,
            Command::Transform(TransformCommand::Rotate {
                axis: Axis::Y,
                degrees: 23.0,
                knob: Some("k1".to_string()),
            })
        );
    }

    #[test]
    fn parses_optional_constants_and_coord_systems_by_count() {
        let program = parse_script(
            "sphere metal 0 0 0 50 hand\ntorus 0 0 0 10 20 world\nbox wood 1 2 3 4 5 6",
        )
        .unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Shape(ShapeCommand::Sphere {
                constants: Some("metal".to_string()),
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 50.0,
                coord_system: Some("hand".to_string()),
            })
        );
        assert!(matches!(
            program.commands[1].node,
            Command::Shape(ShapeCommand::Torus {
                constants: None,
                coord_system: Some(_),
                ..
            })
        ));
        assert!(matches!(
            program.commands[2].node,
            Command::Shape(ShapeCommand::Box {
                constants: Some(_),
                coord_system: None,
                ..
            })
        ));
    }

    #[test]
    fn parses_vary_interpolation_modes() {
        let program =
            parse_script("vary k 0 9 0 1 exponential\nvary spin 0 9 0 90 power 3").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Animation(AnimationCommand::Vary {
                knob: "k".to_string(),
                start_frame: 0,
                end_frame: 9,
                start_val: 0.0,
                end_val: 1.0,
                interpolation: VaryInterpolation::Exponential,
            })
        );
        assert_eq!(
            program.commands[1].node,
            Command::Animation(AnimationCommand::Vary {
                knob: "spin".to_string(),
                start_frame: 0,
                end_frame: 9,
                start_val: 0.0,
                end_val: 90.0,
                interpolation: VaryInterpolation::Power(3.0),
            })
        );
    }

    #[test]
    fn parses_save_coordinate_system_alias() {
        let program = parse_script("save_coordinate_system hand").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Render(RenderCommand::SaveCoordSystem("hand".to_string()))
        );
    }

    #[test]
    fn parses_line_endpoint_coord_systems() {
        let program = parse_script("line metal 0 0 0 hand 100 100 0 world").unwrap();

        assert!(matches!(
            &program.commands[0].node,
            Command::Shape(ShapeCommand::Line {
                constants: Some(constants),
                p0,
                p1,
            }) if constants == "metal"
                && p0.coord_system.as_deref() == Some("hand")
                && p1.coord_system.as_deref() == Some("world")
        ));
    }

    #[test]
    fn parses_line_without_constants_but_with_endpoint_coord_systems() {
        let program = parse_script("line 0 0 0 cs0 1 2 3 cs1").unwrap();

        assert!(matches!(
            &program.commands[0].node,
            Command::Shape(ShapeCommand::Line {
                constants: None,
                p0,
                p1,
            }) if p0.coord_system.as_deref() == Some("cs0")
                && p1.coord_system.as_deref() == Some("cs1")
        ));
    }

    #[test]
    fn line_with_leading_constants_keeps_inner_parse_error() {
        let errors = parse_script("line metal 0 0 0 1 2").unwrap_err();

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("expected point"));
    }

    #[test]
    fn rejects_line_with_extra_trailing_token() {
        let errors = parse_script("line 0 0 0 1 2 3 extra junk").unwrap_err();

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unexpected extra"));
    }

    #[test]
    fn parses_mesh_filename_token() {
        let program = parse_script("mesh metal :teapot.obj world").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Shape(ShapeCommand::Mesh {
                constants: Some("metal".to_string()),
                filename: "teapot.obj".to_string(),
                coord_system: Some("world".to_string()),
            })
        );
    }

    #[test]
    fn parses_lighting_and_misc_commands() {
        let program = parse_script(
            "light 255 128 0 1 2 3\nambient 10 20 30\nconstants mat 1 2 3 4 5 6 7 8 9 100 110 120\nshading phong\nsave_coord_system hand\ncamera 1 2 3 4 5 6\nsave face.png\ngereate_rayfiles",
        )
        .unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Render(RenderCommand::Light {
                name: None,
                color: Vec3::new(255.0, 128.0, 0.0),
                position: Vec3::new(1.0, 2.0, 3.0),
                knob: None,
            })
        );
        assert_eq!(
            program.commands[3].node,
            Command::Render(RenderCommand::Shading(ShadingMode::Phong))
        );
        assert_eq!(
            &program.commands.last().unwrap().node,
            &Command::Output(OutputCommand::GenerateRayfiles)
        );
    }

    #[test]
    fn parses_toon_and_raytrace_shading_modes() {
        let toon = parse_script("shading toon").unwrap();
        let raytrace = parse_script("shading raytrace").unwrap();

        assert_eq!(
            toon.commands[0].node,
            Command::Render(RenderCommand::Shading(ShadingMode::Toon))
        );
        assert_eq!(
            raytrace.commands[0].node,
            Command::Render(RenderCommand::Shading(ShadingMode::Raytrace))
        );
    }

    #[test]
    fn parses_11_anim_named_light_form() {
        let program = parse_script("light key 1 2 3 4 5 6").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Render(RenderCommand::Light {
                name: Some("key".to_string()),
                color: Vec3::new(4.0, 5.0, 6.0),
                position: Vec3::new(1.0, 2.0, 3.0),
                knob: None,
            })
        );
    }

    #[test]
    fn parses_light_position_knobs() {
        let unnamed = parse_script("light 255 128 0 1 2 3 glow").unwrap();
        let named = parse_script("light key 1 2 3 glow 255 128 0").unwrap();

        assert_eq!(
            unnamed.commands[0].node,
            Command::Render(RenderCommand::Light {
                name: None,
                color: Vec3::new(255.0, 128.0, 0.0),
                position: Vec3::new(1.0, 2.0, 3.0),
                knob: Some("glow".to_string()),
            })
        );
        assert_eq!(
            named.commands[0].node,
            Command::Render(RenderCommand::Light {
                name: Some("key".to_string()),
                color: Vec3::new(255.0, 128.0, 0.0),
                position: Vec3::new(1.0, 2.0, 3.0),
                knob: Some("glow".to_string()),
            })
        );
    }

    #[test]
    fn accepts_zero_vector_light_position_because_lighting_normalization_is_safe() {
        let program = parse_script("light 255 255 255 0 0 0").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Render(RenderCommand::Light {
                name: None,
                color: Vec3::new(255.0, 255.0, 255.0),
                position: Vec3::new(0.0, 0.0, 0.0),
                knob: None,
            })
        );
    }

    #[test]
    fn parses_11_anim_validation_script() {
        let program = parse_script(include_str!("../../tests/test.mdl")).unwrap();

        assert!(program.commands.iter().any(|command| matches!(
            command.node,
            Command::Animation(AnimationCommand::Tween { .. })
        )));
        assert!(program.commands.iter().any(|command| matches!(
            command.node,
            Command::Output(OutputCommand::GenerateRayfiles)
        )));
    }

    #[test]
    fn parses_saveknobs_alias() {
        let program = parse_script("saveknobs start").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Animation(AnimationCommand::SaveKnobs("start".to_string()))
        );
    }

    #[test]
    fn parses_11_anim_texture_opcode() {
        let program = parse_script("texture tex.ppm 0 0 0 1 0 0 1 1 0 0 1 0\nweb").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Shape(ShapeCommand::Texture {
                filename: "tex.ppm".to_string(),
                points: [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                ],
            })
        );
        assert_eq!(
            program.commands[1].node,
            Command::Control(ControlCommand::Apply)
        );
    }

    #[test]
    fn rejects_invalid_rotate_axis_and_knob_syntax() {
        let errors = parse_script("rotate x|y 20\nrotate x 33 [k1]").unwrap_err();

        assert_eq!(errors.len(), 2);
        assert!(errors[0].message.contains("invalid rotate axis"));
        assert!(errors[1].message.contains("identifier"));
    }

    #[test]
    fn rejects_non_byte_rgb_channels() {
        let errors = parse_script(
            "color 999 0 0\nambient 12.7 0 0\nlight 0 0 -20 0 0 0\nconstants mat 1 1 1 1 1 1 1 1 1 256 0 0",
        )
        .unwrap_err();

        assert_eq!(errors.len(), 4);
        assert!(
            errors
                .iter()
                .all(|error| error.message.contains("0 to 255"))
        );
    }

    #[test]
    fn parses_origin_light_position() {
        let program = parse_script("light 255 255 255 0 0 0").unwrap();

        assert_eq!(
            program.commands[0].node,
            Command::Render(RenderCommand::Light {
                name: None,
                color: Vec3::new(255.0, 255.0, 255.0),
                position: Vec3::new(0.0, 0.0, 0.0),
                knob: None,
            })
        );
    }

    #[test]
    fn rejects_mesh_without_colon_filename() {
        let errors = parse_script("mesh teapot.obj").unwrap_err();

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("starting with `:`"));
    }

    #[test]
    fn lexer_keeps_save_paths_as_words() {
        let tokens = lex_line(1, "save final/face.png").unwrap();

        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn parses_old_parser_extension_commands() {
        let program = parse_script(
            "ident\napply\nclear\nreset\nreflect x\nshear z 1 2 k\ncolor white\ncircle 50 50 0 10\nhermite 0 0 10 10 1 0 0 1\nbezier 0 0 10 20 30 20 40 0\nbeziern 3 0 0 10 20 30 20 40 0\ncylinder metal 0 0 0 10 20 hand\ncone 0 0 0 10 20\npyramid 0 0 0 10 20\nmesh_reverse :tri.obj\ninclude child.mdl\nfilter grayscale\nquit",
        )
        .unwrap();

        assert!(matches!(
            program.commands[0].node,
            Command::Control(ControlCommand::Ident)
        ));
        assert!(matches!(
            program.commands[5].node,
            Command::Transform(TransformCommand::Shear {
                axis: Axis::Z,
                knob: Some(_),
                ..
            })
        ));
        assert!(matches!(
            &program.commands[6].node,
            Command::Render(RenderCommand::Color(ColorSpec::Name(name))) if name == "white"
        ));
        assert!(matches!(
            program.commands[11].node,
            Command::Shape(ShapeCommand::Cylinder {
                constants: Some(_),
                coord_system: Some(_),
                ..
            })
        ));
        assert!(matches!(
            program.commands[14].node,
            Command::Shape(ShapeCommand::MeshReverse { .. })
        ));
        assert!(matches!(
            program.commands[17].node,
            Command::Control(ControlCommand::Quit)
        ));
    }

    #[test]
    fn rejects_oversized_beziern_degree_without_overflow() {
        let source = format!("beziern {}", usize::MAX);
        let errors = parse_script(&source).unwrap_err();

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("beziern degree"));
    }

    #[test]
    fn parses_bezier_surface_extension_command() {
        let mut source = String::from("bezier_surface 4");
        for i in 0..16 {
            write!(&mut source, " {i} {i} 0").unwrap();
        }

        let program = parse_script(&source).unwrap();

        assert!(matches!(
            &program.commands[0].node,
            Command::Curve(CurveCommand::BezierSurface { steps: 4, controls })
                if controls.len() == 16
        ));
    }
}

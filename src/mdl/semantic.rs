//! Semantic analysis and animation planning for MDL.

use super::{
    animation::{AnimationPlan, KnobMap},
    ast::{
        AnimationCommand, Command, ControlCommand, Program, RenderCommand, Spanned,
        VaryInterpolation,
    },
    diagnostic::Diagnostic,
    lexer::Span,
};
use std::{
    collections::{BTreeSet, HashMap},
    path::PathBuf,
};

/// Maximum animation frames precomputed by the semantic pass.
pub const MAX_FRAMES: usize = 10_000;

/// A semantically compiled MDL program.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledProgram {
    /// Runtime commands in source order, with plan-only animation declarations removed.
    commands: Vec<Spanned<Command>>,
    /// Precomputed animation frame plan.
    animation: AnimationPlan,
}

impl CompiledProgram {
    /// Runtime commands in source order, with plan-only animation declarations removed.
    #[must_use]
    pub fn commands(&self) -> &[Spanned<Command>] {
        &self.commands
    }

    /// Precomputed animation frame plan.
    #[must_use]
    pub const fn animation(&self) -> &AnimationPlan {
        &self.animation
    }
}

#[derive(Debug, Clone)]
struct VarySpec {
    knob: String,
    start_frame: usize,
    end_frame: usize,
    start_val: f64,
    end_val: f64,
    interpolation: VaryInterpolation,
    location: SourceLocation,
}

#[derive(Debug, Clone)]
struct TweenSpec {
    start_frame: usize,
    end_frame: usize,
    list0: KnobMap,
    list1: KnobMap,
    location: SourceLocation,
}

#[derive(Debug, Clone)]
enum AnimationOp {
    Vary(VarySpec),
    Tween(TweenSpec),
    UnresolvedTween {
        start_frame: usize,
        end_frame: usize,
        location: SourceLocation,
    },
}

#[derive(Debug, Clone)]
struct SourceLocation {
    span: Span,
    source_name: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
enum SemanticSymbol {
    Knob(f64),
    KnobList(KnobMap),
    Constants,
    Light,
    CoordSystem,
}

/// Compiles a parsed program into commands plus a frame-knob table.
///
/// # Errors
/// Returns semantic diagnostics for invalid animation ranges, zero frame counts,
/// or missing tween knob lists.
#[allow(clippy::too_many_lines)]
pub fn compile(program: Program) -> Result<CompiledProgram, Vec<Diagnostic>> {
    let mut basename = "frame".to_string();
    let mut frames = 1;
    let mut symbols = HashMap::new();
    let mut animation_ops = Vec::new();
    let mut runtime_commands = Vec::new();
    let mut frames_location = None;
    let mut saw_frames = false;
    let mut animation_range_location = None;
    let mut errors = Vec::new();

    for command in program.commands {
        let location = SourceLocation {
            span: command.span,
            source_name: command.source_name.clone(),
        };
        let Spanned {
            node: command,
            span,
            source_name,
        } = command;
        if command.is_quit() {
            runtime_commands.push(Spanned {
                node: command,
                span,
                source_name,
            });
            break;
        }

        match command {
            Command::Animation(AnimationCommand::Basename(name)) => basename = name,
            Command::Animation(AnimationCommand::Frames(count)) => {
                frames = count;
                frames_location = Some(location);
                saw_frames = true;
            }
            Command::Animation(AnimationCommand::Set { knob, value }) => {
                symbols.insert(knob.clone(), SemanticSymbol::Knob(value));
                runtime_commands.push(Spanned {
                    node: Command::Animation(AnimationCommand::Set { knob, value }),
                    span,
                    source_name,
                });
            }
            Command::Animation(AnimationCommand::SetKnobs(value)) => {
                for symbol in symbols.values_mut() {
                    if let SemanticSymbol::Knob(knob) = symbol {
                        *knob = value;
                    }
                }
                runtime_commands.push(Spanned {
                    node: Command::Animation(AnimationCommand::SetKnobs(value)),
                    span,
                    source_name,
                });
            }
            Command::Animation(AnimationCommand::SaveKnobs(name)) => {
                let snapshot = current_knobs(&symbols);
                symbols.insert(name.clone(), SemanticSymbol::KnobList(snapshot));
                runtime_commands.push(Spanned {
                    node: Command::Animation(AnimationCommand::SaveKnobs(name)),
                    span,
                    source_name,
                });
            }
            Command::Render(RenderCommand::Constants {
                name,
                material,
                color,
            }) => {
                symbols.insert(name.clone(), SemanticSymbol::Constants);
                runtime_commands.push(Spanned {
                    node: Command::Render(RenderCommand::Constants {
                        name,
                        material,
                        color,
                    }),
                    span,
                    source_name,
                });
            }
            Command::Render(RenderCommand::SaveCoordSystem(name)) => {
                symbols.insert(name.clone(), SemanticSymbol::CoordSystem);
                runtime_commands.push(Spanned {
                    node: Command::Render(RenderCommand::SaveCoordSystem(name)),
                    span,
                    source_name,
                });
            }
            Command::Render(RenderCommand::Light {
                name,
                color,
                position,
                knob,
            }) => {
                if let Some(name) = &name {
                    symbols.insert(name.clone(), SemanticSymbol::Light);
                }
                runtime_commands.push(Spanned {
                    node: Command::Render(RenderCommand::Light {
                        name,
                        color,
                        position,
                        knob,
                    }),
                    span,
                    source_name,
                });
            }
            Command::Animation(AnimationCommand::Vary {
                knob,
                start_frame,
                end_frame,
                start_val,
                end_val,
                interpolation,
            }) => {
                animation_range_location.get_or_insert_with(|| location.clone());
                animation_ops.push(AnimationOp::Vary(VarySpec {
                    knob,
                    start_frame,
                    end_frame,
                    start_val,
                    end_val,
                    interpolation,
                    location,
                }));
            }
            Command::Animation(AnimationCommand::Tween {
                start_frame,
                end_frame,
                knoblist0,
                knoblist1,
            }) => {
                animation_range_location.get_or_insert_with(|| location.clone());
                let list0 = lookup_knob_list(&symbols, &knoblist0);
                let list1 = lookup_knob_list(&symbols, &knoblist1);
                if list0.is_none() {
                    errors.push(diagnostic_at(
                        Some(&location),
                        format!("unknown tween knob list `{knoblist0}`"),
                    ));
                }
                if list1.is_none() {
                    errors.push(diagnostic_at(
                        Some(&location),
                        format!("unknown tween knob list `{knoblist1}`"),
                    ));
                }
                if let (Some(list0), Some(list1)) = (list0, list1) {
                    animation_ops.push(AnimationOp::Tween(TweenSpec {
                        start_frame,
                        end_frame,
                        list0,
                        list1,
                        location,
                    }));
                } else {
                    animation_ops.push(AnimationOp::UnresolvedTween {
                        start_frame,
                        end_frame,
                        location,
                    });
                }
            }
            Command::Control(ControlCommand::Reset) => {
                symbols.clear();
                runtime_commands.push(Spanned {
                    node: Command::Control(ControlCommand::Reset),
                    span,
                    source_name,
                });
            }
            command => runtime_commands.push(Spanned {
                node: command,
                span,
                source_name,
            }),
        }
    }

    if frames == 0 {
        errors.push(diagnostic_at(
            frames_location.as_ref(),
            "`frames` must be positive",
        ));
    } else if frames > MAX_FRAMES {
        errors.push(diagnostic_at(
            frames_location.as_ref(),
            format!("`frames` must be <= {MAX_FRAMES}"),
        ));
    }
    if !saw_frames && animation_range_location.is_some() {
        errors.push(diagnostic_at(
            animation_range_location.as_ref(),
            "`vary` and `tween` require a `frames` command",
        ));
    }

    if (1..=MAX_FRAMES).contains(&frames) {
        for op in &animation_ops {
            match op {
                AnimationOp::Vary(vary) => {
                    validate_frame_range(
                        vary.start_frame,
                        vary.end_frame,
                        frames,
                        "vary",
                        &vary.location,
                        &mut errors,
                    );
                }
                AnimationOp::Tween(tween) => {
                    validate_frame_range(
                        tween.start_frame,
                        tween.end_frame,
                        frames,
                        "tween",
                        &tween.location,
                        &mut errors,
                    );
                }
                AnimationOp::UnresolvedTween {
                    start_frame,
                    end_frame,
                    location,
                } => {
                    validate_frame_range(
                        *start_frame,
                        *end_frame,
                        frames,
                        "tween",
                        location,
                        &mut errors,
                    );
                }
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut frame_knobs = vec![HashMap::new(); frames];
    for op in animation_ops {
        match op {
            AnimationOp::Vary(vary) => apply_vary(&mut frame_knobs, &vary),
            AnimationOp::Tween(tween) => apply_tween(&mut frame_knobs, &tween),
            AnimationOp::UnresolvedTween { .. } => unreachable!("unresolved tween returned errors"),
        }
    }

    Ok(CompiledProgram {
        commands: runtime_commands,
        animation: AnimationPlan::new(basename, frames, frame_knobs),
    })
}

fn current_knobs(symbols: &HashMap<String, SemanticSymbol>) -> KnobMap {
    symbols
        .iter()
        .filter_map(|(name, symbol)| match symbol {
            SemanticSymbol::Knob(value) => Some((name.clone(), *value)),
            SemanticSymbol::KnobList(_)
            | SemanticSymbol::Constants
            | SemanticSymbol::Light
            | SemanticSymbol::CoordSystem => None,
        })
        .collect()
}

fn lookup_knob_list(symbols: &HashMap<String, SemanticSymbol>, name: &str) -> Option<KnobMap> {
    match symbols.get(name) {
        Some(SemanticSymbol::KnobList(knobs)) => Some(knobs.clone()),
        Some(
            SemanticSymbol::Knob(_)
            | SemanticSymbol::Constants
            | SemanticSymbol::Light
            | SemanticSymbol::CoordSystem,
        )
        | None => None,
    }
}

fn apply_vary(frame_knobs: &mut [KnobMap], vary: &VarySpec) {
    for (offset, knobs) in frame_knobs[vary.start_frame..=vary.end_frame]
        .iter_mut()
        .enumerate()
    {
        let frame = vary.start_frame + offset;
        let t = interpolation_t(frame, vary.start_frame, vary.end_frame);
        let t = apply_vary_interpolation(t, vary.interpolation);
        knobs.insert(vary.knob.clone(), lerp(vary.start_val, vary.end_val, t));
    }
}

fn apply_vary_interpolation(t: f64, interpolation: VaryInterpolation) -> f64 {
    match interpolation {
        VaryInterpolation::Linear => t,
        VaryInterpolation::Exponential => {
            let curve = 4.0_f64;
            ((curve * t).exp() - 1.0) / (curve.exp() - 1.0)
        }
        VaryInterpolation::Logarithmic => (1.0 + 9.0 * t).ln() / 10.0_f64.ln(),
        VaryInterpolation::Smoothstep => t * t * (3.0 - 2.0 * t),
        VaryInterpolation::Power(exponent) => t.powf(exponent),
    }
}

fn apply_tween(frame_knobs: &mut [KnobMap], tween: &TweenSpec) {
    let list0 = &tween.list0;
    let list1 = &tween.list1;
    let knobs = knob_union(list0, list1);
    for (offset, frame_knobs) in frame_knobs[tween.start_frame..=tween.end_frame]
        .iter_mut()
        .enumerate()
    {
        let frame = tween.start_frame + offset;
        let t = interpolation_t(frame, tween.start_frame, tween.end_frame);
        for knob in &knobs {
            let start = list0.get(knob).copied().unwrap_or(0.0);
            let end = list1.get(knob).copied().unwrap_or(0.0);
            frame_knobs.insert(knob.clone(), lerp(start, end, t));
        }
    }
}

fn validate_frame_range(
    start: usize,
    end: usize,
    frames: usize,
    command: &str,
    location: &SourceLocation,
    errors: &mut Vec<Diagnostic>,
) {
    if start >= end {
        errors.push(diagnostic_at(
            Some(location),
            format!("`{command}` start frame must be < end frame"),
        ));
    }
    if end >= frames {
        errors.push(diagnostic_at(
            Some(location),
            format!("`{command}` end frame {end} is outside {frames} frames"),
        ));
    }
}

fn diagnostic_at(location: Option<&SourceLocation>, message: impl Into<String>) -> Diagnostic {
    if let Some(location) = location {
        let mut diagnostic = Diagnostic::new(
            location.span.line,
            location.span.col_start,
            location.span.col_end,
            message,
        );
        if let Some(source_name) = &location.source_name {
            diagnostic = diagnostic.with_source(source_name);
        }
        diagnostic
    } else {
        Diagnostic::line(1, message)
    }
}

fn interpolation_t(frame: usize, start: usize, end: usize) -> f64 {
    let elapsed = u32::try_from(frame - start).expect("frame range is capped by MAX_FRAMES");
    let span = u32::try_from(end - start).expect("frame range is capped by MAX_FRAMES");
    f64::from(elapsed) / f64::from(span)
}

fn lerp(start: f64, end: f64, t: f64) -> f64 {
    start + (end - start) * t
}

fn knob_union(list0: &KnobMap, list1: &KnobMap) -> Vec<String> {
    let mut knobs = BTreeSet::new();
    knobs.extend(list0.keys().cloned());
    knobs.extend(list1.keys().cloned());
    knobs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::{MAX_FRAMES, compile};
    use crate::mdl::{
        ast::{Command, TransformCommand},
        parser::parse_script,
    };

    fn assert_approx_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn compile_defaults_to_single_frame() {
        let program = parse_script("move 1 2 3").unwrap();
        let compiled = compile(program).unwrap();

        assert_eq!(compiled.animation().basename(), "frame");
        assert_eq!(compiled.animation().frames(), 1);
        assert!(compiled.animation().frame_knobs()[0].is_empty());
    }

    #[test]
    fn compile_builds_vary_frame_knobs_inclusively() {
        let program = parse_script("frames 10\nbasename spin\nvary knob 0 9 0 90").unwrap();
        let compiled = compile(program).unwrap();

        assert_eq!(compiled.animation().basename(), "spin");
        assert_eq!(compiled.animation().frames(), 10);
        assert_approx_eq(compiled.animation().frame_knobs()[0]["knob"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[4]["knob"], 40.0);
        assert_approx_eq(compiled.animation().frame_knobs()[9]["knob"], 90.0);
    }

    #[test]
    fn compile_applies_vary_interpolation_modes() {
        let program = parse_script(
            "frames 11\nvary exp 0 10 0 1 exponential\nvary log 0 10 0 1 logarithmic\nvary pow 0 10 0 1 power 3",
        )
        .unwrap();
        let compiled = compile(program).unwrap();

        let midpoint = &compiled.animation().frame_knobs()[5];
        assert!(midpoint["exp"] < 0.5);
        assert!(midpoint["log"] > 0.5);
        assert_approx_eq(midpoint["pow"], 0.125);
        assert_approx_eq(compiled.animation().frame_knobs()[0]["exp"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[10]["log"], 1.0);
    }

    #[test]
    fn vary_only_overrides_frames_inside_its_range() {
        let program = parse_script("frames 4\nset k 5\nvary k 1 2 0 10").unwrap();
        let compiled = compile(program).unwrap();

        assert!(!compiled.animation().frame_knobs()[0].contains_key("k"));
        assert_approx_eq(compiled.animation().frame_knobs()[1]["k"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[2]["k"], 10.0);
        assert!(!compiled.animation().frame_knobs()[3].contains_key("k"));
    }

    #[test]
    fn compile_tweens_between_saved_knob_lists() {
        let program = parse_script(
            "frames 3\nset spin 0\nsave_knobs start\nset spin 90\nset grow 2\nsave_knobs end\ntween 0 2 start end",
        )
        .unwrap();
        let compiled = compile(program).unwrap();

        assert_approx_eq(compiled.animation().frame_knobs()[0]["spin"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[1]["spin"], 45.0);
        assert_approx_eq(compiled.animation().frame_knobs()[2]["spin"], 90.0);
        assert_approx_eq(compiled.animation().frame_knobs()[1]["grow"], 1.0);
    }

    #[test]
    fn compile_tween_accepts_saveknobs_alias() {
        let program =
            parse_script("frames 2\nset spin 0\nsaveknobs start\nset spin 10\nsaveknobs end\ntween 0 1 start end")
                .unwrap();
        let compiled = compile(program).unwrap();

        assert_approx_eq(compiled.animation().frame_knobs()[0]["spin"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[1]["spin"], 10.0);
    }

    #[test]
    fn save_knobs_snapshots_reflect_runtime_symbol_overwrites() {
        let program = parse_script(
            "frames 2\nset k 1\nconstants k 1 1 1 1 1 1 1 1 1\nsave_knobs start\nset k 2\nsave_knobs end\ntween 0 1 start end",
        )
        .unwrap();
        let compiled = compile(program).unwrap();

        assert_approx_eq(compiled.animation().frame_knobs()[0]["k"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[1]["k"], 2.0);
    }

    #[test]
    fn save_knobs_name_overwrites_same_named_knob() {
        let program = parse_script(
            "frames 2\nset list 5\nsave_knobs list\nsave_knobs end\ntween 0 1 list end",
        )
        .unwrap();
        let compiled = compile(program).unwrap();

        assert_approx_eq(compiled.animation().frame_knobs()[0]["list"], 5.0);
        assert_approx_eq(compiled.animation().frame_knobs()[1]["list"], 0.0);
    }

    #[test]
    fn save_coord_system_names_remove_same_named_base_knobs() {
        let program = parse_script(
            "frames 2\nset hand 7\nsave_coord_system hand\nsave_knobs start\nset hand 9\nsave_knobs end\ntween 0 1 start end",
        )
        .unwrap();
        let compiled = compile(program).unwrap();

        assert_approx_eq(compiled.animation().frame_knobs()[0]["hand"], 0.0);
        assert_approx_eq(compiled.animation().frame_knobs()[1]["hand"], 9.0);
    }

    #[test]
    fn constants_names_remove_same_named_knob_lists() {
        let program = parse_script(
            "frames 2\nset k 1\nsave_knobs a\nconstants a 1 1 1 1 1 1 1 1 1\nset k 2\nsave_knobs b\ntween 0 1 a b",
        )
        .unwrap();
        let errors = compile(program).unwrap_err();

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unknown tween knob list `a`"));
    }

    #[test]
    fn reset_clears_semantic_knobs_and_knob_lists() {
        let program = parse_script(
            "frames 2\nset k 1\nsave_knobs old\nreset\nsave_knobs empty\nset k 2\nsave_knobs end\ntween 0 1 empty end\ntween 0 1 old end",
        )
        .unwrap();
        let errors = compile(program).unwrap_err();

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unknown tween knob list `old`"));
    }

    #[test]
    fn compile_applies_animation_ops_in_source_order() {
        let vary_then_tween = parse_script(
            "frames 3\nset k 20\nsave_knobs a\nset k 40\nsave_knobs b\nvary k 0 2 0 10\ntween 0 2 a b",
        )
        .unwrap();
        let tween_then_vary = parse_script(
            "frames 3\nset k 20\nsave_knobs a\nset k 40\nsave_knobs b\ntween 0 2 a b\nvary k 0 2 0 10",
        )
        .unwrap();

        let vary_then_tween = compile(vary_then_tween).unwrap();
        let tween_then_vary = compile(tween_then_vary).unwrap();

        assert_approx_eq(vary_then_tween.animation().frame_knobs()[1]["k"], 30.0);
        assert_approx_eq(tween_then_vary.animation().frame_knobs()[1]["k"], 5.0);
    }

    #[test]
    fn tween_uses_knob_list_snapshot_at_source_location() {
        let program = parse_script(
            "frames 2\nset k 0\nsave_knobs a\nset k 10\nsave_knobs b\ntween 0 1 a b\nset k 100\nsave_knobs b",
        )
        .unwrap();
        let compiled = compile(program).unwrap();

        assert_approx_eq(compiled.animation().frame_knobs()[1]["k"], 10.0);
    }

    #[test]
    fn compile_removes_plan_only_commands_from_runtime_stream() {
        let program =
            parse_script("frames 2\nbasename spin\nvary k 0 1 0 1\nmove 1 0 0 k").unwrap();
        let compiled = compile(program).unwrap();

        assert_eq!(compiled.commands().len(), 1);
        assert!(matches!(
            compiled.commands()[0].node,
            Command::Transform(TransformCommand::Move { .. })
        ));
    }

    #[test]
    fn compile_stops_semantic_scan_at_quit() {
        let program = parse_script("quit\nframes 100\nvary k 0 99 0 1").unwrap();
        let compiled = compile(program).unwrap();

        assert_eq!(compiled.animation().frames(), 1);
        assert!(compiled.animation().frame_knobs()[0].is_empty());
    }

    #[test]
    fn compile_rejects_invalid_animation_ranges() {
        let program = parse_script("frames 3\nvary spin 0 3 0 90\ntween 2 1 a b").unwrap();
        let errors = compile(program).unwrap_err();

        assert!(errors.iter().any(|error| error.message.contains("outside")));
        assert!(errors.iter().any(|error| error.message.contains('<')));
        assert!(errors.iter().any(|error| error.line == 2));
        assert!(errors.iter().any(|error| error.line == 3));
    }

    #[test]
    fn compile_reports_frame_errors_at_frames_command_span() {
        let program = parse_script("move 1 0 0\nframes 0").unwrap();
        let errors = compile(program).unwrap_err();

        assert_eq!(errors[0].line, 2);
    }

    #[test]
    fn compile_rejects_huge_frame_counts() {
        let program = parse_script(&format!("frames {}", MAX_FRAMES + 1)).unwrap();
        let errors = compile(program).unwrap_err();

        assert!(errors[0].message.contains(&MAX_FRAMES.to_string()));
    }

    #[test]
    fn compile_rejects_missing_tween_lists() {
        let program = parse_script("frames 2\ntween 0 1 missing_a missing_b").unwrap();
        let errors = compile(program).unwrap_err();

        assert_eq!(errors.len(), 2);
        assert!(errors[0].message.contains("missing_a"));
        assert!(errors[1].message.contains("missing_b"));
    }

    #[test]
    fn compile_rejects_vary_without_frames() {
        let program = parse_script("set k 1\nvary k 0 0 0 1").unwrap();
        let errors = compile(program).unwrap_err();

        assert_eq!(errors[0].line, 2);
        assert!(errors[0].message.contains("require a `frames` command"));
    }

    #[test]
    fn compile_rejects_tween_without_frames() {
        let program =
            parse_script("set k 0\nsave_knobs start\nset k 1\nsave_knobs end\ntween 0 0 start end")
                .unwrap();
        let errors = compile(program).unwrap_err();

        assert_eq!(errors[0].line, 5);
        assert!(errors[0].message.contains("require a `frames` command"));
    }
}

//! Typed MDL command representation.

use super::lexer::Span;
use crate::graphics::{
    colors::LinearRgb,
    lighting::{DEFAULT_SPECULAR_EXPONENT, SurfaceMaterial},
};
use std::path::PathBuf;

/// A parsed MDL program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Commands in source order.
    pub commands: Vec<Spanned<Command>>,
}

/// A parsed node with source location metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    /// Parsed node.
    pub node: T,
    /// Source span for the node's command token.
    pub span: Span,
    /// Optional source filename.
    pub source_name: Option<PathBuf>,
}

impl<T> Spanned<T> {
    /// Creates a spanned node without a source filename.
    #[must_use]
    pub const fn new(node: T, span: Span) -> Self {
        Self {
            node,
            span,
            source_name: None,
        }
    }

    /// Attaches a source filename.
    #[must_use]
    pub fn with_source(mut self, source_name: impl Into<PathBuf>) -> Self {
        self.source_name = Some(source_name.into());
        self
    }

    /// Maps the node while preserving source metadata.
    #[must_use]
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            node: map(self.node),
            span: self.span,
            source_name: self.source_name,
        }
    }
}

/// One typed MDL command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Stack/control command.
    Control(ControlCommand),
    /// Current-transform command.
    Transform(TransformCommand),
    /// Edge/curve command.
    Curve(CurveCommand),
    /// Geometry command.
    Shape(ShapeCommand),
    /// Animation and knob command.
    Animation(AnimationCommand),
    /// Rendering state command.
    Render(RenderCommand),
    /// Camera command.
    Camera(CameraCommand),
    /// File/output command.
    Output(OutputCommand),
    /// Include another source file.
    Include(String),
    /// Apply a canvas filter.
    Filter(FilterCommand),
}

impl Command {
    /// Returns true when this command stops execution.
    #[must_use]
    pub const fn is_quit(&self) -> bool {
        matches!(self, Self::Control(ControlCommand::Quit))
    }
}

/// Stack and control-flow commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlCommand {
    /// No-op compatibility command.
    Apply,
    /// Stop executing remaining commands.
    Quit,
    /// Copy the current coordinate-system stack top.
    Push,
    /// Pop the current coordinate-system stack top.
    Pop,
    /// Reset the current coordinate-system stack top to identity.
    Ident,
    /// Clear the current canvas.
    Clear,
    /// Clear canvas and reset runtime state.
    Reset,
}

/// Transform commands.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum TransformCommand {
    /// Translate the current coordinate system.
    Move {
        x: f64,
        y: f64,
        z: f64,
        knob: Option<String>,
    },
    /// Scale the current coordinate system.
    Scale {
        x: f64,
        y: f64,
        z: f64,
        knob: Option<String>,
    },
    /// Rotate the current coordinate system.
    Rotate {
        axis: Axis,
        degrees: f64,
        knob: Option<String>,
    },
    /// Reflect the current coordinate system across an axis plane.
    Reflect { axis: Axis },
    /// Shear the current coordinate system.
    Shear {
        axis: Axis,
        sh0: f64,
        sh1: f64,
        knob: Option<String>,
    },
}

/// Curve and edge commands.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum CurveCommand {
    /// Draw a circle.
    Circle { center: Vec3, radius: f64 },
    /// Draw a Hermite curve.
    Hermite {
        p0: Vec2,
        p1: Vec2,
        r0: Vec2,
        r1: Vec2,
    },
    /// Draw a cubic Bezier curve.
    Bezier {
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
    },
    /// Draw an arbitrary-degree Bezier curve.
    BezierN { degree: usize, points: Vec<Vec2> },
    /// Draw a Bezier surface.
    BezierSurface { steps: usize, controls: Vec<Vec3> },
}

/// Shape and mesh commands.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum ShapeCommand {
    /// Draw a sphere.
    Sphere {
        constants: Option<String>,
        center: Vec3,
        radius: f64,
        coord_system: Option<String>,
    },
    /// Draw a torus.
    Torus {
        constants: Option<String>,
        center: Vec3,
        r0: f64,
        r1: f64,
        coord_system: Option<String>,
    },
    /// Draw a box.
    Box {
        constants: Option<String>,
        corner: Vec3,
        h: f64,
        w: f64,
        d: f64,
        coord_system: Option<String>,
    },
    /// Draw a line segment.
    Line {
        constants: Option<String>,
        p0: PointRef,
        p1: PointRef,
    },
    /// Load and draw a mesh.
    Mesh {
        constants: Option<String>,
        filename: String,
        coord_system: Option<String>,
    },
    /// Load and draw a mesh with reversed triangle winding.
    MeshReverse {
        constants: Option<String>,
        filename: String,
        coord_system: Option<String>,
    },
    /// Parsed `11_anim` texture command. The reference parser accepts this, but
    /// the reference interpreter does not render it.
    Texture { filename: String, points: [Vec3; 4] },
    /// Draw a cylinder.
    Cylinder {
        constants: Option<String>,
        center: Vec3,
        radius: f64,
        height: f64,
        coord_system: Option<String>,
    },
    /// Draw a cone.
    Cone {
        constants: Option<String>,
        center: Vec3,
        radius: f64,
        height: f64,
        coord_system: Option<String>,
    },
    /// Draw a pyramid.
    Pyramid {
        constants: Option<String>,
        center: Vec3,
        base_length: f64,
        height: f64,
        coord_system: Option<String>,
    },
}

/// Animation and knob commands.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum AnimationCommand {
    /// Set the animation basename.
    Basename(String),
    /// Set the animation frame count.
    Frames(usize),
    /// Set a knob value.
    Set { knob: String, value: f64 },
    /// Save current knob values under a list name.
    SaveKnobs(String),
    /// Interpolate between two knob lists.
    Tween {
        start_frame: usize,
        end_frame: usize,
        knoblist0: String,
        knoblist1: String,
    },
    /// Vary one knob over a frame range.
    Vary {
        knob: String,
        start_frame: usize,
        end_frame: usize,
        start_val: f64,
        end_val: f64,
        interpolation: VaryInterpolation,
    },
    /// Set all known knobs to one value.
    SetKnobs(f64),
}

/// Curve used by `vary` to convert frame progress into knob progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VaryInterpolation {
    /// Use unmodified linear progress.
    Linear,
    /// Start slowly and accelerate.
    Exponential,
    /// Start quickly and decelerate.
    Logarithmic,
    /// Smooth cubic easing with zero slope at both ends.
    Smoothstep,
    /// Raise linear progress to a custom exponent.
    Power(f64),
}

/// Rendering state commands.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum RenderCommand {
    /// Change the current drawing color.
    Color(ColorSpec),
    /// Define a point light.
    Light {
        name: Option<String>,
        color: Vec3,
        position: Vec3,
        knob: Option<String>,
    },
    /// Define ambient light.
    Ambient { color: Vec3 },
    /// Define reusable material constants.
    Constants {
        name: String,
        material: Material,
        color: Vec3,
    },
    /// Set the shading mode.
    Shading(ShadingMode),
    /// Save a copy of the current coordinate-system stack top.
    SaveCoordSystem(String),
}

/// Camera commands.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum CameraCommand {
    /// Configure the camera.
    Camera { eye: Vec3, aim: Vec3 },
    /// Set camera focal length.
    Focal(f64),
}

/// Output commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputCommand {
    /// Save the current image to a file.
    Save(String),
    /// Display the current image.
    Display,
    /// Request ray-tracer source generation.
    GenerateRayfiles,
}

/// Canvas filter command.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterCommand {
    /// Filter name.
    pub name: String,
    /// Optional numeric filter parameter.
    pub value: Option<f64>,
}

/// Rotation axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// X axis.
    X,
    /// Y axis.
    Y,
    /// Z axis.
    Z,
}

/// MDL shading mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadingMode {
    /// Draw polygon edges only.
    Wireframe,
    /// Flat polygon shading.
    Flat,
    /// Gouraud shading.
    Gouraud,
    /// Phong shading.
    Phong,
    /// Ray-traced rendering.
    Raytrace,
}

/// Three numeric coordinates or color channels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    /// X coordinate or red channel.
    pub x: f64,
    /// Y coordinate or green channel.
    pub y: f64,
    /// Z coordinate or blue channel.
    pub z: f64,
}

/// Two numeric coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Vec2 {
    /// Creates a new 2-value tuple.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl Vec3 {
    /// Creates a new 3-value tuple.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// A color command payload.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorSpec {
    /// Named color constant.
    Name(String),
    /// RGB color channels.
    Rgb(Vec3),
}

/// A point with an optional coordinate-system reference.
#[derive(Debug, Clone, PartialEq)]
pub struct PointRef {
    /// Point coordinates.
    pub point: Vec3,
    /// Optional coordinate-system name.
    pub coord_system: Option<String>,
}

/// Per-channel material reflection coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Red ambient coefficient.
    pub kar: f64,
    /// Red diffuse coefficient.
    pub kdr: f64,
    /// Red specular coefficient.
    pub ksr: f64,
    /// Green ambient coefficient.
    pub kag: f64,
    /// Green diffuse coefficient.
    pub kdg: f64,
    /// Green specular coefficient.
    pub ksg: f64,
    /// Blue ambient coefficient.
    pub kab: f64,
    /// Blue diffuse coefficient.
    pub kdb: f64,
    /// Blue specular coefficient.
    pub ksb: f64,
}

impl Material {
    /// Creates a material from the nine MDL reflection coefficients.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        kar: f64,
        kdr: f64,
        ksr: f64,
        kag: f64,
        kdg: f64,
        ksg: f64,
        kab: f64,
        kdb: f64,
        ksb: f64,
    ) -> Self {
        Self {
            kar,
            kdr,
            ksr,
            kag,
            kdg,
            ksg,
            kab,
            kdb,
            ksb,
        }
    }
}

impl From<Material> for SurfaceMaterial {
    fn from(material: Material) -> Self {
        Self::new(
            LinearRgb::new(material.kar, material.kag, material.kab),
            LinearRgb::new(material.kdr, material.kdg, material.kdb),
            LinearRgb::new(material.ksr, material.ksg, material.ksb),
            f64::from(DEFAULT_SPECULAR_EXPONENT),
        )
    }
}

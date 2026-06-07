#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]
#![warn(clippy::pedantic)]
//! A Rust graphics playground for raster drawing, mesh scenes, MDL scripts, and path tracing.
//!
//! `gartus` started as a classroom-style computer graphics engine: a [`Canvas`], matrix and vector
//! math, polygon drawing, lighting, and a Motion Description Language (MDL) front end. It now also
//! includes a small physically based path tracer following the *Ray Tracing* book series.
//!
//! [`Canvas`]: crate::graphics::display::Canvas
//!
//! # Main Layers
//!
//! - [`gmath`] contains the shared math layer: vectors, points, matrices, rays, analytic geometry,
//!   random sampling, Perlin noise, and directional PDFs.
//! - [`graphics`] contains renderer-facing APIs: [`Canvas`], colors, raster drawing, cameras,
//!   lighting, renderer-neutral surface materials/textures/scenes, and the path tracer.
//! - [`mdl`] contains the MDL compiler front end and runtime execution path for script-driven
//!   renders.
//!
//! # Raster Rendering
//!
//! Use [`Canvas`] for direct pixel and line work, or [`SurfaceScene`] for mesh/material scene data
//! that should remain independent of the renderer. A [`SurfaceScene`] can be rasterized directly:
//!
//! ```rust
//! use gartus::prelude::*;
//!
//! let camera = Camera3D::new(400, 400);
//! let scene = SurfaceScene::new();
//! let canvas = scene.rasterize(&camera);
//! assert_eq!(canvas.width(), 400);
//! ```
//!
//! [`SurfaceScene`]: crate::graphics::scene::SurfaceScene
//!
//! # Path Tracing
//!
//! Start with [`PathTracer`] and [`RayCamera`]. For one-shot renderer-neutral mesh renders, pass a
//! [`SurfaceScene`] to [`PathTracer::render_scene`]. For repeated renders of the same surface
//! content, compile once with [`SurfaceScene::to_ray_scene`] and render the resulting
//! [`RayScene`] directly so its cached BVH can accelerate traversal. For ray-specific materials,
//! emissive geometry, textured ray materials, or procedural scenes, build a `RayScene` directly.
//!
//! Indoor scenes with small emitters usually converge faster when you pass a dedicated
//! [`SamplingTargetList`] to [`PathTracer::render_with_lights`] instead of using the whole world as
//! the light target.
//!
//! ```rust
//! use gartus::prelude::*;
//!
//! let camera = RayCamera::new(200, 1.0)
//!     .with_samples_per_pixel(8)
//!     .with_max_depth(8);
//! let surface_scene = SurfaceScene::new();
//! let ray_scene = surface_scene.to_ray_scene();
//! let image = PathTracer::new(camera).render(&ray_scene);
//! assert_eq!(image.width(), 200);
//! ```
//!
//! [`PathTracer`]: crate::graphics::raytracing::PathTracer
//! [`RayCamera`]: crate::graphics::camera::RayCamera
//! [`RayScene`]: crate::graphics::raytracing::RayScene
//! [`SamplingTargetList`]: crate::graphics::raytracing::SamplingTargetList
//! [`SurfaceScene::to_ray_scene`]: crate::graphics::scene::SurfaceScene::to_ray_scene
//! [`PathTracer::render_scene`]: crate::graphics::raytracing::PathTracer::render_scene
//! [`PathTracer::render_with_lights`]: crate::graphics::raytracing::PathTracer::render_with_lights
//!
//! # Scripts
//!
//! New script code should use [`mdl`]. The legacy two-line parser is available only behind the
//! `old_parser` feature.
//!
//! # Preludes
//!
//! Use the root prelude for examples and small programs:
//!
//! ```rust
//! use gartus::prelude::*;
//! ```
//!
//! The root prelude intentionally excludes low-level PDF internals. Import sampling code from
//! [`gmath::sampling`] or [`graphics::raytracing::pdf`] when implementing new probability
//! distributions.

#[cfg(feature = "external")]
/// External asset loaders, including PPM and mesh import helpers.
pub mod external;
/// This module hosts all the math needed for computer graphics
pub mod gmath;
/// Renderer-facing structures for drawing, rasterization, animation, and path tracing.
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
/// advanced features that are not fully integrated into parser
/// or graphics modules.
pub mod utils;

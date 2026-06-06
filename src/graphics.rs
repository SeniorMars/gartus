//! The graphics module hosts all the needed structs to play
//! around with computer graphics.

/// Explicit frame recording and GIF encoding.
pub mod animation;
/// Perspective projection helpers for simple 3D scenes.
pub mod camera;
/// Includes RGB and HSL color types.
pub mod colors;
/// Includes the [`display::Canvas`] struct, which represents your drawing board.
pub mod display;
/// Hosts all the functions needed to start drawing onto a canvas.
pub mod draw;
/// Some preset filters that can be applied to a Canvas
#[cfg(feature = "filters")]
pub mod filters;
/// Flat Phong reflection lighting helpers.
pub mod lighting;
/// Renderer-neutral material data.
pub mod material;
/// Small 2D primitives and bitmap text helpers.
pub mod primitives;
/// Minimal path/ray tracing helpers.
pub mod raytracing;
/// Renderer-neutral scene data.
pub mod scene;
#[cfg(test)]
mod tests;
/// 2D texture sampling helpers.
pub mod texture;
/// Textured triangle and quad rasterization.
pub mod textured_raster;
/// An agent that can move throughout a canvas.
#[cfg(feature = "turtle")]
pub mod turtle;

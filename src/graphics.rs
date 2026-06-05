//! The graphics module hosts all the needed struts to playing
//! around with computer graphics.

/// Explicit frame recording and GIF encoding.
pub mod animation;
/// Perspective projection helpers for simple 3D scenes.
pub mod camera;
/// Includes the [Pixel] and [HSL] struts, which are the basic foundation to color
pub mod colors;
/// Includes the [Canvas] strut, which represents your "drawing board".
pub mod display;
/// Hosts all the functions needed to start drawing onto the [Canvas]
pub mod draw;
/// Some preset filters that can be applied to a Canvas
#[cfg(feature = "filters")]
pub mod filters;
/// Flat Phong reflection lighting helpers.
pub mod lighting;
/// Small 2D primitives and bitmap text helpers.
pub mod primitives;
#[cfg(test)]
mod tests;
/// 2D texture sampling helpers.
pub mod texture;
/// Textured triangle and quad rasterization.
pub mod textured_raster;
/// An agent that can move throughout the [Canvas]
#[cfg(feature = "turtle")]
pub mod turtle;

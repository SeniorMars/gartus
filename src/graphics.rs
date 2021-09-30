//! The graphics module hosts all the needed struts to playing
//! around with computer graphics.

/// Includes the [Pixel] and [HSL] struts, which are the basic foundation to color
pub mod colors;
/// Includes the [Canvas] strut, which represents your "drawing board".
pub mod display;
/// Hosts all the functions needed to start drawing onto the [Canvas]
pub mod draw;
/// Includes the [Matrix] struct with a surrounding mini matrix library
/// to make it easier for a user to draw onto the Canvas.
pub mod matrix;
/// Hosts the ray struct for ray tracing
pub mod ray;
/// Hosts all the functions needed to start with 3D transformations with matrices.
pub mod transformations;
/// An agent that can move throughout the [Canvas]
pub mod turtle;
/// Hosts the vector struct for ray tracing
pub mod vector;

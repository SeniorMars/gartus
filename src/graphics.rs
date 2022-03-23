//! The graphics module hosts all the needed struts to playing
//! around with computer graphics.

/// Includes the [Canvas] and [Pixel] struts, which come together to serve as 
/// your "drawing board".
pub mod display;
/// Hosts all the functions needed to start drawing onto the [Canvas]
pub mod draw;
/// Includes the [Matrix] struct with a surrounding mini matrix library
/// to make it easier for a user to draw onto the Canvas.
pub mod matrix;
/// Hosts all the functions needed to start with 3D transformations with matrices.
pub mod transformations;

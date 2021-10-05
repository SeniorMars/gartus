//! The graphics module hosts all the math needed for computer graphics

/// Includes the [Matrix] struct with a surrounding mini matrix library
/// to make it easier for a user to draw onto the Canvas.
pub mod matrix;
/// Hosts the ray struct for ray tracing
pub mod ray;
/// Hosts all the functions needed to start with 3D transformations with matrices.
pub mod transformations;
/// Hosts the vector struct for ray tracing
pub mod vector;

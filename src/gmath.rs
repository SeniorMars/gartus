//! The gmath graphics module hosts all the math needed for computer graphics
// PROPS To Ruoshui for various inspirations

/// Hosts various helpers to make math easier.
pub mod helpers;
/// Includes the [Matrix] struct with a surrounding mini matrix library
/// to make it easier for a user to draw onto the Canvas.
pub mod matrix;
/// Hosts the [Parametric] struct
pub mod parametric;
/// Hosts the [Ray] struct for path/ray tracing
pub mod ray;
/// Contains algorithms for making shapes
pub mod shapes;
/// Hosts all the functions needed to start applying 3D transformations to matrices.
pub mod transformations;
/// Hosts the [Vector] struct for ray tracing
pub mod vector;

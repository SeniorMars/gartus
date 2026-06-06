//! The gmath graphics module hosts all the math needed for computer graphics
// PROPS To Ruoshui for various inspirations

/// Hosts the [`EdgeMatrix`] type — a dynamically-growing 4×N point list for edge drawing.
pub mod edge_matrix;
/// Shared analytic geometry descriptors.
pub mod geometry;
/// Hosts various helpers to make math easier.
pub mod helpers;
/// Includes the [Matrix] struct with a surrounding mini matrix library
/// to make it easier for a user to draw onto the Canvas.
pub mod matrix;
/// Hosts the [Parametric] struct
pub mod parametric;
/// Deterministic Perlin noise for procedural textures.
pub mod perlin;
/// Hosts the [`PolygonMatrix`] type — a dynamically-growing 4×N point list for polygon drawing.
pub mod polygon_matrix;
/// Hosts the [Quaternion] struct for ray tracing and 3D transformations
pub mod quaternion;
/// Deterministic random sampling helpers.
pub mod random;
/// Hosts the [Ray] struct for path/ray tracing
pub mod ray;
/// Probability density functions and sampling distributions.
pub mod sampling;
/// Hosts a coordinate-system stack for hierarchical 3D transforms.
pub mod stack;
/// Hosts all the functions needed to start applying 3D transformations to matrices.
pub mod transformations;
/// Hosts the [Vector] struct for ray tracing
pub mod vector;

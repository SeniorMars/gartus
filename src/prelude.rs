//! Contains common types that can be glob-imported (`*`) for convience
pub use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::{
        animation::FrameRecorder,
        colors::{ColorSpace, Rgb},
        display::Canvas,
    },
};

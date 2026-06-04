//! Contains common types that can be glob-imported (`*`) for convience
pub use crate::{
    gmath::{
        edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix, stack::MatrixStack,
    },
    graphics::{
        animation::{AnimationRenderOptions, FrameRecorder},
        camera::{Camera3D, ProjectedSegment, ScreenPoint, sort_segments_back_to_front},
        colors::{ColorSpace, Rgb},
        display::Canvas,
    },
};

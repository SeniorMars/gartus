use super::colors::Rgb;
use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::display::Canvas,
};

/// A simple perspective camera for projecting 3D points onto a 2D canvas.
#[derive(Debug, Clone, Copy)]
pub struct Camera3D {
    width: u32,
    height: u32,
    camera_distance: f64,
    focal_length: f64,
    center_y_factor: f64,
    near_depth: f64,
}

/// A projected 2D point plus its camera-space depth.
#[derive(Debug, Clone, Copy)]
pub struct ScreenPoint {
    /// Horizontal screen coordinate.
    pub x: f64,
    /// Vertical screen coordinate.
    pub y: f64,
    /// Camera-space depth used for sorting and shading.
    pub depth: f64,
}

/// A projected colored line segment.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedSegment {
    /// First projected endpoint.
    pub a: ScreenPoint,
    /// Second projected endpoint.
    pub b: ScreenPoint,
    /// Segment draw color.
    pub color: Rgb,
}

impl Camera3D {
    /// Creates a camera centered in a canvas.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            camera_distance: 900.0,
            focal_length: 700.0,
            center_y_factor: 0.5,
            near_depth: 80.0,
        }
    }

    /// Sets the distance added to incoming z values before projection.
    #[must_use]
    pub fn with_camera_distance(mut self, camera_distance: f64) -> Self {
        self.camera_distance = camera_distance;
        self
    }

    /// Sets the focal length used for perspective scaling.
    #[must_use]
    pub fn with_focal_length(mut self, focal_length: f64) -> Self {
        self.focal_length = focal_length;
        self
    }

    /// Sets the vertical screen center as a fraction of canvas height.
    #[must_use]
    pub fn with_center_y_factor(mut self, center_y_factor: f64) -> Self {
        self.center_y_factor = center_y_factor;
        self
    }

    /// Sets the minimum projected depth.
    #[must_use]
    pub fn with_near_depth(mut self, near_depth: f64) -> Self {
        self.near_depth = near_depth;
        self
    }

    /// Projects a homogeneous point into 2D screen coordinates.
    #[must_use]
    pub fn project(&self, point: &[f64]) -> Option<ScreenPoint> {
        if point.len() < 3 {
            return None;
        }
        let depth = point[2] + self.camera_distance;
        if depth < self.near_depth {
            return None;
        }
        let scale = self.focal_length / depth;
        Some(ScreenPoint {
            x: f64::from(self.width) * 0.5 + point[0] * scale,
            y: f64::from(self.height) * self.center_y_factor - point[1] * scale,
            depth,
        })
    }
}

impl ProjectedSegment {
    /// Creates a projected segment if both source points project in front of the camera.
    #[must_use]
    pub fn from_points(camera: &Camera3D, p0: &[f64], p1: &[f64], color: Rgb) -> Option<Self> {
        Some(Self {
            a: camera.project(p0)?,
            b: camera.project(p1)?,
            color,
        })
    }

    /// Returns the average projected depth of the segment.
    #[must_use]
    pub fn average_depth(&self) -> f64 {
        (self.a.depth + self.b.depth) * 0.5
    }
}

/// Sorts projected segments back-to-front for painter-style wireframe rendering.
pub fn sort_segments_back_to_front(segments: &mut [ProjectedSegment]) {
    segments.sort_by(|a, b| {
        b.average_depth()
            .partial_cmp(&a.average_depth())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

impl Canvas {
    /// Draws already-projected colored segments.
    pub fn draw_projected_segments<I>(&mut self, segments: I)
    where
        I: IntoIterator<Item = ProjectedSegment>,
    {
        for segment in segments {
            self.draw_line(
                segment.color,
                segment.a.x,
                segment.a.y,
                segment.b.x,
                segment.b.y,
            );
        }
    }

    /// Projects and draws transformed edge lines without allocating a transformed edge matrix.
    pub fn draw_projected_edges(
        &mut self,
        edges: &EdgeMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        color: Rgb,
    ) {
        for (p0, p1) in edges.transformed_edges(transform) {
            if let Some(segment) = ProjectedSegment::from_points(camera, &p0, &p1, color) {
                self.draw_projected_segments([segment]);
            }
        }
    }

    /// Projects and draws transformed mesh triangle wireframes without allocating a transformed mesh.
    pub fn draw_projected_mesh_wireframe(
        &mut self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        color: Rgb,
        stride: usize,
    ) {
        let stride = stride.max(1);
        for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
            if idx % stride != 0 {
                continue;
            }
            let Some(ab) = ProjectedSegment::from_points(camera, &p0, &p1, color) else {
                continue;
            };
            let Some(bc) = ProjectedSegment::from_points(camera, &p1, &p2, color) else {
                continue;
            };
            let Some(ca) = ProjectedSegment::from_points(camera, &p2, &p0, color) else {
                continue;
            };
            self.draw_projected_segments([ab, bc, ca]);
        }
    }
}

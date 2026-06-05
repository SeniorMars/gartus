use super::colors::Rgb;
#[cfg(feature = "fancy_math")]
use crate::gmath::{
    ray::Ray,
    vector::{Point, Vector},
};
use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::display::Canvas,
};
#[cfg(feature = "fancy_math")]
use std::io::{self, Write};

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

/// A simple pinhole camera that emits one ray through each image pixel.
#[cfg(feature = "fancy_math")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCamera {
    image_width: u32,
    image_height: u32,
    camera_center: Point,
    pixel00_loc: Point,
    pixel_delta_u: Vector,
    pixel_delta_v: Vector,
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

    /// Projects transformed mesh triangle edges into colored wireframe segments.
    ///
    /// `color_for_triangle` receives the triangle index and average projected triangle depth.
    pub fn project_mesh_wireframe_segments<F>(
        &self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        stride: usize,
        mut color_for_triangle: F,
    ) -> Vec<ProjectedSegment>
    where
        F: FnMut(usize, f64) -> Rgb,
    {
        let stride = stride.max(1);
        let mut segments = Vec::new();
        for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
            if idx % stride != 0 {
                continue;
            }
            let Some(a) = self.project(&p0) else {
                continue;
            };
            let Some(b) = self.project(&p1) else {
                continue;
            };
            let Some(c) = self.project(&p2) else {
                continue;
            };
            let depth = (a.depth + b.depth + c.depth) / 3.0;
            let color = color_for_triangle(idx, depth);
            segments.push(ProjectedSegment { a, b, color });
            segments.push(ProjectedSegment { a: b, b: c, color });
            segments.push(ProjectedSegment { a: c, b: a, color });
        }
        segments
    }
}

#[cfg(feature = "fancy_math")]
impl RayCamera {
    /// Creates a camera with the requested image width and ideal aspect ratio.
    ///
    /// The image height is rounded down from `image_width / aspect_ratio`, with a
    /// minimum height of one pixel. The viewport is sized from the actual integer
    /// image dimensions so pixel spacing remains square.
    ///
    /// # Panics
    ///
    /// Panics if `image_width` is zero or `aspect_ratio` is not positive and finite.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(image_width: u32, aspect_ratio: f64) -> Self {
        assert!(image_width > 0, "image width must be positive");
        assert!(
            aspect_ratio.is_finite() && aspect_ratio > 0.0,
            "aspect ratio must be positive and finite"
        );

        let image_height = ((f64::from(image_width) / aspect_ratio) as u32).max(1);
        let focal_length = 1.0;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * (f64::from(image_width) / f64::from(image_height));
        let camera_center = Point::new(0.0, 0.0, 0.0);

        let viewport_u = Vector::new(viewport_width, 0.0, 0.0);
        let viewport_v = Vector::new(0.0, -viewport_height, 0.0);
        let pixel_delta_u = viewport_u / f64::from(image_width);
        let pixel_delta_v = viewport_v / f64::from(image_height);

        let viewport_upper_left = camera_center
            - Vector::new(0.0, 0.0, focal_length)
            - viewport_u / 2.0
            - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        Self {
            image_width,
            image_height,
            camera_center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
        }
    }

    /// Returns the rendered image width in pixels.
    #[must_use]
    pub fn image_width(self) -> u32 {
        self.image_width
    }

    /// Returns the rendered image height in pixels.
    #[must_use]
    pub fn image_height(self) -> u32 {
        self.image_height
    }

    /// Returns the camera origin point.
    #[must_use]
    pub fn camera_center(self) -> Point {
        self.camera_center
    }

    /// Returns a ray from the camera center through the center of pixel `(x, y)`.
    ///
    /// Pixel coordinates are in storage order: `(0, 0)` is the upper-left pixel,
    /// rows scan left to right, and rows advance downward.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is outside the camera image dimensions.
    #[must_use]
    pub fn ray_for_pixel(self, x: u32, y: u32) -> Ray {
        assert!(x < self.image_width, "pixel x must be inside the image");
        assert!(y < self.image_height, "pixel y must be inside the image");

        let pixel_center = self.pixel00_loc
            + f64::from(x) * self.pixel_delta_u
            + f64::from(y) * self.pixel_delta_v;
        Ray::new(self.camera_center, pixel_center - self.camera_center)
    }

    /// Renders a canvas by evaluating `ray_color` for each emitted camera ray.
    #[must_use]
    pub fn render<F>(self, mut ray_color: F) -> Canvas
    where
        F: FnMut(&Ray) -> Vector,
    {
        Canvas::from_fn(self.image_width, self.image_height, |x, y| {
            Rgb::from(ray_color(&self.ray_for_pixel(x, y)))
        })
    }

    /// Renders a canvas while writing scanline progress messages to `log`.
    ///
    /// Use `std::io::stderr()` for book-style progress reporting that stays separate
    /// from generated PPM image output.
    ///
    /// # Errors
    ///
    /// Returns any write error produced by `log`.
    pub fn render_with_progress<F, W>(self, mut log: W, mut ray_color: F) -> io::Result<Canvas>
    where
        F: FnMut(&Ray) -> Vector,
        W: Write,
    {
        let mut pixels = Vec::with_capacity(self.image_width as usize * self.image_height as usize);
        for y in 0..self.image_height {
            write!(log, "\rScanlines remaining: {} ", self.image_height - y)?;
            log.flush()?;
            for x in 0..self.image_width {
                pixels.push(Rgb::from(ray_color(&self.ray_for_pixel(x, y))));
            }
        }
        writeln!(log, "\rDone.                 ")?;

        Ok(Canvas::from_pixels(
            self.image_width,
            self.image_height,
            pixels,
        ))
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

    /// Projects, depth-sorts, and draws a transformed mesh as triangle wireframe segments.
    ///
    /// `color_for_triangle` receives the triangle index and average projected triangle depth.
    pub fn draw_projected_mesh_wireframe_depth_sorted<F>(
        &mut self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        stride: usize,
        color_for_triangle: F,
    ) where
        F: FnMut(usize, f64) -> Rgb,
    {
        let mut segments =
            camera.project_mesh_wireframe_segments(mesh, transform, stride, color_for_triangle);
        sort_segments_back_to_front(&mut segments);
        self.draw_projected_segments(segments);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_mesh_wireframe_returns_three_segments_per_visible_triangle() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        let camera = Camera3D::new(100, 100);
        let segments = camera.project_mesh_wireframe_segments(
            &mesh,
            &Matrix::identity_matrix(4),
            1,
            |_, _| Rgb::WHITE,
        );

        assert_eq!(segments.len(), 3);
    }

    #[cfg(feature = "fancy_math")]
    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[cfg(feature = "fancy_math")]
    #[test]
    fn ray_camera_uses_actual_integer_image_ratio() {
        let camera = RayCamera::new(400, 16.0 / 9.0);
        assert_eq!(camera.image_width(), 400);
        assert_eq!(camera.image_height(), 225);
    }

    #[cfg(feature = "fancy_math")]
    #[test]
    fn ray_camera_sends_center_pixel_forward() {
        let camera = RayCamera::new(400, 16.0 / 9.0);
        let ray = camera.ray_for_pixel(200, 112);

        assert_close(ray.origin().x(), 0.0);
        assert_close(ray.origin().y(), 0.0);
        assert_close(ray.origin().z(), 0.0);
        assert!(ray.direction().z() < 0.0);
        assert!(ray.direction().x().abs() < 0.01);
        assert!(ray.direction().y().abs() < 0.01);
    }
}

use super::colors::Rgb;
use crate::gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix};
use crate::graphics::display::Canvas;
use std::collections::HashSet;

const PERSPECTIVE_EPS: f64 = 1e-12;

struct WrappedRestore<'a> {
    wrapped: *mut bool,
    original: bool,
    _marker: std::marker::PhantomData<&'a mut bool>,
}

impl Drop for WrappedRestore<'_> {
    fn drop(&mut self) {
        // SAFETY: the guard is created from `self.wrapped` and never outlives the
        // `fill` call that owns the mutable `Canvas` borrow.
        unsafe {
            *self.wrapped = self.original;
        }
    }
}

#[allow(dead_code)]
impl Canvas {
    /// Fills in the area of a 2D figure given a random point inside the figure.
    ///
    /// # Arguments
    ///
    /// * `x` - A signed i32 int that represents the x of the random point
    /// * `y` - A signed i32 int that represents the y of the random point
    /// * `fill_color` - A [`Rgb`] will be the color the polygon will be filled in
    /// * `boundary_color` - A [`Rgb`] that is the represents the outline of the shape
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::graphics::display::Canvas;
    /// let background_color = Rgb::new(0, 0, 0);
    /// let mut image = Canvas::new(25, 25, background_color);
    /// let color = Rgb::new(0, 64, 255);
    /// image.fill(10, 10, color, background_color)
    /// ```
    pub fn fill(&mut self, x: i64, y: i64, fill_color: Rgb, boundary_color: Rgb) {
        let wrapped_ptr: *mut bool = std::ptr::addr_of_mut!(self.wrapped);
        let _wrapped_restore = WrappedRestore {
            original: self.wrapped,
            wrapped: wrapped_ptr,
            _marker: std::marker::PhantomData,
        };
        self.wrapped = false;

        let mut points = vec![(x, y)];
        let mut visited = HashSet::from([(x, y)]);
        while let Some((x, y)) = points.pop() {
            let Some(pixel) = self.get_pixel(x, y) else {
                continue;
            };
            if *pixel == boundary_color || *pixel == fill_color {
                continue;
            }
            self.plot(&fill_color, x, y);
            for (nx, ny) in [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)] {
                if visited.insert((nx, ny)) {
                    points.push((nx, ny));
                }
            }
            // points.push((x - 1, y - 1));
            // points.push((x - 1, y + 1));
            // points.push((x + 1, y - 1));
            // points.push((x + 1, y + 1));
        }
    }

    /// Fills in the area of a 2D figure using a faster scanline-based algorithm.
    ///
    /// This is generally more efficient than the stack-based [`fill`] method.
    pub fn scanline_fill(&mut self, x: i64, y: i64, fill_color: Rgb, boundary_color: Rgb) {
        if let Some(pixel) = self.get_pixel(x, y) {
            if *pixel == boundary_color || *pixel == fill_color {
                return;
            }
        } else {
            return;
        }

        // We use a similar guard as `fill` to disable wrapping during the operation
        let wrapped_ptr: *mut bool = std::ptr::addr_of_mut!(self.wrapped);
        let _wrapped_restore = WrappedRestore {
            original: self.wrapped,
            wrapped: wrapped_ptr,
            _marker: std::marker::PhantomData,
        };
        self.wrapped = false;

        let mut stack = vec![(x, y)];

        while let Some((x, y)) = stack.pop() {
            let mut lx = x;
            while lx > 0 {
                if let Some(p) = self.get_pixel(lx - 1, y) {
                    if *p == boundary_color || *p == fill_color {
                        break;
                    }
                    lx -= 1;
                } else {
                    break;
                }
            }

            let mut rx = x;
            while rx < i64::from(self.width()) - 1 {
                if let Some(p) = self.get_pixel(rx + 1, y) {
                    if *p == boundary_color || *p == fill_color {
                        break;
                    }
                    rx += 1;
                } else {
                    break;
                }
            }

            for i in lx..=rx {
                self.plot(&fill_color, i, y);
            }

            self.scanline_seed_helper(&mut stack, lx, rx, y + 1, fill_color, boundary_color);
            self.scanline_seed_helper(&mut stack, lx, rx, y - 1, fill_color, boundary_color);
        }
    }

    fn scanline_seed_helper(
        &self,
        stack: &mut Vec<(i64, i64)>,
        lx: i64,
        rx: i64,
        y: i64,
        fill_color: Rgb,
        boundary_color: Rgb,
    ) {
        let mut added = false;
        for i in lx..=rx {
            if let Some(p) = self.get_pixel(i, y) {
                if *p != boundary_color && *p != fill_color {
                    if !added {
                        stack.push((i, y));
                        added = true;
                    }
                } else {
                    added = false;
                }
            } else {
                added = false;
            }
        }
    }

    /// Draws all lines provided in a given [`EdgeMatrix`] onto the [`Canvas`].
    ///
    /// # Arguments
    ///
    /// * `edges` - An [`EdgeMatrix`] reference that has at least two points to draw onto the [`Canvas`]
    ///
    /// # Panics
    /// * If the edge matrix does not have two points to draw
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::gmath::edge_matrix::EdgeMatrix;
    /// let mut image = Canvas::new(25, 25, Rgb::default());
    /// let color = Rgb::new(0, 64, 255);
    /// image.set_line_pixel(color);
    /// let edges = EdgeMatrix::new();
    /// // image.draw_lines(&edges)
    /// ```
    pub fn draw_lines(&mut self, edges: &EdgeMatrix) {
        self.try_draw_lines(edges);
    }

    /// Applies `transform` to `edges`, then draws the transformed lines.
    pub fn draw_transformed(&mut self, edges: &EdgeMatrix, transform: &Matrix) {
        self.draw_lines(&edges.apply(transform));
    }

    /// Draws all lines in `edges` onto the [`Canvas`], returning before drawing if the assertion fails.
    ///
    /// # Panics
    /// Panics if the edge matrix does not contain an even number of points.
    pub fn try_draw_lines(&mut self, edges: &EdgeMatrix) {
        assert!(
            edges.cols().is_multiple_of(2),
            "edge matrix must contain pairs of points"
        );

        for (p0, p1) in edges.iter_edges() {
            self.draw_line(self.line, p0[0], p0[1], p1[0], p1[1]);
        }
    }

    /// Draws all lines in provided in a given [`EdgeMatrix`] onto the [`Canvas`] with perspective division.
    pub fn draw_lines_perspective(&mut self, edges: &EdgeMatrix) {
        self.try_draw_lines_perspective(edges);
    }

    /// Draws all lines in `edges` with perspective division onto the [`Canvas`].
    ///
    /// # Panics
    /// Panics if the edge matrix does not contain an even number of points.
    pub fn try_draw_lines_perspective(&mut self, edges: &EdgeMatrix) {
        assert!(
            edges.cols().is_multiple_of(2),
            "edge matrix must contain pairs of points"
        );

        for (p0, p1) in edges.iter_edges() {
            let Some((x0, y0)) = perspective_xy(p0) else {
                continue;
            };
            let Some((x1, y1)) = perspective_xy(p1) else {
                continue;
            };
            self.draw_line(self.line, x0, y0, x1, y1);
        }
    }

    /// Draws all triangles in `polygons` onto the [`Canvas`] with backface culling.
    ///
    /// # Panics
    /// Panics if the polygon matrix does not contain a multiple of 3 points.
    pub fn draw_polygons(&mut self, polygons: &PolygonMatrix) {
        let data = polygons.as_matrix().data();
        assert!(
            data.len().is_multiple_of(12),
            "polygon matrix must contain multiples of 3 points"
        );

        // Loop over raw f64 data; fixed chunk[N] offsets
        // Triangle layout: [x0,y0,z0,w0, x1,y1,z1,w1, x2,y2,z2,w2]
        // n⃗ · v⃗ = nz = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)   (v⃗ = <0,0,1>)
        for c in data.chunks_exact(12) {
            let vis = (c[4] - c[0]) * (c[9] - c[1]) - (c[5] - c[1]) * (c[8] - c[0]) > 0.0;
            if vis {
                self.draw_line(self.line, c[0], c[1], c[4], c[5]);
                self.draw_line(self.line, c[4], c[5], c[8], c[9]);
                self.draw_line(self.line, c[8], c[9], c[0], c[1]);
            }
        }
    }

    /// Draws a line onto the [Canvas] provided two sets of points.
    ///
    /// # Arguments
    ///
    /// * `color` - A [`Rgb`] that will will represent the color of the new line
    /// * `x0` - A f64 float that represents the start x coordinate of the line
    /// * `y0` - A f64 float that represents the start y coordinate of the line
    /// * `x1` - A f64 float that represents the end x coordinate of the line
    /// * `y1` - A f64 float that represents the end y coordinate of the line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let mut image = Canvas::new(25, 25, Rgb::default());
    /// let color = Rgb::new(0, 64, 255);
    /// image.draw_line(color, 0.0, 0.0, 24.0, 24.0)
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn draw_line(&mut self, color: Rgb, x0: f64, y0: f64, x1: f64, y1: f64) {
        let line_radius = self.line_radius();

        let (x0, y0, x1, y1) = if x0 > x1 {
            (x1, y1, x0, y0)
        } else {
            (x0, y0, x1, y1)
        };

        let (mut x0, mut y0, x1, y1) = (
            x0.round() as i64,
            y0.round() as i64,
            x1.round() as i64,
            y1.round() as i64,
        );

        let (delta_y, delta_x) = (2 * (y1 - y0), -2 * (x1 - x0));

        if (x1 - x0).abs() >= (y1 - y0).abs() {
            if delta_y > 0 {
                // octant 1
                let mut d = delta_y + delta_x / 2;
                for x in x0..=x1 {
                    for dx in -line_radius..=line_radius {
                        self.plot(&color, x, y0 + dx);
                    }
                    if d > 0 {
                        y0 += 1;
                        d += delta_x;
                    }
                    d += delta_y;
                }
            } else {
                // octant 8
                let mut d = delta_y - delta_x / 2;
                for x in x0..=x1 {
                    for dx in -line_radius..=line_radius {
                        self.plot(&color, x, y0 + dx);
                    }
                    if d < 0 {
                        y0 -= 1;
                        d -= delta_x;
                    }
                    d += delta_y;
                }
            }
        } else if delta_y > 0 {
            // octant 2
            let mut d = delta_y / 2 + delta_x;
            for y in y0..=y1 {
                for dy in -line_radius..=line_radius {
                    self.plot(&color, x0 + dy, y);
                }
                if d < 0 {
                    x0 += 1;
                    d += delta_y;
                }
                d += delta_x;
            }
        } else {
            // octant 7
            let mut d = delta_y / 2 - delta_x;
            for y in (y1..=y0).rev() {
                for dy in -line_radius..=line_radius {
                    self.plot(&color, x0 + dy, y);
                }
                if d > 0 {
                    x0 += 1;
                    d += delta_y;
                }
                d -= delta_x;
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn line_radius(&self) -> i64 {
        let width = self.line_width().round().max(1.0) as i64;
        let odd_width = if width % 2 == 0 { width + 1 } else { width };
        (odd_width - 1) / 2
    }
}

fn perspective_xy(point: &[f64]) -> Option<(f64, f64)> {
    let w = point[3];
    if w.abs() < PERSPECTIVE_EPS {
        return None;
    }
    Some((point[0] / w, point[1] / w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::animation::FrameRecorder;
    use std::collections::BTreeSet;
    use std::fs;

    fn line_points(x0: f64, y0: f64, x1: f64, y1: f64) -> BTreeSet<(i64, i64)> {
        let mut canvas = Canvas::new_with_bg(8, 8, Rgb::WHITE);
        canvas.upper_left_origin = true;
        canvas.wrapped = false;
        canvas.draw_line(Rgb::BLACK, x0, y0, x1, y1);
        black_points(&canvas)
    }

    fn black_points(canvas: &Canvas) -> BTreeSet<(i64, i64)> {
        let mut points = BTreeSet::new();
        for y in 0..canvas.height() {
            for x in 0..canvas.width() {
                if canvas.get_pixel(x.into(), y.into()) == Some(&Rgb::BLACK) {
                    points.insert((x.into(), y.into()));
                }
            }
        }
        points
    }

    fn points<const N: usize>(items: [(i64, i64); N]) -> BTreeSet<(i64, i64)> {
        BTreeSet::from(items)
    }

    #[test]
    fn draw_line_covers_horizontal_vertical_and_single_point() {
        assert_eq!(
            line_points(1.0, 2.0, 5.0, 2.0),
            points([(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)])
        );
        assert_eq!(
            line_points(3.0, 1.0, 3.0, 5.0),
            points([(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)])
        );
        assert_eq!(line_points(4.0, 4.0, 4.0, 4.0), points([(4, 4)]));
    }

    #[test]
    fn draw_line_covers_shallow_and_steep_octants() {
        assert_eq!(
            line_points(1.0, 1.0, 5.0, 3.0),
            points([(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)])
        );
        assert_eq!(
            line_points(1.0, 5.0, 5.0, 3.0),
            points([(1, 5), (2, 5), (3, 4), (4, 4), (5, 3)])
        );
        assert_eq!(
            line_points(1.0, 1.0, 3.0, 5.0),
            points([(1, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
        );
        assert_eq!(
            line_points(1.0, 5.0, 3.0, 1.0),
            points([(1, 5), (1, 4), (2, 3), (2, 2), (3, 1)])
        );
    }

    #[test]
    fn draw_line_reverse_directions_match_forward_lines() {
        assert_eq!(
            line_points(5.0, 3.0, 1.0, 1.0),
            line_points(1.0, 1.0, 5.0, 3.0)
        );
        assert_eq!(
            line_points(5.0, 3.0, 1.0, 5.0),
            line_points(1.0, 5.0, 5.0, 3.0)
        );
        assert_eq!(
            line_points(3.0, 5.0, 1.0, 1.0),
            line_points(1.0, 1.0, 3.0, 5.0)
        );
        assert_eq!(
            line_points(3.0, 1.0, 1.0, 5.0),
            line_points(1.0, 5.0, 3.0, 1.0)
        );
    }

    #[test]
    fn draw_line_uses_odd_width_radius() {
        let mut canvas = Canvas::new_with_bg(5, 5, Rgb::WHITE);
        canvas.upper_left_origin = true;
        canvas.wrapped = false;
        canvas.set_line_width(2.0);
        canvas.draw_line(Rgb::BLACK, 2.0, 2.0, 2.0, 2.0);

        assert_eq!(black_points(&canvas), points([(2, 1), (2, 2), (2, 3)]));
    }

    #[test]
    fn thick_steep_lines_use_horizontal_brush() {
        let mut canvas = Canvas::new_with_bg(5, 5, Rgb::WHITE);
        canvas.upper_left_origin = true;
        canvas.wrapped = false;
        canvas.set_line_width(3.0);
        canvas.draw_line(Rgb::BLACK, 2.0, 1.0, 2.0, 3.0);

        assert_eq!(
            black_points(&canvas),
            points([
                (1, 1),
                (2, 1),
                (3, 1),
                (1, 2),
                (2, 2),
                (3, 2),
                (1, 3),
                (2, 3),
                (3, 3)
            ])
        );
    }

    #[test]
    fn fill_uses_clipped_coordinates_even_when_canvas_wraps() {
        let mut canvas = Canvas::new_with_bg(3, 1, Rgb::WHITE);
        canvas.upper_left_origin = true;
        canvas.wrapped = true;
        canvas.plot(&Rgb::BLACK, 1, 0);
        canvas.fill(2, 0, Rgb::new(255, 0, 0), Rgb::BLACK);

        assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::WHITE));
        assert_eq!(canvas.get_pixel(1, 0), Some(&Rgb::BLACK));
        assert_eq!(canvas.get_pixel(2, 0), Some(&Rgb::new(255, 0, 0)));
        assert!(canvas.wrapped);
    }

    #[test]
    #[should_panic(expected = "edge matrix must contain pairs of points")]
    fn draw_lines_rejects_odd_point_count() {
        let mut edges = EdgeMatrix::new();
        edges.push_point(1.0, 1.0, 0.0);
        let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);
        canvas.draw_lines(&edges);
    }

    #[test]
    fn draw_transformed_applies_matrix_before_drawing() {
        let mut edges = EdgeMatrix::new();
        edges.push_edge(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);
        canvas.upper_left_origin = true;
        canvas.wrapped = false;
        canvas.draw_transformed(&edges, &Matrix::translate(1.0, 2.0, 0.0));

        assert_eq!(black_points(&canvas), points([(1, 2), (2, 2)]));
    }

    #[test]
    fn draw_lines_no_longer_saves_animation_frames() {
        fs::create_dir_all("anim").expect("create animation dir");
        let prefix = format!("test-frame-count-{}-", std::process::id());
        let mut edges = EdgeMatrix::new();
        edges.push_edge(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);
        edges.push_edge(1.0, 1.0, 0.0, 2.0, 2.0, 0.0);

        let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);
        canvas.try_draw_lines(&edges);

        assert!(!std::path::Path::new(&format!("anim/{prefix}00000000.ppm")).exists());
    }

    #[test]
    fn frame_recorder_captures_explicit_frames() {
        let prefix = format!("test-recorder-{}-", std::process::id());
        let mut recorder = FrameRecorder::new("anim", prefix.clone());
        let canvas = Canvas::new_with_bg(2, 2, Rgb::WHITE);

        recorder.capture(&canvas).expect("capture frame");

        assert_eq!(recorder.frame_index(), 1);
        let _ = fs::remove_file(format!("anim/{prefix}00000000.ppm"));
    }

    #[test]
    fn frame_recorder_can_capture_drawn_transformed_edges() {
        let prefix = format!("test-recorder-drawn-{}-", std::process::id());
        let mut recorder = FrameRecorder::new("anim", prefix.clone());
        let canvas = Canvas::new_with_bg(3, 3, Rgb::WHITE);
        let mut edges = EdgeMatrix::new();
        edges.push_edge(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        recorder
            .capture_drawn(&canvas, &edges, &Matrix::translate(1.0, 1.0, 0.0))
            .expect("capture transformed frame");

        assert_eq!(recorder.frame_index(), 1);
        let _ = fs::remove_file(format!("anim/{prefix}00000000.ppm"));
    }
}

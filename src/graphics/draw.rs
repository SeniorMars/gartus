use super::colors::{Hsl, Rgb};
use crate::gmath::{
    edge_matrix::EdgeMatrix,
    geometry::TriangleGeometry,
    matrix::Matrix,
    polygon_matrix::PolygonMatrix,
    vector::{Point, Vector},
};
use crate::graphics::{
    display::{Canvas, PolygonColorMode, ShadingMode, ZSpan},
    lighting::PreparedLighting,
};
use std::{
    collections::{HashMap, hash_map::Entry},
    hash::{BuildHasherDefault, Hasher},
};

pub use super::textured_raster::TexturedVertex;

const PERSPECTIVE_EPS: f64 = 1e-12;

/// Cached vertex-normal adjacency for a polygon mesh.
///
/// The plan stores which triangle vertex occurrences should share an accumulated smoothed normal.
/// It can be reused when the same mesh is transformed and redrawn across frames; transformed
/// triangle normals are still recomputed from the current polygon data.
#[derive(Clone, Debug, PartialEq)]
pub struct VertexNormalPlan {
    normal_indices: Vec<usize>,
    normal_count: usize,
}

impl VertexNormalPlan {
    /// Builds a vertex-normal plan for polygon matrix data.
    ///
    /// # Panics
    /// Panics if `data` is not a sequence of homogeneous triangle vertices.
    #[must_use]
    pub fn from_polygon_data(data: &[f64]) -> Self {
        assert!(
            data.len().is_multiple_of(12),
            "polygon data must contain multiples of 3 homogeneous points"
        );

        let point_count = data.len() / 4;
        let mut normal_indices = Vec::with_capacity(point_count);
        let mut normal_by_vertex = VertexNormalMap::<usize>::with_capacity_and_hasher(
            point_count,
            BuildHasherDefault::default(),
        );

        for c in data.chunks_exact(12) {
            let points = [(c[0], c[1], c[2]), (c[4], c[5], c[6]), (c[8], c[9], c[10])];
            for point in points {
                let next_index = normal_by_vertex.len();
                let normal_index = match normal_by_vertex.entry(vertex_key(point)) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => *entry.insert(next_index),
                };
                normal_indices.push(normal_index);
            }
        }

        Self {
            normal_indices,
            normal_count: normal_by_vertex.len(),
        }
    }

    /// Returns one normalized vertex normal per polygon point occurrence for `data`.
    ///
    /// Returns `None` if the plan was not built for the same number of point occurrences.
    #[must_use]
    pub fn normals_for_polygon_data(&self, data: &[f64]) -> Option<Vec<Vector>> {
        if data.len() / 4 != self.normal_indices.len() || !data.len().is_multiple_of(12) {
            return None;
        }

        let mut accumulated = vec![Vector::default(); self.normal_count];
        for (triangle_index, c) in data.chunks_exact(12).enumerate() {
            let points = [(c[0], c[1], c[2]), (c[4], c[5], c[6]), (c[8], c[9], c[10])];
            let normal = triangle_normal(points[0], points[1], points[2]);
            if normal.dot(normal) < f64::EPSILON * f64::EPSILON {
                continue;
            }

            let base = triangle_index * 3;
            for offset in 0..3 {
                accumulated[self.normal_indices[base + offset]] += normal;
            }
        }

        for normal in &mut accumulated {
            *normal = if normal.dot(*normal) < f64::EPSILON * f64::EPSILON {
                Vector::new(0.0, 0.0, 1.0)
            } else {
                normal.normalized()
            };
        }

        Some(
            self.normal_indices
                .iter()
                .map(|&normal_index| accumulated[normal_index])
                .collect(),
        )
    }
}

#[derive(Clone, Copy)]
struct ScanPoint {
    x: f64,
    y: i32,
    z: f64,
}

#[derive(Clone, Copy)]
struct ScanEdge {
    current: ScanPoint,
    dx: f64,
    dz: f64,
}

impl ScanEdge {
    fn new(start: ScanPoint, end: ScanPoint) -> Self {
        let dy = end.y - start.y;
        let (dx, dz) = if dy == 0 {
            (0.0, 0.0)
        } else {
            (
                (end.x - start.x) / f64::from(dy),
                (end.z - start.z) / f64::from(dy),
            )
        };
        Self {
            current: start,
            dx,
            dz,
        }
    }

    fn point(self) -> ScanPoint {
        self.current
    }

    fn step(&mut self) {
        self.current.x += self.dx;
        self.current.y += 1;
        self.current.z += self.dz;
    }
}

#[derive(Clone, Copy)]
struct ColorScanPoint {
    x: f64,
    y: i32,
    z: f64,
    color: [f64; 3],
}

#[derive(Clone, Copy)]
struct ColorScanEdge {
    current: ColorScanPoint,
    dx: f64,
    dz: f64,
    dcolor: [f64; 3],
}

impl ColorScanEdge {
    fn new(start: ColorScanPoint, end: ColorScanPoint) -> Self {
        let dy = end.y - start.y;
        let (dx, dz, dcolor) = if dy == 0 {
            (0.0, 0.0, [0.0; 3])
        } else {
            let dy = f64::from(dy);
            (
                (end.x - start.x) / dy,
                (end.z - start.z) / dy,
                [
                    (end.color[0] - start.color[0]) / dy,
                    (end.color[1] - start.color[1]) / dy,
                    (end.color[2] - start.color[2]) / dy,
                ],
            )
        };
        Self {
            current: start,
            dx,
            dz,
            dcolor,
        }
    }

    fn point(self) -> ColorScanPoint {
        self.current
    }

    fn step(&mut self) {
        self.current.x += self.dx;
        self.current.y += 1;
        self.current.z += self.dz;
        add3(&mut self.current.color, self.dcolor);
    }
}

#[derive(Clone, Copy)]
struct NormalScanPoint {
    x: f64,
    y: i32,
    z: f64,
    normal: Vector,
}

#[derive(Clone, Copy)]
struct NormalScanEdge {
    current: NormalScanPoint,
    dx: f64,
    dz: f64,
    dnormal: Vector,
}

impl NormalScanEdge {
    fn new(start: NormalScanPoint, end: NormalScanPoint) -> Self {
        let dy = end.y - start.y;
        let (dx, dz, dnormal) = if dy == 0 {
            (0.0, 0.0, Vector::default())
        } else {
            let dy = f64::from(dy);
            (
                (end.x - start.x) / dy,
                (end.z - start.z) / dy,
                (end.normal - start.normal) / dy,
            )
        };
        Self {
            current: start,
            dx,
            dz,
            dnormal,
        }
    }

    fn point(self) -> NormalScanPoint {
        self.current
    }

    fn step(&mut self) {
        self.current.x += self.dx;
        self.current.y += 1;
        self.current.z += self.dz;
        self.current.normal += self.dnormal;
    }
}

trait ScanSortable {
    fn scan_x(&self) -> f64;
    fn scan_y(&self) -> i32;
}

impl ScanSortable for ScanPoint {
    fn scan_x(&self) -> f64 {
        self.x
    }

    fn scan_y(&self) -> i32 {
        self.y
    }
}

impl ScanSortable for ColorScanPoint {
    fn scan_x(&self) -> f64 {
        self.x
    }

    fn scan_y(&self) -> i32 {
        self.y
    }
}

impl ScanSortable for NormalScanPoint {
    fn scan_x(&self) -> f64 {
        self.x
    }

    fn scan_y(&self) -> i32 {
        self.y
    }
}

fn sort3_by_y<T: ScanSortable>(points: &mut [T; 3]) {
    if points[0].scan_y() > points[1].scan_y() {
        points.swap(0, 1);
    }
    if points[1].scan_y() > points[2].scan_y() {
        points.swap(1, 2);
    }
    if points[0].scan_y() > points[1].scan_y() {
        points.swap(0, 1);
    }
}

fn sort3_by_x<T: ScanSortable>(points: &mut [T; 3]) {
    if points[0].scan_x() > points[1].scan_x() {
        points.swap(0, 1);
    }
    if points[1].scan_x() > points[2].scan_x() {
        points.swap(1, 2);
    }
    if points[0].scan_x() > points[1].scan_x() {
        points.swap(0, 1);
    }
}

#[derive(Clone, Copy)]
struct NormalScanState {
    x: f64,
    y: i64,
    z: f64,
    normal: Vector,
}

impl NormalScanState {
    #[allow(clippy::cast_precision_loss)]
    fn point(self) -> Vector {
        Vector::new(self.x, self.y as f64, self.z)
    }
}

#[allow(dead_code)]
impl Canvas {
    /// Fills in the area of a 2D figure given a random point inside the figure.
    ///
    /// This delegates to [`Self::scanline_fill`], which avoids the extra per-pixel visited set
    /// used by the older stack-based flood fill implementation.
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
        self.scanline_fill(x, y, fill_color, boundary_color);
    }

    /// Fills in the area of a 2D figure using a faster scanline-based algorithm.
    ///
    /// This is generally more efficient than the stack-based [`Self::fill`] method.
    pub fn scanline_fill(&mut self, x: i64, y: i64, fill_color: Rgb, boundary_color: Rgb) {
        if let Some(pixel) = self.get_pixel(x, y) {
            if *pixel == boundary_color || *pixel == fill_color {
                return;
            }
        } else {
            return;
        }

        let previous_wrapped = self.wrapped();
        self.set_wrapped(false);

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

        self.set_wrapped(previous_wrapped);
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
        self.draw_lines_checked(edges);
    }

    /// Applies `transform` to `edges`, then draws the transformed lines.
    pub fn draw_transformed(&mut self, edges: &EdgeMatrix, transform: &Matrix) {
        self.draw_lines(&edges.apply(transform));
    }

    /// Draws all lines in `edges` onto the [`Canvas`] after validating edge pairs.
    ///
    /// # Panics
    /// Panics if the edge matrix does not contain an even number of points.
    pub fn draw_lines_checked(&mut self, edges: &EdgeMatrix) {
        assert!(
            edges.cols().is_multiple_of(2),
            "edge matrix must contain pairs of points"
        );

        for (p0, p1) in edges.iter_edges() {
            self.draw_line_z(
                self.line_color(),
                (p0[0], p0[1], p0[2]),
                (p1[0], p1[1], p1[2]),
            );
        }
    }

    /// Deprecated alias for [`Self::draw_lines_checked`].
    #[deprecated(
        since = "0.1.0",
        note = "use draw_lines_checked; this method panics instead of returning Result"
    )]
    #[doc(hidden)]
    pub fn try_draw_lines(&mut self, edges: &EdgeMatrix) {
        self.draw_lines_checked(edges);
    }

    /// Draws all lines in provided in a given [`EdgeMatrix`] onto the [`Canvas`] with perspective division.
    pub fn draw_lines_perspective(&mut self, edges: &EdgeMatrix) {
        self.draw_lines_perspective_checked(edges);
    }

    /// Draws all lines in `edges` with perspective division onto the [`Canvas`].
    ///
    /// # Panics
    /// Panics if the edge matrix does not contain an even number of points.
    pub fn draw_lines_perspective_checked(&mut self, edges: &EdgeMatrix) {
        assert!(
            edges.cols().is_multiple_of(2),
            "edge matrix must contain pairs of points"
        );

        for (p0, p1) in edges.iter_edges() {
            let Some((x0, y0, z0)) = perspective_xyz(p0) else {
                continue;
            };
            let Some((x1, y1, z1)) = perspective_xyz(p1) else {
                continue;
            };
            self.draw_line_z(self.line_color(), (x0, y0, z0), (x1, y1, z1));
        }
    }

    /// Deprecated alias for [`Self::draw_lines_perspective_checked`].
    #[deprecated(
        since = "0.1.0",
        note = "use draw_lines_perspective_checked; this method panics instead of returning Result"
    )]
    #[doc(hidden)]
    pub fn try_draw_lines_perspective(&mut self, edges: &EdgeMatrix) {
        self.draw_lines_perspective_checked(edges);
    }

    /// Fills all triangles in `polygons` onto the [`Canvas`] with backface culling.
    ///
    /// # Panics
    /// Panics if the polygon matrix does not contain a multiple of 3 points.
    pub fn draw_polygons(&mut self, polygons: &PolygonMatrix) {
        self.draw_polygons_with_vertex_normal_plan(polygons, None);
    }

    /// Fills all triangles in `polygons` using a cached vertex-normal plan when supplied.
    ///
    /// If `vertex_normal_plan` does not match the polygon point count, this falls back to building
    /// normals from the current polygon data.
    ///
    /// # Panics
    /// Panics if the polygon matrix does not contain a multiple of 3 points.
    pub fn draw_polygons_with_vertex_normal_plan(
        &mut self,
        polygons: &PolygonMatrix,
        vertex_normal_plan: Option<&VertexNormalPlan>,
    ) {
        let data = polygons.as_matrix().data();
        assert!(
            data.len().is_multiple_of(12),
            "polygon matrix must contain multiples of 3 points"
        );

        let shading_mode = self.shading_mode();
        let color_mode = self.polygon_color_mode();
        let line_color = self.line_color();
        let lighting = if matches!(
            (shading_mode, color_mode),
            (
                ShadingMode::Gouraud | ShadingMode::Phong | ShadingMode::Toon,
                _
            ) | (_, PolygonColorMode::PhongReflection)
        ) {
            Some(self.lighting_ref().prepare())
        } else {
            None
        };
        let vertex_normals = if matches!(
            shading_mode,
            ShadingMode::Gouraud | ShadingMode::Phong | ShadingMode::Toon
        ) {
            Some(
                vertex_normal_plan
                    .and_then(|plan| plan.normals_for_polygon_data(data))
                    .unwrap_or_else(|| vertex_normals_by_point(data)),
            )
        } else {
            None
        };

        for (index, c) in data.chunks_exact(12).enumerate() {
            let p0 = (c[0], c[1], c[2]);
            let p1 = (c[4], c[5], c[6]);
            let p2 = (c[8], c[9], c[10]);
            if shading_mode == ShadingMode::Wireframe {
                self.draw_polygon_edges(line_color, p0, p1, p2);
                continue;
            }

            let normal = triangle_normal(p0, p1, p2);
            if normal[2] <= 0.0 {
                continue;
            }

            match shading_mode {
                ShadingMode::Wireframe => unreachable!("wireframe handled before culling"),
                ShadingMode::Flat => {
                    let color = match &lighting {
                        Some(lighting) => {
                            lighting.illuminate_at(normal, triangle_centroid(p0, p1, p2))
                        }
                        None => triangle_color(color_mode, line_color, index),
                    };
                    self.draw_scanline_triangle(color, p0, p1, p2);
                }
                ShadingMode::Gouraud | ShadingMode::Phong | ShadingMode::Toon => self
                    .draw_smooth_triangle(
                        shading_mode,
                        lighting
                            .as_ref()
                            .expect("lighting prepared for smooth shading"),
                        vertex_normals
                            .as_ref()
                            .expect("vertex normals prepared for smooth shading"),
                        index,
                        [p0, p1, p2],
                    ),
            }
        }
    }

    /// Draws one raw filled triangle with a fixed color.
    ///
    /// This bypasses [`PolygonMatrix`] construction for callers that already have a single raw
    /// screen-space triangle. It does not perform backface culling; use
    /// [`Self::draw_triangle_culled`] when winding should match [`Self::draw_polygons`].
    pub fn draw_triangle_raw(
        &mut self,
        color: Rgb,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        if !triangle_points_are_finite([p0, p1, p2]) {
            return;
        }
        self.draw_scanline_triangle(color, p0, p1, p2);
    }

    /// Draws one filled triangle with a fixed color.
    ///
    /// This is a compatibility alias for [`Self::draw_triangle_raw`]. It does not perform backface
    /// culling.
    pub fn draw_triangle(
        &mut self,
        color: Rgb,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        self.draw_triangle_raw(color, p0, p1, p2);
    }

    /// Draws one filled triangle with backface culling.
    ///
    /// Back-facing triangles are culled to match [`Self::draw_polygons`].
    pub fn draw_triangle_culled(
        &mut self,
        color: Rgb,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        if !triangle_points_are_finite([p0, p1, p2]) || triangle_normal(p0, p1, p2)[2] <= 0.0 {
            return;
        }
        self.draw_scanline_triangle(color, p0, p1, p2);
    }

    /// Draws one flat Phong-lit triangle using the canvas lighting state.
    ///
    /// Back-facing triangles are culled to match [`Self::draw_polygons`].
    pub fn draw_lit_triangle_culled(
        &mut self,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        if !triangle_points_are_finite([p0, p1, p2]) {
            return;
        }
        let normal = triangle_normal(p0, p1, p2);
        if normal[2] <= 0.0 {
            return;
        }
        let color = self
            .lighting_ref()
            .prepare()
            .illuminate_at(normal, triangle_centroid(p0, p1, p2));
        self.draw_scanline_triangle(color, p0, p1, p2);
    }

    /// Draws one flat Phong-lit triangle using the canvas lighting state.
    ///
    /// This is a compatibility alias for [`Self::draw_lit_triangle_culled`].
    pub fn draw_lit_triangle(
        &mut self,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        self.draw_lit_triangle_culled(p0, p1, p2);
    }

    fn draw_polygon_edges(
        &mut self,
        color: Rgb,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        self.draw_line_z(color, p0, p1);
        self.draw_line_z(color, p1, p2);
        self.draw_line_z(color, p2, p0);
    }

    fn draw_smooth_triangle(
        &mut self,
        shading_mode: ShadingMode,
        lighting: &PreparedLighting,
        vertex_normals: &[Vector],
        triangle_index: usize,
        points: [(f64, f64, f64); 3],
    ) {
        let normal_index = triangle_index * 3;
        let normals = [
            vertex_normals[normal_index],
            vertex_normals[normal_index + 1],
            vertex_normals[normal_index + 2],
        ];

        match shading_mode {
            ShadingMode::Gouraud => self.draw_gouraud_triangle(
                points[0],
                points[1],
                points[2],
                [
                    lighting.illuminate_unit_at(normals[0], tuple_to_vector(points[0])),
                    lighting.illuminate_unit_at(normals[1], tuple_to_vector(points[1])),
                    lighting.illuminate_unit_at(normals[2], tuple_to_vector(points[2])),
                ],
            ),
            ShadingMode::Phong => {
                self.draw_phong_triangle(lighting, points[0], points[1], points[2], normals);
            }
            ShadingMode::Toon => {
                self.draw_toon_triangle(lighting, points[0], points[1], points[2], normals);
            }
            ShadingMode::Wireframe | ShadingMode::Flat => {
                unreachable!("smooth triangle helper only handles smooth shading")
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn draw_scanline_triangle(
        &mut self,
        color: Rgb,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        if ![p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]
            .iter()
            .all(|value| value.is_finite())
        {
            return;
        }

        let mut points = [
            ScanPoint {
                x: p0.0.round(),
                y: p0.1.round() as i32,
                z: p0.2,
            },
            ScanPoint {
                x: p1.0.round(),
                y: p1.1.round() as i32,
                z: p1.2,
            },
            ScanPoint {
                x: p2.0.round(),
                y: p2.1.round() as i32,
                z: p2.2,
            },
        ];
        sort3_by_y(&mut points);

        let [bottom, middle, top] = points;
        if bottom.y == top.y {
            sort3_by_x(&mut points);
            self.draw_scanline(color, points[0], points[2], bottom.y);
            return;
        }
        if bottom.y == middle.y {
            self.draw_scanline(color, bottom, middle, bottom.y);
            let mut edge0 = ScanEdge::new(bottom, top);
            let mut edge1 = ScanEdge::new(middle, top);
            edge0.step();
            edge1.step();
            for y in (bottom.y + 1)..=top.y {
                self.draw_scanline(color, edge0.point(), edge1.point(), y);
                edge0.step();
                edge1.step();
            }
            return;
        }

        let mut long = ScanEdge::new(bottom, top);
        let mut short = ScanEdge::new(bottom, middle);
        for y in bottom.y..=middle.y {
            self.draw_scanline(color, long.point(), short.point(), y);
            long.step();
            short.step();
        }

        let mut short = ScanEdge::new(middle, top);
        short.step();
        for y in (middle.y + 1)..=top.y {
            self.draw_scanline(color, long.point(), short.point(), y);
            long.step();
            short.step();
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn draw_scanline(&mut self, color: Rgb, mut p0: ScanPoint, mut p1: ScanPoint, y: i32) {
        if p0.x > p1.x {
            std::mem::swap(&mut p0, &mut p1);
        }

        let x0 = p0.x.round() as i64;
        let x1 = p1.x.round() as i64;
        if x0 > x1 {
            return;
        }

        let steps = x1 - x0;
        let dz = if steps == 0 {
            0.0
        } else {
            (p1.z - p0.z) / steps as f64
        };
        let line_radius = self.line_radius();
        let y = i64::from(y);
        let z = p0.z;

        if !self.wrapped() {
            let height = i64::from(self.height());
            if height == 0 || y + line_radius < 0 || y - line_radius >= height {
                return;
            }

            for dy in -line_radius..=line_radius {
                self.plot_z_span_clipped(color, x0, x1, y + dy, z, dz);
            }
            return;
        }

        for dy in -line_radius..=line_radius {
            let y = y + dy;
            let mut z = z;
            for x in x0..=x1 {
                self.plot_z(&color, x, y, z);
                z += dz;
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn draw_gouraud_triangle(
        &mut self,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
        colors: [Rgb; 3],
    ) {
        if !triangle_points_are_finite([p0, p1, p2]) {
            return;
        }

        let mut points = [
            ColorScanPoint {
                x: p0.0.round(),
                y: p0.1.round() as i32,
                z: p0.2,
                color: rgb_to_f64(colors[0]),
            },
            ColorScanPoint {
                x: p1.0.round(),
                y: p1.1.round() as i32,
                z: p1.2,
                color: rgb_to_f64(colors[1]),
            },
            ColorScanPoint {
                x: p2.0.round(),
                y: p2.1.round() as i32,
                z: p2.2,
                color: rgb_to_f64(colors[2]),
            },
        ];
        if !points.iter().all(ColorScanPoint::is_finite) {
            return;
        }
        sort3_by_y(&mut points);

        let [bottom, middle, top] = points;
        if bottom.y == top.y {
            sort3_by_x(&mut points);
            self.draw_gouraud_scanline(points[0], points[2], bottom.y);
            return;
        }
        if bottom.y == middle.y {
            self.draw_gouraud_scanline(bottom, middle, bottom.y);
            let mut edge0 = ColorScanEdge::new(bottom, top);
            let mut edge1 = ColorScanEdge::new(middle, top);
            edge0.step();
            edge1.step();
            for y in (bottom.y + 1)..=top.y {
                self.draw_gouraud_scanline(edge0.point(), edge1.point(), y);
                edge0.step();
                edge1.step();
            }
            return;
        }

        let mut long = ColorScanEdge::new(bottom, top);
        let mut short = ColorScanEdge::new(bottom, middle);
        for y in bottom.y..=middle.y {
            self.draw_gouraud_scanline(long.point(), short.point(), y);
            long.step();
            short.step();
        }

        let mut short = ColorScanEdge::new(middle, top);
        short.step();
        for y in (middle.y + 1)..=top.y {
            self.draw_gouraud_scanline(long.point(), short.point(), y);
            long.step();
            short.step();
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn draw_gouraud_scanline(&mut self, mut p0: ColorScanPoint, mut p1: ColorScanPoint, y: i32) {
        if p0.x > p1.x {
            std::mem::swap(&mut p0, &mut p1);
        }

        let x0 = p0.x.round() as i64;
        let x1 = p1.x.round() as i64;
        if x0 > x1 {
            return;
        }

        let steps = x1 - x0;
        let dz = if steps == 0 {
            0.0
        } else {
            (p1.z - p0.z) / steps as f64
        };
        let dcolor = if steps == 0 {
            [0.0; 3]
        } else {
            [
                (p1.color[0] - p0.color[0]) / steps as f64,
                (p1.color[1] - p0.color[1]) / steps as f64,
                (p1.color[2] - p0.color[2]) / steps as f64,
            ]
        };
        let line_radius = self.line_radius();
        let y = i64::from(y);

        if !self.wrapped() {
            let height = i64::from(self.height());
            if height == 0 || y + line_radius < 0 || y - line_radius >= height {
                return;
            }
            for dy in -line_radius..=line_radius {
                self.plot_z_span_clipped_with(
                    ZSpan {
                        x0,
                        x1,
                        y: y + dy,
                        z: p0.z,
                        dz,
                    },
                    p0.color,
                    |color, step| add_scaled3(color, dcolor, step),
                    |color| rgb_from_f64(*color),
                );
            }
            return;
        }

        for dy in -line_radius..=line_radius {
            let mut z = p0.z;
            let mut color = p0.color;
            for x in x0..=x1 {
                self.plot_z(&rgb_from_f64(color), x, y + dy, z);
                z += dz;
                add3(&mut color, dcolor);
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn draw_phong_triangle(
        &mut self,
        lighting: &PreparedLighting,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
        normals: [Vector; 3],
    ) {
        if !triangle_points_are_finite([p0, p1, p2]) {
            return;
        }

        let mut points = [
            NormalScanPoint {
                x: p0.0.round(),
                y: p0.1.round() as i32,
                z: p0.2,
                normal: normals[0],
            },
            NormalScanPoint {
                x: p1.0.round(),
                y: p1.1.round() as i32,
                z: p1.2,
                normal: normals[1],
            },
            NormalScanPoint {
                x: p2.0.round(),
                y: p2.1.round() as i32,
                z: p2.2,
                normal: normals[2],
            },
        ];
        if !points.iter().all(NormalScanPoint::is_finite) {
            return;
        }
        sort3_by_y(&mut points);

        let [bottom, middle, top] = points;
        if bottom.y == top.y {
            sort3_by_x(&mut points);
            self.draw_phong_scanline(lighting, points[0], points[2], bottom.y);
            return;
        }
        if bottom.y == middle.y {
            self.draw_phong_scanline(lighting, bottom, middle, bottom.y);
            let mut edge0 = NormalScanEdge::new(bottom, top);
            let mut edge1 = NormalScanEdge::new(middle, top);
            edge0.step();
            edge1.step();
            for y in (bottom.y + 1)..=top.y {
                self.draw_phong_scanline(lighting, edge0.point(), edge1.point(), y);
                edge0.step();
                edge1.step();
            }
            return;
        }

        let mut long = NormalScanEdge::new(bottom, top);
        let mut short = NormalScanEdge::new(bottom, middle);
        for y in bottom.y..=middle.y {
            self.draw_phong_scanline(lighting, long.point(), short.point(), y);
            long.step();
            short.step();
        }

        let mut short = NormalScanEdge::new(middle, top);
        short.step();
        for y in (middle.y + 1)..=top.y {
            self.draw_phong_scanline(lighting, long.point(), short.point(), y);
            long.step();
            short.step();
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn draw_toon_triangle(
        &mut self,
        lighting: &PreparedLighting,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
        normals: [Vector; 3],
    ) {
        if !triangle_points_are_finite([p0, p1, p2]) {
            return;
        }

        let mut points = [
            NormalScanPoint {
                x: p0.0.round(),
                y: p0.1.round() as i32,
                z: p0.2,
                normal: normals[0],
            },
            NormalScanPoint {
                x: p1.0.round(),
                y: p1.1.round() as i32,
                z: p1.2,
                normal: normals[1],
            },
            NormalScanPoint {
                x: p2.0.round(),
                y: p2.1.round() as i32,
                z: p2.2,
                normal: normals[2],
            },
        ];
        if !points.iter().all(NormalScanPoint::is_finite) {
            return;
        }
        sort3_by_y(&mut points);

        let [bottom, middle, top] = points;
        if bottom.y == top.y {
            sort3_by_x(&mut points);
            self.draw_toon_scanline(lighting, points[0], points[2], bottom.y);
            return;
        }
        if bottom.y == middle.y {
            self.draw_toon_scanline(lighting, bottom, middle, bottom.y);
            let mut edge0 = NormalScanEdge::new(bottom, top);
            let mut edge1 = NormalScanEdge::new(middle, top);
            edge0.step();
            edge1.step();
            for y in (bottom.y + 1)..=top.y {
                self.draw_toon_scanline(lighting, edge0.point(), edge1.point(), y);
                edge0.step();
                edge1.step();
            }
            return;
        }

        let mut long = NormalScanEdge::new(bottom, top);
        let mut short = NormalScanEdge::new(bottom, middle);
        for y in bottom.y..=middle.y {
            self.draw_toon_scanline(lighting, long.point(), short.point(), y);
            long.step();
            short.step();
        }

        let mut short = NormalScanEdge::new(middle, top);
        short.step();
        for y in (middle.y + 1)..=top.y {
            self.draw_toon_scanline(lighting, long.point(), short.point(), y);
            long.step();
            short.step();
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn draw_phong_scanline(
        &mut self,
        lighting: &PreparedLighting,
        mut p0: NormalScanPoint,
        mut p1: NormalScanPoint,
        y: i32,
    ) {
        if p0.x > p1.x {
            std::mem::swap(&mut p0, &mut p1);
        }

        let x0 = p0.x.round() as i64;
        let x1 = p1.x.round() as i64;
        if x0 > x1 {
            return;
        }

        let steps = x1 - x0;
        let dz = if steps == 0 {
            0.0
        } else {
            (p1.z - p0.z) / steps as f64
        };
        let dnormal = if steps == 0 {
            Vector::default()
        } else {
            (p1.normal - p0.normal) / steps as f64
        };
        let line_radius = self.line_radius();
        let y = i64::from(y);

        if !self.wrapped() {
            let height = i64::from(self.height());
            if height == 0 || y + line_radius < 0 || y - line_radius >= height {
                return;
            }
            for dy in -line_radius..=line_radius {
                self.plot_z_span_clipped_with(
                    ZSpan {
                        x0,
                        x1,
                        y: y + dy,
                        z: p0.z,
                        dz,
                    },
                    NormalScanState {
                        x: x0 as f64,
                        y: y + dy,
                        z: p0.z,
                        normal: p0.normal,
                    },
                    |state, step| {
                        state.x += step;
                        state.z += dz * step;
                        state.normal += dnormal * step;
                    },
                    |state| lighting.illuminate_at(state.normal, state.point()),
                );
            }
            return;
        }

        for dy in -line_radius..=line_radius {
            let mut z = p0.z;
            let mut normal = p0.normal;
            for x in x0..=x1 {
                let point = Vector::new(x as f64, (y + dy) as f64, z);
                self.plot_z(&lighting.illuminate_at(normal, point), x, y + dy, z);
                z += dz;
                normal += dnormal;
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn draw_toon_scanline(
        &mut self,
        lighting: &PreparedLighting,
        mut p0: NormalScanPoint,
        mut p1: NormalScanPoint,
        y: i32,
    ) {
        if p0.x > p1.x {
            std::mem::swap(&mut p0, &mut p1);
        }

        let x0 = p0.x.round() as i64;
        let x1 = p1.x.round() as i64;
        if x0 > x1 {
            return;
        }

        let steps = x1 - x0;
        let dz = if steps == 0 {
            0.0
        } else {
            (p1.z - p0.z) / steps as f64
        };
        let dnormal = if steps == 0 {
            Vector::default()
        } else {
            (p1.normal - p0.normal) / steps as f64
        };
        let line_radius = self.line_radius();
        let y = i64::from(y);

        if !self.wrapped() {
            let height = i64::from(self.height());
            if height == 0 || y + line_radius < 0 || y - line_radius >= height {
                return;
            }
            for dy in -line_radius..=line_radius {
                self.plot_z_span_clipped_with(
                    ZSpan {
                        x0,
                        x1,
                        y: y + dy,
                        z: p0.z,
                        dz,
                    },
                    NormalScanState {
                        x: x0 as f64,
                        y: y + dy,
                        z: p0.z,
                        normal: p0.normal,
                    },
                    |state, step| {
                        state.x += step;
                        state.z += dz * step;
                        state.normal += dnormal * step;
                    },
                    |state| lighting.illuminate_toon_at(state.normal, state.point()),
                );
            }
            return;
        }

        for dy in -line_radius..=line_radius {
            let mut z = p0.z;
            let mut normal = p0.normal;
            for x in x0..=x1 {
                let point = Vector::new(x as f64, (y + dy) as f64, z);
                self.plot_z(&lighting.illuminate_toon_at(normal, point), x, y + dy, z);
                z += dz;
                normal += dnormal;
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
        self.draw_line_z(color, (x0, y0, 0.0), (x1, y1, 0.0));
    }

    /// Draws a z-buffered line onto the [Canvas] from `(x, y, z)` to `(x, y, z)`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn draw_line_z(&mut self, color: Rgb, p0: (f64, f64, f64), p1: (f64, f64, f64)) {
        let line_radius = self.line_radius();

        if ![p0.0, p0.1, p0.2, p1.0, p1.1, p1.2]
            .iter()
            .all(|value| value.is_finite())
        {
            return;
        }

        let (x0, y0, z0, x1, y1, z1) = if p0.0 > p1.0 {
            (p1.0, p1.1, p1.2, p0.0, p0.1, p0.2)
        } else {
            (p0.0, p0.1, p0.2, p1.0, p1.1, p1.2)
        };

        let (mut x0, mut y0, x1, y1) = (
            x0.round() as i64,
            y0.round() as i64,
            x1.round() as i64,
            y1.round() as i64,
        );

        let (delta_y, delta_x) = (2 * (y1 - y0), -2 * (x1 - x0));

        if (x1 - x0).abs() >= (y1 - y0).abs() {
            let steps = (x1 - x0).abs();
            let dz = if steps == 0 {
                0.0
            } else {
                (z1 - z0) / steps as f64
            };
            let mut z = z0;
            if delta_y > 0 {
                // octant 1
                let mut d = delta_y + delta_x / 2;
                for x in x0..=x1 {
                    for dx in -line_radius..=line_radius {
                        self.plot_z(&color, x, y0 + dx, z);
                    }
                    if d > 0 {
                        y0 += 1;
                        d += delta_x;
                    }
                    d += delta_y;
                    z += dz;
                }
            } else {
                // octant 8
                let mut d = delta_y - delta_x / 2;
                for x in x0..=x1 {
                    for dx in -line_radius..=line_radius {
                        self.plot_z(&color, x, y0 + dx, z);
                    }
                    if d < 0 {
                        y0 -= 1;
                        d -= delta_x;
                    }
                    d += delta_y;
                    z += dz;
                }
            }
        } else if delta_y > 0 {
            // octant 2
            let steps = (y1 - y0).abs();
            let dz = if steps == 0 {
                0.0
            } else {
                (z1 - z0) / steps as f64
            };
            let mut z = z0;
            let mut d = delta_y / 2 + delta_x;
            for y in y0..=y1 {
                for dy in -line_radius..=line_radius {
                    self.plot_z(&color, x0 + dy, y, z);
                }
                if d < 0 {
                    x0 += 1;
                    d += delta_y;
                }
                d += delta_x;
                z += dz;
            }
        } else {
            // octant 7
            let steps = (y1 - y0).abs();
            let dz = if steps == 0 {
                0.0
            } else {
                (z1 - z0) / steps as f64
            };
            let mut z = z0;
            let mut d = delta_y / 2 - delta_x;
            for y in (y1..=y0).rev() {
                for dy in -line_radius..=line_radius {
                    self.plot_z(&color, x0 + dy, y, z);
                }
                if d > 0 {
                    x0 += 1;
                    d += delta_y;
                }
                d -= delta_x;
                z += dz;
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

fn perspective_xyz(point: &[f64]) -> Option<(f64, f64, f64)> {
    let w = point[3];
    if w.abs() < PERSPECTIVE_EPS {
        return None;
    }
    Some((point[0] / w, point[1] / w, point[2] / w))
}

impl ColorScanPoint {
    fn is_finite(&self) -> bool {
        self.x.is_finite()
            && self.z.is_finite()
            && self.color.iter().all(|channel| channel.is_finite())
    }
}

impl NormalScanPoint {
    fn is_finite(&self) -> bool {
        self.x.is_finite()
            && self.z.is_finite()
            && self.normal[0].is_finite()
            && self.normal[1].is_finite()
            && self.normal[2].is_finite()
    }
}

pub(super) fn triangle_normal(
    p0: (f64, f64, f64),
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
) -> Vector {
    TriangleGeometry::from_tuples([p0, p1, p2]).area_weighted_normal()
}

fn triangle_centroid(p0: (f64, f64, f64), p1: (f64, f64, f64), p2: (f64, f64, f64)) -> Vector {
    let centroid = TriangleGeometry::from_tuples([p0, p1, p2]).centroid();
    Vector::new(centroid.x(), centroid.y(), centroid.z())
}

fn tuple_to_vector(point: (f64, f64, f64)) -> Vector {
    let point = Point::new(point.0, point.1, point.2);
    Vector::new(point.x(), point.y(), point.z())
}

fn triangle_points_are_finite(points: [(f64, f64, f64); 3]) -> bool {
    points
        .into_iter()
        .all(|point| point.0.is_finite() && point.1.is_finite() && point.2.is_finite())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct VertexKey([u64; 3]);

pub(super) type VertexNormalMap<T> = HashMap<VertexKey, T, BuildHasherDefault<VertexKeyHasher>>;

#[derive(Debug)]
pub(super) struct VertexKeyHasher(u64);

impl Default for VertexKeyHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for VertexKeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }
}

fn vertex_key(point: (f64, f64, f64)) -> VertexKey {
    VertexKey([point.0.to_bits(), point.1.to_bits(), point.2.to_bits()])
}

#[cfg(test)]
pub(super) fn vertex_normals(data: &[f64]) -> VertexNormalMap<Vector> {
    let mut normals = VertexNormalMap::<Vector>::with_capacity_and_hasher(
        data.len() / 4,
        BuildHasherDefault::default(),
    );

    for c in data.chunks_exact(12) {
        let p0 = (c[0], c[1], c[2]);
        let p1 = (c[4], c[5], c[6]);
        let p2 = (c[8], c[9], c[10]);
        let normal = triangle_normal(p0, p1, p2);
        if normal.length() < f64::EPSILON {
            continue;
        }

        for point in [p0, p1, p2] {
            normals
                .entry(vertex_key(point))
                .and_modify(|accumulated| *accumulated += normal)
                .or_insert(normal);
        }
    }

    for normal in normals.values_mut() {
        *normal = normal.normalized();
    }

    normals
}

fn vertex_normals_by_point(data: &[f64]) -> Vec<Vector> {
    let point_count = data.len() / 4;
    let mut normal_indices = Vec::with_capacity(point_count);
    let mut normal_by_vertex = VertexNormalMap::<usize>::with_capacity_and_hasher(
        point_count,
        BuildHasherDefault::default(),
    );
    let mut accumulated = Vec::<Vector>::new();

    for c in data.chunks_exact(12) {
        let points = [(c[0], c[1], c[2]), (c[4], c[5], c[6]), (c[8], c[9], c[10])];
        let normal = triangle_normal(points[0], points[1], points[2]);
        let has_surface_normal = normal.dot(normal) >= f64::EPSILON * f64::EPSILON;

        for point in points {
            let next_index = normal_by_vertex.len();
            let normal_index = match normal_by_vertex.entry(vertex_key(point)) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    accumulated.push(Vector::default());
                    *entry.insert(next_index)
                }
            };
            if has_surface_normal {
                accumulated[normal_index] += normal;
            }
            normal_indices.push(normal_index);
        }
    }

    for normal in &mut accumulated {
        *normal = if normal.dot(*normal) < f64::EPSILON * f64::EPSILON {
            Vector::new(0.0, 0.0, 1.0)
        } else {
            normal.normalized()
        };
    }

    normal_indices
        .into_iter()
        .map(|normal_index| accumulated[normal_index])
        .collect()
}

#[cfg(test)]
pub(super) fn vertex_normal(normals: &VertexNormalMap<Vector>, point: (f64, f64, f64)) -> Vector {
    normals
        .get(&vertex_key(point))
        .copied()
        .unwrap_or(Vector::new(0.0, 0.0, 1.0))
}

fn rgb_to_f64(color: Rgb) -> [f64; 3] {
    [
        f64::from(color.red),
        f64::from(color.green),
        f64::from(color.blue),
    ]
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rgb_from_f64(color: [f64; 3]) -> Rgb {
    Rgb::new(
        color[0].round().clamp(0.0, 255.0) as u8,
        color[1].round().clamp(0.0, 255.0) as u8,
        color[2].round().clamp(0.0, 255.0) as u8,
    )
}

fn add3(value: &mut [f64; 3], delta: [f64; 3]) {
    value[0] += delta[0];
    value[1] += delta[1];
    value[2] += delta[2];
}

fn add_scaled3(value: &mut [f64; 3], delta: [f64; 3], scale: f64) {
    value[0] += delta[0] * scale;
    value[1] += delta[1] * scale;
    value[2] += delta[2] * scale;
}

pub(super) fn triangle_color(mode: PolygonColorMode, base: Rgb, index: usize) -> Rgb {
    match mode {
        PolygonColorMode::LineColor | PolygonColorMode::PhongReflection => base,
        PolygonColorMode::DeterministicRandom => random_triangle_color(index),
        PolygonColorMode::TintedFromLine => tinted_triangle_color(base, index),
    }
}

fn random_triangle_color(index: usize) -> Rgb {
    let seed = triangle_color_seed(index);
    Rgb::new(
        ((seed >> 16) & 0xff) as u8,
        ((seed >> 8) & 0xff) as u8,
        (seed & 0xff) as u8,
    )
}

fn tinted_triangle_color(base: Rgb, index: usize) -> Rgb {
    if index.is_multiple_of(8) {
        return base;
    }

    let base_hsl = Hsl::from(base);
    let seed = triangle_color_seed(index);
    let hue = (u32::from(base_hsl.hue) + seed % 360) % 360;
    let saturation_jitter = i32::try_from((seed >> 12) % 31).expect("jitter fits i32") - 15;
    let light_jitter = i32::try_from((seed >> 20) % 29).expect("jitter fits i32") - 14;

    let saturation = (i32::from(base_hsl.saturation) + 20 + saturation_jitter).clamp(45, 95);
    let light = (i32::from(base_hsl.light) + light_jitter).clamp(30, 78);
    let varied = Rgb::from(Hsl::new(
        u16::try_from(hue).expect("hue is less than 360"),
        u16::try_from(saturation).expect("saturation is clamped to 0..=100"),
        u16::try_from(light).expect("light is clamped to 0..=100"),
    ));

    base.lerp(varied, 0.68)
}

fn triangle_color_seed(index: usize) -> u32 {
    let mut x = u32::try_from(index).unwrap_or(u32::MAX) ^ 0x9e37_79b9;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

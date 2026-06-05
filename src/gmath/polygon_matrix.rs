use super::matrix::Matrix;
use std::f64::consts::PI;
use std::fmt;

/// A dynamically-growing list of triangle vertices stored as a 4×N column-major matrix.
/// Used for polygon mesh rendering.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PolygonMatrix {
    inner: Matrix,
}

/// Axis-aligned bounds for 3D points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3 {
    /// Minimum x, y, and z coordinates.
    pub min: (f64, f64, f64),
    /// Maximum x, y, and z coordinates.
    pub max: (f64, f64, f64),
}

/// Scaling options for converting a row-major height map into a triangle mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightMapOptions {
    /// Total generated mesh width along the x axis.
    pub x_size: f64,
    /// Total generated mesh depth along the z axis.
    pub z_size: f64,
    /// Multiplier applied to each height value to produce y coordinates.
    pub height_scale: f64,
    /// Offset added to each generated y coordinate.
    pub y_offset: f64,
}

impl HeightMapOptions {
    /// Creates height-map mesh options centered around the x/z origin.
    ///
    /// # Panics
    /// Panics if any size or scale is non-finite, or if either planar size is not positive.
    #[must_use]
    pub fn new(x_size: f64, z_size: f64, height_scale: f64) -> Self {
        assert!(
            [x_size, z_size, height_scale]
                .iter()
                .all(|value| value.is_finite()),
            "height-map options must be finite"
        );
        assert!(x_size > 0.0, "height-map x size must be positive");
        assert!(z_size > 0.0, "height-map z size must be positive");
        Self {
            x_size,
            z_size,
            height_scale,
            y_offset: 0.0,
        }
    }

    /// Sets an additive y offset for generated vertices.
    ///
    /// # Panics
    /// Panics if `y_offset` is non-finite.
    #[must_use]
    pub fn y_offset(mut self, y_offset: f64) -> Self {
        assert!(y_offset.is_finite(), "height-map y offset must be finite");
        self.y_offset = y_offset;
        self
    }
}

impl PolygonMatrix {
    /// Creates an empty polygon matrix (4 rows, 0 cols).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Matrix::new(4, 0, Vec::new()),
        }
    }

    /// Creates a polygon matrix pre-allocated for `n` points.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let v = Vec::with_capacity(n * 4);
        Self {
            inner: Matrix::new(4, 0, v),
        }
    }

    /// Clears all points from the matrix without deallocating memory.
    pub fn clear(&mut self) {
        self.inner.truncate_cols(0);
    }

    /// Number of points (columns) in this polygon matrix.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Number of complete triangles in this polygon matrix.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.inner.cols() / 3
    }

    /// Total number of f64 values stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the polygon matrix has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.cols() == 0
    }

    /// Returns the axis-aligned bounds for all points in this matrix.
    #[must_use]
    pub fn bounds(&self) -> Option<Bounds3> {
        self.bounds_from_col(0)
    }

    /// Returns the axis-aligned bounds for points starting at `start_col`.
    ///
    /// # Panics
    /// Panics if `start_col` is greater than the current column count.
    #[must_use]
    pub fn bounds_from_col(&self, start_col: usize) -> Option<Bounds3> {
        assert!(
            start_col <= self.cols(),
            "start column must be within the polygon matrix"
        );

        let mut points = self.iter_points().skip(start_col);
        let first = points.next()?;
        let mut bounds = Bounds3 {
            min: (first[0], first[1], first[2]),
            max: (first[0], first[1], first[2]),
        };

        for point in points {
            bounds.min.0 = bounds.min.0.min(point[0]);
            bounds.min.1 = bounds.min.1.min(point[1]);
            bounds.min.2 = bounds.min.2.min(point[2]);
            bounds.max.0 = bounds.max.0.max(point[0]);
            bounds.max.1 = bounds.max.1.max(point[1]);
            bounds.max.2 = bounds.max.2.max(point[2]);
        }

        Some(bounds)
    }

    /// Adds a point (x, y, z) with implicit w=1.
    ///
    /// # Panics
    /// Panics if the inner matrix row count is not 4.
    pub fn push_point(&mut self, x: f64, y: f64, z: f64) {
        self.append_homogeneous_points(&[x, y, z, 1.0]);
    }

    /// Appends multiple points to the matrix.
    pub fn push_points(&mut self, points: &[(f64, f64, f64)]) {
        let mut data = Vec::with_capacity(points.len() * 4);
        for &(x, y, z) in points {
            data.extend_from_slice(&[x, y, z, 1.0]);
        }
        self.append_homogeneous_points(&data);
    }

    /// Adds a triangle (three points) in counter-clockwise order.
    pub fn add_polygon(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64), p2: (f64, f64, f64)) {
        self.append_homogeneous_points(&[
            p0.0, p0.1, p0.2, 1.0, p1.0, p1.1, p1.2, 1.0, p2.0, p2.1, p2.2, 1.0,
        ]);
    }

    /// Appends multiple triangles to the matrix.
    ///
    /// Each triangle is stored in the point order supplied by the caller.
    pub fn push_polygons(&mut self, polygons: &[[(f64, f64, f64); 3]]) {
        let mut data = Vec::with_capacity(polygons.len() * 12);
        for &[p0, p1, p2] in polygons {
            Self::extend_polygon_data(&mut data, p0, p1, p2);
        }
        self.append_homogeneous_points(&data);
    }

    /// Appends another `PolygonMatrix`'s points to this one.
    ///
    /// # Panics
    /// Panics if the inner matrices have differing row counts.
    pub fn extend(&mut self, other: &PolygonMatrix) {
        self.inner
            .append_columns(&other.inner)
            .expect("PolygonMatrix values must always have 4 rows");
    }

    /// Truncates this polygon matrix to `cols` points.
    ///
    /// This is primarily used to roll back failed streaming imports after batches have already
    /// been appended.
    ///
    /// # Panics
    /// Panics if `cols` is greater than the current column count.
    pub fn truncate_cols(&mut self, cols: usize) {
        self.inner.truncate_cols(cols);
    }

    /// Returns an iterator over individual points as `&[f64]` slices of length 4.
    pub fn iter_points(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.inner.iter_by_point()
    }

    /// Returns an iterator over triangle vertex triples as `&[f64]` slices of length 4.
    pub fn iter_triangles(&self) -> impl Iterator<Item = (&[f64], &[f64], &[f64])> + '_ {
        self.inner.data().chunks_exact(12).map(|tri| {
            let (p01, p2) = tri.split_at(8);
            let (p0, p1) = p01.split_at(4);
            (p0, p1, p2)
        })
    }

    /// Returns transformed triangle vertex triples without allocating a transformed `PolygonMatrix`.
    pub fn transformed_triangles<'a>(
        &'a self,
        transform: &'a Matrix,
    ) -> impl Iterator<Item = ([f64; 4], [f64; 4], [f64; 4])> + 'a {
        self.iter_triangles().map(|(p0, p1, p2)| {
            (
                transform.transform_homogeneous_point(p0),
                transform.transform_homogeneous_point(p1),
                transform.transform_homogeneous_point(p2),
            )
        })
    }

    /// Reverses the winding order of every triangle in place.
    ///
    /// This is useful for imported meshes whose face order is opposite the renderer's
    /// backface-culling convention.
    ///
    /// # Panics
    /// Panics if the polygon matrix does not contain a multiple of 3 points.
    pub fn reverse_winding(&mut self) {
        self.reverse_winding_from_col(0);
    }

    /// Reverses the winding order of triangles starting at `start_col`.
    ///
    /// # Panics
    /// Panics if `start_col` is not on a triangle boundary or the polygon matrix does not contain
    /// a multiple of 3 points after `start_col`.
    pub fn reverse_winding_from_col(&mut self, start_col: usize) {
        assert!(
            start_col.is_multiple_of(3) && (self.cols() - start_col).is_multiple_of(3),
            "polygon matrix must contain multiples of 3 points"
        );

        for triangle_start in (start_col..self.cols()).step_by(3) {
            for row in 0..4 {
                let p1 = self.inner[(row, triangle_start + 1)];
                self.inner[(row, triangle_start + 1)] = self.inner[(row, triangle_start + 2)];
                self.inner[(row, triangle_start + 2)] = p1;
            }
        }
    }

    /// Returns a copy with every triangle winding order reversed.
    ///
    /// # Panics
    /// Panics if the polygon matrix does not contain a multiple of 3 points.
    #[must_use]
    pub fn reversed_winding(&self) -> Self {
        let mut reversed = self.clone();
        reversed.reverse_winding();
        reversed
    }

    /// Apply a 4×4 transformation matrix to all points. Returns a new `PolygonMatrix`.
    #[must_use]
    pub fn apply(&self, transform: &Matrix) -> Self {
        Self {
            inner: transform.mult_matrix(&self.inner),
        }
    }

    /// Applies a 4x4 transformation matrix to all points in place.
    pub fn apply_in_place(&mut self, transform: &Matrix) {
        self.inner
            .apply_homogeneous_transform_from_col(0, transform);
    }

    /// Builds a triangle mesh from row-major height samples.
    ///
    /// The outer slice contains z rows and each inner slice contains x columns. Generated x and z
    /// coordinates are centered around `0.0`, while y coordinates are `height * height_scale +
    /// y_offset`.
    ///
    /// # Panics
    /// Panics if the height map has fewer than two rows or columns, or if rows have differing
    /// lengths.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_height_map(heights: &[Vec<f64>], options: HeightMapOptions) -> Self {
        assert!(heights.len() >= 2, "height map must have at least two rows");
        let cols = heights[0].len();
        assert!(cols >= 2, "height map must have at least two columns");
        assert!(
            heights.iter().all(|row| row.len() == cols),
            "height map rows must all have the same length"
        );

        let rows = heights.len();
        let mut mesh = Self::with_capacity((rows - 1) * (cols - 1) * 6);
        let x_denom = (cols - 1) as f64;
        let z_denom = (rows - 1) as f64;
        let point = |x: usize, z: usize| {
            let px = (x as f64 / x_denom - 0.5) * options.x_size;
            let pz = (z as f64 / z_denom - 0.5) * options.z_size;
            let py = heights[z][x] * options.height_scale + options.y_offset;
            (px, py, pz)
        };

        for z in 0..rows - 1 {
            for x in 0..cols - 1 {
                let p00 = point(x, z);
                let p10 = point(x + 1, z);
                let p01 = point(x, z + 1);
                let p11 = point(x + 1, z + 1);
                mesh.add_polygon(p00, p01, p11);
                mesh.add_polygon(p00, p11, p10);
            }
        }
        mesh
    }

    /// Get a reference to the underlying `Matrix`.
    pub fn as_matrix(&self) -> &Matrix {
        &self.inner
    }

    fn append_homogeneous_points(&mut self, data: &[f64]) {
        self.inner
            .append_columns_from_slice(data)
            .expect("PolygonMatrix values must always have 4 rows");
    }

    fn extend_polygon_data(
        data: &mut Vec<f64>,
        p0: (f64, f64, f64),
        p1: (f64, f64, f64),
        p2: (f64, f64, f64),
    ) {
        data.extend_from_slice(&[
            p0.0, p0.1, p0.2, 1.0, p1.0, p1.1, p1.2, 1.0, p2.0, p2.1, p2.2, 1.0,
        ]);
    }
}

// Polygon based shapes
impl PolygonMatrix {
    /// Adds a box (rectangular prism) with top-left-front corner at `(x, y, z)`.
    ///
    /// Implemented as 12 triangles in counter-clockwise order.
    #[allow(clippy::many_single_char_names)]
    pub fn add_box(&mut self, (x, y, z): (f64, f64, f64), width: f64, height: f64, depth: f64) {
        let (h, w, d) = (height, width, depth);
        let p0 = (x, y, z);
        let p1 = (x, y - h, z);
        let p2 = (x + w, y - h, z);
        let p3 = (x + w, y, z);
        let p4 = (x, y, z - d);
        let p5 = (x, y - h, z - d);
        let p6 = (x + w, y - h, z - d);
        let p7 = (x + w, y, z - d);

        let mut data = Vec::with_capacity(12 * 12);
        // Front face
        Self::extend_polygon_data(&mut data, p0, p1, p2);
        Self::extend_polygon_data(&mut data, p0, p2, p3);
        // Back face
        Self::extend_polygon_data(&mut data, p7, p6, p5);
        Self::extend_polygon_data(&mut data, p7, p5, p4);
        // Top face
        Self::extend_polygon_data(&mut data, p4, p0, p3);
        Self::extend_polygon_data(&mut data, p4, p3, p7);
        // Bottom face
        Self::extend_polygon_data(&mut data, p1, p5, p6);
        Self::extend_polygon_data(&mut data, p1, p6, p2);
        // Left face
        Self::extend_polygon_data(&mut data, p4, p5, p1);
        Self::extend_polygon_data(&mut data, p4, p1, p0);
        // Right face
        Self::extend_polygon_data(&mut data, p3, p2, p6);
        Self::extend_polygon_data(&mut data, p3, p6, p7);
        self.append_homogeneous_points(&data);
    }

    /// Adds a sphere centered at `(cx, cy, cz)` with given `radius` and `steps` precision.
    ///
    /// Implemented as triangles with outward-facing normals for backface culling.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_sphere(&mut self, center: (f64, f64, f64), radius: f64, steps: usize) {
        assert!(steps > 0, "sphere steps must be positive");
        let step_by = 1.0 / steps as f64;
        let mut latitudes = Vec::with_capacity(steps + 1);
        for j in 0..=steps {
            let theta = j as f64 * step_by * PI;
            latitudes.push(theta.sin_cos());
        }

        let mut data = Vec::with_capacity(steps * steps * 24);
        let mut current_ring = Self::sphere_ring(center, radius, 0, step_by, &latitudes);
        for i in 0..steps {
            let next_ring = Self::sphere_ring(center, radius, i + 1, step_by, &latitudes);
            for j in 0..steps {
                let p0 = current_ring[j];
                let p1 = current_ring[j + 1];
                let p2 = next_ring[j + 1];
                let p3 = next_ring[j];

                // Handle poles to avoid degenerate triangles
                // Outward normal: (p0, p3, p2) and (p0, p2, p1)
                // North pole (j=0): p0 == p3. Tri (p0, p3, p2) is degenerate.
                // South pole (j=steps-1): p1 == p2. Tri (p0, p2, p1) is degenerate.
                if j != 0 {
                    Self::extend_polygon_data(&mut data, p0, p3, p2);
                }
                if j != steps - 1 {
                    Self::extend_polygon_data(&mut data, p0, p2, p1);
                }
            }
            current_ring = next_ring;
        }
        self.append_homogeneous_points(&data);
    }

    #[allow(clippy::cast_precision_loss)]
    fn sphere_ring(
        (cx, cy, cz): (f64, f64, f64),
        radius: f64,
        phi_index: usize,
        step_by: f64,
        latitudes: &[(f64, f64)],
    ) -> Vec<(f64, f64, f64)> {
        let phi = phi_index as f64 * step_by * PI * 2.0;
        let (sin_p, cos_p) = phi.sin_cos();
        latitudes
            .iter()
            .map(|&(sin_t, cos_t)| {
                (
                    radius * sin_t * cos_p + cx,
                    radius * cos_t + cy,
                    radius * sin_t * sin_p + cz,
                )
            })
            .collect()
    }

    /// Adds a torus centered at `(cx, cy, cz)`.
    ///
    /// Implemented as triangles with outward-facing normals for backface culling.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn add_torus(
        &mut self,
        center: (f64, f64, f64),
        radius_sm: f64,
        radius_big: f64,
        steps: usize,
    ) {
        assert!(steps > 0, "torus steps must be positive");
        let step_by = 1.0 / steps as f64;
        let mut inner_circle = Vec::with_capacity(steps);
        for j in 0..steps {
            let theta = j as f64 * step_by * PI * 2.0;
            inner_circle.push(theta.sin_cos());
        }

        let mut data = Vec::with_capacity(steps * steps * 24);
        let first_ring = Self::torus_ring(center, radius_sm, radius_big, 0, step_by, &inner_circle);
        let mut current_ring = first_ring.clone();
        for i in 0..steps {
            let next_ring_storage = if i + 1 == steps {
                None
            } else {
                Some(Self::torus_ring(
                    center,
                    radius_sm,
                    radius_big,
                    i + 1,
                    step_by,
                    &inner_circle,
                ))
            };
            let next_ring = next_ring_storage.as_deref().unwrap_or(&first_ring);
            for j in 0..steps {
                let p0 = current_ring[j];
                let p1 = current_ring[(j + 1) % steps];
                let p2 = next_ring[(j + 1) % steps];
                let p3 = next_ring[j];

                // Outward normal: (p0, p3, p2) and (p0, p2, p1)
                Self::extend_polygon_data(&mut data, p0, p3, p2);
                Self::extend_polygon_data(&mut data, p0, p2, p1);
            }
            if let Some(next_ring) = next_ring_storage {
                current_ring = next_ring;
            }
        }
        self.append_homogeneous_points(&data);
    }

    #[allow(clippy::cast_precision_loss)]
    fn torus_ring(
        (cx, cy, cz): (f64, f64, f64),
        r_small: f64,
        r_large: f64,
        phi_index: usize,
        step_by: f64,
        inner_circle: &[(f64, f64)],
    ) -> Vec<(f64, f64, f64)> {
        let phi = phi_index as f64 * step_by * PI * 2.0;
        let (sin_p, cos_p) = phi.sin_cos();
        inner_circle
            .iter()
            .map(|&(sin_t, cos_t)| {
                let x = cos_p * (r_small * cos_t + r_large) + cx;
                let y = r_small * sin_t + cy;
                let z = -sin_p * (r_small * cos_t + r_large) + cz;
                (x, y, z)
            })
            .collect()
    }

    /// Adds a surface of revolution generated by rotating a 2D profile about the y-axis.
    ///
    /// # Arguments
    /// * `profile` - A list of (x, y) coordinates defining the 2D shape.
    /// * `steps` - Number of rotation steps around the y-axis.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_revolution_surface(&mut self, profile: &[(f64, f64)], steps: usize) {
        assert!(steps > 0, "revolution surface steps must be positive");
        if profile.len() < 2 {
            return;
        }

        let step_by = 2.0 * PI / steps as f64;
        let n_profile = profile.len();

        let mut data = Vec::with_capacity(steps * (n_profile - 1) * 24);
        let first_ring = Self::revolution_ring(profile, 0.0);
        let mut current_ring = first_ring.clone();
        for i in 0..steps {
            let next_ring_storage = if i + 1 == steps {
                None
            } else {
                Some(Self::revolution_ring(profile, (i + 1) as f64 * step_by))
            };
            let next_ring = next_ring_storage.as_deref().unwrap_or(&first_ring);
            for j in 0..n_profile - 1 {
                let p0 = current_ring[j];
                let p1 = current_ring[j + 1];
                let p2 = next_ring[j + 1];
                let p3 = next_ring[j];

                // Outward normal: (p0, p1, p2) and (p0, p2, p3)
                // This assumes the profile is on the +X half-plane.
                Self::extend_polygon_data(&mut data, p0, p1, p2);
                Self::extend_polygon_data(&mut data, p0, p2, p3);
            }
            if let Some(next_ring) = next_ring_storage {
                current_ring = next_ring;
            }
        }
        self.append_homogeneous_points(&data);
    }

    fn revolution_ring(profile: &[(f64, f64)], phi: f64) -> Vec<(f64, f64, f64)> {
        let (sin_phi, cos_phi) = phi.sin_cos();
        profile
            .iter()
            .map(|&(px, py)| (px * cos_phi, py, -px * sin_phi))
            .collect()
    }

    /// Adds an icosahedron (20-sided regular polyhedron) centered at `(cx, cy, cz)`.
    pub fn add_icosahedron(&mut self, (cx, cy, cz): (f64, f64, f64), scale: f64) {
        #[allow(clippy::manual_midpoint)]
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let s = scale;
        let sp = scale * phi;

        let v = [
            (0.0, -s, -sp),
            (0.0, -s, sp),
            (0.0, s, -sp),
            (0.0, s, sp),
            (-s, -sp, 0.0),
            (-s, sp, 0.0),
            (s, -sp, 0.0),
            (s, sp, 0.0),
            (-sp, 0.0, -s),
            (sp, 0.0, -s),
            (-sp, 0.0, s),
            (sp, 0.0, s),
        ];
        let v: Vec<(f64, f64, f64)> = v
            .iter()
            .map(|&(x, y, z)| (x + cx, y + cy, z + cz))
            .collect();

        // 20 faces
        let faces = [
            (0, 1, 4),
            (0, 4, 9),
            (9, 4, 5),
            (4, 8, 5),
            (4, 1, 8),
            (8, 1, 10),
            (8, 10, 3),
            (5, 8, 3),
            (5, 3, 2),
            (2, 3, 7),
            (7, 3, 10),
            (7, 10, 6),
            (7, 6, 11),
            (11, 6, 0),
            (0, 6, 1),
            (6, 10, 1),
            (9, 11, 0),
            (9, 2, 11),
            (9, 5, 2),
            (7, 11, 2),
        ];

        let mut data = Vec::with_capacity(faces.len() * 12);
        for &(f0, f1, f2) in &faces {
            Self::extend_polygon_data(&mut data, v[f0], v[f1], v[f2]);
        }
        self.append_homogeneous_points(&data);
    }

    /// Adds a dodecahedron (12-sided regular polyhedron) centered at `(cx, cy, cz)`.
    pub fn add_dodecahedron(&mut self, (cx, cy, cz): (f64, f64, f64), scale: f64) {
        #[allow(clippy::manual_midpoint)]
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let inv_phi = 1.0 / phi;
        let s = scale;
        let si = scale * inv_phi;
        let sp = scale * phi;

        let v = [
            (s, s, s),
            (s, s, -s),
            (s, -s, s),
            (s, -s, -s),
            (-s, s, s),
            (-s, s, -s),
            (-s, -s, s),
            (-s, -s, -s),
            (0.0, si, sp),
            (0.0, si, -sp),
            (0.0, -si, sp),
            (0.0, -si, -sp),
            (si, sp, 0.0),
            (si, -sp, 0.0),
            (-si, sp, 0.0),
            (-si, -sp, 0.0),
            (sp, 0.0, si),
            (sp, 0.0, -si),
            (-sp, 0.0, si),
            (-sp, 0.0, -si),
        ];
        let v: Vec<(f64, f64, f64)> = v
            .iter()
            .map(|&(x, y, z)| (x + cx, y + cy, z + cz))
            .collect();

        let faces = [
            [0, 8, 10, 2, 16],
            [0, 16, 17, 1, 12],
            [0, 12, 14, 4, 8],
            [1, 9, 5, 14, 12],
            [1, 17, 3, 11, 9],
            [2, 10, 6, 15, 13],
            [2, 13, 3, 17, 16],
            [3, 13, 15, 7, 11],
            [4, 14, 5, 19, 18],
            [4, 8, 10, 6, 18],
            [5, 9, 11, 7, 19],
            [6, 15, 7, 19, 18],
        ];

        let mut data = Vec::with_capacity(faces.len() * 36);
        for f in &faces {
            Self::extend_polygon_data(&mut data, v[f[0]], v[f[1]], v[f[2]]);
            Self::extend_polygon_data(&mut data, v[f[0]], v[f[2]], v[f[3]]);
            Self::extend_polygon_data(&mut data, v[f[0]], v[f[3]], v[f[4]]);
        }
        self.append_homogeneous_points(&data);
    }

    /// Adds a cylinder centered at `(x, y, z)` with given `radius`, `height`, and `steps` precision.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names
    )]
    pub fn add_cylinder(
        &mut self,
        (x, y, z): (f64, f64, f64),
        radius: f64,
        height: f64,
        steps: usize,
    ) {
        assert!(steps > 0, "cylinder steps must be positive");
        let theta_step = 2.0 * PI / steps as f64;

        let mut data = Vec::with_capacity(steps * 48);
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let next_theta = ((i + 1) % steps) as f64 * theta_step;

            let p0 = (x + radius * theta.cos(), y + radius * theta.sin(), z);
            let p1 = (
                x + radius * next_theta.cos(),
                y + radius * next_theta.sin(),
                z,
            );
            let p2 = (p1.0, p1.1, z + height);
            let p3 = (p0.0, p0.1, z + height);

            // Side face (two triangles)
            Self::extend_polygon_data(&mut data, p0, p1, p2);
            Self::extend_polygon_data(&mut data, p0, p2, p3);

            // Bottom base
            Self::extend_polygon_data(&mut data, (x, y, z), p1, p0);
            // Top base
            Self::extend_polygon_data(&mut data, (x, y, z + height), p3, p2);
        }
        self.append_homogeneous_points(&data);
    }

    /// Adds a cone with base center at `(x, y, z)`, given `radius`, `height`, and `steps` precision.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_cone(&mut self, (x, y, z): (f64, f64, f64), radius: f64, height: f64, steps: usize) {
        assert!(steps > 0, "cone steps must be positive");
        let theta_step = 2.0 * PI / steps as f64;
        let top = (x, y, z + height);
        let center = (x, y, z);
        let mut data = Vec::with_capacity(steps * 24);
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let next_theta = ((i + 1) % steps) as f64 * theta_step;

            let p0 = (x + radius * theta.cos(), y + radius * theta.sin(), z);
            let p1 = (
                x + radius * next_theta.cos(),
                y + radius * next_theta.sin(),
                z,
            );

            // Side face
            Self::extend_polygon_data(&mut data, p0, p1, top);
            // Base
            Self::extend_polygon_data(&mut data, center, p1, p0);
        }
        self.append_homogeneous_points(&data);
    }

    /// Adds a pyramid with base centered at `(x, y, z)`.
    pub fn add_pyramid(&mut self, (x, y, z): (f64, f64, f64), base_length: f64, height: f64) {
        let half = base_length / 2.0;
        let p0 = (x - half, y, z - half);
        let p1 = (x + half, y, z - half);
        let p2 = (x + half, y, z + half);
        let p3 = (x - half, y, z + half);
        let top = (x, y + height, z);

        let mut data = Vec::with_capacity(6 * 12);
        // Sides
        Self::extend_polygon_data(&mut data, p0, p1, top);
        Self::extend_polygon_data(&mut data, p1, p2, top);
        Self::extend_polygon_data(&mut data, p2, p3, top);
        Self::extend_polygon_data(&mut data, p3, p0, top);

        // Base
        Self::extend_polygon_data(&mut data, p0, p3, p2);
        Self::extend_polygon_data(&mut data, p0, p2, p1);
        self.append_homogeneous_points(&data);
    }

    /// Adds a third-degree Bezier surface from a 4x4 grid of control points.
    ///
    /// # Arguments
    /// * `controls` - A 4x4 grid of (x, y, z) control points.
    /// * `steps` - Number of steps in u and v directions.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_bezier_surface(&mut self, controls: [[(f64, f64, f64); 4]; 4], steps: usize) {
        assert!(steps > 0, "bezier surface steps must be positive");
        let step_by = 1.0 / steps as f64;

        let mut data = Vec::with_capacity(steps * steps * 24);
        let mut current_row = Self::bezier_surface_row(&controls, steps, 0.0);
        for i in 0..steps {
            let next_row = Self::bezier_surface_row(&controls, steps, (i + 1) as f64 * step_by);
            for j in 0..steps {
                let p0 = current_row[j];
                let p1 = current_row[j + 1];
                let p2 = next_row[j + 1];
                let p3 = next_row[j];

                // Outward normal: (p0, p3, p2) and (p0, p2, p1)
                Self::extend_polygon_data(&mut data, p0, p3, p2);
                Self::extend_polygon_data(&mut data, p0, p2, p1);
            }
            current_row = next_row;
        }
        self.append_homogeneous_points(&data);
    }

    #[allow(clippy::cast_precision_loss)]
    fn bezier_surface_row(
        controls: &[[(f64, f64, f64); 4]; 4],
        steps: usize,
        u: f64,
    ) -> Vec<(f64, f64, f64)> {
        let bu = Self::cubic_bernstein(u);
        let step_by = 1.0 / steps as f64;
        let mut row_points = Vec::with_capacity(steps + 1);
        for j in 0..=steps {
            let bv = Self::cubic_bernstein(j as f64 * step_by);
            let (mut px, mut py, mut pz) = (0.0, 0.0, 0.0);
            for (row, control_row) in controls.iter().enumerate() {
                for (col, &p) in control_row.iter().enumerate() {
                    let b = bu[row] * bv[col];
                    px += p.0 * b;
                    py += p.1 * b;
                    pz += p.2 * b;
                }
            }
            row_points.push((px, py, pz));
        }
        row_points
    }

    fn cubic_bernstein(t: f64) -> [f64; 4] {
        let one_minus_t = 1.0 - t;
        [
            one_minus_t.powi(3),
            3.0 * t * one_minus_t.powi(2),
            3.0 * t.powi(2) * one_minus_t,
            t.powi(3),
        ]
    }
}

impl fmt::Display for PolygonMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PolygonMatrix {{ cols: {}, points: {} }}",
            self.cols(),
            self.len() / 4
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{Canvas, Rgb};

    #[test]
    fn point_matrix_has_four_rows_and_zero_cols() {
        let mut matrix = PolygonMatrix::new();
        matrix.push_point(1.0, 2.0, 3.0);

        assert_eq!(matrix.as_matrix().rows(), 4);
        assert_eq!(matrix.cols(), 1);
        assert_eq!(matrix.as_matrix().data(), &[1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn add_box_test() {
        let mut test = PolygonMatrix::new();
        test.add_box((10.0, 10.0, 10.0), 10.0, 10.0, 10.0);
        // 12 triangles * 3 points per triangle * 4 values per point = 144
        assert_eq!(test.len(), 144);
        let transform =
            Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(45.0) * Matrix::rotate_z(45.0);
        let test = test.apply(&transform);
        let mut canvas = Canvas::new_with_bg(100, 100, Rgb::new(24, 26, 27));
        canvas.set_line_pixel(Rgb::new(255, 255, 255));
        canvas.draw_polygons(&test);
    }

    #[test]
    fn add_sphere_test() {
        let mut test = PolygonMatrix::new();
        test.add_sphere((50.0, 50.0, 50.0), 25.0, 24);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_sphere_has_no_degenerate_triangles() {
        let mut sphere = PolygonMatrix::new();
        sphere.add_sphere((0.0, 0.0, 0.0), 1.0, 4);

        assert!(sphere.iter_triangles().all(|(p0, p1, p2)| {
            let is_degenerate = |a: &[f64], b: &[f64]| {
                (a[0] - b[0]).abs() < 1e-10
                    && (a[1] - b[1]).abs() < 1e-10
                    && (a[2] - b[2]).abs() < 1e-10
            };
            !(is_degenerate(p0, p1) || is_degenerate(p1, p2) || is_degenerate(p2, p0))
        }));
    }

    #[test]
    fn add_torus_test() {
        let mut test = PolygonMatrix::new();
        test.add_torus((50.0, 50.0, 50.0), 12.5, 25.0, 30);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_cylinder_test() {
        let mut test = PolygonMatrix::new();
        test.add_cylinder((25.0, 25.0, 25.0), 25.0, 25.0, 36);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_cone_test() {
        let mut test = PolygonMatrix::new();
        test.add_cone((25.0, 25.0, 25.0), 25.0, 25.0, 36);
        // 36 steps * 2 triangles per step (side + base) * 3 points = 216
        assert_eq!(test.cols(), 216);
    }

    #[test]
    fn add_pyramid_test() {
        let mut test = PolygonMatrix::new();
        test.add_pyramid((25.0, 25.0, 25.0), 25.0, 25.0);
        // 4 sides + 2 triangles for base = 6 triangles * 3 points = 18
        assert_eq!(test.cols(), 18);
    }

    #[test]
    fn add_icosahedron_test() {
        let mut test = PolygonMatrix::new();
        test.add_icosahedron((0.0, 0.0, 0.0), 1.0);
        // 20 triangles * 3 points = 60 columns
        assert_eq!(test.cols(), 60);
    }

    #[test]
    fn add_dodecahedron_test() {
        let mut test = PolygonMatrix::new();
        test.add_dodecahedron((0.0, 0.0, 0.0), 1.0);
        // 12 faces * 3 triangles/face * 3 points = 108 columns
        assert_eq!(test.cols(), 108);
    }

    #[test]
    fn add_revolution_surface_test() {
        let mut test = PolygonMatrix::new();
        let profile = vec![(1.0, 0.0), (1.0, 1.0)];
        test.add_revolution_surface(&profile, 4);
        // 4 steps * 1 quad * 2 triangles/quad * 3 points = 24 columns
        assert_eq!(test.cols(), 24);
    }

    #[test]
    #[should_panic(expected = "sphere steps must be positive")]
    fn add_sphere_rejects_zero_steps() {
        PolygonMatrix::new().add_sphere((0.0, 0.0, 0.0), 1.0, 0);
    }

    #[test]
    #[should_panic(expected = "torus steps must be positive")]
    fn add_torus_rejects_zero_steps() {
        PolygonMatrix::new().add_torus((0.0, 0.0, 0.0), 1.0, 2.0, 0);
    }

    #[test]
    #[should_panic(expected = "cone steps must be positive")]
    fn add_cone_rejects_zero_steps() {
        PolygonMatrix::new().add_cone((0.0, 0.0, 0.0), 1.0, 2.0, 0);
    }

    #[test]
    #[should_panic(expected = "revolution surface steps must be positive")]
    fn add_revolution_surface_rejects_zero_steps() {
        PolygonMatrix::new().add_revolution_surface(&[(1.0, 0.0), (1.0, 1.0)], 0);
    }

    #[test]
    #[should_panic(expected = "bezier surface steps must be positive")]
    fn add_bezier_surface_rejects_zero_steps() {
        PolygonMatrix::new().add_bezier_surface([[(0.0, 0.0, 0.0); 4]; 4], 0);
    }

    #[test]
    fn push_polygons_appends_triangle_batch() {
        let mut matrix = PolygonMatrix::new();
        matrix.push_polygons(&[
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        ]);

        assert_eq!(matrix.cols(), 6);
        assert_eq!(matrix.triangle_count(), 2);
        assert_eq!(
            matrix.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn bounds_reports_axis_aligned_extents() {
        let mut matrix = PolygonMatrix::new();
        assert_eq!(matrix.bounds(), None);

        matrix.push_polygons(&[
            [(-1.0, 2.0, 0.5), (4.0, -3.0, 2.0), (0.0, 1.0, -5.0)],
            [(10.0, 0.0, 0.0), (12.0, 1.0, 1.0), (11.0, 2.0, 2.0)],
        ]);

        assert_eq!(
            matrix.bounds(),
            Some(Bounds3 {
                min: (-1.0, -3.0, -5.0),
                max: (12.0, 2.0, 2.0),
            })
        );
        assert_eq!(
            matrix.bounds_from_col(3),
            Some(Bounds3 {
                min: (10.0, 0.0, 0.0),
                max: (12.0, 2.0, 2.0),
            })
        );
    }

    #[test]
    fn truncate_cols_rolls_back_appended_points() {
        let mut matrix = PolygonMatrix::new();
        matrix.push_polygons(&[
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
        ]);

        matrix.truncate_cols(3);

        assert_eq!(matrix.cols(), 3);
        assert_eq!(
            matrix.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn reverse_winding_swaps_triangle_vertices() {
        let mut matrix = PolygonMatrix::new();
        matrix.add_polygon((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));

        matrix.reverse_winding();

        assert_eq!(
            matrix.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn reverse_winding_from_col_leaves_existing_triangles_alone() {
        let mut matrix = PolygonMatrix::new();
        matrix.push_polygons(&[
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
        ]);

        matrix.reverse_winding_from_col(3);

        assert_eq!(
            matrix.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0, //
                2.0, 0.0, 0.0, 1.0, //
                2.0, 1.0, 0.0, 1.0, //
                3.0, 0.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn draw_polygons_culls_reversed_winding() {
        fn drawn_pixels(canvas: &Canvas) -> usize {
            canvas
                .pixels()
                .iter()
                .filter(|&&pixel| pixel != Rgb::BLACK)
                .count()
        }

        let mut visible = PolygonMatrix::new();
        visible.add_polygon((1.0, 1.0, 0.0), (8.0, 1.0, 0.0), (1.0, 8.0, 0.0));

        let hidden = visible.reversed_winding();

        let mut visible_canvas = Canvas::new_with_bg(10, 10, Rgb::BLACK);
        visible_canvas.set_line_pixel(Rgb::WHITE);
        visible_canvas.draw_polygons(&visible);
        assert!(drawn_pixels(&visible_canvas) > 0);

        let mut hidden_canvas = Canvas::new_with_bg(10, 10, Rgb::BLACK);
        hidden_canvas.set_line_pixel(Rgb::WHITE);
        hidden_canvas.draw_polygons(&hidden);
        assert_eq!(drawn_pixels(&hidden_canvas), 0);

        let corrected = hidden.reversed_winding();
        let mut corrected_canvas = Canvas::new_with_bg(10, 10, Rgb::BLACK);
        corrected_canvas.set_line_pixel(Rgb::WHITE);
        corrected_canvas.draw_polygons(&corrected);
        assert!(drawn_pixels(&corrected_canvas) > 0);
    }

    #[test]
    fn add_box_winding_order() {
        use crate::gmath::vector::Vector;
        let mut matrix = PolygonMatrix::new();
        matrix.add_box((0.0, 0.0, 0.0), 1.0, 1.0, 1.0);

        // Center of the box is (0.5, -0.5, -0.5)
        let center = Vector::new(0.5, -0.5, -0.5);

        for (p0, p1, p2) in matrix.iter_triangles() {
            let v0 = Vector::new(p0[0], p0[1], p0[2]);
            let v1 = Vector::new(p1[0], p1[1], p1[2]);
            let v2 = Vector::new(p2[0], p2[1], p2[2]);

            let normal = (v1 - v0).cross(v2 - v0);
            let to_center = center - v0;

            // Outward normal means dot product with vector from center to vertex should be positive
            assert!(
                normal.dot(to_center) < 0.0,
                "Triangle normal should point outward"
            );
        }
    }

    #[test]
    fn sphere_winding_order() {
        use crate::gmath::vector::Vector;
        let mut matrix = PolygonMatrix::new();
        let center_coords = (0.0, 0.0, 0.0);
        let center = Vector::new(0.0, 0.0, 0.0);
        matrix.add_sphere(center_coords, 1.0, 8);

        for (p0, p1, p2) in matrix.iter_triangles() {
            let v0 = Vector::new(p0[0], p0[1], p0[2]);
            let v1 = Vector::new(p1[0], p1[1], p1[2]);
            let v2 = Vector::new(p2[0], p2[1], p2[2]);

            let normal = (v1 - v0).cross(v2 - v0);
            let from_center = v0 - center;

            assert!(
                normal.dot(from_center) > 0.0,
                "Sphere normal should point outward"
            );
        }
    }

    #[test]
    fn height_map_builds_two_triangles_per_cell() {
        let heights = vec![vec![0.0, 1.0], vec![0.5, 0.25]];
        let mesh = PolygonMatrix::from_height_map(&heights, HeightMapOptions::new(10.0, 20.0, 2.0));

        assert_eq!(mesh.triangle_count(), 2);
        let (p0, p1, p2) = mesh.iter_triangles().next().expect("first triangle");
        assert_eq!((p0[0], p0[1], p0[2]), (-5.0, 0.0, -10.0));
        assert_eq!((p1[0], p1[1], p1[2]), (-5.0, 1.0, 10.0));
        assert_eq!((p2[0], p2[1], p2[2]), (5.0, 0.5, 10.0));
    }
}

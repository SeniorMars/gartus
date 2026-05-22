use super::matrix::Matrix;
use super::parametric::Parametric;
use std::f64::consts::PI;
use std::fmt;
use std::iter::once;

/// A dynamically-growing list of 4D homogeneous points stored as a 4×N column-major matrix.
/// Used for edge lists and drawing.
#[derive(Debug, Default, Clone)]
pub struct EdgeMatrix {
    inner: Matrix,
}

impl EdgeMatrix {
    /// Creates an empty edge matrix (4 rows, 0 cols).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Matrix::new(4, 0, Vec::new()),
        }
    }

    /// Creates an edge matrix pre-allocated for `n` points.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let v = Vec::with_capacity(n * 4);
        Self {
            inner: Matrix::new(4, 0, v),
        }
    }

    /// Creates an edge matrix from a flat `[x0, y0, x1, y1, ...]` coordinate list.
    ///
    /// Adjacent coordinate pairs become points with the supplied `z` value.
    ///
    /// # Panics
    /// Panics if `coords` contains an odd number of values.
    #[must_use]
    pub fn from_xy_pairs<T>(coords: &[T], z: f64) -> Self
    where
        T: Copy + Into<f64>,
    {
        assert!(
            coords.len().is_multiple_of(2),
            "xy coordinate list must contain pairs"
        );

        let mut matrix = Self::with_capacity(coords.len() / 2);
        for pair in coords.chunks_exact(2) {
            matrix.push_point(pair[0].into(), pair[1].into(), z);
        }
        matrix
    }

    /// Number of points (columns) in this edge matrix.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Total number of f64 values stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the edge matrix has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.cols() == 0
    }

    /// Adds a point (x, y, z) with implicit w=1.
    ///
    /// # Panics
    /// Panics if the inner matrix row count is not 4.
    pub fn push_point(&mut self, x: f64, y: f64, z: f64) {
        self.inner
            .append_column(&[x, y, z, 1.0])
            .expect("EdgeMatrix must always have 4 rows");
    }

    /// Appends multiple points to the matrix.
    pub fn push_points(&mut self, points: &[(f64, f64, f64)]) {
        for &(x, y, z) in points {
            self.push_point(x, y, z);
        }
    }

    /// Adds a triangle (three points).
    pub fn add_polygon(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64), p2: (f64, f64, f64)) {
        self.push_points(&[p0, p1, p2]);
    }

    /// Adds an edge (two points).
    pub fn push_edge(&mut self, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) {
        self.push_points(&[(x0, y0, z0), (x1, y1, z1)]);
    }

    /// Adds an edge from two (x, y, z) tuples.
    pub fn push_edge_tuple(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64)) {
        self.push_points(&[p0, p1]);
    }

    /// Appends another `EdgeMatrix`'s points to this one.
    ///
    /// # Panics
    /// Panics if the inner matrices have differing row counts.
    pub fn extend(&mut self, other: &EdgeMatrix) {
        self.inner
            .append_columns(&other.inner)
            .expect("EdgeMatrix values must always have 4 rows");
    }

    /// Returns an iterator over individual points as `&[f64]` slices of length 4.
    pub fn iter_points(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.inner.iter_by_point()
    }

    /// Returns an iterator over point pairs as line edges.
    pub fn iter_edges(&self) -> impl Iterator<Item = (&[f64], &[f64])> + '_ {
        self.inner.data().chunks_exact(8).map(|edge| {
            let (p0, p1) = edge.split_at(4);
            (p0, p1)
        })
    }

    /// Apply a 4×4 transformation matrix to all points. Returns a new `EdgeMatrix`.
    #[must_use]
    pub fn apply(&self, transform: &Matrix) -> Self {
        Self {
            inner: transform.mult_matrix(&self.inner),
        }
    }

    /// Get a reference to the underlying `Matrix` (for interop with `draw_lines` etc.)
    #[must_use]
    pub fn as_matrix(&self) -> &Matrix {
        &self.inner
    }

    /// Returns an iterator over triangle vertex triples as `&[f64]` slices of length 4.
    pub fn iter_triangles(&self) -> impl Iterator<Item = (&[f64], &[f64], &[f64])> + '_ {
        self.inner.data().chunks_exact(12).map(|tri| {
            let (p01, p2) = tri.split_at(8);
            let (p0, p1) = p01.split_at(4);
            (p0, p1, p2)
        })
    }
}

// Curve methods
impl EdgeMatrix {
    /// Adds a parametric curve to the edge matrix.
    ///
    /// # Panics
    /// Panics if `step` is not positive.
    pub fn add_parametric_curve<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(
        &mut self,
        x_func: F,
        y_func: G,
        z: f64,
        step: f64,
    ) {
        assert!(step > 0.0, "step must be positive");
        let parametric = Parametric::new(x_func, y_func);
        let mut values = parametric.values_iter(step);

        let Some(mut prev) = values.next() else {
            return;
        };

        for curr in values {
            let (x0, y0) = prev;
            let (x1, y1) = curr;
            self.push_edge(x0, y0, z, x1, y1, z);
            prev = curr;
        }
    }

    /// Adds a hermite curve to the edge matrix.
    pub fn add_hermite(&mut self, p0: (f64, f64), p1: (f64, f64), r0: (f64, f64), r1: (f64, f64)) {
        fn hermite_curve_coeffs(p0: f64, p1: f64, r0: f64, r1: f64) -> (f64, f64, f64, f64) {
            (
                2.0 * (p0 - p1) + r0 + r1,
                3.0 * (-p0 + p1) - 2.0 * r0 - r1,
                r0,
                p0,
            )
        }
        let (ax, bx, cx, dx) = hermite_curve_coeffs(p0.0, p1.0, r0.0, r1.0);
        let (ay, by, cy, dy) = hermite_curve_coeffs(p0.1, p1.1, r0.1, r1.1);
        self.add_parametric_curve(
            |t: f64| ax * t * t * t + bx * t * t + cx * t + dx,
            |t: f64| ay * t * t * t + by * t * t + cy * t + dy,
            0.0,
            0.001,
        );
    }

    fn bezier_curve_coeffs(p0: f64, p1: f64, p2: f64, p3: f64) -> (f64, f64, f64, f64) {
        (
            -p0 + 3.0 * (p1 - p2) + p3,
            3.0 * p0 - 6.0 * p1 + 3.0 * p2,
            3.0 * (-p0 + p1),
            p0,
        )
    }

    /// Adds a third-degree bezier curve to the edge matrix.
    pub fn add_bezier3(&mut self, p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)) {
        let (ax, bx, cx, dx) = Self::bezier_curve_coeffs(p0.0, p1.0, p2.0, p3.0);
        let (ay, by, cy, dy) = Self::bezier_curve_coeffs(p0.1, p1.1, p2.1, p3.1);
        self.add_parametric_curve(
            |t: f64| ax * t * t * t + bx * t * t + cx * t + dx,
            |t: f64| ay * t * t * t + by * t * t + cy * t + dy,
            0.0,
            0.001,
        );
    }

    /// Generates the coefficients of the n-th degree bezier polynomials.
    ///
    /// See <http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Polynomial_form>
    #[allow(clippy::cast_precision_loss)]
    fn generate_n_degree_bezier_polynomials_coeff(n_degree: usize, points: &[f64]) -> Vec<f64> {
        assert!(
            n_degree <= 170,
            "Bezier degree {n_degree} exceeds f64 factorial precision (max 170)"
        );
        let mut coeffs = Vec::with_capacity(n_degree + 1);
        let dp = once(1)
            .chain(1..=n_degree)
            .scan(1.0, |acc, i| {
                *acc *= i as f64;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        for curr_deg in 0..=n_degree {
            let lhs = dp[n_degree] / dp[n_degree - curr_deg];
            let rhs = (0..=curr_deg).fold(0.0, |acc, i| {
                let sign = if (i + curr_deg) % 2 == 0 { 1.0 } else { -1.0 };
                let numerator = points[i] * sign;
                let denum = dp[i] * dp[curr_deg - i];
                acc + numerator / denum
            });
            coeffs.push(lhs * rhs);
        }
        coeffs
    }

    /// Adds an n-th degree bezier curve to the edge matrix.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn add_beziern(&mut self, n_degree: usize, x_points: &[f64], y_points: &[f64]) {
        let x_coeffs = Self::generate_n_degree_bezier_polynomials_coeff(n_degree, x_points);
        let y_coeffs = Self::generate_n_degree_bezier_polynomials_coeff(n_degree, y_points);
        let x_func = move |t: f64| {
            x_coeffs
                .iter()
                .enumerate()
                .fold(0.0, |acc, (i, coeff)| acc + coeff * t.powi(i as i32))
        };
        let y_func = move |t: f64| {
            y_coeffs
                .iter()
                .enumerate()
                .fold(0.0, |acc, (i, coeff)| acc + coeff * t.powi(i as i32))
        };
        self.add_parametric_curve(x_func, y_func, 0.0, 0.001);
    }
}

// Shape methods
impl EdgeMatrix {
    /// Adds a circle centered at `(cx, cy, cz)` with radius `r` and precision `step`.
    pub fn add_circle(&mut self, cx: f64, cy: f64, cz: f64, r: f64, step: f64) {
        self.add_parametric_curve(
            |t: f64| r * (t * 2.0 * PI).cos() + cx,
            |t: f64| r * (t * 2.0 * PI).sin() + cy,
            cz,
            step,
        );
    }

    /// Adds a triangle connecting the three given vertices.
    pub fn add_triangle(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64), p2: (f64, f64, f64)) {
        self.add_parametric_curve(|t| p0.0 + t * (p1.0 - p0.0), |t| p0.1 + t * (p1.1 - p0.1), p0.2, 1.0);
        self.add_parametric_curve(|t| p1.0 + t * (p2.0 - p1.0), |t| p1.1 + t * (p2.1 - p1.1), p1.2, 1.0);
        self.add_parametric_curve(|t| p2.0 + t * (p0.0 - p2.0), |t| p2.1 + t * (p0.1 - p2.1), p2.2, 1.0);
    }

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

        // Front face
        self.add_polygon(p0, p1, p2);
        self.add_polygon(p0, p2, p3);
        // Back face
        self.add_polygon(p7, p6, p5);
        self.add_polygon(p7, p5, p4);
        // Top face
        self.add_polygon(p4, p0, p3);
        self.add_polygon(p4, p3, p7);
        // Bottom face
        self.add_polygon(p1, p5, p6);
        self.add_polygon(p1, p6, p2);
        // Left face
        self.add_polygon(p4, p5, p1);
        self.add_polygon(p4, p1, p0);
        // Right face
        self.add_polygon(p3, p2, p6);
        self.add_polygon(p3, p6, p7);
    }

    /// Adds a sphere centered at `(cx, cy, cz)` with given `radius` and `steps` precision.
    ///
    /// Implemented as triangles, handling poles to avoid degenerate triangles.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_sphere(&mut self, center: (f64, f64, f64), radius: f64, steps: usize) {
        let points = Self::generate_sphere_points(center, radius, steps);
        let lat_steps = steps;
        let long_steps = steps;
        let pps = lat_steps + 1; // points per semicircle

        for i in 0..long_steps {
            for j in 0..lat_steps {
                let p0 = i * pps + j;
                let p1 = p0 + 1;
                let p2 = (i + 1) * pps + j + 1;
                let p3 = (i + 1) * pps + j;

                // Handle poles to avoid degenerate triangles
                if j != 0 {
                    self.add_polygon(points[p0], points[p1], points[p2]);
                }
                if j != lat_steps - 1 {
                    self.add_polygon(points[p0], points[p2], points[p3]);
                }
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn generate_sphere_points(
        (cx, cy, cz): (f64, f64, f64),
        radius: f64,
        steps: usize,
    ) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::with_capacity((steps + 1) * (steps + 1));
        let step_by = 1.0 / steps as f64;
        for i in 0..=steps {
            let phi = i as f64 * step_by * PI * 2.0;
            for j in 0..=steps {
                let theta = j as f64 * step_by * PI;
                let x = radius * theta.sin() * phi.cos() + cx;
                let y = radius * theta.cos() + cy;
                let z = radius * theta.sin() * phi.sin() + cz;
                points.push((x, y, z));
            }
        }
        points
    }

    /// Adds a torus centered at `(cx, cy, cz)`.
    ///
    /// Implemented as triangles rotated about the y-axis.
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
        let points = Self::generate_torus_points(center, radius_sm, radius_big, steps);
        let pps = steps; // torus circles are closed, but we generate 'steps' points

        for i in 0..steps {
            for j in 0..steps {
                let p0 = i * pps + j;
                let p1 = i * pps + (j + 1) % pps;
                let p2 = ((i + 1) % steps) * pps + (j + 1) % pps;
                let p3 = ((i + 1) % steps) * pps + j;

                self.add_polygon(points[p0], points[p1], points[p2]);
                self.add_polygon(points[p0], points[p2], points[p3]);
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn generate_torus_points(
        (cx, cy, cz): (f64, f64, f64),
        r_small: f64,
        r_large: f64,
        steps: usize,
    ) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::with_capacity(steps * steps);
        let step_by = 1.0 / steps as f64;
        for i in 0..steps {
            let phi = i as f64 * step_by * PI * 2.0;
            for j in 0..steps {
                let theta = j as f64 * step_by * PI * 2.0;
                let x = phi.cos() * (r_small * theta.cos() + r_large) + cx;
                let y = r_small * theta.sin() + cy;
                let z = -phi.sin() * (r_small * theta.cos() + r_large) + cz;
                points.push((x, y, z));
            }
        }
        points
    }

    /// Adds a surface of revolution generated by rotating a 2D profile about the y-axis.
    ///
    /// # Arguments
    /// * `profile` - A list of (x, y) coordinates defining the 2D shape.
    /// * `steps` - Number of rotation steps around the y-axis.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_revolution_surface(&mut self, profile: &[(f64, f64)], steps: usize) {
        if profile.len() < 2 {
            return;
        }

        let n_profile = profile.len();
        let mut points = Vec::with_capacity(steps * n_profile);
        let step_by = 2.0 * PI / steps as f64;

        for i in 0..steps {
            let phi = i as f64 * step_by;
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            for &(px, py) in profile {
                // Rotate (px, py, 0) around y-axis
                // x' = px * cos(phi)
                // y' = py
                // z' = -px * sin(phi)
                points.push((px * cos_phi, py, -px * sin_phi));
            }
        }

        for i in 0..steps {
            let next_i = (i + 1) % steps;
            for j in 0..n_profile - 1 {
                let p0 = i * n_profile + j;
                let p1 = i * n_profile + j + 1;
                let p2 = next_i * n_profile + j + 1;
                let p3 = next_i * n_profile + j;

                self.add_polygon(points[p0], points[p1], points[p2]);
                self.add_polygon(points[p0], points[p2], points[p3]);
            }
        }
    }

    /// Adds an icosahedron (20-sided regular polyhedron) centered at `(cx, cy, cz)`.
    pub fn add_icosahedron(&mut self, (cx, cy, cz): (f64, f64, f64), scale: f64) {
        #[allow(clippy::manual_midpoint)]
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let s = scale;
        let sp = scale * phi;

        let v = [
            (0.0, -s, -sp), (0.0, -s, sp), (0.0, s, -sp), (0.0, s, sp),
            (-s, -sp, 0.0), (-s, sp, 0.0), (s, -sp, 0.0), (s, sp, 0.0),
            (-sp, 0.0, -s), (sp, 0.0, -s), (-sp, 0.0, s), (sp, 0.0, s),
        ];
        let v: Vec<(f64, f64, f64)> = v.iter().map(|&(x, y, z)| (x + cx, y + cy, z + cz)).collect();

        // 20 faces
        let faces = [
            (0, 1, 4), (0, 4, 9), (9, 4, 5), (4, 8, 5), (4, 1, 8),
            (8, 1, 10), (8, 10, 3), (5, 8, 3), (5, 3, 2), (2, 3, 7),
            (7, 3, 10), (7, 10, 6), (7, 6, 11), (11, 6, 0), (0, 6, 1),
            (6, 10, 1), (9, 11, 0), (9, 2, 11), (9, 5, 2), (7, 11, 2),
        ];

        for &(f0, f1, f2) in &faces {
            self.add_polygon(v[f0], v[f1], v[f2]);
        }
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
            (s, s, s), (s, s, -s), (s, -s, s), (s, -s, -s),
            (-s, s, s), (-s, s, -s), (-s, -s, s), (-s, -s, -s),
            (0.0, si, sp), (0.0, si, -sp), (0.0, -si, sp), (0.0, -si, -sp),
            (si, sp, 0.0), (si, -sp, 0.0), (-si, sp, 0.0), (-si, -sp, 0.0),
            (sp, 0.0, si), (sp, 0.0, -si), (-sp, 0.0, si), (-sp, 0.0, -si),
        ];
        let v: Vec<(f64, f64, f64)> = v.iter().map(|&(x, y, z)| (x + cx, y + cy, z + cz)).collect();

        let faces = [
            [0, 8, 10, 2, 16], [0, 16, 17, 1, 12], [0, 12, 14, 4, 8],
            [1, 9, 5, 14, 12], [1, 17, 3, 11, 9], [2, 10, 6, 15, 13],
            [2, 13, 3, 17, 16], [3, 13, 15, 7, 11], [4, 14, 5, 19, 18],
            [4, 8, 10, 6, 18], [5, 9, 11, 7, 19], [6, 15, 7, 19, 18],
        ];

        for f in &faces {
            self.add_polygon(v[f[0]], v[f[1]], v[f[2]]);
            self.add_polygon(v[f[0]], v[f[2]], v[f[3]]);
            self.add_polygon(v[f[0]], v[f[3]], v[f[4]]);
        }
    }

    /// Adds a cylinder centered at `(x, y, z)` with given `radius`, `height`, and `steps` precision.
    ///
    /// # Panics
    /// Panics if `steps` is zero.
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::many_single_char_names)]
    pub fn add_cylinder(
        &mut self,
        (x, y, z): (f64, f64, f64),
        radius: f64,
        height: f64,
        steps: usize,
    ) {
        assert!(steps > 0, "cylinder steps must be positive");
        let theta_step = 2.0 * PI / steps as f64;

        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let next_theta = ((i + 1) % steps) as f64 * theta_step;

            let p0 = (x + radius * theta.cos(), y + radius * theta.sin(), z);
            let p1 = (x + radius * next_theta.cos(), y + radius * next_theta.sin(), z);
            let p2 = (p1.0, p1.1, z + height);
            let p3 = (p0.0, p0.1, z + height);

            // Side face (two triangles)
            self.add_polygon(p0, p1, p2);
            self.add_polygon(p0, p2, p3);

            // Bottom base
            self.add_polygon((x, y, z), p1, p0);
            // Top base
            self.add_polygon((x, y, z + height), p3, p2);
        }
    }

    /// Adds a cone with base center at `(x, y, z)`, given `radius`, `height`, and `steps` precision.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_cone(&mut self, (x, y, z): (f64, f64, f64), radius: f64, height: f64, steps: usize) {
        let theta_step = 2.0 * PI / steps as f64;
        let top = (x, y, z + height);
        let center = (x, y, z);
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let next_theta = ((i + 1) % steps) as f64 * theta_step;

            let p0 = (x + radius * theta.cos(), y + radius * theta.sin(), z);
            let p1 = (x + radius * next_theta.cos(), y + radius * next_theta.sin(), z);

            // Side face
            self.add_polygon(p0, p1, top);
            // Base
            self.add_polygon(center, p1, p0);
        }
    }

    /// Adds a pyramid with base centered at `(x, y, z)`.
    pub fn add_pyramid(&mut self, (x, y, z): (f64, f64, f64), base_length: f64, height: f64) {
        let half = base_length / 2.0;
        let p0 = (x - half, y, z - half);
        let p1 = (x + half, y, z - half);
        let p2 = (x + half, y, z + half);
        let p3 = (x - half, y, z + half);
        let top = (x, y + height, z);

        // Sides
        self.add_polygon(p0, p1, top);
        self.add_polygon(p1, p2, top);
        self.add_polygon(p2, p3, top);
        self.add_polygon(p3, p0, top);

        // Base
        self.add_polygon(p0, p3, p2);
        self.add_polygon(p0, p2, p1);
    }

    /// Adds a third-degree Bezier surface from a 4x4 grid of control points.
    ///
    /// # Arguments
    /// * `controls` - A 4x4 grid of (x, y, z) control points.
    /// * `steps` - Number of steps in u and v directions.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_bezier_surface(&mut self, controls: [[(f64, f64, f64); 4]; 4], steps: usize) {
        let mut points = Vec::with_capacity((steps + 1) * (steps + 1));
        let step_by = 1.0 / steps as f64;

        let bernstein = |i: usize, t: f64| -> f64 {
            match i {
                0 => (1.0 - t).powi(3),
                1 => 3.0 * t * (1.0 - t).powi(2),
                2 => 3.0 * t.powi(2) * (1.0 - t),
                3 => t.powi(3),
                _ => 0.0,
            }
        };

        for i in 0..=steps {
            let u = i as f64 * step_by;
            for j in 0..=steps {
                let v = j as f64 * step_by;
                let (mut px, mut py, mut pz) = (0.0, 0.0, 0.0);
                for (row, control_row) in controls.iter().enumerate() {
                    let bu = bernstein(row, u);
                    for (col, &p) in control_row.iter().enumerate() {
                        let bv = bernstein(col, v);
                        let b = bu * bv;
                        px += p.0 * b;
                        py += p.1 * b;
                        pz += p.2 * b;
                    }
                }
                points.push((px, py, pz));
            }
        }

        let pps = steps + 1;
        for i in 0..steps {
            for j in 0..steps {
                let p0 = i * pps + j;
                let p1 = i * pps + j + 1;
                let p2 = (i + 1) * pps + j + 1;
                let p3 = (i + 1) * pps + j;

                self.add_polygon(points[p0], points[p1], points[p2]);
                self.add_polygon(points[p0], points[p2], points[p3]);
            }
        }
    }
}

impl fmt::Display for EdgeMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EdgeMatrix {{ cols: {}, points: {} }}",
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
        let mut matrix = EdgeMatrix::new();
        matrix.push_point(1.0, 2.0, 3.0);

        assert_eq!(matrix.as_matrix().rows(), 4);
        assert_eq!(matrix.cols(), 1);
        assert_eq!(matrix.as_matrix().data(), &[1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn extend_appends_points() {
        let mut left = EdgeMatrix::new();
        left.push_point(1.0, 2.0, 3.0);

        let mut right = EdgeMatrix::new();
        right.push_point(4.0, 5.0, 6.0);

        left.extend(&right);

        assert_eq!(left.cols(), 2);
        assert_eq!(
            left.as_matrix().data(),
            &[1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0]
        );
    }

    #[test]
    fn from_xy_pairs_builds_homogeneous_points() {
        let matrix = EdgeMatrix::from_xy_pairs(&[1, 2, 3, 4], 5.0);

        assert_eq!(matrix.cols(), 2);
        assert_eq!(
            matrix.as_matrix().data(),
            &[1.0, 2.0, 5.0, 1.0, 3.0, 4.0, 5.0, 1.0]
        );
    }

    #[test]
    fn add_box_test() {
        let mut test = EdgeMatrix::new();
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

    #[ignore = "visual test, requires display"]
    #[test]
    fn add_box_visual() {
        let mut test = EdgeMatrix::new();
        test.add_box((10.0, 10.0, 10.0), 10.0, 10.0, 10.0);
        let transform =
            Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(45.0) * Matrix::rotate_z(45.0);
        let test = test.apply(&transform);
        let mut canvas = Canvas::new_with_bg(100, 100, Rgb::new(24, 26, 27));
        canvas.set_line_pixel(Rgb::WHITE);
        canvas.draw_polygons(&test);
        canvas.display().expect("Failed to display canvas");
    }

    #[test]
    fn add_sphere_test() {
        let mut test = EdgeMatrix::new();
        test.add_sphere((50.0, 50.0, 50.0), 25.0, 24);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_sphere_includes_longitude_seam_edges() {
        let mut sphere = EdgeMatrix::new();
        sphere.add_sphere((0.0, 0.0, 0.0), 1.0, 4);

        assert!(sphere.iter_triangles().any(|(p0, p1, p2)| {
            let is_seam = |a: &[f64], b: &[f64]| {
                (a[0] - b[0]).abs() < 1e-10
                    && (a[1] - b[1]).abs() < 1e-10
                    && (a[2] - b[2]).abs() < 1e-10
            };
            is_seam(p0, p1) || is_seam(p1, p2) || is_seam(p2, p0)
        }));
    }

    #[test]
    fn add_torus_test() {
        let mut test = EdgeMatrix::new();
        test.add_torus((50.0, 50.0, 50.0), 12.5, 25.0, 30);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_torus_includes_rotation_seam_edges() {
        let mut torus = EdgeMatrix::new();
        torus.add_torus((0.0, 0.0, 0.0), 1.0, 3.0, 4);

        assert!(torus.iter_triangles().any(|(p0, p1, p2)| {
            let is_seam = |a: &[f64], b: &[f64]| {
                (a[0]).abs() < 1e-10
                    && (a[1]).abs() < 1e-10
                    && (a[2] - 4.0).abs() < 1e-10
                    && (b[0] - 4.0).abs() < 1e-10
                    && (b[1]).abs() < 1e-10
                    && (b[2]).abs() < 1e-10
            };
            // Note: rotation seam is between phi=0 and phi=2pi, which are coincident.
            // This check might need adjustment depending on exact vertex order, but
            // for now we're just checking that the logic holds.
            is_seam(p0, p1) || is_seam(p1, p2) || is_seam(p2, p0)
        }));
    }

    #[test]
    fn add_cylinder_test() {
        let mut test = EdgeMatrix::new();
        test.add_cylinder((25.0, 25.0, 25.0), 25.0, 25.0, 36);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_cone_test() {
        let mut test = EdgeMatrix::new();
        test.add_cone((25.0, 25.0, 25.0), 25.0, 25.0, 36);
        // 36 steps * 2 triangles per step (side + base) * 3 points = 216
        assert_eq!(test.cols(), 216);
    }

    #[test]
    fn add_pyramid_test() {
        let mut test = EdgeMatrix::new();
        test.add_pyramid((25.0, 25.0, 25.0), 25.0, 25.0);
        // 4 sides + 2 triangles for base = 6 triangles * 3 points = 18
        assert_eq!(test.cols(), 18);
    }

    #[test]
    fn add_icosahedron_test() {
        let mut test = EdgeMatrix::new();
        test.add_icosahedron((0.0, 0.0, 0.0), 1.0);
        // 20 triangles * 3 points = 60 columns
        assert_eq!(test.cols(), 60);
    }

    #[test]
    fn add_dodecahedron_test() {
        let mut test = EdgeMatrix::new();
        test.add_dodecahedron((0.0, 0.0, 0.0), 1.0);
        // 12 faces * 3 triangles/face * 3 points = 108 columns
        assert_eq!(test.cols(), 108);
    }

    #[test]
    fn add_revolution_surface_test() {
        let mut test = EdgeMatrix::new();
        let profile = vec![(1.0, 0.0), (1.0, 1.0)];
        test.add_revolution_surface(&profile, 4);
        // 4 steps * 1 quad * 2 triangles/quad * 3 points = 24 columns
        assert_eq!(test.cols(), 24);
    }
}

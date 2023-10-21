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

    /// Adds an edge (two points).
    pub fn push_edge(&mut self, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) {
        self.push_point(x0, y0, z0);
        self.push_point(x1, y1, z1);
    }

    /// Adds an edge from two (x, y, z) tuples.
    pub fn push_edge_tuple(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64)) {
        self.push_point(p0.0, p0.1, p0.2);
        self.push_point(p1.0, p1.1, p1.2);
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
    #[cfg(feature = "fancy_math")]
    #[allow(clippy::too_many_arguments)]
    pub fn add_triangle(
        &mut self,
        x1: f64,
        y1: f64,
        z1: f64,
        x2: f64,
        y2: f64,
        z2: f64,
        x3: f64,
        y3: f64,
        z3: f64,
    ) {
        self.add_parametric_curve(|t| x1 + t * (x2 - x1), |t| y1 + t * (y2 - y1), z1, 1.0);
        self.add_parametric_curve(|t| x2 + t * (x3 - x2), |t| y2 + t * (y3 - y2), z2, 1.0);
        self.add_parametric_curve(|t| x3 + t * (x1 - x3), |t| y3 + t * (y1 - y3), z3, 1.0);
    }

    /// Adds a box (rectangular prism) with top-left-front corner at `(x, y, z)`.
    #[allow(clippy::many_single_char_names)]
    pub fn add_box(&mut self, (x, y, z): (f64, f64, f64), width: f64, height: f64, depth: f64) {
        let (h, w, d) = (height, width, depth);
        let p1 = (x, y, z);
        let p2 = (x, y - h, z);
        let p3 = (x + w, y, z);
        let p4 = (x + w, y - h, z);
        let p5 = (x, y, z - d);
        let p6 = (x, y - h, z - d);
        let p7 = (x + w, y, z - d);
        let p8 = (x + w, y - h, z - d);

        // front face
        self.push_edge_tuple(p1, p2);
        self.push_edge_tuple(p2, p4);
        self.push_edge_tuple(p1, p3);
        self.push_edge_tuple(p3, p4);

        // back face
        self.push_edge_tuple(p5, p6);
        self.push_edge_tuple(p6, p8);
        self.push_edge_tuple(p5, p7);
        self.push_edge_tuple(p7, p8);

        // connecting edges
        self.push_edge_tuple(p1, p5);
        self.push_edge_tuple(p2, p6);
        self.push_edge_tuple(p3, p7);
        self.push_edge_tuple(p4, p8);
    }

    /// Adds a sphere centered at `(cx, cy, cz)` with given `radius` and `steps` precision.
    ///
    /// Connects actual mesh neighbors via longitude and latitude lines.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_sphere(&mut self, (cx, cy, cz): (f64, f64, f64), radius: f64, steps: usize) {
        let pps = steps + 1;
        let step_by = 2.0 * PI / steps as f64;
        let mut points: Vec<(f64, f64, f64)> = Vec::with_capacity((steps + 1) * pps);

        for rot in 0..=steps {
            let phi = rot as f64 * step_by;
            for cir in 0..=steps {
                let theta = cir as f64 * step_by;
                let x = radius * theta.cos() + cx;
                let y = radius * theta.sin() * phi.sin() + cy;
                let z = radius * theta.sin() * phi.cos() + cz;
                points.push((x, y, z));
            }
        }

        for rot in 0..steps {
            for cir in 0..steps {
                let p = points[rot * pps + cir];
                let p_lat = points[rot * pps + cir + 1]; // next latitude
                let p_lon = points[(rot + 1) * pps + cir]; // next longitude
                self.push_edge_tuple(p, p_lat);
                self.push_edge_tuple(p, p_lon);
            }
        }
    }

    /// Adds a torus centered at `(cx, cy, cz)`.
    ///
    /// Connects actual mesh neighbors (circle-ring and torus-rotation lines).
    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn add_torus(
        &mut self,
        (cx, cy, cz): (f64, f64, f64),
        radius_sm: f64,
        radius_big: f64,
        steps: usize,
    ) {
        let pps = steps + 1;
        let step_by = 2.0 * PI / steps as f64;
        let mut points: Vec<(f64, f64, f64)> = Vec::with_capacity((steps + 1) * pps);

        for torus_ang_norm in 0..=steps {
            let torus_ang = torus_ang_norm as f64 * step_by;
            for cir_ang_norm in 0..=steps {
                let circ_ang = cir_ang_norm as f64 * step_by;
                let x = torus_ang.cos() * (radius_sm * circ_ang.cos() + radius_big) + cx;
                let y = radius_sm * circ_ang.sin() + cy;
                let z = -torus_ang.sin() * (radius_sm * circ_ang.cos() + radius_big) + cz;
                points.push((x, y, z));
            }
        }

        for rot in 0..steps {
            for cir in 0..steps {
                let p = points[rot * pps + cir];
                let p_cir = points[rot * pps + cir + 1]; // next circle point
                let p_tor = points[(rot + 1) * pps + cir]; // next torus rotation
                self.push_edge_tuple(p, p_cir);
                self.push_edge_tuple(p, p_tor);
            }
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
        let height_step = height / steps as f64;

        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let next_theta = ((i + 1) % steps) as f64 * theta_step;
            for j in 0..=steps {
                let h = j as f64 * height_step;
                let p = (x + radius * theta.cos(), y + radius * theta.sin(), z + h);
                let p_theta = (
                    x + radius * next_theta.cos(),
                    y + radius * next_theta.sin(),
                    z + h,
                );
                self.push_edge_tuple(p, p_theta);

                if j < steps {
                    let p_height = (p.0, p.1, z + (j + 1) as f64 * height_step);
                    self.push_edge_tuple(p, p_height);
                }
            }
        }

        // bottom and top bases
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let x_point = x + radius * theta.cos();
            let y_point = y + radius * theta.sin();
            let next_theta = ((i + 1) % steps) as f64 * theta_step;
            let next_x = x + radius * next_theta.cos();
            let next_y = y + radius * next_theta.sin();
            self.push_edge(x, y, z, x_point, y_point, z);
            self.push_edge(x_point, y_point, z, next_x, next_y, z);
            self.push_edge(next_x, next_y, z, x, y, z);
            self.push_edge(x, y, z + height, x_point, y_point, z + height);
            self.push_edge(x_point, y_point, z + height, next_x, next_y, z + height);
            self.push_edge(next_x, next_y, z + height, x, y, z + height);
        }
    }

    /// Adds a cone with base center at `(x, y, z)`, given `radius`, `height`, and `steps` precision.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_cone(&mut self, (x, y, z): (f64, f64, f64), radius: f64, height: f64, steps: usize) {
        let theta_step = 2.0 * PI / steps as f64;
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let x_point = x + radius * theta.cos();
            let y_point = y + radius * theta.sin();
            let z_point = z;
            let next_theta = ((i + 1) % steps) as f64 * theta_step;
            let next_x = x + radius * next_theta.cos();
            let next_y = y + radius * next_theta.sin();
            self.push_edge(x, y, z, x_point, y_point, z_point);
            self.push_edge(x_point, y_point, z_point, x, y, z + height);
            self.push_edge(x_point, y_point, z_point, next_x, next_y, z);
        }
    }

    /// Adds a pyramid with base centered at `(x, y, z)`.
    pub fn add_pyramid(&mut self, (x, y, z): (f64, f64, f64), base_length: f64, height: f64) {
        let half = base_length / 2.0;
        let corners = [
            (x - half, y, z - half),
            (x + half, y, z - half),
            (x + half, y, z + half),
            (x - half, y, z + half),
        ];
        for &corner in &corners {
            self.push_edge(corner.0, corner.1, corner.2, x, y + height, z);
        }
        for i in 0..corners.len() {
            let next = (i + 1) % corners.len();
            self.push_edge_tuple(corners[i], corners[next]);
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
        assert_eq!(test.len(), 24 * 4);
        let transform =
            Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(45.0) * Matrix::rotate_z(45.0);
        let test = test.apply(&transform);
        let mut canvas = Canvas::new_with_bg(100, 100, Rgb::new(24, 26, 27));
        canvas.set_line_pixel(Rgb::new(255, 255, 255));
        canvas.draw_lines(&test);
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
        canvas.draw_lines(&test);
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

        assert!(sphere.iter_edges().any(|(p0, p1)| {
            (p0[0] - p1[0]).abs() < 1e-10
                && (p0[1] - p1[1]).abs() < 1e-10
                && (p0[2] - p1[2]).abs() < 1e-10
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

        assert!(torus.iter_edges().any(|(p0, p1)| {
            (p0[0]).abs() < 1e-10
                && (p0[1]).abs() < 1e-10
                && (p0[2] - 4.0).abs() < 1e-10
                && (p1[0] - 4.0).abs() < 1e-10
                && (p1[1]).abs() < 1e-10
                && (p1[2]).abs() < 1e-10
        }));
    }

    #[test]
    fn add_cylinder_test() {
        let mut test = EdgeMatrix::new();
        test.add_cylinder((25.0, 25.0, 25.0), 25.0, 25.0, 36);
        assert!(test.cols() > 0);
    }

    #[test]
    fn add_cylinder_does_not_create_diagonal_side_edges() {
        let mut cylinder = EdgeMatrix::new();
        cylinder.add_cylinder((0.0, 0.0, 0.0), 1.0, 4.0, 4);

        for (p0, p1) in cylinder.iter_edges() {
            let xy_changed = !((p0[0] - p1[0]).abs() < 1e-10 && (p0[1] - p1[1]).abs() < 1e-10);
            let z_changed = (p0[2] - p1[2]).abs() > 1e-10;

            assert!(
                !(xy_changed && z_changed),
                "cylinder edge should be either around a ring or along height, got {p0:?} -> {p1:?}"
            );
        }
    }

    #[test]
    fn add_cone_test() {
        let mut test = EdgeMatrix::new();
        test.add_cone((25.0, 25.0, 25.0), 25.0, 25.0, 36);
        assert_eq!(test.cols(), 36 * 3 * 2);
    }

    #[test]
    fn add_pyramid_test() {
        let mut test = EdgeMatrix::new();
        test.add_pyramid((25.0, 25.0, 25.0), 25.0, 25.0);
        assert_eq!(test.cols(), 8 * 2);
    }
}

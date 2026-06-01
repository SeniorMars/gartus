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

    /// Clears all points from the matrix without deallocating memory.
    pub fn clear(&mut self) {
        self.inner.truncate_cols(0);
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
        let mut data = Vec::with_capacity(coords.len() * 2);
        for pair in coords.chunks_exact(2) {
            Self::extend_point_data(&mut data, pair[0].into(), pair[1].into(), z);
        }
        matrix.append_homogeneous_points(&data);
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

    /// Adds an edge (two points).
    pub fn push_edge(&mut self, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) {
        self.append_homogeneous_points(&[x0, y0, z0, 1.0, x1, y1, z1, 1.0]);
    }

    /// Adds an edge from two (x, y, z) tuples.
    pub fn push_edge_tuple(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64)) {
        self.append_homogeneous_points(&[p0.0, p0.1, p0.2, 1.0, p1.0, p1.1, p1.2, 1.0]);
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
    pub fn as_matrix(&self) -> &Matrix {
        &self.inner
    }

    fn append_homogeneous_points(&mut self, data: &[f64]) {
        self.inner
            .append_columns_from_slice(data)
            .expect("EdgeMatrix values must always have 4 rows");
    }

    fn extend_point_data(data: &mut Vec<f64>, x: f64, y: f64, z: f64) {
        data.extend_from_slice(&[x, y, z, 1.0]);
    }

    fn extend_edge_data(data: &mut Vec<f64>, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) {
        data.extend_from_slice(&[x0, y0, z0, 1.0, x1, y1, z1, 1.0]);
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

        let mut data = Vec::new();
        for curr in values {
            let (x0, y0) = prev;
            let (x1, y1) = curr;
            Self::extend_edge_data(&mut data, x0, y0, z, x1, y1, z);
            prev = curr;
        }
        self.append_homogeneous_points(&data);
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

    /// Adds a circle centered at `(cx, cy, cz)` with radius `r` and precision `step`.
    pub fn add_circle(&mut self, cx: f64, cy: f64, cz: f64, r: f64, step: f64) {
        self.add_parametric_curve(
            |t: f64| r * (t * 2.0 * PI).cos() + cx,
            |t: f64| r * (t * 2.0 * PI).sin() + cy,
            cz,
            step,
        );
    }

    /// Adds a triangle connecting the three given vertices. (Wireframe)
    pub fn add_triangle(&mut self, p0: (f64, f64, f64), p1: (f64, f64, f64), p2: (f64, f64, f64)) {
        let mut data = Vec::with_capacity(24);
        Self::extend_edge_data(&mut data, p0.0, p0.1, p0.2, p1.0, p1.1, p0.2);
        Self::extend_edge_data(&mut data, p1.0, p1.1, p1.2, p2.0, p2.1, p1.2);
        Self::extend_edge_data(&mut data, p2.0, p2.1, p2.2, p0.0, p0.1, p2.2);
        self.append_homogeneous_points(&data);
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
}

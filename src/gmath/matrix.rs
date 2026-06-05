use super::vector::Vector;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    fmt,
    ops::{Add, AddAssign, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    slice::{self},
};

const EPS: f64 = 1e-10;
#[cfg(feature = "rayon")]
const PARALLEL_POINT_TRANSFORM_THRESHOLD: usize = 4_096;
#[cfg(feature = "rayon")]
const PARALLEL_MATRIX_MULTIPLY_WORK_THRESHOLD: usize = 16_384;

#[must_use]
#[derive(Default, Clone, Debug)]
/// A type that represents a m x n Matrix
pub struct Matrix {
    /// The rows (m) component of the Matrix
    rows: usize,
    /// The column (n) component of the Matrix
    cols: usize,
    /// The actual data the Matrix includes
    pub(crate) data: Vec<f64>,
}

#[allow(dead_code)]
impl Matrix {
    /// Returns a new row x column [Matrix] with a vector that contains the data.
    ///
    /// # Panics
    ///
    /// Panics if the size of data isn't the same as rows * cols
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Matrix must be filled completely");
        Self { rows, cols, data }
    }

    /// Returns a new row x column [Matrix] initialized to zero.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self { rows, cols, data }
    }

    /// Fill the [Matrix] with a vector of floats.
    ///
    /// Ideally, should be used after `with_capacity` to fill the [Matrix] with data.
    ///
    /// * `data`: A vector comprised of floats that is the body of the [Matrix]
    ///
    /// # Panics
    ///
    /// Panics if the size of data isn't the same as rows * cols
    pub fn fill_data(&mut self, data: Vec<f64>) {
        assert_eq!(
            self.rows * self.cols,
            data.len(),
            "Matrix must be filled completely"
        );
        self.data = data;
    }

    /// Returns the number of points (cols) currently in the [Matrix].
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the rows in the [Matrix].
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of elements currently in the [`Matrix`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the [`Matrix`] contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a new N by N identity [Matrix].
    pub fn identity_matrix(size: usize) -> Self {
        let mut matrix: Matrix = Matrix::new(size, size, vec![0.0; size * size]);
        for i in 0..size {
            matrix.set(i, i, 1.0);
        }
        matrix
    }

    /// Returns the inverse of a squared [`Matrix`].
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        let (l_mat, u_mat, p_vec, _) = self.lu_decomp()?;
        Self::solve_with_lu(&l_mat, &u_mat, &p_vec, &Self::identity_matrix(n))
    }

    /// Returns the trace of the [`Matrix`].
    #[must_use]
    pub fn trace(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }
        let mut sum = 0.0;
        for i in 0..self.rows {
            sum += self[(i, i)];
        }
        Some(sum)
    }

    /// Returns the Frobenius norm of the [`Matrix`].
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Returns `true` if the [`Matrix`] is symmetric.
    #[must_use]
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for i in 0..self.rows {
            for j in 0..i {
                if (self[(i, j)] - self[(j, i)]).abs() > EPS {
                    return false;
                }
            }
        }
        true
    }

    /// Returns `true` if the [`Matrix`] is singular (determinant is 0).
    #[must_use]
    pub fn is_singular(&self) -> bool {
        self.determinant().is_none_or(|det| det.abs() < EPS)
    }

    /// Returns the reduced row-echelon form of the [`Matrix`].
    pub fn rref(&self) -> Self {
        self.rref_with_tolerance(EPS)
    }

    /// Returns the reduced row-echelon form of the [`Matrix`] using `eps` as the zero tolerance.
    pub fn rref_with_tolerance(&self, eps: f64) -> Self {
        let mut matrix = self.clone();
        let _full_rank = Self::gauss_jordan_general(&mut matrix, eps);

        for value in &mut matrix.data {
            if value.abs() <= eps {
                *value = 0.0;
            }
        }

        matrix
    }

    /// Returns the rank of the [`Matrix`].
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank_with_tolerance(EPS)
    }

    /// Returns the rank of the [`Matrix`] using `eps` as the zero tolerance.
    #[must_use]
    pub fn rank_with_tolerance(&self, eps: f64) -> usize {
        let rref = self.rref_with_tolerance(eps);
        (0..rref.rows)
            .filter(|&row| (0..rref.cols).any(|col| rref[(row, col)].abs() > eps))
            .count()
    }

    /// Solves the linear system AX = B, where B is a [`Matrix`].
    #[must_use]
    pub fn solve_matrix(&self, b_mat: &Matrix) -> Option<Matrix> {
        if self.rows != self.cols || b_mat.rows != self.rows {
            return None;
        }

        let (l_mat, u_mat, p_vec, _) = self.lu_decomp()?;
        Self::solve_with_lu(&l_mat, &u_mat, &p_vec, b_mat)
    }

    /// Returns the determinant of a squared [Matrix].
    #[must_use]
    pub fn determinant(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }

        let (_, u_mat, _, swaps) = self.lu_decomp()?;
        let mut det = if swaps % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..self.rows {
            det *= u_mat[(i, i)];
        }
        Some(det)
    }

    /// Returns the LU decomposition of a squared [`Matrix`].
    /// Returns (L, U, P, swaps) where P is the permutation vector and swaps is the number of row swaps.
    /// Returns `None` if the matrix is singular or not square.
    #[must_use]
    pub fn lu_decomp(&self) -> Option<(Self, Self, Vec<usize>, usize)> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        let mut l_mat = Self::identity_matrix(n);
        let mut u_mat = self.clone();
        let mut p_vec: Vec<usize> = (0..n).collect();
        let mut swaps = 0;

        for i in 0..n {
            let mut max_row = i;
            let mut max_val = unsafe { u_mat.get_unchecked(i, i).abs() };
            for k in i + 1..n {
                let val = unsafe { u_mat.get_unchecked(k, i).abs() };
                if val > max_val {
                    max_val = val;
                    max_row = k;
                }
            }

            if max_val < EPS {
                return None;
            }

            if max_row != i {
                u_mat.swap_rows(i, max_row);
                p_vec.swap(i, max_row);
                for k in 0..i {
                    unsafe {
                        let tmp = l_mat.get_unchecked(i, k);
                        l_mat.set_unchecked(i, k, l_mat.get_unchecked(max_row, k));
                        l_mat.set_unchecked(max_row, k, tmp);
                    }
                }
                swaps += 1;
            }

            let pivot = unsafe { u_mat.get_unchecked(i, i) };
            for j in i + 1..n {
                unsafe {
                    let factor = u_mat.get_unchecked(j, i) / pivot;
                    l_mat.set_unchecked(j, i, factor);
                    u_mat.set_unchecked(j, i, 0.0);
                    for k in i + 1..n {
                        let val = u_mat.get_unchecked(j, k) - factor * u_mat.get_unchecked(i, k);
                        u_mat.set_unchecked(j, k, val);
                    }
                }
            }
        }

        Some((l_mat, u_mat, p_vec, swaps))
    }

    /// Solves the linear system Ax = b.
    #[must_use]
    pub fn solve(&self, target_b: &Matrix) -> Option<Matrix> {
        if self.rows != self.cols || target_b.rows != self.rows || target_b.cols != 1 {
            return None;
        }

        let (l_mat, u_mat, p_vec, _) = self.lu_decomp()?;
        Self::solve_with_lu(&l_mat, &u_mat, &p_vec, target_b)
    }

    fn solve_with_lu(
        l_mat: &Matrix,
        u_mat: &Matrix,
        p_vec: &[usize],
        target_b: &Matrix,
    ) -> Option<Matrix> {
        let n = l_mat.rows;
        if l_mat.rows != l_mat.cols
            || u_mat.rows != n
            || u_mat.cols != n
            || p_vec.len() != n
            || target_b.rows != n
        {
            return None;
        }

        let mut y_data = vec![0.0; n * target_b.cols];
        let mut x_data = vec![0.0; n * target_b.cols];

        for col in 0..target_b.cols {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..i {
                    unsafe {
                        sum += l_mat.get_unchecked(i, j) * *y_data.get_unchecked(col * n + j);
                    }
                }
                unsafe {
                    *y_data.get_unchecked_mut(col * n + i) =
                        target_b.get_unchecked(p_vec[i], col) - sum;
                }
            }

            for i in (0..n).rev() {
                let mut sum = 0.0;
                for j in i + 1..n {
                    unsafe {
                        sum += u_mat.get_unchecked(i, j) * *x_data.get_unchecked(col * n + j);
                    }
                }
                unsafe {
                    *x_data.get_unchecked_mut(col * n + i) =
                        (*y_data.get_unchecked(col * n + i) - sum) / u_mat.get_unchecked(i, i);
                }
            }
        }

        Some(Matrix::new(n, target_b.cols, x_data))
    }

    /// Returns the QR decomposition of a squared [`Matrix`].
    #[must_use]
    pub fn qr_decomp(&self) -> Option<(Self, Self)> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        let mut q_mat = Self::identity_matrix(n);
        let mut r_mat = self.clone();

        if n <= 1 {
            return Some((q_mat, r_mat));
        }

        for i in 0..n - 1 {
            let mut v_data = Vec::with_capacity(n - i);
            for j in i..n {
                v_data.push(r_mat[(j, i)]);
            }
            let mut v_vec = Matrix::new(n - i, 1, v_data);
            let x_norm = v_vec.norm();
            if x_norm < EPS {
                continue;
            }

            let sign = if v_vec[(0, 0)] >= 0.0 { 1.0 } else { -1.0 };
            v_vec[(0, 0)] += sign * x_norm;

            let v_norm = v_vec.norm();
            if v_norm < EPS {
                continue;
            }
            v_vec /= v_norm;

            for j in i..n {
                let mut dot = 0.0;
                for k in i..n {
                    dot += v_vec[(k - i, 0)] * r_mat[(k, j)];
                }
                for k in i..n {
                    r_mat[(k, j)] -= 2.0 * v_vec[(k - i, 0)] * dot;
                }
            }

            for j in 0..n {
                let mut dot = 0.0;
                for k in i..n {
                    dot += q_mat[(j, k)] * v_vec[(k - i, 0)];
                }
                for k in i..n {
                    q_mat[(j, k)] -= 2.0 * dot * v_vec[(k - i, 0)];
                }
            }
        }

        Some((q_mat, r_mat))
    }

    /// Returns the real eigenvalues of a symmetric squared [`Matrix`] using the QR algorithm.
    ///
    /// Returns `None` for non-square or non-symmetric matrices. This method does not model
    /// complex eigenvalues.
    #[must_use]
    pub fn eigenvalues(&self) -> Option<Vec<f64>> {
        if self.rows != self.cols || !self.is_symmetric() {
            return None;
        }
        let n = self.rows;
        let mut ak = self.clone();

        for _ in 0..200 {
            let (q, r) = ak.qr_decomp()?;
            ak = r * q;

            let mut off_diag_norm = 0.0;
            for i in 0..n {
                for j in 0..i {
                    off_diag_norm += ak[(i, j)].powi(2);
                }
            }
            if off_diag_norm.sqrt() < EPS {
                break;
            }
        }

        let mut ev = Vec::with_capacity(n);
        for i in 0..n {
            ev.push(ak[(i, i)]);
        }
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Some(ev)
    }

    /// Returns the transpose [`Matrix`] of self.
    pub fn transpose(&self) -> Self {
        let mut new_data = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                let original_idx = col * self.rows + row;
                let transposed_idx = row * self.cols + col;
                new_data[transposed_idx] = self.data[original_idx];
            }
        }
        Matrix::new(self.cols, self.rows, new_data)
    }

    /// Makes self an identity [Matrix] if the matrix is N by N.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn identifize(&mut self) {
        assert_eq!(self.rows, self.cols, "An identity matrix must be N x N");
        let size = self.rows;
        self.data.fill(0.0);
        for i in 0..size {
            self.set(i, i, 1.0);
        }
    }

    pub(crate) fn flat_index(&self, row: usize, col: usize) -> usize {
        assert!(
            row < self.rows && col < self.cols,
            "matrix index ({row}, {col}) out of bounds for {}x{} matrix",
            self.rows,
            self.cols
        );
        col * self.rows + row
    }

    /// Returns the corresponding self.data element given a row and column.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.flat_index(row, col)]
    }

    /// Returns the element at `(row, col)` without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[must_use]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> f64 {
        let i = col * self.rows + row;
        unsafe { *self.data.get_unchecked(i) }
    }

    /// Sets the corresponding self.data element a new value given a row and column.
    pub fn set(&mut self, row: usize, col: usize, new_point: f64) {
        let i = self.flat_index(row, col);
        self.data[i] = new_point;
    }

    /// Sets the element at `(row, col)` without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    pub unsafe fn set_unchecked(&mut self, row: usize, col: usize, val: f64) {
        let i = col * self.rows + row;
        unsafe {
            *self.data.get_unchecked_mut(i) = val;
        }
    }

    /// Swaps two rows in self.data.
    ///
    /// # Arguments
    ///
    /// * `row_one` - The index of the first row to be swapped.
    /// * `row_two` - The index of the second row to be swapped.
    ///
    /// # Panics
    ///
    /// Panics if either row index is out of bounds.
    pub fn swap_rows(&mut self, row_one: usize, row_two: usize) {
        if row_one == row_two {
            return;
        }

        assert!(
            row_one < self.rows && row_two < self.rows,
            "row index out of bounds for {}x{} matrix",
            self.rows,
            self.cols
        );

        let rows = self.rows;
        let cols = self.cols;
        for col in 0..cols {
            let idx_one = col * rows + row_one;
            let idx_two = col * rows + row_two;
            self.data.swap(idx_one, idx_two);
        }
    }

    /// Get a reference to the matrix's data.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        self.data.as_ref()
    }

    /// Applies this 4x4 matrix to one homogeneous point without allocating a matrix.
    ///
    /// # Panics
    /// Panics if this matrix is not 4x4 or if `point` has fewer than 4 values.
    #[must_use]
    pub fn transform_homogeneous_point(&self, point: &[f64]) -> [f64; 4] {
        assert_eq!(self.rows, 4, "transform must have 4 rows");
        assert_eq!(self.cols, 4, "transform must have 4 columns");
        assert!(point.len() >= 4, "homogeneous point must have 4 values");

        let (x, y, z, w) = (point[0], point[1], point[2], point[3]);
        [
            self[(0, 0)] * x + self[(0, 1)] * y + self[(0, 2)] * z + self[(0, 3)] * w,
            self[(1, 0)] * x + self[(1, 1)] * y + self[(1, 2)] * z + self[(1, 3)] * w,
            self[(2, 0)] * x + self[(2, 1)] * y + self[(2, 2)] * z + self[(2, 3)] * w,
            self[(3, 0)] * x + self[(3, 1)] * y + self[(3, 2)] * z + self[(3, 3)] * w,
        ]
    }

    pub(crate) fn append_column(&mut self, column: &[f64]) -> Result<(), &'static str> {
        if self.rows != column.len() {
            return Err("new column length must match matrix rows");
        }
        self.data.extend_from_slice(column);
        self.cols += 1;
        Ok(())
    }

    pub(crate) fn append_columns(&mut self, other: &Self) -> Result<(), &'static str> {
        if self.rows != other.rows {
            return Err("appended matrix must have the same number of rows");
        }
        self.data.extend_from_slice(&other.data);
        self.cols += other.cols;
        Ok(())
    }

    pub(crate) fn append_columns_from_slice(
        &mut self,
        columns: &[f64],
    ) -> Result<(), &'static str> {
        if self.rows == 0 {
            return if columns.is_empty() {
                Ok(())
            } else {
                Err("cannot append columns to a matrix with zero rows")
            };
        }
        if !columns.len().is_multiple_of(self.rows) {
            return Err("appended data length must be a multiple of matrix rows");
        }
        self.data.extend_from_slice(columns);
        self.cols += columns.len() / self.rows;
        Ok(())
    }

    pub(crate) fn truncate_cols(&mut self, cols: usize) {
        assert!(
            cols <= self.cols,
            "cannot truncate matrix to more columns than it contains"
        );
        self.data.truncate(cols * self.rows);
        self.cols = cols;
    }

    pub(crate) fn apply_homogeneous_transform_from_col(
        &mut self,
        start_col: usize,
        transform: &Matrix,
    ) {
        if start_col >= self.cols {
            return;
        }

        assert_eq!(self.rows, 4, "homogeneous point matrices must have 4 rows");
        assert_eq!(transform.rows(), 4, "transform must have 4 rows");
        assert_eq!(transform.cols(), 4, "transform must have 4 columns");

        let m00 = transform[(0, 0)];
        let m01 = transform[(0, 1)];
        let m02 = transform[(0, 2)];
        let m03 = transform[(0, 3)];
        let m10 = transform[(1, 0)];
        let m11 = transform[(1, 1)];
        let m12 = transform[(1, 2)];
        let m13 = transform[(1, 3)];
        let m20 = transform[(2, 0)];
        let m21 = transform[(2, 1)];
        let m22 = transform[(2, 2)];
        let m23 = transform[(2, 3)];
        let m30 = transform[(3, 0)];
        let m31 = transform[(3, 1)];
        let m32 = transform[(3, 2)];
        let m33 = transform[(3, 3)];

        let points = &mut self.data[start_col * 4..];

        #[cfg(feature = "rayon")]
        {
            let point_count = points.len() / 4;
            if point_count >= PARALLEL_POINT_TRANSFORM_THRESHOLD {
                points.par_chunks_exact_mut(4).for_each(|point| {
                    let (x, y, z, w) = (point[0], point[1], point[2], point[3]);
                    point[0] = m00 * x + m01 * y + m02 * z + m03 * w;
                    point[1] = m10 * x + m11 * y + m12 * z + m13 * w;
                    point[2] = m20 * x + m21 * y + m22 * z + m23 * w;
                    point[3] = m30 * x + m31 * y + m32 * z + m33 * w;
                });
                return;
            }
        }

        for point in points.chunks_exact_mut(4) {
            let (x, y, z, w) = (point[0], point[1], point[2], point[3]);
            point[0] = m00 * x + m01 * y + m02 * z + m03 * w;
            point[1] = m10 * x + m11 * y + m12 * z + m13 * w;
            point[2] = m20 * x + m21 * y + m22 * z + m23 * w;
            point[3] = m30 * x + m31 * y + m32 * z + m33 * w;
        }
    }

    pub(crate) fn gauss_jordan_general(matrix: &mut Matrix, eps: f64) -> bool {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let mut lead = 0;

        for row in 0..rows {
            let pivot = loop {
                if lead >= cols {
                    return false;
                }

                let pivot = (row..rows)
                    .max_by(|&a, &b| {
                        matrix[(a, lead)]
                            .abs()
                            .partial_cmp(&matrix[(b, lead)].abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("row range is non-empty");

                if matrix[(pivot, lead)].abs() > eps {
                    break pivot;
                }

                lead += 1;
            };

            matrix.swap_rows(pivot, row);

            let div = matrix[(row, lead)];
            for j in 0..cols {
                matrix[(row, j)] /= div;
            }

            for i in 0..rows {
                if i != row {
                    let factor = matrix[(i, lead)];
                    for j in 0..cols {
                        matrix[(i, j)] -= factor * matrix[(row, j)];
                    }
                }
            }

            lead += 1;
        }

        true
    }

    /// Returns the result of multiplying self by another [`Matrix`].
    ///
    /// # Panics
    ///
    /// Panics if the length of cols of self is not equal to the rows of other.
    pub fn mult_matrix(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "cols of self must equal rows of other"
        );

        let mut result = Matrix::zeros(self.rows, other.cols);

        #[cfg(feature = "rayon")]
        {
            let work = self.rows * self.cols * other.cols;
            if work >= PARALLEL_MATRIX_MULTIPLY_WORK_THRESHOLD {
                result
                    .data
                    .par_chunks_mut(self.rows)
                    .enumerate()
                    .for_each(|(j, result_col)| {
                        for k in 0..self.cols {
                            unsafe {
                                let b = other.get_unchecked(k, j);
                                for i in 0..self.rows {
                                    *result_col.get_unchecked_mut(i) +=
                                        self.get_unchecked(i, k) * b;
                                }
                            }
                        }
                    });
                return result;
            }
        }

        for j in 0..other.cols {
            for k in 0..self.cols {
                unsafe {
                    let b = other.get_unchecked(k, j);
                    for i in 0..self.rows {
                        let val = result.get_unchecked(i, j) + self.get_unchecked(i, k) * b;
                        result.set_unchecked(i, j, val);
                    }
                }
            }
        }
        result
    }

    /// Returns the result of multiplying the transpose of self by another [`Matrix`].
    ///
    /// # Panics
    ///
    /// Panics if the length of rows of self is not equal to the rows of other.
    pub fn mult_transpose_left(&self, other: &Self) -> Self {
        assert_eq!(
            self.rows, other.rows,
            "rows of self must equal rows of other for self^T * other"
        );

        let mut result = Matrix::zeros(self.cols, other.cols);
        for j in 0..other.cols {
            for k in 0..self.rows {
                let b = other[(k, j)];
                for i in 0..self.cols {
                    result[(i, j)] += self[(k, i)] * b;
                }
            }
        }
        result
    }

    /// Returns the resulting [`Vector`] when multiplying by self.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not 4x4.
    #[must_use]
    pub fn mult_vector(&self, vector: Vector) -> Vector {
        assert_eq!(
            self.rows, 4,
            "Matrix must have 4 rows for homogeneous multiply"
        );
        assert_eq!(
            self.cols, 4,
            "Matrix must have 4 cols for homogeneous multiply"
        );
        let x = self[(0, 0)] * vector[0]
            + self[(0, 1)] * vector[1]
            + self[(0, 2)] * vector[2]
            + self[(0, 3)];
        let y = self[(1, 0)] * vector[0]
            + self[(1, 1)] * vector[1]
            + self[(1, 2)] * vector[2]
            + self[(1, 3)];
        let z = self[(2, 0)] * vector[0]
            + self[(2, 1)] * vector[1]
            + self[(2, 2)] * vector[2]
            + self[(2, 3)];
        let w = self[(3, 0)] * vector[0]
            + self[(3, 1)] * vector[1]
            + self[(3, 2)] * vector[2]
            + self[(3, 3)];
        if w != 0.0 && (w - 1.0).abs() > f64::EPSILON {
            Vector::new(x / w, y / w, z / w)
        } else {
            Vector::new(x, y, z)
        }
    }

    /// Returns the sum of a matrix data
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.iter().sum()
    }

    /// applies the absolute value to each point in matrix's data
    pub fn abs(&mut self) {
        for x in &mut self.data {
            *x = x.abs();
        }
    }
}

// Iterator stuff
impl Matrix {
    /// Returns a iterator that iterates over the [Matrix]'s points.
    pub fn iter(&self) -> impl Iterator<Item = &f64> + '_ {
        self.data.iter()
    }

    /// Returns a mut iterator that iterates over the [Matrix]'s points.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> + '_ {
        self.data.iter_mut()
    }

    /// Returns a iterator that iterates over a specific row.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = &f64> + '_ {
        self.iter().skip(row).step_by(self.rows)
    }

    /// Returns a mutable iterator that iterates over a specific row.
    pub fn iter_row_mut(&mut self, row: usize) -> impl Iterator<Item = &mut f64> + '_ {
        let r = self.rows;
        self.iter_mut().skip(row).step_by(r)
    }

    /// Returns a iterator that iterates over a specific column.
    pub fn iter_col(&self, column: usize) -> impl Iterator<Item = &f64> + '_ {
        let start = column * self.rows;
        self.data[start..self.rows + start].iter()
    }

    /// Returns a mutable iterator that iterates over a specific column.
    pub fn iter_col_mut(&mut self, column: usize) -> impl Iterator<Item = &mut f64> + '_ {
        let start = column * self.rows;
        self.data[start..self.rows + start].iter_mut()
    }

    /// Returns a iterator that iterates over the [Matrix]'s columns
    pub fn iter_cols(&self) -> slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(self.rows)
    }

    /// Returns a mutable iterator that iterates over the [Matrix]'s cols
    pub fn iter_cols_mut(&mut self) -> slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(self.rows)
    }

    /// Returns a iterator that iterates over the [Matrix]'s points
    pub fn iter_by_point(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.data.chunks_exact(self.rows)
    }

    /// Returns a iterator that iterates over the [Matrix]'s rows
    pub fn iter_rows(&self) -> impl Iterator<Item = impl Iterator<Item = &f64> + '_> + '_ {
        (0..self.rows).map(move |row| self.data.iter().skip(row).step_by(self.rows))
    }

    /// Returns `true` if every element differs by at most `eps`.
    #[must_use]
    pub fn approx_eq(&self, other: &Self, eps: f64) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| (a - b).abs() <= eps)
    }
}

// Operators
impl IntoIterator for Matrix {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        let (row, col) = index;
        &self.data[self.flat_index(row, col)]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        let (row, col) = index;
        let index = self.flat_index(row, col);
        &mut self.data[index]
    }
}

impl Add for Matrix {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert!(
            self.cols == other.cols && self.rows == other.rows,
            "To add Matices must be the same size"
        );
        let new_data = self.iter().zip(other.iter()).map(|(a, b)| a + b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl AddAssign<Self> for Matrix {
    fn add_assign(&mut self, other: Self) {
        assert!(
            self.cols == other.cols && self.rows == other.rows,
            "To add Matices must be the same size"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b);
    }
}

impl AddAssign<&Self> for Matrix {
    fn add_assign(&mut self, other: &Self) {
        assert!(
            self.cols == other.cols && self.rows == other.rows,
            "To add Matrices must be the same size"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b);
    }
}

impl AddAssign<f64> for Matrix {
    fn add_assign(&mut self, other: f64) {
        self.iter_mut().for_each(|e| *e += other);
    }
}

impl Sub for Matrix {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        assert!(
            self.cols == other.cols && self.rows == other.rows,
            "To subtract Matices must be the same size"
        );
        let new_data = self.iter().zip(other.iter()).map(|(a, b)| a - b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, other: Self) {
        assert!(
            self.cols == other.cols && self.rows == other.rows,
            "To subtract Matices must be the same size"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a -= b);
    }
}

impl SubAssign<f64> for Matrix {
    fn sub_assign(&mut self, other: f64) {
        self.iter_mut().for_each(|e| *e -= other);
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, other: Self) -> Self::Output {
        self.mult_matrix(other)
    }
}

impl Mul<&Self> for Matrix {
    type Output = Matrix;
    fn mul(self, other: &Self) -> Self::Output {
        self.mult_matrix(other)
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, other: Self) -> Self::Output {
        self.mult_matrix(&other)
    }
}

impl MulAssign for Matrix {
    fn mul_assign(&mut self, other: Self) {
        *self = self.mult_matrix(&other);
    }
}

impl MulAssign<&Self> for Matrix {
    fn mul_assign(&mut self, other: &Matrix) {
        *self = self.mult_matrix(other);
    }
}

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, other: f64) {
        self.iter_mut().for_each(|e| *e *= other);
    }
}

impl DivAssign<f64> for Matrix {
    fn div_assign(&mut self, other: f64) {
        self.iter_mut().for_each(|e| *e /= other);
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| (a - b).abs() < EPS)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{:.6}\t", self.get(row, col))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use super::*;
    use std::iter::Iterator;

    #[test]
    fn new_matrix() {
        let nums: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let edge: Matrix = Matrix::new(3, 3, nums);
        let ident = Matrix::identity_matrix(3);
        let bruh = edge.transpose();
        assert_ne!(ident, bruh);
        assert_eq!(
            format!("{ident}"),
            "1.000000\t0.000000\t0.000000\t\n0.000000\t1.000000\t0.000000\t\n0.000000\t0.000000\t1.000000\t\n"
        );
    }

    #[test]
    fn rows_and_cols() {
        let nums: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let edge: Matrix = Matrix::new(3, 3, nums);
        let row_one: Vec<f64> = edge.iter_col(1).copied().collect();
        assert_eq!(row_one, vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn swap() {
        let mut ident = Matrix::identity_matrix(3);
        ident.swap_rows(0, 2);
        assert_eq!(
            ident,
            Matrix::new(3, 3, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn iterators() {
        let ident = Matrix::identity_matrix(4);
        let points = ident.iter_by_point().collect::<Vec<_>>();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0], &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn inverse_test() {
        let test = Matrix::new(3, 3, vec![5.0, 4.0, 7.0, 7.0, 3.0, 5.0, 9.0, 8.0, 6.0]);
        let inverse = test.inverse().expect("should be invertible");
        let ident = &test * &inverse;
        assert!(ident.approx_eq(&Matrix::identity_matrix(3), 1e-10));
    }

    #[test]
    fn solve_matrix_handles_multiple_rhs_columns() {
        let a = Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let expected_x = Matrix::new(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
        let b = a.mult_matrix(&expected_x);

        let actual_x = a.solve_matrix(&b).expect("matrix should be invertible");

        assert!(actual_x.approx_eq(&expected_x, 1e-10));
    }

    #[test]
    fn qr_decomp_test() {
        let a = Matrix::new(
            3,
            3,
            vec![12.0, 6.0, -4.0, -51.0, 167.0, 24.0, 4.0, -68.0, -41.0],
        );
        let (q, r) = a.qr_decomp().expect("square matrix");
        let qt_q = q.transpose() * &q;
        assert!(qt_q.approx_eq(&Matrix::identity_matrix(3), 1e-10));
        let q_r = q * r;
        assert!(q_r.approx_eq(&a, 1e-10));
    }

    #[test]
    fn qr_decomp_handles_zero_by_zero_matrix() {
        let a = Matrix::new(0, 0, Vec::new());
        let (q, r) = a.qr_decomp().expect("square matrix");

        assert_eq!(q.rows(), 0);
        assert_eq!(q.cols(), 0);
        assert_eq!(r.rows(), 0);
        assert_eq!(r.cols(), 0);
    }

    #[test]
    fn eigenvalues_handles_zero_by_zero_matrix() {
        let a = Matrix::new(0, 0, Vec::new());
        let ev = a.eigenvalues().expect("symmetric matrix");

        assert!(ev.is_empty());
    }

    #[test]
    fn eigenvalues_test() {
        let a = Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let ev = a.eigenvalues().expect("symmetric matrix");
        assert!((ev[0] - 3.0).abs() < 1e-10);
        assert!((ev[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eigenvalues_rejects_non_symmetric_matrix() {
        let rotation = Matrix::new(2, 2, vec![0.0, 1.0, -1.0, 0.0]);

        assert!(rotation.eigenvalues().is_none());
    }

    #[test]
    fn test_column_major_layout() {
        // Create a 2x3 matrix:
        // [ 1.0  3.0  5.0 ]
        // [ 2.0  4.0  6.0 ]
        // In column-major, data should be: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 0), 2.0);
        assert_eq!(matrix.get(0, 1), 3.0);
        assert_eq!(matrix.get(1, 1), 4.0);
        assert_eq!(matrix.get(0, 2), 5.0);
        assert_eq!(matrix.get(1, 2), 6.0);

        assert_eq!(matrix.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_matrix_multiplication_complex() {
        // [ 1  2 ]   [ 5  6 ]   [ 1*5+2*7  1*6+2*8 ]   [ 19  22 ]
        // [ 3  4 ] * [ 7  8 ] = [ 3*5+4*7  3*6+4*8 ] = [ 43  50 ]
        let a = Matrix::new(2, 2, vec![1.0, 3.0, 2.0, 4.0]); // col-major: col0=[1,3], col1=[2,4]
        let b = Matrix::new(2, 2, vec![5.0, 7.0, 6.0, 8.0]); // col-major: col0=[5,7], col1=[6,8]
        let c = &a * &b;

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn large_matrix_multiplication_matches_expected_values() {
        let rows = 24;
        let inner = 32;
        let cols = 24;
        let mut a_data = Vec::with_capacity(rows * inner);
        let mut b_data = Vec::with_capacity(inner * cols);

        for k in 0..inner {
            for i in 0..rows {
                a_data.push(i as f64 * 0.5 + k as f64 * 0.25);
            }
        }
        for j in 0..cols {
            for k in 0..inner {
                b_data.push(j as f64 * 0.125 - k as f64 * 0.375);
            }
        }

        let a = Matrix::new(rows, inner, a_data);
        let b = Matrix::new(inner, cols, b_data);
        let c = a.mult_matrix(&b);

        for &(i, j) in &[(0, 0), (3, 5), (11, 17), (23, 23)] {
            let expected = (0..inner)
                .map(|k| (i as f64 * 0.5 + k as f64 * 0.25) * (j as f64 * 0.125 - k as f64 * 0.375))
                .sum::<f64>();
            assert!((c[(i, j)] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn large_homogeneous_transform_updates_all_points() {
        let point_count = 4_096;
        let mut data = Vec::with_capacity(point_count * 4);
        for col in 0..point_count {
            let c = col as f64;
            data.extend_from_slice(&[c, c + 1.0, c + 2.0, 1.0]);
        }
        let mut points = Matrix::new(4, point_count, data);
        let transform = Matrix::new(
            4,
            4,
            vec![
                2.0, 7.0, 11.0, 13.0, 3.0, 8.0, 12.0, 14.0, 4.0, 9.0, 15.0, 16.0, 5.0, 10.0, 17.0,
                18.0,
            ],
        );

        points.apply_homogeneous_transform_from_col(0, &transform);

        for &col in &[0, 97, point_count - 1] {
            let c = col as f64;
            let x = c;
            let y = c + 1.0;
            let z = c + 2.0;
            assert_eq!(points[(0, col)], 2.0 * x + 3.0 * y + 4.0 * z + 5.0);
            assert_eq!(points[(1, col)], 7.0 * x + 8.0 * y + 9.0 * z + 10.0);
            assert_eq!(points[(2, col)], 11.0 * x + 12.0 * y + 15.0 * z + 17.0);
            assert_eq!(points[(3, col)], 13.0 * x + 14.0 * y + 16.0 * z + 18.0);
        }
    }

    #[test]
    fn test_identity_property() {
        let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let i3 = Matrix::identity_matrix(3);
        let i2 = Matrix::identity_matrix(2);

        // I * A = A
        assert_eq!(&i3 * &a, a);
        // A * I = A
        assert_eq!(&a * &i2, a);
    }

    #[test]
    fn test_transpose_layout() {
        // [ 1  3  5 ]      [ 1  2 ]
        // [ 2  4  6 ]  ^T  [ 3  4 ]
        //                  [ 5  6 ]
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose();

        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        assert_eq!(transposed.data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_determinant_known_values() {
        let a = Matrix::new(2, 2, vec![3.0, 1.0, 8.0, 2.0]); // col-major: [[3, 1], [8, 2]] -> [3 8; 1 2]
        // det = 3*2 - 8*1 = -2
        assert!((a.determinant().unwrap() - (-2.0)).abs() < EPS);

        let b = Matrix::identity_matrix(4);
        assert!((b.determinant().unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_inverse_identity() {
        let i = Matrix::identity_matrix(4);
        assert_eq!(i.inverse().unwrap(), i);
    }

    #[test]
    fn test_inverse_singular() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]); // col-major: col0=[1,2], col1=[2,4]
        assert!(a.inverse().is_none());
        assert!(a.is_singular());
    }

    #[test]
    fn rref_reduces_matrix() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);

        let rref = matrix.rref();

        assert_eq!(rref.data(), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn rank_counts_independent_rows() {
        let matrix = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0]);

        assert_eq!(matrix.rank(), 2);
    }

    #[test]
    fn test_iter_row() {
        let matrix = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let row0: Vec<_> = matrix.iter_row(0).collect();
        assert_eq!(row0, vec![&1.0, &4.0, &7.0]);
    }

    #[test]
    fn test_iter_rows() {
        let matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let rows = matrix
            .iter_rows()
            .map(Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>();
        assert_eq!(rows, vec![vec![&1.0, &3.0, &5.0], vec![&2.0, &4.0, &6.0]]);
    }
}

use super::vector::Vector;
use std::{
    fmt,
    ops::{Add, AddAssign, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    slice::{self},
};

const EPS: f64 = 1e-10;

#[derive(Default, Clone, Debug)]
/// A type that represents a m x n Matrix
pub struct Matrix {
    /// The rows (m) component of the Matrix
    rows: usize,
    /// The column (n) component of the Matrix
    cols: usize,
    /// The actual data the Matrix includes
    data: Vec<f64>,
}

// /// An iterator over the rows of a matrix for mutable access.
// ///
// /// * `matrix`: matrix to iterate over
// /// * `current_row`: current row index
// pub(crate) struct RowIterMut<'a> {
//     matrix: &'a mut Matrix,
//     current_row: usize,
// }
//
// impl<'a> Iterator for RowIterMut<'a> {
//     type Item = Vec<&'a mut f64>;
//     // &'a mut [f64];
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.current_row < self.matrix.rows {
//             let rows = self.matrix.rows;
//             let cols = self.matrix.cols;
//
//
//             let data = &mut self.matrix.data;
//
//             let mut row = Vec::with_capacity(cols);
//             for col in 0..cols {
//                 let idx = col * rows + self.current_row;
//                 row.push(&mut data[idx]);
//             }
//             self.current_row += 1;
//
//             Some(row)
//         } else {
//             None
//         }
//     }
// }

// #[derive(Debug)]
// /// A matrix that has a constant size
// pub struct ConstMatrix<const SIZE: usize> {
//     /// The actual data the Matrix includes
//     data: [f64; SIZE],
// }

#[allow(dead_code)]
impl Matrix {
    /// Returns a new row x column [Matrix] with a vector that contains the data.
    ///
    /// # Arguments
    ///
    /// * `rows` - An unsigned usize int that represents
    ///   the number of rows in the [Matrix]
    /// * `cols` - An unsigned usize int that represents
    ///   the number of columns in the [Matrix]
    /// * `data` - A vector comprised of floats that is the body of the [Matrix]
    ///
    /// # Panics
    /// If the size of data isn't the same as rows * cols
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// ```
    #[must_use]
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Matrix must be filled completely");
        Self { rows, cols, data }
    }

    /// Returns a new row x column [Matrix] initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `rows` - An unsigned usize int that represents
    ///   the number of rows in the [Matrix]
    /// * `cols` - An unsigned usize int that represents
    ///   the number of columns in the [Matrix]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let matrix = Matrix::zeros(2, 2);
    /// ```
    #[must_use]
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
    /// * `data`: If the size of data isn't the same as rows * cols
    pub fn fill_data(&mut self, data: Vec<f64>) {
        assert_eq!(
            self.rows * self.cols,
            data.len(),
            "Matrix must be filled completely"
        );
        self.data = data;
    }

    /// Returns the number of points (cols) currently in the [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// let num = matrix.cols();
    /// ```
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the rows in the [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// let num = matrix.rows();
    /// ```
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
    ///
    /// # Arguments
    ///
    /// * `size` - An unsigned usize int that represents
    ///   the size of the identity [Matrix]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// ```
    #[must_use]
    pub fn identity_matrix(size: usize) -> Self {
        let mut matrix: Matrix = Matrix::new(size, size, vec![0.0; size * size]);
        (0..size).for_each(|i| {
            matrix.set(i, i, 1.0);
        });
        matrix
    }

    /// Returns the inverse of a squared [`Matrix`].
    /// ```rust
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let inv = matrix.inverse();
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let (rows, cols) = (self.rows, self.cols);
        if rows != cols {
            return None;
        }

        let len = rows;

        let mut rref = Matrix::zeros(len, len * 2);
        let mut inv = Matrix::zeros(len, len);

        for idx in 0..len {
            for jdx in 0..len {
                rref[(idx, jdx)] = self[(idx, jdx)];
            }
            rref[(idx, idx + len)] = 1.0;
        }

        Self::gauss_jordan_general(&mut rref, EPS);

        for idx in 0..len {
            if (rref[(idx, idx)] - 1.0).abs() > EPS {
                return None;
            }
            for jdx in 0..len {
                if jdx != idx && rref[(idx, jdx)].abs() > EPS {
                    return None;
                }
            }
        }

        for idx in 0..len {
            for jdx in 0..len {
                inv[(idx, jdx)] = rref[(idx, jdx + len)];
            }
        }

        Some(inv)
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

            for kdx in 0..rows {
                if kdx != row {
                    let mult = matrix[(kdx, lead)];
                    for jdx in 0..cols {
                        matrix[(kdx, jdx)] -= matrix[(row, jdx)] * mult;
                    }
                }
            }

            lead += 1;
        }

        true
    }

    // LU decomposition
    // fn lu(&self) -> (Matrix, Matrix, Matrix) {
    //
    // }

    // fn pivotize(&self) -> Matrix {
    //     let mut p = Matrix::identity_matrix(self.rows);
    //
    //     for j in 0..self.rows {
    //         let mut row = j;
    //         for i in j + 1..self.rows {
    //             if self[(i, j)].abs() > self[(row, j)].abs() {
    //                 row = i;
    //             }
    //         }
    //
    //         if j != row {
    //             p.swap_rows(j, row);
    //         }
    //     }
    //     p
    // }

    /// Returns the determinant of a squared [Matrix] or None if the [Matrix] is not squared.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let det = matrix.determinant().unwrap();
    /// ```
    #[must_use]
    pub fn determinant(&self) -> Option<f64> {
        if self.rows == self.cols {
            Some(self.determinant_helper())
        } else {
            None
        }
    }

    /// Computes the determinant of a squared [Matrix]
    /// in O(n^3) time.
    fn determinant_helper(&self) -> f64 {
        const EPS: f64 = 1e-12;
        let mut det = 1.0;
        let mut gauss = self.clone();

        for idx in 0..self.rows {
            let mut k = idx;
            for jdx in idx + 1..self.rows {
                if gauss[(jdx, idx)].abs() > gauss[(k, idx)].abs() {
                    k = jdx;
                }
            }

            if gauss[(k, idx)].abs() < EPS {
                return 0.0;
            }

            gauss.swap_rows(idx, k);

            if idx != k {
                det = -det;
            }

            let pivot = gauss[(idx, idx)];
            det *= pivot;

            for row in idx + 1..self.rows {
                let factor = gauss[(row, idx)] / pivot;
                gauss[(row, idx)] = 0.0;
                if factor.abs() <= EPS {
                    continue;
                }
                for col in idx + 1..self.rows {
                    gauss[(row, col)] -= factor * gauss[(idx, col)];
                }
            }
        }
        det
    }

    /// Returns the transpose [`Matrix`] of self.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let transpose = ident.transpose();
    /// ```
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut new_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i * self.cols + j] = self[(i, j)];
            }
        }
        Matrix::new(self.cols, self.rows, new_data)
    }

    /// Makes self an identity [Matrix] if the matrix is N by N.
    ///
    /// # Panics
    /// self.rows and self.cols must be the same to convert the Matrix into an indentity matrix
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let mut matrix = Matrix::new(2, 2, vector);
    /// matrix.identifize();
    /// ```
    pub fn identifize(&mut self) {
        assert_eq!(self.rows, self.cols, "An identity matrix must be N x N");
        let cols = self.cols;
        for (index, element) in self.iter_mut().enumerate() {
            *element = if index / cols == index % cols {
                1.0
            } else {
                0.0
            }
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

    /// Fills every element of self.data with a specific float.
    ///
    /// # Arguments
    ///
    /// * `float` - A f64 float that override every element in self.data
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let mut matrix = Matrix::new(2, 2, vector);
    /// matrix.fill(0.0);
    /// ```
    pub fn fill(&mut self, float: f64) {
        self.data.fill(float);
    }

    /// Swaps two rows in self.data.
    ///
    /// # Arguments
    ///
    /// * `row_one` - The index of the first row to be swapped.
    /// * `row_two` - The index of the second row to be swapped.
    ///
    /// # Panics
    /// Panics if either row is out of bounds.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.swap_rows(0, 1);
    /// ```
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
        let data = &mut self.data;

        for col in 0..cols {
            let idx_one = col * rows + row_one;
            let idx_two = col * rows + row_two;
            data.swap(idx_one, idx_two);
        }
    }

    /// Returns the corresponding self.data element
    /// given a row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - The index of the row of the data point to be accessed
    /// * `column` - The index of the column of the data point to be accessed
    ///
    /// # Panics
    /// If index is out of bounds
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let num = ident.get(0, 0);
    /// ```
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.flat_index(row, col)]
    }

    /// Sets the corresponding self.data element a new value
    /// given a row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - The index of the row of the data point to be changed
    /// * `column` - The index of the column of the data point to be changed
    /// * `new_point` - The new value to be added
    ///
    /// # Panics
    /// If index is out of bounds
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.set(0, 0, 100.0);
    /// ```
    pub fn set(&mut self, row: usize, col: usize, new_point: f64) {
        let i = self.flat_index(row, col);
        self.data[i] = new_point;
    }

    /// Get a reference to the matrix's data.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        self.data.as_ref()
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
}

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

// impl Index<usize> for Matrix {
//     type Output = f64;
//
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.data[index]
//     }
// }
//
// impl IndexMut<usize> for Matrix {
//     fn index_mut(&mut self, index: usize) -> &mut f64 {
//         &mut self.data[index]
//     }
// }

// Iterator stuff
#[allow(dead_code)]
impl Matrix {
    // pub fn for_each<F>(&mut self, function: F)
    // where
    //     F: Fn(f64) -> f64,
    // {
    //     self.iter_by_point_mut()
    //         .for_each(|point: &mut [f64]| point.iter_mut().for_each(|e| *e = function(*e)))
    // }
    // pub fn from_iter(&self) -> impl IntoIterator<Item = &[f64]> {
    //     self.data.chunks_exact(self.cols)
    // }
    /// Returns a iterator that iterates over the [Matrix]'s points.
    pub fn iter(&self) -> impl Iterator<Item = &f64> + '_ {
        self.data.iter()
    }

    /// Returns a mut iterator that iterates over the [Matrix]'s points.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> + '_ {
        self.data.iter_mut()
    }

    /// Returns a iterator that iterates over a specific row.
    ///
    /// # Arguments
    ///
    /// * `row` - The index of the row to be interated over
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_row(0);
    /// ```
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = &f64> + '_ {
        self.iter().skip(row).step_by(self.rows)
    }

    /// Returns a mutable iterator that iterates over a specific row.
    ///
    /// # Arguments
    ///
    /// * `row` - The index of the row to be interated over
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_row_mut(0);
    /// ```
    pub fn iter_row_mut(&mut self, row: usize) -> impl Iterator<Item = &mut f64> + '_ {
        let r = self.rows;
        self.iter_mut().skip(row).step_by(r)
    }

    /// Returns a iterator that iterates over a specific column.
    ///
    /// # Arguments
    ///
    /// * `column` - The index of the column to be interated over
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_col(0);
    /// ```
    pub fn iter_col(&self, column: usize) -> impl Iterator<Item = &f64> + '_ {
        let start = column * self.rows;
        self.data[start..self.rows + start].iter()
    }

    /// Returns a mutable iterator that iterates over a specific column.
    ///
    /// # Arguments
    ///
    /// * `column` - The index of the column to be interated over
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_col_mut(0);
    /// ```
    pub fn iter_col_mut(&mut self, column: usize) -> impl Iterator<Item = &mut f64> + '_ {
        let start = column * self.rows;
        self.data[start..self.rows + start].iter_mut()
    }

    /// Returns a iterator that iterates over the [Matrix]'s columns
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_cols();
    /// ```
    pub fn iter_cols(&self) -> slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(self.rows)
    }

    /// Returns a mutable iterator that iterates over the [Matrix]'s cols
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// let mut iter = ident.iter_cols_mut();
    /// ```
    pub fn iter_cols_mut(&mut self) -> slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(self.rows)
    }

    /// Returns a iterator that iterates over the [Matrix]'s points
    pub fn iter_by_point(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.data.chunks_exact(self.rows)
    }

    /// Returns a iterator that iterates over the [Matrix]'s rows
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_rows();
    /// ```
    pub fn iter_rows(&self) -> impl Iterator<Item = impl Iterator<Item = &f64> + '_> + '_ {
        (0..self.rows).map(move |row| self.data.iter().skip(row).step_by(self.rows))
    }
}

// Equal
impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl Matrix {
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

// multiply two matrices
impl Matrix {
    /// Returns the result of multiplying self by another [`Matrix`].
    /// Self's rows must be the same size as the other's [`Matrix`]'s cols.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to a Matrix to be multipled with self
    ///
    /// # Panics
    /// If the length of cols of self is not equal to the rows of other
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.mult_matrix(&Matrix::identity_matrix(4));
    /// ```
    #[must_use]
    pub fn mult_matrix(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "cols of self must equal rows of other"
        );

        let mut result = Matrix::zeros(self.rows, other.cols);
        for j in 0..other.cols {
            for k in 0..self.cols {
                let b = other[(k, j)];
                for i in 0..self.rows {
                    result[(i, j)] += self[(i, k)] * b;
                }
            }
        }
        result

        // let data = (0..self.rows * other.cols)
        //     .map(|index| {
        //         self.iter_row(index % self.rows)
        //             .zip(other.iter_col(index / self.rows))
        //             .fold(0.0, |acc, (s, o)| acc + s * o)
        //     })
        //     .collect();
        //
        // Matrix {
        //     rows: self.rows,
        //     cols: other.cols,
        //     data,
        // }
    }

    /// Returns the result of multiplying the transpose of self by another [`Matrix`].
    /// `self.rows` must equal `other.rows`.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to a Matrix to be multipled with self
    ///
    /// # Panics
    /// If the length of rows of self is not equal to the cols of other
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.mult_transpose_left(&Matrix::identity_matrix(4));
    /// ```
    ///
    /// # Note
    /// The way this algorithm is implemented is not the most efficient as I wanted to try
    /// functional programming
    #[must_use]
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
    /// # Arguments
    ///
    /// * `vector` - A mutable vector that will be mutliplied with self.
    ///
    /// # Panics
    /// Currenlty only supports identity matrices
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// use crate::gartus::gmath::vector::Vector;
    /// let ident1 = Matrix::identity_matrix(4);
    /// let mut vector = Vector::new(0.0, 0.1, 0.2);
    /// ident1.mult_vector(vector);
    /// ```
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
}

// Operators
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

// other operators
impl Matrix {
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
mod tests {
    use std::iter::Iterator;

    use super::*;

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
        assert!(format!("{ident:?}").contains("rows: 3"));
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

        let mut random = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0]);
        random.swap_rows(0, 2);

        let expected = Matrix::new(3, 3, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0, -3.0, -2.0, -1.0]);

        assert_eq!(random, expected);
    }

    #[test]
    fn iterators() {
        let ident = Matrix::identity_matrix(4);
        let points = ident.iter_by_point().collect::<Vec<_>>();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0], &[1.0, 0.0, 0.0, 0.0]);
        assert!((ident.iter().copied().sum::<f64>() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn operators() {
        let mut matrix = Matrix::identity_matrix(2);
        let bruh = Matrix {
            rows: 2,
            cols: 2,
            data: [2.0, 0.0, 0.0, 2.0].to_vec(),
        };
        matrix -= bruh;
        assert_eq!(matrix, Matrix::new(2, 2, vec![-1.0, 0.0, 0.0, -1.0]));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn indexing_invalid_row_panics() {
        let matrix = Matrix::identity_matrix(3);
        let _ = matrix[(4, 0)];
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn indexing_invalid_col_panics() {
        let matrix = Matrix::identity_matrix(3);
        let _ = matrix[(0, 4)];
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn mul_for_now() {
        let mut a = Matrix::new(1, 3, vec![3.0, 4.0, 2.0]);
        let c = Matrix::new(
            3,
            4,
            vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        a *= c;
        assert_eq!(a, Matrix::new(1, 4, vec![89.0, 91.0, 48.0, 18.0]));
    }

    #[test]
    fn multipled() {
        let m1_contents = vec![2.0, -1.0, 7.0, 4.0, -2.0, -12.0];
        let m1 = Matrix::new(3, 2, m1_contents);

        let m2_contents = vec![5.0, -3.0];
        let m2 = Matrix::new(2, 1, m2_contents);

        assert_eq!(m1 * m2, Matrix::new(3, 1, vec![-2.0, 1.0, 71.0]));
    }

    #[test]
    fn iter_test() {
        let values = Matrix::identity_matrix(4).into_iter().collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
            ]
        );
    }

    #[test]
    fn inverse_test() {
        let test = Matrix::new(3, 3, vec![5.0, 4.0, 7.0, 7.0, 3.0, 5.0, 9.0, 8.0, 6.0]);

        if let Some(inverse) = test.inverse() {
            let ones = (test * inverse)
                .data
                .iter()
                .map(|x| x.round().abs())
                .collect::<Vec<f64>>();
            assert_eq!(ones, Matrix::identity_matrix(3).data());
        }

        let test2 = vec![1.0, 2.0, 3.0, 4.0, 1.0, 6.0, 7.0, 8.0, 9.0];
        let test2 = Matrix::new(3, 3, test2);

        if let Some(inverse) = test2.inverse() {
            let ones = (test2 * inverse)
                .data
                .iter()
                .map(|x| x.round().abs())
                .collect::<Vec<f64>>();
            assert_eq!(ones, Matrix::identity_matrix(3).data());
        }

        let mut ident = Matrix::identity_matrix(3);
        ident.set(2, 2, 0.0);

        assert_eq!(ident.inverse(), None);

        let data = vec![3.0, 6.0, 2.0, 4.0];
        let matrix = Matrix::new(2, 2, data);
        assert_eq!(matrix.inverse(), None);

        let data = vec![1.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0];
        let matrix = Matrix::new(3, 3, data);
        assert_eq!(matrix.inverse(), None);
    }

    #[test]
    fn hermite() {
        let data = [
            0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ];
        let h = Matrix::new(4, 4, data.to_vec());
        let her = Matrix::hermite();
        assert_eq!(her, h);
    }

    #[test]
    fn add_matrix() {
        let d = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let e = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let a = d.mult_transpose_left(&e);
        assert_eq!(a, Matrix::new(2, 2, vec![50.0, 122.0, 68.0, 167.0]));
    }

    #[test]
    fn transpose_test() {
        let a = Matrix::new(2, 2, vec![-15.0, 14.0, 70.0, 91.0]);
        let b = a.transpose();
        assert_eq!(b, Matrix::new(2, 2, vec![-15.0, 70.0, 14.0, 91.0]));
    }

    #[test]
    fn comp140() {
        let m1_contents = vec![5.0, 14.0, 10.0, 7.0];
        let m1 = Matrix::new(2, 2, m1_contents);

        let m2_contents = vec![-3.0, 1.0, 7.0, 13.0];
        let m2 = Matrix::new(2, 2, m2_contents);

        let correct = Matrix::new(2, 2, vec![-5.0, -35.0, 165.0, 189.0]);
        assert_eq!(m1 * m2, correct);
    }

    #[test]
    fn det_test() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let det = a.determinant().expect("not a squared matrix");
        assert!((det - (-2.0)).abs() < 1e-10);

        let b = Matrix::new(2, 2, vec![3.0, 4.0, 8.0, 6.0]);
        let det = b.determinant().expect("not a squared matrix");
        assert!((det - (-14.0)).abs() < 1e-10);

        let c = Matrix::new(3, 3, vec![1.0, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, 4.0, 2.0]);
        let det = c.determinant().expect("not a squared matrix");
        assert!((det - 6.0).abs() < 1e-10);
    }

    #[test]
    fn determinant_handles_pivoting_without_stale_subdiagonal_values() {
        let matrix = Matrix::new(
            4,
            4,
            vec![
                0.0, 2.0, 3.0, 1.0, 1.0, 0.0, 4.0, 2.0, 5.0, 6.0, 0.0, 3.0, 2.0, 1.0, 7.0, 0.0,
            ],
        );

        assert!((matrix.determinant().unwrap() + 211.0).abs() < 1e-10);
    }

    #[test]
    fn test_iter_row() {
        let matrix = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], // column-major
        };

        let row0: Vec<_> = matrix.iter_row(0).collect();
        let row1: Vec<_> = matrix.iter_row(1).collect();
        let row2: Vec<_> = matrix.iter_row(2).collect();

        assert_eq!(row0, vec![&1.0, &4.0, &7.0]);
        assert_eq!(row1, vec![&2.0, &5.0, &8.0]);
        assert_eq!(row2, vec![&3.0, &6.0, &9.0]);
    }

    #[test]
    fn test_iter_col() {
        let matrix = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };

        let col0: Vec<_> = matrix.iter_col(0).collect();
        let col1: Vec<_> = matrix.iter_col(1).collect();
        let col2: Vec<_> = matrix.iter_col(2).collect();

        assert_eq!(col0, vec![&1.0, &2.0, &3.0]);
        assert_eq!(col1, vec![&4.0, &5.0, &6.0]);
        assert_eq!(col2, vec![&7.0, &8.0, &9.0]);
    }

    #[test]
    fn test_iter_col_mut() {
        let mut matrix = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };

        matrix.iter_col_mut(1).for_each(|x| *x *= 2.0);

        assert_eq!(
            matrix.data,
            vec![1.0, 2.0, 3.0, 8.0, 10.0, 12.0, 7.0, 8.0, 9.0]
        );
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
            .map(std::iter::Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>();

        assert_eq!(rows, vec![vec![&1.0, &3.0, &5.0], vec![&2.0, &4.0, &6.0]]);

        let matrix = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };

        let rows = matrix
            .iter_rows()
            .map(std::iter::Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>();

        assert_eq!(
            rows,
            vec![
                vec![&1.0, &4.0, &7.0],
                vec![&2.0, &5.0, &8.0],
                vec![&3.0, &6.0, &9.0]
            ]
        );

        let matrix = Matrix {
            rows: 3,
            cols: 3,
            data: vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        };

        let rows = matrix
            .iter_rows()
            .map(std::iter::Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>();

        assert_eq!(
            rows,
            vec![
                vec![&1.0, &2.0, &3.0],
                vec![&4.0, &5.0, &6.0],
                vec![&7.0, &8.0, &9.0]
            ]
        );

        let large_i = Matrix::identity_matrix(20);

        let rows = large_i
            .iter_rows()
            .map(Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>();

        let mut expected = vec![];

        for i in 0..20 {
            let mut row = vec![&0.0; 20];
            row[i] = &1.0;
            expected.push(row);
        }

        assert_eq!(rows, expected, "identity matrix rows are not as expected");
    }

    #[test]
    fn test_gauss_jordan_square() {
        let mut matrix = Matrix::new(3, 3, vec![2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]);
        assert!(Matrix::gauss_jordan_general(&mut matrix, EPS));

        // Verify that the matrix is in reduced row-echelon form
        let expected_matrix = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(matrix.approx_eq(&expected_matrix, EPS));
    }

    #[test]
    fn test_matrix_inversion() {
        let matrices = [
            Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 3.0]),
            Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0]),
            // (
            //     Matrix::new(3, 3, vec![2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 3.0]),
            //     Matrix::new(3, 3, vec![3.0, -1.0, -1.0, 2.0, 1.0, -1.0, 0.0, 1.0, 1.0]),
            // ),
            // (
            //     Matrix::new(4, 4, vec![1.0, 0.0, 2.0, -1.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 1.0]),
            //     Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, -1.5, 0.0, 0.0, 0.333333, 0.0, 0.0, -0.333333, 0.0]),
            // ),
            // (
            //     Matrix::new(5, 5, vec![
            //         1.0, 0.0, 0.0, 0.0, 0.0,
            //         0.0, 1.0, 0.0, 0.0, 0.0,
            //         0.0, 0.0, 1.0, 0.0, 0.0,
            //         0.0, 0.0, 0.0, 1.0, 0.0,
            //         0.0, 0.0, 0.0, 0.0, 1.0,
            //     ]),
            //     Matrix::new(5, 5, vec![
            //         1.0, 0.0, 0.0, 0.0, 0.0,
            //         0.0, 1.0, 0.0, 0.0, 0.0,
            //         0.0, 0.0, 1.0, 0.0, 0.0,
            //         0.0, 0.0, 0.0, 1.0, 0.0,
            //         0.0, 0.0, 0.0, 0.0, 1.0,
            //     ]),
            // ),
        ];

        for matrix in matrices {
            let inverse = matrix.inverse().expect("matrix should be invertible");
            let identity = Matrix::identity_matrix(matrix.rows());
            assert!((matrix * inverse).approx_eq(&identity, EPS));
        }
    }
}

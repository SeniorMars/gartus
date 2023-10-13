use super::{parametric::Parametric, vector::Vector};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    slice,
};

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
    /// the number of rows in the [Matrix]
    /// * `cols` - An unsigned usize int that represents
    /// the number of columns in the [Matrix]
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

    /// Returns a new row x column [Matrix] with a vector `with_capacity` of row * column
    ///
    /// # Arguments
    ///
    /// * `rows` - An unsigned usize int that represents
    /// the number of rows in the [Matrix]
    /// * `cols` - An unsigned usize int that represents
    /// the number of columns in the [Matrix]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let matrix = Matrix::with_capacity(2, 2);
    /// ```
    #[must_use]
    pub fn with_capacity(rows: usize, cols: usize) -> Self {
        let data = Vec::with_capacity(rows * cols);
        Self { rows, cols, data }
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

    /// Returns a new N by N identity [Matrix].
    ///
    /// # Arguments
    ///
    /// * `size` - An unsigned usize int that represents
    /// the size of the identity [Matrix]
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

    #[must_use]
    /// Returns the inverse of a squared [Matrix].
    /// ```rust
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let inv = matrix.inverse();
    /// ```
    pub fn inverse(&self) -> Option<Self> {
        let (rows, cols) = (self.rows, self.cols);
        if rows != cols {
            return None;
        }

        let len = rows;

        let mut rref = Matrix::new(len, len * 2, vec![0.0; len * len * 2]);

        for idx in 0..len {
            for jdx in 0..len {
                rref[(idx, jdx)] = self[(idx, jdx)];
            }
            rref[(idx, idx + len)] = 1.0;
        }

        Self::gauss_jordan_general(&mut rref).ok()?;

        let mut inv = Matrix::new(len, len, vec![0.0; len * len]);

        for idx in 0..len {
            for jdx in 0..len {
                inv[(idx, jdx)] = rref[(idx, jdx + len)];
            }
        }

        Some(inv)
    }

    pub(crate) fn gauss_jordan_general(matrix: &mut Matrix) -> Result<(), String> {
        let mut lead = 0;
        let (rows, cols) = (matrix.rows, matrix.cols);
        for row in 0..rows {
            if cols <= lead {
                break;
                // return Err("Inversion Impossible".to_string());
            }

            // pick a pivot
            let mut idx = row;

            // check if pivot is zero
            if matrix[(idx, lead)] == 0.0 {
                return Err("Inversion Impossible".to_string());
            }
            while matrix[(idx, lead)] == 0.0 {
                idx += 1;
                if rows == idx {
                    idx = row;
                    lead += 1;
                    if cols == lead {
                        break;
                        // return Err("Inversion Impossible".to_string());
                    }
                }
            }

            matrix.swap_rows(row, idx);

            // set elments among the diagonal to one
            let div = matrix[(row, lead)];
            if div != 0.0 {
                for jdx in 0..cols {
                    matrix[(row, jdx)] /= div;
                }
            }

            // eliminate among the diagonals
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
        Ok(())
    }

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
        let mut det = 1.0;
        let mut gauss = self.clone();

        for idx in 0..self.rows {
            let mut k = idx;
            for jdx in idx + 1..self.rows {
                if gauss[(jdx, idx)].abs() > gauss[(k, idx)].abs() {
                    k = jdx;
                }
            }

            if gauss[(k, idx)].abs() < f64::EPSILON {
                det = 0.0;
                break;
            }

            gauss.swap_rows(idx, k);

            if idx != k {
                det = -det;
            }

            det *= gauss[(idx, idx)];

            for jdx in idx + 1..self.rows {
                gauss[(idx, jdx)] /= gauss[(idx, idx)];
            }

            for jdx in 0..self.rows {
                if jdx != idx && gauss[(jdx, idx)].abs() > f64::EPSILON {
                    for kdx in idx + 1..self.rows {
                        gauss[(jdx, kdx)] -= gauss[(idx, kdx)] * gauss[(jdx, idx)];
                    }
                }
            }
        }
        det
    }

    #[must_use]
    /// Returns the transpose [Matrix] of self.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let transpose = ident.transpose();
    /// ```
    pub fn transpose(&self) -> Self {
        let mut new_data = vec![0.0; self.rows * self.cols];
        (0..self.rows).for_each(|i| {
            (0..self.cols).for_each(|j| {
                new_data[self.index(i, j)] = self.get(j, i);
            });
        });

        Matrix::new(self.rows, self.cols, new_data)
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

    pub(crate) fn index(&self, row: usize, col: usize) -> usize {
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
        self.data = vec![float; self.rows * self.cols];
    }

    /// Swaps two rows in self.data.
    ///
    /// # Arguments
    ///
    /// * `row_one` - The index of the first row to be swapped.
    /// * `row_two` - The index of the second row to be swapped.
    ///
    /// # Panics
    /// Will not panic
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.swap_cols(0, 1);
    /// ```
    pub fn swap_rows(&mut self, row_one: usize, row_two: usize) {
        if row_one == row_two {
            return;
        }

        if row_two < row_one {
            return self.swap_rows(row_two, row_one);
        }

        let mut points = self.iter_by_point_mut();
        points
            .nth(row_one)
            .unwrap()
            .swap_with_slice(points.nth(row_two - row_one - 1).unwrap());
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
        assert!(row < self.rows && col < self.cols, "Index out of bound");
        self.data[self.index(row, col)]
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
        assert!(row < self.rows && col < self.cols, "Index out of bound");
        let i = self.index(row, col);
        self.data[i] = new_point;
    }

    /// Get a reference to the matrix's data.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        self.data.as_ref()
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
        &self.data[col * self.rows + row]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        let (row, col) = index;
        &mut self.data[col * self.rows + row]
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
    fn iter(&self) -> impl Iterator<Item = &f64> + '_ {
        self.data.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> + '_ {
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

    /// Returns a iterator that iterates over the [Matrix]'s points.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_by_point();
    /// ```
    pub fn iter_by_point(&self) -> slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(self.rows)
    }

    /// Returns a mutable iterator that iterates over the [Matrix]'s points.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// let mut iter = ident.iter_by_point_mut();
    /// ```
    pub fn iter_by_point_mut(&mut self) -> slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(self.rows)
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

// add + append
#[allow(dead_code)]
impl Matrix {
    /// Adds a new point (x, y, z) to a [Matrix].
    ///
    /// # Arguments
    ///
    /// * `x` - A f64 float representing the x corrdinate of a point
    /// * `y` - A f64 float representing the y corrdinate of a point
    /// * `z` - A f64 float representing the z corrdinate of a point
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// matrix.add_point(0.0, 0.1, 0.2);
    /// ```
    pub fn add_point(&mut self, x: f64, y: f64, z: f64) {
        self.data.push(x);
        self.data.push(y);
        self.data.push(z);
        self.data.push(1.0);
        self.cols += 1;
    }

    /// Adds a new edge to an edge [Matrix].
    ///
    /// # Arguments
    ///
    /// * `x0` - A f64 float representing the start x corrdinate of an edge
    /// * `y0` - A f64 float representing the start y corrdinate of an edge
    /// * `z0` - A f64 float representing the start z corrdinate of an edge
    /// * `x1` - A f64 float representing the end x corrdinate of an edge
    /// * `y1` - A f64 float representing the end y corrdinate of an edge
    /// * `z1` - A f64 float representing the end z corrdinate of an edge
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// matrix.add_edge(0.0, 0.1, 0.2, 0.3, 0.4, 0.5);
    /// ```
    pub fn add_edge(&mut self, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) {
        self.add_point(x0, y0, z0);
        self.add_point(x1, y1, z1);
    }

    /// Adds a new edge in the form of a f64 vector to an edge [Matrix].
    ///
    /// # Arguments
    ///
    /// * `edge` - A vector with six floats representing two points
    /// to be added to the edge [Matrix]
    ///
    /// # Panics
    /// If the len of edge is not 6
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// let vector = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    /// matrix.add_edge_vec(&vector);
    /// ```
    pub fn add_edge_vec(&mut self, edge: &[f64]) {
        assert_eq!(6, edge.len());
        self.add_point(edge[0], edge[1], edge[2]);
        self.add_point(edge[3], edge[4], edge[5]);
    }

    /// Appends a vector to the edge [Matrix].
    ///
    /// # Arguments
    ///
    /// * `point` - a mutable vector that has three floats, which will be append to the [Matrix]
    ///
    /// # Panics
    /// If the length of Vector is not equal to the length of rows
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// use crate::gartus::gmath::vector::Vector;
    /// let mut matrix = Matrix::new(4, 0, Vec::new());
    /// let vector = Vector::new(1.0, 2.0, 3.0);
    /// matrix.append_point(&vector);
    /// ```
    pub fn append_point(&mut self, vector: &Vector) {
        assert_eq!(
            self.rows() - 1,
            vector.data.len(),
            "self.cols and new row's len are not equal"
        );
        self.add_point(vector[0], vector[1], vector[2]);
    }

    /// Adds other's data to Self
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to a Matrix with the new dataset
    ///
    /// # Panics
    /// If the rows of both matrices are not equal
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.add_dataset(&Matrix::identity_matrix(4));
    /// ```
    pub fn add_dataset(&mut self, other: &Self) {
        assert!(
            self.rows == other.rows,
            "To add a dataset Matices must have the same number of rows"
        );
        self.data.extend(&other.data);
        self.cols += other.cols;
    }

    /// Adds a parametric curve to Matrix
    ///
    /// # Arguments
    ///
    /// * `x_func` - A function that returns a f64 x value given t
    /// * `y_func` - A function that returns a f64 y value given t
    /// * `z` - The z value to be added to the matrix. TODO: perhaps, make h(z)?
    /// * `step` - A value representing precision of the curve.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// use crate::gartus::gmath::parametric::Parametric;
    /// let mut matrix = Matrix::new(4, 0, Vec::new());
    /// ```
    pub fn add_parametric_curve<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(
        &mut self,
        x_func: F,
        y_func: G,
        z: f64,
        step: f64,
    ) {
        let parametric = Parametric::new(x_func, y_func);
        parametric
            .values_iter(step)
            .collect::<Vec<(f64, f64)>>()
            .windows(2)
            .for_each(|points| {
                let (x0, y0) = points[0];
                let (x1, y1) = points[1];
                self.add_edge(x0, y0, z, x1, y1, z);
            });
    }

    /// Adds a hermite curve to Matrix
    ///
    /// # Arguments
    ///
    ///
    /// * `p0` - a point (x, y) that represents the start of the curve
    /// * `p1` - a point (x, y) that represents the start of the curve
    /// * `r0` - the slope of p0
    /// * `r1` - the slope of p1
    pub fn add_hermite(&mut self, p0: (f64, f64), p1: (f64, f64), r0: (f64, f64), r1: (f64, f64)) {
        // These are the numbers you get when you multiply by the Inverse Hermite Matrix
        fn hermite_curve_coeffs(p0: f64, p1: f64, r0: f64, r1: f64) -> (f64, f64, f64, f64) {
            (
                // Take advantage that p1 is greater than p0...so
                // another way to write this is:
                // 2.0 * p0 + -2 * p1 + ro + r1, but since the first temrs will cancel out
                //   technically this is fine
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
            0.0001,
        );
    }

    // (-P0 + 3P1 - 3P2 + P3)t^3 + (3P0 - 6P1 + 3P2)t^2 + (-3P0 + 3P1)t + P0
    // These are the numbers you get when you multiply by the Inverse Hermite Matrix
    fn bezier_curve_coeffs(p0: f64, p1: f64, p2: f64, p3: f64) -> (f64, f64, f64, f64) {
        // simple optimization
        (
            -p0 + 3.0 * (p1 - p2) + p3,
            3.0 * p0 - 6.0 * p1 + 3.0 * p2,
            3.0 * (-p0 + p1),
            p0,
        )
    }

    /// Adds a third degree bezier curve to Matrix
    ///
    /// # Arguments
    ///
    ///
    /// * `p0` - a point (x, y) that represents the start of the curve
    /// * `p1` - a point (x, y) that represents the first control point of the curve
    /// * `p2` - a point (x, y) that represents the second control point of the curve
    /// * `p3` - a point (x, y) that represents the end of the curve
    ///
    pub fn add_bezier3(&mut self, p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)) {
        // TODO: should handle both axis
        let (ax, bx, cx, dx) = Self::bezier_curve_coeffs(p0.0, p1.0, p2.0, p3.0);
        let (ay, by, cy, dy) = Self::bezier_curve_coeffs(p0.1, p1.1, p2.1, p3.1);

        self.add_parametric_curve(
            |t: f64| ax * t * t * t + bx * t * t + cx * t + dx,
            |t: f64| ay * t * t * t + by * t * t + cy * t + dy,
            0.0,
            0.001,
        );
    }

    fn generate_n_degree_bezizer_polynomials_coeff(n_degree: u8, points: &[f64]) -> Vec<f64> {
        // let points_matrix = Matrix::new(n_degree.into(), 1, points);
        //
        // let basis: Matrix;
        todo!()
        // let mut coeffs: Vec<f64> = Vec::new();

        // for i in 0..=n_degree {
        //     coeffs.push(binom(n_degree.into(), i.into()))
        // }

        // return (basis * points_matrix).data().to_vec();
    }

    // #[test]
    // fn

    fn generate_n_degree_bezizer_fun(n_degree: u8, coeffs: &[f64]) -> impl Fn(f64) -> f64 {
        move |x| x
    }

    /// Adds an n-th degree bezier curve to Matrix
    ///
    /// # Arguments
    ///
    ///
    /// * `n` - a u8 that represents the degree of the bezier curve
    /// * `points` - a list that has the control, start, and end xy corrdinates
    ///
    pub fn add_beziern(&mut self, n_degree: u8, x_points: &[f64], y_points: &[f64]) {
        todo!()
        // let x_coeffs = generate_n_degree_bezizer_polynomials_coeff(n_degree, x_points);
        // let y_coeffs = generate_n_degree_bezizer_polynomials_coeff(n_degree, y_points);
        //
        // let t_nth_bezier_x = generate_n_degree_bezizer_fun(n_degree, x_coeffs);
        // let t_nth_bezier_y = generate_n_degree_bezizer_fun(n_degree, y_coeffs);

        // self.add_parametric_curve(t_nth_bezier_x, t_nth_bezier_y, 0.0, 0.001);
    }
}

// multiply two matrices
impl Matrix {
    #[must_use]
    /// Returns the result of multiplying self by another [Matrix].
    /// Self's rows must be the same size as the other's Matrix's cols.
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
    pub fn mult_matrix(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "cols of self must equal rows of other"
        );

        let mut result = Matrix::new(self.rows, other.cols, vec![0.0; self.rows * other.cols]);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let r = self[(i, k)];
                for j in 0..other.cols {
                    result[(i, j)] += r * other[(k, j)];
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

    #[must_use]
    /// Returns the result of multiplying the tranpose of self by another [Matrix].
    /// Self's columns must be the same size as the other's Matrix's rwos.
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
    /// let mut ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.mult_trans(&Matrix::identity_matrix(4));
    /// ```
    pub fn mult_trans(&self, other: &Self) -> Self {
        assert_eq!(
            self.rows, other.cols,
            "cols of self must equal rows of other"
        );
        let data = (0..self.cols * other.rows)
            .map(|index| {
                self.iter_col(index % self.rows)
                    .zip(other.iter_row(index / self.rows))
                    .fold(0.0, |acc, (s, o)| acc + s * o)
            })
            .collect();

        Matrix {
            rows: self.cols,
            cols: other.rows,
            data,
        }
    }

    /// Returns the resulting [vector] when multiplying by self.
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
    pub fn mult_vector(&self, mut vector: Vector) {
        assert_eq!(
            self.rows(),
            self.cols(),
            "Multiply only with identity matrix transformation"
        );
        let copy = vector;
        for (i, element) in vector.data.iter_mut().enumerate().take(3) {
            *element =
                self.get(0, i) * copy[0] + self.get(1, i) * copy[1] + self.get(2, i) * copy[2];
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
            self.cols == other.cols,
            "To add Matices must be the same size"
        );
        self.data.extend(&other.data);
        self.cols += other.cols;
    }
}

impl AddAssign<f64> for Matrix {
    fn add_assign(&mut self, other: f64) {
        self.iter_mut().for_each(|e| *e += other);
    }
}

impl AddAssign<[f64; 3]> for Matrix {
    fn add_assign(&mut self, other: [f64; 3]) {
        self.add_point(other[0], other[1], other[2]);
    }
}

impl AddAssign<Vec<f64>> for Matrix {
    fn add_assign(&mut self, other: Vec<f64>) {
        self.add_edge_vec(&other);
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
    fn mul(self, other: &Matrix) -> Self::Output {
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
        *self = self.clone() * other;
    }
}

impl MulAssign<&Self> for Matrix {
    fn mul_assign(&mut self, other: &Matrix) {
        *self = self.clone() * other;
    }
}

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, other: f64) {
        self.iter_by_point_mut()
            .for_each(|row| row.iter_mut().for_each(|e| *e *= other));
    }
}

impl Div for Matrix {
    type Output = Self;

    fn div(self, _other: Self) -> Self {
        todo!()
    }
}

impl DivAssign<f64> for Matrix {
    fn div_assign(&mut self, other: f64) {
        self.iter_by_point_mut()
            .for_each(|row| row.iter_mut().for_each(|e| *e /= other));
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
        self.data = self.iter_mut().map(|x| x.abs()).collect();
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
    use super::*;

    #[test]
    // #[should_panic]
    fn new_matrix() {
        let nums: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let edge: Matrix = Matrix::new(3, 3, nums);
        // println!("{}", edge);
        let ident = Matrix::identity_matrix(3);
        let bruh = edge.transpose();
        println!("{ident}");
        println!("{ident:?}");
        assert!(ident != bruh, "Not Equal");
        // assert_eq!(edge.data, bruh.data)
    }

    #[test]
    fn rows_and_cols() {
        let nums: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let edge: Matrix = Matrix::new(3, 3, nums.clone());
        let row_one: Vec<f64> = edge.iter_col(1).copied().collect();
        let mut points = edge.iter_by_point();
        println!("{:?}", points.next());
        println!("{:?}", points.next());
        assert_eq!(row_one, nums[3..6]);
    }

    #[test]
    fn swap() {
        let mut ident = Matrix::identity_matrix(3);
        ident.swap_rows(0, 2);
        println!("ident:\n{ident}");
        // println!("test:\n{}", test);
        assert_eq!(
            ident,
            Matrix::new(3, 3, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        );
    }

    #[test]
    // #[should_panic]
    fn iterators() {
        let ident = Matrix::identity_matrix(4);
        let src: Vec<&[f64]> = ident.iter_by_point().collect();
        // println!("{:?}", src);
        println!("{src:?}");
        // let src: Vec<String> = ident.iter().map(|x| format!("{}", x)).collect();
        // println!("{:?}",src)
        // src.map(|x| println!("{}", x))
        assert_eq!(5 / ident.rows, 5 % ident.cols);
    }

    #[test]
    // #[should_panic]
    fn operators() {
        let mut matrix = Matrix::identity_matrix(2);
        let bruh = Matrix {
            rows: 2,
            cols: 2,
            data: [2.0, 0.0, 0.0, 2.0].to_vec(),
        };
        // let bruh = Matrix::identity_matrix(2);
        println!("{matrix:?}");
        matrix -= bruh;
        println!("{matrix:?}");
        // let new_matrix = matrix + bruh;
        // println!("{:?}", new_matrix);
        let test = Matrix {
            rows: 2,
            cols: 2,
            data: [2.0, 0.0, 0.0, 2.0].to_vec(),
        };
        assert_ne!(matrix, test);
    }

    #[test]
    #[should_panic]
    fn add_points() {
        let mut matrix = Matrix::new(0, 4, Vec::with_capacity(8));
        let x = [0.0, 0.1, 0.2];
        let y = vec![0.3, 1.3, 2.3];
        matrix += x;
        matrix += y;
        matrix += x;
        matrix += x;
        println!("{matrix}");
        matrix.identifize();
        println!("{matrix}");
    }

    #[test]
    // #[should_panic]
    #[allow(clippy::many_single_char_names)]
    fn mul_for_now() {
        let mut a = Matrix::new(1, 3, vec![3.0, 4.0, 2.0]);
        // let _b = Matrix::new(3, 1, vec![3.0, 4.0, 2.0]);
        let c = Matrix::new(
            3,
            4,
            vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        // let _d: Matrix = Matrix::new(
        //     4,
        //     3,
        //     vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        // );
        println!("{a}");
        println!("{c}");
        a *= c;
        println!("{a}");
        // let e = b * d;
        // println!("{}", c);
        // assert_eq!(c, e)
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
        Matrix::identity_matrix(4).into_iter().for_each(|i| {
            println!("{i}");
        });
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
        println!("{her}");
        println!("{h}");
    }

    #[test]
    fn add_matrix() {
        let d = Matrix::new(2, 2, vec![5.0, 10.0, 14.0, 7.0]);
        let e = Matrix::new(2, 2, vec![-3.0, 7.0, 1.0, 13.0]);
        let f = d.clone() * e.clone();
        let a = d.mult_trans(&e);
        println!("{f}");
        println!("{a}");
    }

    #[test]
    fn transpose_test() {
        let a = Matrix::new(2, 2, vec![-15.0, 14.0, 70.0, 91.0]);
        println!("{a}");
        let b = a.transpose();
        println!("{b}");
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
        println!("{a}");

        // det = ad - bc = 1 * 4 - 2 * 3 = -2
        let det = a.determinant().expect("not a squared matrix");

        println!("{det}");

        let b = Matrix::new(2, 2, vec![3.0, 4.0, 8.0, 6.0]);
        println!("{b}");

        // det = ad - bc = 18 - 32 = -14
        let det = b.determinant().expect("not a squared matrix");

        println!("{det}");

        let c = Matrix::new(3, 3, vec![1.0, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, 4.0, 2.0]);

        println!("{c}");

        // det = a(ei - fh) - b(di - fg) + c(dh - eg)
        // 6
        let det = c.determinant().expect("not a squared matrix");

        println!("{det}");
    }
}

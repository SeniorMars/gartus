use super::{helpers::hermite_curve_coeffs, parametric::Parametric, vector::Vector};
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// ```
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Matrix must be filled completely");
        Self { rows, cols, data }
    }

    /// Returns a new row x column [Matrix] with a vector with_capacity of row * column
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let matrix = Matrix::with_capacity(2, 2);
    /// ```
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// let num = matrix.cols();
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the rows in the [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// let num = matrix.rows();
    /// ```
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
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// ```
    pub fn identity_matrix(size: usize) -> Self {
        let mut matrix: Matrix = Matrix::new(size, size, vec![0.0; size * size]);
        for i in 0..size {
            matrix.set(i, i, 1.0);
        }
        matrix
    }

    /// Returns the inverse of a squared [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let inverse = ident.inverse();
    /// ```
    pub fn inverse(&self) -> Self {
        eprintln!("This doesn't really work lol -- pls don't use");
        let (rows, cols) = (self.rows, self.cols);
        assert_eq!(rows, cols, "The matrix must be N x N");
        let mut aug = Matrix::new(rows, cols * 2, vec![0.0; rows * (cols * 2)]);
        for i in 0..cols {
            for j in 0..cols {
                aug.set(i, j, self.get(i, j))
            }
            aug.set(i, i + cols, 1.0)
        }
        Self::gauss_jordan_general(&mut aug);
        let mut unaug = Matrix::new(rows, cols, vec![0.0; rows * cols]);
        for i in 0..rows {
            for j in 0..rows {
                unaug.set(i, j, aug.get(i, j + cols));
            }
        }
        unaug
    }

    fn gauss_jordan_general(matrix: &mut Self) {
        let mut lead = 0;
        let (rows, cols) = (matrix.rows, matrix.cols);

        for row in 0..rows {
            if cols <= lead {
                break;
            }
            let mut i = row;
            while matrix.get(i, lead) == 0.0 {
                i += 1;
                if rows == i {
                    i = row;
                    lead += 1;
                    if cols == lead {
                        break;
                    }
                }
            }

            matrix.data.swap(i, row);
            // let temp = matrix[i].to_owned();
            // matrix[i] = matrix[r].to_owned();
            // matrix[r] = temp.to_owned();

            if matrix.get(row, lead) != 0.0 {
                let div = matrix.get(row, lead);
                for j in 0..cols {
                    matrix.set(row, j, matrix.get(row, j) / div);
                }
            }

            for k in 0..rows {
                if k != row {
                    let mult = matrix.get(k, lead);
                    for j in 0..cols {
                        matrix.set(k, j, matrix.get(k, j) - matrix.get(row, j) * mult);
                    }
                }
            }
            lead += 1;
        }
    }

    /// Returns the transpose [Matrix] of self.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let mut matrix = Matrix::new(2, 2, vector);
    /// matrix.fill(0.0);
    /// ```
    pub fn fill(&mut self, float: f64) {
        self.data = vec![float; self.rows * self.cols]
    }

    /// Swaps two rows in self.data.
    ///
    /// # Arguments
    ///
    /// * `row_one` - The index of the first row to be swapped.
    /// * `row_two` - The index of the second row to be swapped.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.swap_cols(0, 1);
    /// ```
    pub fn swap_cols(&mut self, col_one: usize, col_two: usize) {
        let mut points = self.iter_by_point_mut();
        points
            .nth(col_one)
            .unwrap()
            .swap_with_slice(points.nth(col_two - col_one - 1).unwrap());
    }

    /// Returns the corresponding self.data element
    /// given a row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - The index of the row of the data point to be accessed
    /// * `column` - The index of the column of the data point to be accessed
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let num = ident.get(0, 0);
    /// ```
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.set(0, 0, 100.0);
    /// ```
    pub fn set(&mut self, row: usize, col: usize, new_point: f64) {
        assert!(row < self.rows && col < self.cols, "Index out of bound");
        let i = self.index(row, col);
        self.data[i] = new_point;
    }
}

impl IntoIterator for Matrix {
    type Item = f64;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl Index<usize> for Matrix {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.data[index]
    }
}

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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// let vector = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    /// matrix.add_edge_vec(vector);
    /// ```
    pub fn add_edge_vec(&mut self, edge: Vec<f64>) {
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// use crate::curves_rs::gmath::vector::Vector;
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
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
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// use crate::curves_rs::gmath::parametric::Parametric;
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
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// use crate::curves_rs::gmath::parametric::Parametric;
    /// let mut matrix = Matrix::new(4, 0, Vec::new());
    /// ```
    pub fn add_hermite(&mut self, p0: (f64, f64), p1: (f64, f64), r0: (f64, f64), r1: (f64, f64)) {
        let (ax, bx, cx, dx) = hermite_curve_coeffs(p0.0, p1.0, r0.0, r1.0);
        let (ay, by, cy, dy) = hermite_curve_coeffs(p0.1, p1.1, r0.1, r1.1);
        self.add_parametric_curve(
            |t: f64| ax * t * t * t + bx * t * t + cx * t + dx,
            |t: f64| ay * t * t * t + by * t * t + cy * t + dy,
            0.0,
            0.0001,
        );
    }
}

impl Matrix {
    /// Returns the result of multiplying self by another [Matrix].
    /// Self's columns must be the same size as the other's Matrix's rwos.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to a Matrix to be multipled with self
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let mut ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.mult_matrix(&Matrix::identity_matrix(4));
    /// ```
    pub fn mult_matrix(&self, other: &Self) -> Self {
        assert_eq!(
            self.rows, other.cols,
            "rows of self must equal cols of other"
        );
        let (rows, cols) = (other.rows, self.cols);
        let mut data = vec![0.0; rows * cols];
        for (index, element) in data.iter_mut().enumerate() {
            *element = self
                .iter_col(index / rows)
                .zip(other.iter_row(index % rows))
                .fold(0.0, |acc, (s, o)| acc + s * o);
        }
        Matrix { rows, cols, data }
    }

    /// Returns the result of multiplying the tranpose of self by another [Matrix].
    /// Self's columns must be the same size as the other's Matrix's rwos.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to a Matrix to be multipled with self
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// let mut ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.mult_trans(&Matrix::identity_matrix(4));
    /// ```
    pub fn mult_trans(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "cols of self must equal rows of other"
        );
        let (rows, cols) = (self.rows, other.cols);
        let mut data = vec![0.0; rows * cols];
        for (index, element) in data.iter_mut().enumerate() {
            *element = self
                .iter_row(index % rows)
                .zip(other.iter_col(index / rows))
                .fold(0.0, |acc, (s, o)| acc + s * o);
        }
        Matrix { rows, cols, data }
    }

    /// Returns the resulting [vector] when multiplying by self.
    ///
    /// # Arguments
    ///
    /// * `vector` - A mutable vector that will be mutliplied with self.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::matrix::Matrix;
    /// use crate::curves_rs::gmath::vector::Vector;
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
                self.get(0, i) * copy[0] + self.get(1, i) * copy[1] + self.get(2, i) * copy[2]
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
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
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
        self.iter_mut().for_each(|e| *e += other)
    }
}

impl AddAssign<[f64; 3]> for Matrix {
    fn add_assign(&mut self, other: [f64; 3]) {
        self.add_point(other[0], other[1], other[2])
    }
}

impl AddAssign<Vec<f64>> for Matrix {
    fn add_assign(&mut self, other: Vec<f64>) {
        self.add_edge_vec(other)
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
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a -= b)
    }
}

impl SubAssign<f64> for Matrix {
    fn sub_assign(&mut self, other: f64) {
        self.iter_mut().for_each(|e| *e -= other)
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
        *self = other * self.clone()
    }
}

impl MulAssign<&Self> for Matrix {
    fn mul_assign(&mut self, other: &Matrix) {
        *self = other * self
    }
}

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, other: f64) {
        self.iter_by_point_mut()
            .for_each(|row| row.iter_mut().for_each(|e| *e *= other))
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
            .for_each(|row| row.iter_mut().for_each(|e| *e /= other))
    }
}

// other operators
impl Matrix {
    /// Returns the sum of a matrix data
    pub fn sum(&self) -> f64 {
        self.iter().sum()
    }

    /// applies the absolute value to each point in matrix's data
    pub fn abs(&mut self) {
        self.data = self.iter_mut().map(|x| x.abs()).collect()
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
        println!("{}", ident);
        println!("{:?}", ident);
        assert!(ident != bruh, "Not Equal")
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
        assert_eq!(row_one, nums[3..6])
    }

    #[test]
    fn swap() {
        let mut ident = Matrix::identity_matrix(3);
        ident.swap_cols(0, 2);
        println!("ident:\n{}", ident);
        // println!("test:\n{}", test);
        assert_eq!(
            ident,
            Matrix::new(3, 3, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        )
    }

    #[test]
    // #[should_panic]
    fn iterators() {
        let ident = Matrix::identity_matrix(4);
        let src: Vec<&[f64]> = ident.iter_by_point().collect();
        // println!("{:?}", src);
        println!("{:?}", src);
        // let src: Vec<String> = ident.iter().map(|x| format!("{}", x)).collect();
        // println!("{:?}",src)
        // src.map(|x| println!("{}", x))
        assert_eq!(5 / ident.rows, 5 % ident.cols)
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
        println!("{:?}", matrix);
        matrix -= bruh;
        println!("{:?}", matrix);
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
        println!("{}", matrix);
        matrix.identifize();
        println!("{}", matrix);
    }

    #[test]
    // #[should_panic]
    #[allow(clippy::many_single_char_names)]
    fn mul_for_now() {
        let a = Matrix::new(3, 1, vec![3.0, 4.0, 2.0]);
        let b = Matrix::new(3, 1, vec![3.0, 4.0, 2.0]);
        let mut c = Matrix::new(
            4,
            3,
            vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        let d: Matrix = Matrix::new(
            4,
            3,
            vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        println!("{}", a);
        println!("{}", c);
        c *= a;
        println!("{}", c);
        // let e = b * d;
        // println!("{}", c);
        // assert_eq!(c, e)
    }

    #[test]
    fn iter_test() {
        for i in Matrix::identity_matrix(4).into_iter() {
            println!("{}", i);
        }
    }

    #[test]
    fn inverse_test() {
        let test = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 1.0, 6.0, 7.0, 8.0, 9.0]);
        let inverse = test.inverse();
        println!("{}", inverse);
        let one = test * inverse;
        println!("{}", one)
    }

    #[test]
    fn hermite() {
        let data = [
            0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ];
        let h = Matrix::new(4, 4, data.to_vec());
        let her = Matrix::hermite();
        println!("{}", her);
        println!("{}", h)
    }

    #[test]
    fn add_matrix() {
        let d = Matrix::new(2, 2, vec![5.0, 10.0, 14.0, 7.0]);
        let e = Matrix::new(2, 2, vec![-3.0, 7.0, 1.0, 13.0]);
        let f = d.clone() * e.clone();
        let a = d.mult_trans(&e);
        println!("{}", f);
        println!("{}", a);
    }

    #[test]
    fn transpose_test() {
        let a = Matrix::new(2, 2, vec![-5.0, -35.0, 165.0, 189.0]);
        println!("{}", a);
        let b = a.transpose();
        println!("{}", b)
    }
}

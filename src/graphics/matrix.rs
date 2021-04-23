use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    slice,
};

#[derive(Default, Clone, Debug)]
/// A m x n Matrix is represented here
pub struct Matrix {
    /// The rows (m) componet of the Matrix
    rows: usize,
    /// The column (n) componet of the Matrix
    cols: usize,
    /// The actual data the Matrix includes
    data: Vec<f64>,
}

// pub struct Matrix<const rows:usize, const cols:usize> {
//     data: Vec<f64>, not going to do generics
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// ```
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Matrix must be filled completely");
        Self { rows, cols, data }
    }

    /// Returns the number of points (rows) currently in the [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let vector = vec![0.0, 0.1, 0.2, 0.3];
    /// let matrix = Matrix::new(2, 2, vector);
    /// let num = matrix.get_num_points();
    /// ```
    pub fn get_num_points(&self) -> usize {
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
    /// use crate::graphics::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// ```
    pub fn identity_matrix(size: usize) -> Self {
        let mut matrix: Matrix = Matrix::new(size, size, vec![0.0; size * size]);
        for i in 0..size {
            matrix.set(i, i, 1.0);
        }
        matrix
    }

    // Returns the inverse [Matrix] of self.
    //
    // # Examples
    //
    // Basic usage:
    // ```
    // use crate::graphics::matrix::Matrix;
    // let ident = Matrix::identity_matrix(4);
    // let inverse = ident.inverse();
    // ```
    fn inverse(&self) -> Self {
        // may not do this
        todo!()
    }

    /// Returns the transpose [Matrix] of self.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let transpose = ident.transpose();
    /// ```
    pub fn transpose(&self) -> Self {
        Matrix::new(self.cols, self.rows, self.data.clone())
    }

    /// Makes self an identity [Matrix] if the matrix is N by N.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
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

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
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
    /// use crate::graphics::matrix::Matrix;
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
    /// use crate::graphics::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.swap_rows(0, 1);
    /// ```
    pub fn swap_rows(&mut self, row_one: usize, row_two: usize) {
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
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
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
    /// use crate::graphics::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// ident.set(0, 0, 100.0);
    /// ```
    pub fn set(&mut self, row: usize, col: usize, new_point: f64) {
        assert!(row < self.rows && col < self.cols, "Index out of bound");
        let i = self.index(row, col);
        self.data[i] = new_point;
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
    /// use crate::graphics::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_row(0);
    /// ```
    pub fn iter_row(&self, r: usize) -> impl Iterator<Item = &f64> + '_ {
        let start = r * self.cols;
        self.data[start..self.cols + start].iter()
    }

    fn iter_row_mut(&mut self, r: usize) -> impl Iterator<Item = &mut f64> + '_ {
        let start = r * self.cols;
        self.data[start..self.cols + start].iter_mut()
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
    /// use crate::graphics::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_col(0);
    /// ```
    pub fn iter_col(&self, column: usize) -> impl Iterator<Item = &f64> + '_ {
        self.iter().skip(column).step_by(self.cols)
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
    /// use crate::graphics::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// let mut iter = ident.iter_col(0);
    /// ```
    pub fn iter_col_mut(&mut self, column: usize) -> impl Iterator<Item = &mut f64> + '_ {
        let col = self.cols;
        self.iter_mut().skip(column).step_by(col)
    }

    /// Returns a iterator that iterates over the [Matrix]'s points.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let ident = Matrix::identity_matrix(4);
    /// let iter = ident.iter_by_point();
    /// ```
    pub fn iter_by_point(&self) -> slice::ChunksExact<'_, f64> {
        self.data.chunks_exact(self.cols)
    }

    /// Returns a mutable iterator that iterates over the [Matrix]'s points.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let mut ident = Matrix::identity_matrix(4);
    /// let mut iter = ident.iter_by_point_mut();
    /// ```
    pub fn iter_by_point_mut(&mut self) -> slice::ChunksExactMut<'_, f64> {
        self.data.chunks_exact_mut(self.cols)
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
    /// use crate::graphics::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// matrix.add_point(0.0, 0.1, 0.2);
    /// ```
    pub fn add_point(&mut self, x: f64, y: f64, z: f64) {
        self.data.push(x);
        self.data.push(y);
        self.data.push(z);
        self.data.push(1.0);
        self.rows += 1;
    }

    /// Appends a point in the form of a vector to the edge [Matrix].
    ///
    /// # Arguments
    ///
    /// * `point` - a mutable vector that has three floats, which will be append to the [Matrix]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// let mut vector = vec![0.0, 0.1, 0.2];
    /// matrix.append_point(&mut vector);
    /// ```
    pub fn append_point(&mut self, point: &mut Vec<f64>) {
        assert_eq!(
            self.cols,
            point.len() + 1,
            "self.cols and new row's len are not equal"
        );
        point.push(1.0);
        self.data.append(point);
        self.rows += 1;
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
    /// use crate::graphics::matrix::Matrix;
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
    /// use crate::graphics::matrix::Matrix;
    /// let mut matrix = Matrix::new(0, 4, Vec::new());
    /// let vector = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    /// matrix.add_edge_vec(vector);
    /// ```
    pub fn add_edge_vec(&mut self, edge: Vec<f64>) {
        assert_eq!(6, edge.len());
        self.add_point(edge[0], edge[1], edge[2]);
        self.add_point(edge[3], edge[4], edge[5]);
    }

    /// Returns a the result of multiplying self by another [Matrix].
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
    /// use crate::graphics::matrix::Matrix;
    /// let mut ident1 = Matrix::identity_matrix(4);
    /// let result = ident1.mult_matrix(&Matrix::identity_matrix(4));
    /// ```
    pub fn mult_matrix(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "Colms of self must equal rows of other"
        );
        let (rows, cols) = (self.rows, other.cols);
        let mut data = vec![0.0; rows * cols];
        for (i, e) in data.iter_mut().enumerate() {
            *e = self
                .iter_row(i / cols)
                .zip(other.iter_col(i % cols))
                .fold(0.0, |acc, (s, o)| acc + s * o);
        }
        Matrix { rows, cols, data }
    }

    /// Returns the resulting vecotr when multiplying by self.
    ///
    /// # Arguments
    ///
    /// * `vector` - A mutable vector that will be mutliplied with self.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let ident1 = Matrix::identity_matrix(4);
    /// let mut vector = vec![0.0, 0.1, 0.2, 1.0];
    /// ident1.mult_vector(vector);
    /// ```
    pub fn mult_vector(&self, mut vector: Vec<f64>) {
        assert_eq!(
            self.rows, self.cols,
            "Multiply only with identity matrix transformation"
        );
        let copy = vector.clone();
        for (i, element) in vector.iter_mut().enumerate().take(4) {
            *element = self.get(0, i) * copy[0]
                + self.get(1, i) * copy[1]
                + self.get(2, i) * copy[2]
                + self.get(3, i) * copy[3]
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

impl AddAssign for Matrix {
    fn add_assign(&mut self, other: Self) {
        assert!(
            self.cols == other.cols && self.rows == other.rows,
            "To add Matices must be the same size"
        );
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
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

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.mult_matrix(rhs)
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

impl MulAssign<&Matrix> for Matrix {
    fn mul_assign(&mut self, other: &Matrix) {
        *self = other * &self
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
        // may not do this
        todo!()
    }
}

impl DivAssign<f64> for Matrix {
    fn div_assign(&mut self, other: f64) {
        self.iter_by_point_mut()
            .for_each(|row| row.iter_mut().for_each(|e| *e /= other))
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for col in 0..self.cols {
            for row in 0..self.rows {
                write!(f, "{}\t", self.get(row, col))?;
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
        let row_one: Vec<f64> = edge.iter_row(1).copied().collect();
        let mut points = edge.iter_by_point();
        println!("{:?}", points.next());
        println!("{:?}", points.next());
        assert_eq!(row_one, nums[3..6])
    }

    #[test]
    fn swap() {
        let mut ident = Matrix::identity_matrix(3);
        ident.swap_rows(0, 2);
        // println!("ident:\n{}", ident);
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
    fn mul_for_now() {
        let a = Matrix::new(1, 3, vec![3.0, 4.0, 2.0]);
        let b = Matrix::new(1, 3, vec![3.0, 4.0, 2.0]);
        let mut c = Matrix::new(
            3,
            4,
            vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        let d = Matrix::new(
            3,
            4,
            vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        // println!("{}", c);
        c *= a;
        // println!("{}", c);
        let e = b * d;
        // println!("{}", c);
        assert_eq!(c, e)
    }
}

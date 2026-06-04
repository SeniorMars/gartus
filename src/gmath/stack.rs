use super::matrix::Matrix;

/// A small coordinate-system stack for hierarchical transforms.
#[derive(Debug, Clone)]
pub struct MatrixStack {
    stack: Vec<Matrix>,
}

impl Default for MatrixStack {
    fn default() -> Self {
        Self::new()
    }
}

impl MatrixStack {
    /// Creates a stack with a 4x4 identity root transform.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stack: vec![Matrix::identity_matrix(4)],
        }
    }

    /// Creates a stack with `root` as its first transform.
    #[must_use]
    pub fn with_root(root: Matrix) -> Self {
        Self { stack: vec![root] }
    }

    /// Duplicates the current transform.
    pub fn push(&mut self) {
        self.stack.push(self.top().clone());
    }

    /// Duplicates the current transform, applies `transform`, and makes that the new top.
    pub fn push_transform(&mut self, transform: Matrix) {
        self.push();
        self.apply(transform);
    }

    /// Removes the current transform, preserving the root.
    ///
    /// Returns `None` when called on the root transform.
    pub fn pop(&mut self) -> Option<Matrix> {
        if self.stack.len() <= 1 {
            return None;
        }
        self.stack.pop()
    }

    /// Applies `transform` to the current top transform.
    pub fn apply(&mut self, transform: Matrix) {
        let top = self.stack.len() - 1;
        self.stack[top] = self.stack[top].clone() * transform;
    }

    /// Returns the current transform.
    ///
    /// # Panics
    /// Panics only if the stack's internal root invariant has been broken.
    pub fn top(&self) -> &Matrix {
        self.stack.last().expect("matrix stack is never empty")
    }

    /// Returns the number of transforms in the stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Returns true if the stack has no transforms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns true when the stack contains only the root transform.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.stack.len() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::MatrixStack;
    use crate::gmath::matrix::Matrix;

    #[test]
    fn pop_preserves_root() {
        let mut stack = MatrixStack::new();
        assert!(stack.pop().is_none());
        assert_eq!(stack.len(), 1);
    }

    #[test]
    fn push_transform_is_scoped() {
        let mut stack = MatrixStack::new();
        stack.push_transform(Matrix::translate(10.0, 0.0, 0.0));
        let translated_x = stack
            .top()
            .transform_homogeneous_point(&[1.0, 2.0, 3.0, 1.0])[0];
        assert!((translated_x - 11.0).abs() < f64::EPSILON);
        assert!(stack.pop().is_some());
        let root_x = stack
            .top()
            .transform_homogeneous_point(&[1.0, 2.0, 3.0, 1.0])[0];
        assert!((root_x - 1.0).abs() < f64::EPSILON);
    }
}

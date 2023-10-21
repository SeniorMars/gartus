/// As always huge thanks to ruoshui

#[derive(Debug)]
/// A type that represents a parametric equation
pub struct Parametric<F: Fn(f64) -> f64, G: Fn(f64) -> f64> {
    /// The first part, x, of the parametric equation: f(t)
    x: F,
    /// The second part, y, of the parametric equation: g(t)
    y: G,
}

impl<'plife, F: Fn(f64) -> f64, G: Fn(f64) -> f64> Parametric<F, G> {
    /// Creates a new Parametric equation with functions that give x and y based on t
    ///
    /// # Arguments
    ///
    /// * `F` - A function that returns x (a f64) based on t -- f(t)
    /// * `G` - A function that returns y (a f64) based on t -- g(t)
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::parametric::Parametric;
    /// fn value(num: f64) -> f64 {
    ///    num
    /// }
    /// let equations = Parametric::new(value, value);
    /// ```
    pub fn new(x: F, y: G) -> Self {
        Self { x, y }
    }

    /// Returns the values of (x, y) at the given T in the parametric equations
    ///
    /// # Arguments
    ///
    /// * `T` - A f64 in the range of [0, 1] that returns (x, y)
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::parametric::Parametric;
    /// fn value(num: f64) -> f64 {
    ///    num
    /// }
    /// let equations = Parametric::new(value, value);
    /// ```
    pub fn value(&self, t: f64) -> (f64, f64) {
        ((self.x)(t), (self.y)(t))
    }

    /// Returns the iterator of (x, y) at the given T in the parametric equations and step
    ///
    /// # Arguments
    ///
    /// * `step` - A f64 in the range of [0, 1] that details the step of the iterator
    pub fn values_iter(&'plife self, step: f64) -> impl Iterator<Item = (f64, f64)> + 'plife {
        ParametricIter::new(self, step)
    }
}

pub(crate) struct ParametricIter<'plife, F: Fn(f64) -> f64, G: Fn(f64) -> f64> {
    parametric: &'plife Parametric<F, G>,
    t: f64,
    step: f64,
}

impl<F: Fn(f64) -> f64, G: Fn(f64) -> f64> Iterator for ParametricIter<'_, F, G> {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.t > 1.0 {
            None
        } else {
            let current_t = self.t;
            self.t += self.step;
            Some(self.parametric.value(current_t))
        }
    }
}

impl<'plife, F: Fn(f64) -> f64, G: Fn(f64) -> f64> ParametricIter<'plife, F, G> {
    fn new(parametric: &'plife Parametric<F, G>, step: f64) -> Self {
        assert!(step > 0.0);
        Self {
            parametric,
            step,
            t: 0.0,
        }
    }
}

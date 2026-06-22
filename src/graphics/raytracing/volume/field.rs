use crate::gmath::vector::Point;
use std::{fmt, sync::Arc};

/// Spatially varying density used by non-uniform participating media.
///
/// `density` returns the local extinction density at a world-space point and ray time. Values at
/// or below zero are treated as empty space by `NonUniformMedium`. `max_density` is the majorant
/// used for Woodcock tracking, so it must be greater than or equal to the maximum density the field
/// can return over the medium bounds.
pub trait DensityField: Send + Sync {
    /// Returns the local density at `point` for `time`.
    fn density(&self, point: Point, time: f64) -> f64;

    /// Returns a positive finite upper bound for [`Self::density`].
    fn max_density(&self) -> f64;
}

impl<T: DensityField + ?Sized> DensityField for Arc<T> {
    fn density(&self, point: Point, time: f64) -> f64 {
        (**self).density(point, time)
    }

    fn max_density(&self) -> f64 {
        (**self).max_density()
    }
}

/// Shared density-field handle.
pub type DensityFieldRef = Arc<dyn DensityField>;

/// Constant density field usable with `NonUniformMedium`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantDensity {
    density: f64,
}

impl ConstantDensity {
    /// Creates a constant density field.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not positive and finite.
    #[must_use]
    pub fn new(density: f64) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "density field maximum must be positive and finite"
        );
        Self { density }
    }

    /// Returns the stored density.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.density
    }
}

impl DensityField for ConstantDensity {
    fn density(&self, _point: Point, _time: f64) -> f64 {
        self.density
    }

    fn max_density(&self) -> f64 {
        self.density
    }
}

/// Closure-backed density field with an explicit majorant.
pub struct FnDensityField<F> {
    density_fn: F,
    max_density: f64,
}

impl<F> fmt::Debug for FnDensityField<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnDensityField")
            .field("max_density", &self.max_density)
            .finish_non_exhaustive()
    }
}

impl<F> FnDensityField<F> {
    /// Creates a density field from a closure and explicit maximum density.
    ///
    /// # Panics
    ///
    /// Panics if `max_density` is not positive and finite.
    #[must_use]
    pub fn new(max_density: f64, density_fn: F) -> Self {
        assert!(
            max_density.is_finite() && max_density > 0.0,
            "density field maximum must be positive and finite"
        );
        Self {
            density_fn,
            max_density,
        }
    }

    /// Returns the explicit maximum density.
    #[must_use]
    pub const fn maximum_density(&self) -> f64 {
        self.max_density
    }
}

impl<F> DensityField for FnDensityField<F>
where
    F: Fn(Point, f64) -> f64 + Send + Sync,
{
    fn density(&self, point: Point, time: f64) -> f64 {
        (self.density_fn)(point, time)
    }

    fn max_density(&self) -> f64 {
        self.max_density
    }
}

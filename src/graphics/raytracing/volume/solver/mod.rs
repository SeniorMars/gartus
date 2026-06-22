//! Stable-fluid grid solvers that can export density fields for volume rendering.

mod advection;
mod linear_solve;
mod mac;
mod mac3;
mod projection;

use super::grid::{GridBounds, GridDensityField, GridInterpolation};

pub use mac::{
    MacFluidEmitter, MacFluidGrid2, MacProjectionStats, MacScalarAdvection, MacStepStats,
};
pub use mac3::{MacCellFlags, MacFluidGrid3, MacScalarGrid3};

const DEFAULT_DT: f64 = 1.0 / 60.0;
const DEFAULT_SOLVER_ITERATIONS: usize = 20;

/// Radial source used to inject smoke density and velocity into a stable-fluid grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StableFluidEmitter {
    center: [f64; 2],
    radius: f64,
    density: f64,
    velocity: [f64; 2],
}

impl StableFluidEmitter {
    /// Creates a radial emitter centered in solver cell coordinates.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    #[must_use]
    pub fn new(center: [f64; 2], radius: f64) -> Self {
        validate_point2(center, "stable-fluid emitter center");
        validate_radius(radius, "stable-fluid emitter radius");
        Self {
            center,
            radius,
            density: 1.0,
            velocity: [0.0, 0.0],
        }
    }

    /// Returns a copy with a different density injection amount.
    ///
    /// # Panics
    ///
    /// Panics if `density` is not finite or is negative.
    #[must_use]
    pub fn with_density(mut self, density: f64) -> Self {
        assert!(
            density.is_finite() && density >= 0.0,
            "stable-fluid emitter density must be finite and non-negative"
        );
        self.density = density;
        self
    }

    /// Returns a copy with a different velocity injection amount.
    ///
    /// # Panics
    ///
    /// Panics if either velocity component is not finite.
    #[must_use]
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> Self {
        validate_point2(velocity, "stable-fluid emitter velocity");
        self.velocity = velocity;
        self
    }

    /// Returns the emitter center in solver cell coordinates.
    #[must_use]
    pub const fn center(self) -> [f64; 2] {
        self.center
    }

    /// Returns the emitter radius in solver cells.
    #[must_use]
    pub const fn radius(self) -> f64 {
        self.radius
    }

    /// Returns the density amount injected at the emitter center.
    #[must_use]
    pub const fn density(self) -> f64 {
        self.density
    }

    /// Returns the velocity amount injected at the emitter center.
    #[must_use]
    pub const fn velocity(self) -> [f64; 2] {
        self.velocity
    }
}

/// A 2D stable-fluids smoke solver backed by density and velocity grids.
///
/// The solver uses semi-Lagrangian advection, Gauss-Seidel diffusion, and projection to reduce
/// velocity divergence. It is intended as a practical smoke-density source, not as a high-accuracy
/// computational fluid dynamics solver. Export it with [`Self::to_density_volume`] to render the
/// simulation as a thick [`GridDensityField`] through [`super::NonUniformMedium`].
#[derive(Clone, Debug)]
pub struct StableFluidGrid2 {
    dims: [usize; 2],
    dt: f64,
    diffusion: f64,
    viscosity: f64,
    solver_iterations: usize,
    density: Vec<f32>,
    previous_density: Vec<f32>,
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    previous_velocity_x: Vec<f32>,
    previous_velocity_y: Vec<f32>,
    solid: Vec<bool>,
}

impl StableFluidGrid2 {
    /// Creates an empty 2D stable-fluid grid.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is smaller than three or if the grid cell count overflows.
    #[must_use]
    pub fn new(dims: [usize; 2]) -> Self {
        validate_dims(dims);
        let cell_count = cell_count_for_dims(dims);
        Self {
            dims,
            dt: DEFAULT_DT,
            diffusion: 0.0,
            viscosity: 0.0,
            solver_iterations: DEFAULT_SOLVER_ITERATIONS,
            density: vec![0.0; cell_count],
            previous_density: vec![0.0; cell_count],
            velocity_x: vec![0.0; cell_count],
            velocity_y: vec![0.0; cell_count],
            previous_velocity_x: vec![0.0; cell_count],
            previous_velocity_y: vec![0.0; cell_count],
            solid: vec![false; cell_count],
        }
    }

    /// Returns a copy with a different simulation timestep, in seconds.
    ///
    /// # Panics
    ///
    /// Panics if `dt` is not positive and finite.
    #[must_use]
    pub fn with_dt(mut self, dt: f64) -> Self {
        assert!(
            dt.is_finite() && dt > 0.0,
            "stable-fluid timestep must be positive and finite"
        );
        self.dt = dt;
        self
    }

    /// Returns a copy with a different scalar-density diffusion coefficient.
    ///
    /// # Panics
    ///
    /// Panics if `diffusion` is not finite or is negative.
    #[must_use]
    pub fn with_diffusion(mut self, diffusion: f64) -> Self {
        assert!(
            diffusion.is_finite() && diffusion >= 0.0,
            "stable-fluid diffusion must be finite and non-negative"
        );
        self.diffusion = diffusion;
        self
    }

    /// Returns a copy with a different velocity viscosity coefficient.
    ///
    /// # Panics
    ///
    /// Panics if `viscosity` is not finite or is negative.
    #[must_use]
    pub fn with_viscosity(mut self, viscosity: f64) -> Self {
        assert!(
            viscosity.is_finite() && viscosity >= 0.0,
            "stable-fluid viscosity must be finite and non-negative"
        );
        self.viscosity = viscosity;
        self
    }

    /// Returns a copy with a different Gauss-Seidel iteration count.
    ///
    /// # Panics
    ///
    /// Panics if `solver_iterations` is zero.
    #[must_use]
    pub fn with_solver_iterations(mut self, solver_iterations: usize) -> Self {
        assert!(
            solver_iterations > 0,
            "stable-fluid solver iterations must be non-zero"
        );
        self.solver_iterations = solver_iterations;
        self
    }

    /// Returns the grid dimensions as `[width, height]`.
    #[must_use]
    pub const fn dims(&self) -> [usize; 2] {
        self.dims
    }

    /// Returns the timestep, in seconds.
    #[must_use]
    pub const fn dt(&self) -> f64 {
        self.dt
    }

    /// Returns the scalar-density diffusion coefficient.
    #[must_use]
    pub const fn diffusion(&self) -> f64 {
        self.diffusion
    }

    /// Returns the velocity viscosity coefficient.
    #[must_use]
    pub const fn viscosity(&self) -> f64 {
        self.viscosity
    }

    /// Returns the Gauss-Seidel iteration count.
    #[must_use]
    pub const fn solver_iterations(&self) -> usize {
        self.solver_iterations
    }

    /// Returns the flattened index for a grid cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn index(&self, cell: [usize; 2]) -> usize {
        self.checked_index(cell)
            .expect("stable-fluid grid index out of bounds")
    }

    /// Returns the density at one grid cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn density_at(&self, cell: [usize; 2]) -> f64 {
        f64::from(self.density[self.index(cell)])
    }

    /// Returns the velocity at one grid cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn velocity_at(&self, cell: [usize; 2]) -> [f64; 2] {
        let index = self.index(cell);
        [
            f64::from(self.velocity_x[index]),
            f64::from(self.velocity_y[index]),
        ]
    }

    /// Returns all density samples in row-major order.
    #[must_use]
    pub fn densities(&self) -> &[f32] {
        &self.density
    }

    /// Returns all x velocity samples in row-major order.
    #[must_use]
    pub fn velocity_x(&self) -> &[f32] {
        &self.velocity_x
    }

    /// Returns all y velocity samples in row-major order.
    #[must_use]
    pub fn velocity_y(&self) -> &[f32] {
        &self.velocity_y
    }

    /// Returns the solid-cell mask in row-major order.
    #[must_use]
    pub fn solid_cells(&self) -> &[bool] {
        &self.solid
    }

    /// Returns whether one grid cell is solid.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn is_solid(&self, cell: [usize; 2]) -> bool {
        self.solid[self.index(cell)]
    }

    /// Replaces the solid-cell mask.
    ///
    /// Solid cells hold no density or velocity. Fluid cells adjacent to solids keep tangential
    /// velocity but have normal velocity into the obstacle suppressed.
    ///
    /// # Panics
    ///
    /// Panics if `solid.len()` does not match the grid cell count.
    pub fn set_solid_mask(&mut self, solid: Vec<bool>) {
        assert_eq!(
            solid.len(),
            self.density.len(),
            "stable-fluid solid mask length must match grid dimensions"
        );
        self.solid = solid;
        self.apply_solid_constraints();
    }

    /// Sets whether one grid cell is solid.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    pub fn set_solid(&mut self, cell: [usize; 2], solid: bool) {
        let index = self.index(cell);
        self.solid[index] = solid;
        self.apply_solid_constraints();
    }

    /// Clears all solid cells.
    pub fn clear_solids(&mut self) {
        self.solid.fill(false);
    }

    /// Sets a rectangular solid-cell region.
    ///
    /// `min` is inclusive and `max` is exclusive, matching Rust range conventions.
    ///
    /// # Panics
    ///
    /// Panics if the rectangle is outside the grid or if any `min` component exceeds `max`.
    pub fn set_solid_rect(&mut self, min: [usize; 2], max: [usize; 2], solid: bool) {
        assert!(
            min[0] <= max[0] && min[1] <= max[1],
            "stable-fluid solid rectangle min must not exceed max"
        );
        assert!(
            max[0] <= self.dims[0] && max[1] <= self.dims[1],
            "stable-fluid solid rectangle must be inside the grid"
        );
        for y in min[1]..max[1] {
            for x in min[0]..max[0] {
                self.solid[index_for_dims(self.dims, x, y)] = solid;
            }
        }
        self.apply_solid_constraints();
    }

    /// Sets a circular solid-cell region.
    ///
    /// `center` and `radius` are expressed in solver cell coordinates.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    pub fn set_solid_circle(&mut self, center: [f64; 2], radius: f64, solid: bool) {
        validate_point2(center, "stable-fluid solid circle center");
        validate_radius(radius, "stable-fluid solid circle radius");
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let cell = [usize_to_f64(x), usize_to_f64(y)];
                let dx = cell[0] - center[0];
                let dy = cell[1] - center[1];
                if dx * dx + dy * dy <= radius * radius {
                    self.solid[index_for_dims(self.dims, x, y)] = solid;
                }
            }
        }
        self.apply_solid_constraints();
    }

    /// Adds density to one grid cell.
    ///
    /// Negative amounts remove density but the stored value is clamped to zero.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or `amount` is not finite.
    pub fn add_density(&mut self, cell: [usize; 2], amount: f64) {
        assert!(
            amount.is_finite(),
            "stable-fluid density amount must be finite"
        );
        let index = self.index(cell);
        if self.solid[index] {
            return;
        }
        let density = f64::from(self.density[index]) + amount;
        self.density[index] = nonnegative_f32(density);
    }

    /// Sets density at one grid cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or `density` is not finite.
    pub fn set_density(&mut self, cell: [usize; 2], density: f64) {
        assert!(
            density.is_finite(),
            "stable-fluid density value must be finite"
        );
        let index = self.index(cell);
        if self.solid[index] {
            self.density[index] = 0.0;
            return;
        }
        self.density[index] = nonnegative_f32(density);
    }

    /// Adds velocity to one grid cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or either velocity component is not
    /// finite.
    pub fn add_velocity(&mut self, cell: [usize; 2], velocity: [f64; 2]) {
        assert!(
            velocity[0].is_finite() && velocity[1].is_finite(),
            "stable-fluid velocity must be finite"
        );
        let index = self.index(cell);
        if self.solid[index] {
            return;
        }
        self.velocity_x[index] = finite_f32(f64::from(self.velocity_x[index]) + velocity[0]);
        self.velocity_y[index] = finite_f32(f64::from(self.velocity_y[index]) + velocity[1]);
        self.apply_solid_velocity_constraints();
    }

    /// Sets velocity at one grid cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or either velocity component is not
    /// finite.
    pub fn set_velocity(&mut self, cell: [usize; 2], velocity: [f64; 2]) {
        assert!(
            velocity[0].is_finite() && velocity[1].is_finite(),
            "stable-fluid velocity must be finite"
        );
        let index = self.index(cell);
        if self.solid[index] {
            self.velocity_x[index] = 0.0;
            self.velocity_y[index] = 0.0;
            return;
        }
        self.velocity_x[index] = finite_f32(velocity[0]);
        self.velocity_y[index] = finite_f32(velocity[1]);
        self.apply_solid_velocity_constraints();
    }

    /// Clears all density samples.
    pub fn clear_density(&mut self) {
        self.density.fill(0.0);
        self.previous_density.fill(0.0);
    }

    /// Clears all velocity samples.
    pub fn clear_velocity(&mut self) {
        self.velocity_x.fill(0.0);
        self.velocity_y.fill(0.0);
        self.previous_velocity_x.fill(0.0);
        self.previous_velocity_y.fill(0.0);
    }

    /// Injects a radial density and velocity source.
    ///
    /// The emitter center and radius are expressed in solver cell coordinates. Density and velocity
    /// smoothly fade to zero at the emitter radius.
    pub fn apply_emitter(&mut self, emitter: StableFluidEmitter) {
        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let cell = [usize_to_f64(x), usize_to_f64(y)];
                let falloff = radial_falloff(cell, emitter.center, emitter.radius, 2.0);
                if falloff <= 0.0 {
                    continue;
                }
                let index = index_for_dims(self.dims, x, y);
                if self.solid[index] {
                    continue;
                }
                let density = f64::from(self.density[index]) + emitter.density * falloff;
                self.density[index] = nonnegative_f32(density);
                self.velocity_x[index] =
                    finite_f32(f64::from(self.velocity_x[index]) + emitter.velocity[0] * falloff);
                self.velocity_y[index] =
                    finite_f32(f64::from(self.velocity_y[index]) + emitter.velocity[1] * falloff);
            }
        }
        set_boundary(self.dims, &mut self.density, BoundaryCondition::Scalar);
        set_velocity_boundary(self.dims, &mut self.velocity_x, &mut self.velocity_y);
        self.apply_solid_constraints();
    }

    /// Adds an outward radial velocity impulse.
    ///
    /// `center` and `radius` are expressed in solver cell coordinates. Positive `strength` pushes
    /// outward; negative `strength` pulls inward.
    ///
    /// # Panics
    ///
    /// Panics if any argument is not finite or if `radius` is not positive.
    pub fn add_radial_impulse(&mut self, center: [f64; 2], radius: f64, strength: f64) {
        validate_point2(center, "stable-fluid radial impulse center");
        validate_radius(radius, "stable-fluid radial impulse radius");
        assert!(
            strength.is_finite(),
            "stable-fluid radial impulse strength must be finite"
        );

        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let cell = [usize_to_f64(x), usize_to_f64(y)];
                let dx = cell[0] - center[0];
                let dy = cell[1] - center[1];
                let distance = dx.hypot(dy);
                if distance <= f64::EPSILON || distance >= radius {
                    continue;
                }

                let falloff = (1.0 - distance / radius).max(0.0).powi(2);
                let impulse = strength * falloff;
                let index = index_for_dims(self.dims, x, y);
                if self.solid[index] {
                    continue;
                }
                self.velocity_x[index] =
                    finite_f32(f64::from(self.velocity_x[index]) + impulse * dx / distance);
                self.velocity_y[index] =
                    finite_f32(f64::from(self.velocity_y[index]) + impulse * dy / distance);
            }
        }
        set_velocity_boundary(self.dims, &mut self.velocity_x, &mut self.velocity_y);
        self.apply_solid_velocity_constraints();
    }

    /// Adds a uniform wind velocity to interior cells.
    ///
    /// # Panics
    ///
    /// Panics if either velocity component is not finite.
    pub fn add_wind(&mut self, velocity: [f64; 2]) {
        validate_point2(velocity, "stable-fluid wind velocity");
        self.add_wind_field(|_cell| velocity);
    }

    /// Adds a spatially varying wind velocity to interior cells.
    ///
    /// The closure receives solver cell coordinates and must return finite velocity components.
    ///
    /// # Panics
    ///
    /// Panics if the wind field returns a non-finite velocity.
    pub fn add_wind_field<F>(&mut self, mut wind: F)
    where
        F: FnMut([f64; 2]) -> [f64; 2],
    {
        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let velocity = wind([usize_to_f64(x), usize_to_f64(y)]);
                validate_point2(velocity, "stable-fluid wind field velocity");
                let index = index_for_dims(self.dims, x, y);
                if self.solid[index] {
                    continue;
                }
                self.velocity_x[index] =
                    finite_f32(f64::from(self.velocity_x[index]) + velocity[0]);
                self.velocity_y[index] =
                    finite_f32(f64::from(self.velocity_y[index]) + velocity[1]);
            }
        }
        set_velocity_boundary(self.dims, &mut self.velocity_x, &mut self.velocity_y);
        self.apply_solid_velocity_constraints();
    }

    /// Adds density-driven upward velocity.
    ///
    /// # Panics
    ///
    /// Panics if `strength` is not finite.
    pub fn apply_buoyancy(&mut self, strength: f64) {
        assert!(
            strength.is_finite(),
            "stable-fluid buoyancy strength must be finite"
        );
        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let index = index_for_dims(self.dims, x, y);
                if self.solid[index] {
                    continue;
                }
                let impulse = self.dt * strength * f64::from(self.density[index]);
                self.velocity_y[index] = finite_f32(f64::from(self.velocity_y[index]) + impulse);
            }
        }
        set_velocity_boundary(self.dims, &mut self.velocity_x, &mut self.velocity_y);
        self.apply_solid_velocity_constraints();
    }

    /// Adds vorticity confinement to preserve small swirling detail.
    ///
    /// This is an art-directed force for visual smoke. Larger strengths add more curl energy.
    ///
    /// # Panics
    ///
    /// Panics if `strength` is not finite.
    pub fn apply_vorticity_confinement(&mut self, strength: f64) {
        assert!(
            strength.is_finite(),
            "stable-fluid vorticity strength must be finite"
        );
        if strength == 0.0 {
            return;
        }

        let mut curl = vec![0.0; self.density.len()];
        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let index = index_for_dims(self.dims, x, y);
                if self.solid[index] {
                    continue;
                }
                let dvy_dx = f64::from(
                    self.velocity_y[index_for_dims(self.dims, x + 1, y)]
                        - self.velocity_y[index_for_dims(self.dims, x - 1, y)],
                ) * 0.5;
                let dvx_dy = f64::from(
                    self.velocity_x[index_for_dims(self.dims, x, y + 1)]
                        - self.velocity_x[index_for_dims(self.dims, x, y - 1)],
                ) * 0.5;
                curl[index] = dvy_dx - dvx_dy;
            }
        }

        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let index = index_for_dims(self.dims, x, y);
                if self.solid[index] {
                    continue;
                }
                let gradient_x = 0.5
                    * (curl[index_for_dims(self.dims, x + 1, y)].abs()
                        - curl[index_for_dims(self.dims, x - 1, y)].abs());
                let gradient_y = 0.5
                    * (curl[index_for_dims(self.dims, x, y + 1)].abs()
                        - curl[index_for_dims(self.dims, x, y - 1)].abs());
                let gradient_length = gradient_x.hypot(gradient_y);
                if gradient_length <= f64::EPSILON {
                    continue;
                }

                let normal_x = gradient_x / gradient_length;
                let normal_y = gradient_y / gradient_length;
                let force_scale = strength * self.dt * curl[index];
                self.velocity_x[index] =
                    finite_f32(f64::from(self.velocity_x[index]) + normal_y * force_scale);
                self.velocity_y[index] =
                    finite_f32(f64::from(self.velocity_y[index]) - normal_x * force_scale);
            }
        }
        set_velocity_boundary(self.dims, &mut self.velocity_x, &mut self.velocity_y);
        self.apply_solid_velocity_constraints();
    }

    /// Projects the current velocity field to reduce divergence.
    pub fn project_velocity(&mut self) {
        self.apply_solid_velocity_constraints();
        projection::project_2d(
            self.dims,
            &self.solid,
            &mut self.velocity_x,
            &mut self.velocity_y,
            &mut self.previous_velocity_x,
            &mut self.previous_velocity_y,
            self.solver_iterations,
        );
    }

    /// Returns the root-mean-square divergence of the current velocity field.
    #[must_use]
    pub fn velocity_divergence_l2(&self) -> f64 {
        projection::divergence_l2_2d(self.dims, &self.solid, &self.velocity_x, &self.velocity_y)
    }

    /// Advances density and velocity by one stable-fluids step.
    pub fn step(&mut self) {
        self.previous_velocity_x.clone_from(&self.velocity_x);
        self.previous_velocity_y.clone_from(&self.velocity_y);
        linear_solve::diffuse_velocity_2d(
            self.dims,
            &self.solid,
            self.dt,
            self.viscosity,
            self.solver_iterations,
            linear_solve::VelocityFieldMut {
                x: &mut self.velocity_x,
                y: &mut self.velocity_y,
            },
            linear_solve::VelocityField {
                x: &self.previous_velocity_x,
                y: &self.previous_velocity_y,
            },
        );
        self.project_velocity();

        self.previous_velocity_x.clone_from(&self.velocity_x);
        self.previous_velocity_y.clone_from(&self.velocity_y);
        advection::advect_velocity_2d(
            self.dims,
            &self.solid,
            self.dt,
            &mut self.velocity_x,
            &mut self.velocity_y,
            &self.previous_velocity_x,
            &self.previous_velocity_y,
        );
        self.project_velocity();

        self.previous_density.clone_from(&self.density);
        linear_solve::diffuse_scalar_2d(
            self.dims,
            &self.solid,
            self.dt,
            self.diffusion,
            self.solver_iterations,
            &mut self.density,
            &self.previous_density,
        );

        self.previous_density.clone_from(&self.density);
        advection::advect_scalar_2d(
            self.dims,
            &self.solid,
            self.dt,
            &mut self.density,
            &self.previous_density,
            &self.velocity_x,
            &self.velocity_y,
        );

        sanitize_density(&mut self.density);
        sanitize_velocity(&mut self.velocity_x);
        sanitize_velocity(&mut self.velocity_y);
        self.apply_solid_constraints();
    }

    /// Exports the 2D density grid as a one-voxel-thick 3D [`GridDensityField`].
    ///
    /// The simulation x axis maps to the grid x axis. The simulation y axis maps to the grid z
    /// axis, leaving the exported grid with dimensions `[width, 1, height]`.
    ///
    /// # Panics
    ///
    /// Panics if `bounds` is invalid according to [`GridBounds::new`].
    #[must_use]
    pub fn to_density_field(&self, bounds: GridBounds) -> GridDensityField {
        let mut density = Vec::with_capacity(self.density.len());
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                density.push(self.density[index_for_dims(self.dims, x, y)]);
            }
        }

        GridDensityField::new(bounds, [self.dims[0], 1, self.dims[1]], density)
            .with_interpolation(GridInterpolation::Trilinear)
    }

    /// Exports the 2D density grid as a thick 3D [`GridDensityField`].
    ///
    /// The simulation x/y axes map to the grid x/y axes, and `depth` extrudes the simulation along
    /// the grid z axis. `falloff` controls how strongly density fades toward the front/back edges;
    /// zero creates uniform thickness.
    ///
    /// # Panics
    ///
    /// Panics if `depth` is zero, if `falloff` is not finite or is negative, or if `bounds` is
    /// invalid according to [`GridBounds::new`].
    #[must_use]
    pub fn to_density_volume(
        &self,
        bounds: GridBounds,
        depth: usize,
        falloff: f64,
    ) -> GridDensityField {
        assert!(depth > 0, "stable-fluid volume depth must be non-zero");
        assert!(
            falloff.is_finite() && falloff >= 0.0,
            "stable-fluid volume falloff must be finite and non-negative"
        );

        let dims = [self.dims[0], self.dims[1], depth];
        let mut density = Vec::with_capacity(self.density.len() * depth);
        for z_depth in 0..depth {
            let thickness = thickness_weight(z_depth, depth, falloff);
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    density.push(self.density[index_for_dims(self.dims, x, y)] * thickness);
                }
            }
        }

        GridDensityField::new(bounds, dims, density)
            .with_interpolation(GridInterpolation::Trilinear)
    }

    fn checked_index(&self, cell: [usize; 2]) -> Option<usize> {
        (cell[0] < self.dims[0] && cell[1] < self.dims[1])
            .then_some(index_for_dims(self.dims, cell[0], cell[1]))
    }

    fn apply_solid_constraints(&mut self) {
        apply_solid_scalar_boundaries(self.dims, &self.solid, &mut self.density);
        apply_solid_scalar_boundaries(self.dims, &self.solid, &mut self.previous_density);
        self.apply_solid_velocity_constraints();
    }

    fn apply_solid_velocity_constraints(&mut self) {
        apply_solid_velocity_boundaries(
            self.dims,
            &self.solid,
            &mut self.velocity_x,
            &mut self.velocity_y,
        );
        apply_solid_velocity_boundaries(
            self.dims,
            &self.solid,
            &mut self.previous_velocity_x,
            &mut self.previous_velocity_y,
        );
    }
}

#[derive(Clone, Copy)]
pub(super) enum BoundaryCondition {
    Scalar,
    HorizontalVelocity,
    VerticalVelocity,
}

pub(super) fn validate_dims(dims: [usize; 2]) {
    assert!(
        dims[0] >= 3 && dims[1] >= 3,
        "stable-fluid grid dimensions must each be at least 3"
    );
    let _ = cell_count_for_dims(dims);
}

pub(super) fn cell_count_for_dims(dims: [usize; 2]) -> usize {
    dims[0]
        .checked_mul(dims[1])
        .expect("stable-fluid grid dimensions overflow")
}

pub(super) fn index_for_dims(dims: [usize; 2], x: usize, y: usize) -> usize {
    x + dims[0] * y
}

pub(super) fn is_solid_cell(dims: [usize; 2], solid: &[bool], x: usize, y: usize) -> bool {
    debug_assert_eq!(solid.len(), cell_count_for_dims(dims));
    solid[index_for_dims(dims, x, y)]
}

pub(super) fn apply_solid_scalar_boundaries(dims: [usize; 2], solid: &[bool], field: &mut [f32]) {
    debug_assert_eq!(solid.len(), cell_count_for_dims(dims));
    debug_assert_eq!(solid.len(), field.len());
    for (value, is_solid) in field.iter_mut().zip(solid) {
        if *is_solid {
            *value = 0.0;
        }
    }
}

pub(super) fn apply_solid_velocity_boundaries(
    dims: [usize; 2],
    solid: &[bool],
    velocity_x: &mut [f32],
    velocity_y: &mut [f32],
) {
    debug_assert_eq!(solid.len(), velocity_x.len());
    debug_assert_eq!(solid.len(), velocity_y.len());

    let width = dims[0];
    let height = dims[1];
    for y in 0..height {
        for x in 0..width {
            let index = index_for_dims(dims, x, y);
            if solid[index] {
                velocity_x[index] = 0.0;
                velocity_y[index] = 0.0;
                continue;
            }

            let blocked_left = x > 0 && is_solid_cell(dims, solid, x - 1, y);
            let blocked_right = x + 1 < width && is_solid_cell(dims, solid, x + 1, y);
            let blocked_bottom = y > 0 && is_solid_cell(dims, solid, x, y - 1);
            let blocked_top = y + 1 < height && is_solid_cell(dims, solid, x, y + 1);
            if (blocked_left && velocity_x[index] < 0.0)
                || (blocked_right && velocity_x[index] > 0.0)
            {
                velocity_x[index] = 0.0;
            }
            if (blocked_bottom && velocity_y[index] < 0.0)
                || (blocked_top && velocity_y[index] > 0.0)
            {
                velocity_y[index] = 0.0;
            }
        }
    }
}

fn validate_point2(value: [f64; 2], label: &str) {
    assert!(
        value[0].is_finite() && value[1].is_finite(),
        "{label} must be finite"
    );
}

fn validate_radius(radius: f64, label: &str) {
    assert!(
        radius.is_finite() && radius > 0.0,
        "{label} must be positive and finite"
    );
}

fn radial_falloff(cell: [f64; 2], center: [f64; 2], radius: f64, exponent: f64) -> f64 {
    let dx = cell[0] - center[0];
    let dy = cell[1] - center[1];
    let distance_squared = dx * dx + dy * dy;
    let radius_squared = radius * radius;
    if distance_squared >= radius_squared {
        return 0.0;
    }

    (1.0 - distance_squared / radius_squared)
        .max(0.0)
        .powf(exponent)
}

fn thickness_weight(slice: usize, depth: usize, falloff: f64) -> f32 {
    if depth == 1 || falloff == 0.0 {
        return 1.0;
    }

    let center = usize_to_f64(depth - 1) * 0.5;
    let max_distance = center.max(usize_to_f64(depth - 1) - center);
    if max_distance <= f64::EPSILON {
        return 1.0;
    }

    let normalized_distance = ((usize_to_f64(slice) - center).abs() / max_distance).clamp(0.0, 1.0);
    finite_f32(
        (1.0 - normalized_distance * normalized_distance)
            .max(0.0)
            .powf(falloff),
    )
}

pub(super) fn set_boundary(dims: [usize; 2], field: &mut [f32], boundary: BoundaryCondition) {
    let width = dims[0];
    let height = dims[1];

    for x in 1..(width - 1) {
        let bottom = index_for_dims(dims, x, 0);
        let bottom_neighbor = index_for_dims(dims, x, 1);
        let top = index_for_dims(dims, x, height - 1);
        let top_neighbor = index_for_dims(dims, x, height - 2);
        field[bottom] = match boundary {
            BoundaryCondition::VerticalVelocity => -field[bottom_neighbor],
            BoundaryCondition::Scalar | BoundaryCondition::HorizontalVelocity => {
                field[bottom_neighbor]
            }
        };
        field[top] = match boundary {
            BoundaryCondition::VerticalVelocity => -field[top_neighbor],
            BoundaryCondition::Scalar | BoundaryCondition::HorizontalVelocity => {
                field[top_neighbor]
            }
        };
    }

    for y in 1..(height - 1) {
        let left = index_for_dims(dims, 0, y);
        let left_neighbor = index_for_dims(dims, 1, y);
        let right = index_for_dims(dims, width - 1, y);
        let right_neighbor = index_for_dims(dims, width - 2, y);
        field[left] = match boundary {
            BoundaryCondition::HorizontalVelocity => -field[left_neighbor],
            BoundaryCondition::Scalar | BoundaryCondition::VerticalVelocity => field[left_neighbor],
        };
        field[right] = match boundary {
            BoundaryCondition::HorizontalVelocity => -field[right_neighbor],
            BoundaryCondition::Scalar | BoundaryCondition::VerticalVelocity => {
                field[right_neighbor]
            }
        };
    }

    field[index_for_dims(dims, 0, 0)] =
        0.5 * (field[index_for_dims(dims, 1, 0)] + field[index_for_dims(dims, 0, 1)]);
    field[index_for_dims(dims, 0, height - 1)] = 0.5
        * (field[index_for_dims(dims, 1, height - 1)] + field[index_for_dims(dims, 0, height - 2)]);
    field[index_for_dims(dims, width - 1, 0)] = 0.5
        * (field[index_for_dims(dims, width - 2, 0)] + field[index_for_dims(dims, width - 1, 1)]);
    field[index_for_dims(dims, width - 1, height - 1)] = 0.5
        * (field[index_for_dims(dims, width - 2, height - 1)]
            + field[index_for_dims(dims, width - 1, height - 2)]);
}

pub(super) fn set_velocity_boundary(
    dims: [usize; 2],
    velocity_x: &mut [f32],
    velocity_y: &mut [f32],
) {
    set_boundary(dims, velocity_x, BoundaryCondition::HorizontalVelocity);
    set_boundary(dims, velocity_y, BoundaryCondition::VerticalVelocity);
}

pub(super) fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("stable-fluid dimension should fit in u32"))
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn finite_f32(value: f64) -> f32 {
    if value.is_finite() {
        value.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
    } else {
        0.0
    }
}

pub(super) fn nonnegative_f32(value: f64) -> f32 {
    finite_f32(value.max(0.0))
}

fn sanitize_density(density: &mut [f32]) {
    for value in density {
        if !value.is_finite() || *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn sanitize_velocity(velocity: &mut [f32]) {
    for value in velocity {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gmath::vector::Point, graphics::raytracing::volume::field::DensityField};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
    }

    fn assert_velocity_close(actual: [f64; 2], expected: [f64; 2]) {
        assert_close(actual[0], expected[0]);
        assert_close(actual[1], expected[1]);
    }

    fn density_center_of_mass_x(sim: &StableFluidGrid2) -> f64 {
        let mut weighted_sum = 0.0;
        let mut mass = 0.0;
        for y in 0..sim.dims()[1] {
            for x in 0..sim.dims()[0] {
                let density = sim.density_at([x, y]);
                weighted_sum += usize_to_f64(x) * density;
                mass += density;
            }
        }
        weighted_sum / mass.max(f64::MIN_POSITIVE)
    }

    #[test]
    fn stable_fluid_density_moves_with_velocity() {
        let mut sim = StableFluidGrid2::new([12, 8])
            .with_dt(0.8)
            .with_solver_iterations(12);
        sim.add_density([4, 4], 10.0);
        let before = density_center_of_mass_x(&sim);

        for y in 1..7 {
            for x in 1..11 {
                sim.set_velocity([x, y], [1.2, 0.0]);
            }
        }

        sim.step();
        let after = density_center_of_mass_x(&sim);

        assert!(after > before, "{after} should be greater than {before}");
    }

    #[test]
    fn projection_reduces_divergence() {
        let mut sim = StableFluidGrid2::new([12, 12]).with_solver_iterations(40);
        for y in 1..11 {
            for x in 1..11 {
                sim.set_velocity([x, y], [usize_to_f64(x) - 5.5, usize_to_f64(y) - 5.5]);
            }
        }

        let before = sim.velocity_divergence_l2();
        sim.project_velocity();
        let after = sim.velocity_divergence_l2();

        assert!(before > 0.0);
        assert!(after < before, "{after} should be less than {before}");
    }

    #[test]
    fn density_remains_finite_after_many_steps() {
        let mut sim = StableFluidGrid2::new([16, 16])
            .with_dt(0.4)
            .with_diffusion(0.001)
            .with_viscosity(0.001)
            .with_solver_iterations(16);

        for step in 0..48 {
            sim.add_density([8, 5], 3.0);
            sim.add_velocity([8, 5], [0.2 * f64::from(step % 3), 1.0]);
            sim.step();
        }

        assert!(
            sim.densities()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(sim.velocity_x().iter().all(|value| value.is_finite()));
        assert!(sim.velocity_y().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_velocity_keeps_density_stable_without_diffusion() {
        let mut sim = StableFluidGrid2::new([8, 8])
            .with_dt(1.0)
            .with_diffusion(0.0)
            .with_viscosity(0.0);
        sim.add_density([4, 4], 5.0);

        sim.step();

        assert_close(sim.density_at([4, 4]), 5.0);
        assert_close(sim.density_at([3, 4]), 0.0);
        assert_close(sim.velocity_divergence_l2(), 0.0);
    }

    #[test]
    fn stable_fluid_exports_thin_grid_density_field() {
        let mut sim = StableFluidGrid2::new([4, 3]);
        sim.add_density([2, 1], 7.0);
        let bounds = GridBounds::new(Point::new(-2.0, 0.0, -1.5), Point::new(2.0, 0.2, 1.5));
        let grid = sim.to_density_field(bounds);

        assert_eq!(grid.dims(), [4, 1, 3]);
        assert_eq!(grid.interpolation(), GridInterpolation::Trilinear);
        assert_close(grid.density(grid.cell_center(2, 0, 1), 0.0), 7.0);
    }

    #[test]
    fn stable_fluid_exports_thick_grid_density_volume() {
        let mut sim = StableFluidGrid2::new([4, 3]);
        sim.add_density([2, 1], 7.0);
        let bounds = GridBounds::new(Point::new(-2.0, -0.5, -1.5), Point::new(2.0, 0.5, 1.5));
        let grid = sim.to_density_volume(bounds, 5, 1.0);

        assert_eq!(grid.dims(), [4, 3, 5]);
        assert_eq!(grid.interpolation(), GridInterpolation::Trilinear);
        let center_index = 2 + 4 * (1 + 3 * 2);
        let edge_index = 2 + 4;
        assert_close(f64::from(grid.densities()[center_index]), 7.0);
        assert_close(f64::from(grid.densities()[edge_index]), 0.0);
        assert_close(grid.density(grid.cell_center(2, 1, 2), 0.0), 7.0);
    }

    #[test]
    fn stable_fluid_emitter_adds_density_and_velocity() {
        let mut sim = StableFluidGrid2::new([12, 12]);
        sim.apply_emitter(
            StableFluidEmitter::new([6.0, 5.0], 3.0)
                .with_density(4.0)
                .with_velocity([1.5, 2.5]),
        );

        assert!(sim.density_at([6, 5]) > 3.9);
        let velocity = sim.velocity_at([6, 5]);
        assert!(velocity[0] > 1.4);
        assert!(velocity[1] > 2.4);
    }

    #[test]
    fn stable_fluid_forces_modify_velocity_and_stay_finite() {
        let mut sim = StableFluidGrid2::new([10, 10]).with_dt(0.25);
        sim.add_density([5, 5], 2.0);

        sim.add_wind([0.25, 0.0]);
        sim.add_radial_impulse([5.0, 5.0], 4.0, 2.0);
        sim.apply_buoyancy(3.0);
        sim.set_velocity([4, 4], [0.0, 1.0]);
        sim.set_velocity([6, 4], [0.0, -1.0]);
        sim.set_velocity([4, 6], [-1.0, 0.0]);
        sim.set_velocity([6, 6], [1.0, 0.0]);
        sim.apply_vorticity_confinement(4.0);

        let center_velocity = sim.velocity_at([5, 5]);
        assert!(center_velocity[1] > 0.0);
        assert!(sim.velocity_x().iter().all(|value| value.is_finite()));
        assert!(sim.velocity_y().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn stable_fluid_solid_cells_block_sources() {
        let mut sim = StableFluidGrid2::new([12, 12]);
        sim.set_solid([6, 6], true);

        sim.add_density([6, 6], 10.0);
        sim.add_velocity([6, 6], [4.0, 3.0]);
        sim.apply_emitter(
            StableFluidEmitter::new([6.0, 6.0], 2.5)
                .with_density(4.0)
                .with_velocity([2.0, 1.0]),
        );

        assert!(sim.is_solid([6, 6]));
        assert_close(sim.density_at([6, 6]), 0.0);
        assert_velocity_close(sim.velocity_at([6, 6]), [0.0, 0.0]);
        assert!(sim.density_at([5, 6]) > 0.0);
    }

    #[test]
    fn stable_fluid_solid_boundaries_suppress_normal_velocity() {
        let mut sim = StableFluidGrid2::new([12, 12]);
        sim.set_solid([6, 6], true);

        sim.set_velocity([5, 6], [3.0, 2.0]);
        assert_velocity_close(sim.velocity_at([5, 6]), [0.0, 2.0]);
        sim.set_velocity([6, 5], [2.0, 3.0]);
        assert_velocity_close(sim.velocity_at([6, 5]), [2.0, 0.0]);
    }

    #[test]
    fn stable_fluid_steps_keep_solid_cells_empty_and_finite() {
        let mut sim = StableFluidGrid2::new([16, 12])
            .with_dt(0.4)
            .with_diffusion(0.001)
            .with_viscosity(0.001)
            .with_solver_iterations(16);
        sim.set_solid_rect([7, 3], [9, 9], true);

        for y in 4..8 {
            sim.add_density([5, y], 4.0);
            sim.set_velocity([5, y], [4.0, 0.0]);
        }
        for _ in 0..8 {
            sim.step();
        }

        for y in 3..9 {
            for x in 7..9 {
                assert_close(sim.density_at([x, y]), 0.0);
                assert_velocity_close(sim.velocity_at([x, y]), [0.0, 0.0]);
            }
        }
        assert!(
            sim.densities()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(sim.velocity_x().iter().all(|value| value.is_finite()));
        assert!(sim.velocity_y().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn builder_methods_store_solver_settings() {
        let sim = StableFluidGrid2::new([5, 6])
            .with_dt(0.25)
            .with_diffusion(0.02)
            .with_viscosity(0.03)
            .with_solver_iterations(9);

        assert_eq!(sim.dims(), [5, 6]);
        assert_close(sim.dt(), 0.25);
        assert_close(sim.diffusion(), 0.02);
        assert_close(sim.viscosity(), 0.03);
        assert_eq!(sim.solver_iterations(), 9);
    }
}

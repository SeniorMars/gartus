use super::{
    DEFAULT_DT, cell_count_for_dims, finite_f32, index_for_dims, nonnegative_f32, radial_falloff,
    thickness_weight, usize_to_f64, validate_dims, validate_point2, validate_radius,
};
use crate::graphics::raytracing::volume::grid::{GridBounds, GridDensityField, GridInterpolation};

const DEFAULT_PRESSURE_ITERATIONS: usize = 200;
const DEFAULT_PRESSURE_TOLERANCE: f64 = 1.0e-5;
const SOLID_PHI_DEFAULT: f32 = 1.0;
const FACE_WEIGHT_EPSILON: f32 = 1.0e-4;

/// Scalar advection scheme used by [`MacFluidGrid2`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MacScalarAdvection {
    /// First-order semi-Lagrangian advection.
    SemiLagrangian,
    /// `MacCormack` advection with neighborhood clamping to suppress ringing.
    MacCormack,
}

/// Diagnostic values from the most recent MAC-grid pressure projection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MacProjectionStats {
    /// Root-mean-square velocity divergence before pressure projection.
    pub divergence_before_l2: f64,
    /// Root-mean-square velocity divergence after pressure projection.
    pub divergence_after_l2: f64,
    /// Root-mean-square pressure-system residual from the PCG solve.
    pub pressure_residual_l2: f64,
    /// Number of PCG iterations used by the pressure solve.
    pub iterations: usize,
}

/// Diagnostic values from a full MAC fluid step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MacStepStats {
    /// Projection applied before advection.
    pub initial_projection: MacProjectionStats,
    /// Projection applied after velocity advection.
    pub final_projection: MacProjectionStats,
}

/// Radial source used to inject density, temperature, and momentum into a MAC fluid grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacFluidEmitter {
    center: [f64; 2],
    radius: f64,
    density: f64,
    temperature: f64,
    velocity: [f64; 2],
}

impl MacFluidEmitter {
    /// Creates a radial emitter centered in solver cell coordinates.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    #[must_use]
    pub fn new(center: [f64; 2], radius: f64) -> Self {
        validate_point2(center, "MAC fluid emitter center");
        validate_radius(radius, "MAC fluid emitter radius");
        Self {
            center,
            radius,
            density: 1.0,
            temperature: 0.0,
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
            "MAC fluid emitter density must be finite and non-negative"
        );
        self.density = density;
        self
    }

    /// Returns a copy with a different temperature injection amount.
    ///
    /// # Panics
    ///
    /// Panics if `temperature` is not finite.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        assert!(
            temperature.is_finite(),
            "MAC fluid emitter temperature must be finite"
        );
        self.temperature = temperature;
        self
    }

    /// Returns a copy with a different velocity injection amount.
    ///
    /// # Panics
    ///
    /// Panics if either velocity component is not finite.
    #[must_use]
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> Self {
        validate_point2(velocity, "MAC fluid emitter velocity");
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

    /// Returns the temperature amount injected at the emitter center.
    #[must_use]
    pub const fn temperature(self) -> f64 {
        self.temperature
    }

    /// Returns the velocity amount injected at the emitter center.
    #[must_use]
    pub const fn velocity(self) -> [f64; 2] {
        self.velocity
    }
}

/// Two-dimensional smoke solver using a Marker-and-Cell layout.
///
/// Density, temperature, and pressure are stored at cell centers. Velocity is stored on cell
/// faces: `u` has dimensions `[width + 1, height]`, and `v` has dimensions
/// `[width, height + 1]`. Obstacles are represented by a center-cell signed distance field that
/// is converted into face open fractions for projection and boundary damping.
#[derive(Clone, Debug)]
pub struct MacFluidGrid2 {
    dims: [usize; 2],
    dt: f64,
    cell_size: [f64; 2],
    pressure_iterations: usize,
    pressure_tolerance: f64,
    scalar_advection: MacScalarAdvection,
    density: Vec<f32>,
    temperature: Vec<f32>,
    pressure: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    previous_u: Vec<f32>,
    previous_v: Vec<f32>,
    solid_phi: Vec<f32>,
    u_weights: Vec<f32>,
    v_weights: Vec<f32>,
    last_projection: MacProjectionStats,
}

impl MacFluidGrid2 {
    /// Creates an empty 2D MAC smoke grid.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is smaller than three or if any grid allocation would overflow.
    #[must_use]
    pub fn new(dims: [usize; 2]) -> Self {
        validate_dims(dims);
        let cell_count = cell_count_for_dims(dims);
        let u_count = u_count_for_dims(dims);
        let v_count = v_count_for_dims(dims);
        let mut grid = Self {
            dims,
            dt: DEFAULT_DT,
            cell_size: [1.0, 1.0],
            pressure_iterations: DEFAULT_PRESSURE_ITERATIONS,
            pressure_tolerance: DEFAULT_PRESSURE_TOLERANCE,
            scalar_advection: MacScalarAdvection::MacCormack,
            density: vec![0.0; cell_count],
            temperature: vec![0.0; cell_count],
            pressure: vec![0.0; cell_count],
            u: vec![0.0; u_count],
            v: vec![0.0; v_count],
            previous_u: vec![0.0; u_count],
            previous_v: vec![0.0; v_count],
            solid_phi: vec![SOLID_PHI_DEFAULT; cell_count],
            u_weights: vec![1.0; u_count],
            v_weights: vec![1.0; v_count],
            last_projection: MacProjectionStats::default(),
        };
        grid.rebuild_face_weights();
        grid
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
            "MAC fluid timestep must be positive and finite"
        );
        self.dt = dt;
        self
    }

    /// Returns a copy with a different grid cell size.
    ///
    /// # Panics
    ///
    /// Panics if either cell-size component is not positive and finite.
    #[must_use]
    pub fn with_cell_size(mut self, cell_size: [f64; 2]) -> Self {
        assert!(
            cell_size[0].is_finite()
                && cell_size[0] > 0.0
                && cell_size[1].is_finite()
                && cell_size[1] > 0.0,
            "MAC fluid cell size must be positive and finite"
        );
        self.cell_size = cell_size;
        self
    }

    /// Returns a copy with a different PCG pressure iteration cap.
    ///
    /// # Panics
    ///
    /// Panics if `pressure_iterations` is zero.
    #[must_use]
    pub fn with_pressure_iterations(mut self, pressure_iterations: usize) -> Self {
        assert!(
            pressure_iterations > 0,
            "MAC fluid pressure iterations must be non-zero"
        );
        self.pressure_iterations = pressure_iterations;
        self
    }

    /// Returns a copy with a different pressure residual tolerance.
    ///
    /// # Panics
    ///
    /// Panics if `pressure_tolerance` is not positive and finite.
    #[must_use]
    pub fn with_pressure_tolerance(mut self, pressure_tolerance: f64) -> Self {
        assert!(
            pressure_tolerance.is_finite() && pressure_tolerance > 0.0,
            "MAC fluid pressure tolerance must be positive and finite"
        );
        self.pressure_tolerance = pressure_tolerance;
        self
    }

    /// Returns a copy with a different scalar advection scheme.
    #[must_use]
    pub const fn with_scalar_advection(mut self, scalar_advection: MacScalarAdvection) -> Self {
        self.scalar_advection = scalar_advection;
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

    /// Returns the cell size.
    #[must_use]
    pub const fn cell_size(&self) -> [f64; 2] {
        self.cell_size
    }

    /// Returns the pressure solver iteration cap.
    #[must_use]
    pub const fn pressure_iterations(&self) -> usize {
        self.pressure_iterations
    }

    /// Returns the pressure solver residual tolerance.
    #[must_use]
    pub const fn pressure_tolerance(&self) -> f64 {
        self.pressure_tolerance
    }

    /// Returns the scalar advection scheme.
    #[must_use]
    pub const fn scalar_advection(&self) -> MacScalarAdvection {
        self.scalar_advection
    }

    /// Returns all density samples in row-major cell-center order.
    #[must_use]
    pub fn densities(&self) -> &[f32] {
        &self.density
    }

    /// Returns all temperature samples in row-major cell-center order.
    #[must_use]
    pub fn temperatures(&self) -> &[f32] {
        &self.temperature
    }

    /// Returns all pressure samples in row-major cell-center order.
    #[must_use]
    pub fn pressures(&self) -> &[f32] {
        &self.pressure
    }

    /// Returns all horizontal face velocities in row-major `[width + 1, height]` order.
    #[must_use]
    pub fn u(&self) -> &[f32] {
        &self.u
    }

    /// Returns all vertical face velocities in row-major `[width, height + 1]` order.
    #[must_use]
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    /// Returns the center-cell signed distance field.
    #[must_use]
    pub fn solid_phi(&self) -> &[f32] {
        &self.solid_phi
    }

    /// Returns horizontal face open fractions in row-major `[width + 1, height]` order.
    #[must_use]
    pub fn u_weights(&self) -> &[f32] {
        &self.u_weights
    }

    /// Returns vertical face open fractions in row-major `[width, height + 1]` order.
    #[must_use]
    pub fn v_weights(&self) -> &[f32] {
        &self.v_weights
    }

    /// Returns projection diagnostics from the most recent call to [`Self::project_velocity`].
    #[must_use]
    pub const fn last_projection(&self) -> MacProjectionStats {
        self.last_projection
    }

    /// Returns the flattened index for a cell-center sample.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn index(&self, cell: [usize; 2]) -> usize {
        self.checked_index(cell)
            .expect("MAC fluid grid index out of bounds")
    }

    /// Returns the flattened index for a horizontal face velocity.
    ///
    /// # Panics
    ///
    /// Panics if `face[0] > width` or `face[1] >= height`.
    #[must_use]
    pub fn u_index(&self, face: [usize; 2]) -> usize {
        assert!(
            face[0] <= self.dims[0] && face[1] < self.dims[1],
            "MAC u-face index out of bounds"
        );
        u_index_for_dims(self.dims, face[0], face[1])
    }

    /// Returns the flattened index for a vertical face velocity.
    ///
    /// # Panics
    ///
    /// Panics if `face[0] >= width` or `face[1] > height`.
    #[must_use]
    pub fn v_index(&self, face: [usize; 2]) -> usize {
        assert!(
            face[0] < self.dims[0] && face[1] <= self.dims[1],
            "MAC v-face index out of bounds"
        );
        v_index_for_dims(self.dims, face[0], face[1])
    }

    /// Returns the density at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn density_at(&self, cell: [usize; 2]) -> f64 {
        f64::from(self.density[self.index(cell)])
    }

    /// Returns the temperature at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn temperature_at(&self, cell: [usize; 2]) -> f64 {
        f64::from(self.temperature[self.index(cell)])
    }

    /// Returns the horizontal velocity at one u face.
    ///
    /// # Panics
    ///
    /// Panics if the face coordinate is outside the u grid dimensions.
    #[must_use]
    pub fn u_at(&self, face: [usize; 2]) -> f64 {
        f64::from(self.u[self.u_index(face)])
    }

    /// Returns the vertical velocity at one v face.
    ///
    /// # Panics
    ///
    /// Panics if the face coordinate is outside the v grid dimensions.
    #[must_use]
    pub fn v_at(&self, face: [usize; 2]) -> f64 {
        f64::from(self.v[self.v_index(face)])
    }

    /// Returns the face-averaged cell-center velocity.
    ///
    /// # Panics
    ///
    /// Panics if the cell coordinate is outside the grid dimensions.
    #[must_use]
    pub fn velocity_at_cell(&self, cell: [usize; 2]) -> [f64; 2] {
        let x = cell[0];
        let y = cell[1];
        assert!(
            x < self.dims[0] && y < self.dims[1],
            "MAC fluid grid index out of bounds"
        );
        self.center_velocity_from_faces(&self.u, &self.v, x, y)
    }

    /// Returns true when one center cell is inside solid geometry.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions.
    #[must_use]
    pub fn is_solid(&self, cell: [usize; 2]) -> bool {
        self.solid_phi[self.index(cell)] <= 0.0
    }

    /// Replaces the center-cell signed distance field and rebuilds face open fractions.
    ///
    /// Negative values are solid, positive values are fluid, and zero lies on the obstacle
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics if `solid_phi.len()` does not match the grid cell count or contains non-finite
    /// values.
    pub fn set_solid_phi(&mut self, solid_phi: Vec<f32>) {
        assert_eq!(
            solid_phi.len(),
            self.solid_phi.len(),
            "MAC fluid SDF length must match grid dimensions"
        );
        assert!(
            solid_phi.iter().all(|value| value.is_finite()),
            "MAC fluid SDF values must be finite"
        );
        self.solid_phi = solid_phi;
        self.rebuild_face_weights();
        self.apply_obstacle_constraints();
    }

    /// Samples a center-cell signed distance field from a closure.
    ///
    /// The closure receives solver cell coordinates. Negative values are solid, positive values are
    /// fluid, and zero lies on the obstacle surface.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a non-finite value.
    pub fn set_solid_sdf<F>(&mut self, mut sdf: F)
    where
        F: FnMut([f64; 2]) -> f64,
    {
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let value = sdf([usize_to_f64(x), usize_to_f64(y)]);
                assert!(value.is_finite(), "MAC fluid SDF values must be finite");
                let index = index_for_dims(self.dims, x, y);
                self.solid_phi[index] = finite_f32(value);
            }
        }
        self.rebuild_face_weights();
        self.apply_obstacle_constraints();
    }

    /// Clears all obstacle geometry.
    pub fn clear_obstacles(&mut self) {
        self.solid_phi.fill(SOLID_PHI_DEFAULT);
        self.rebuild_face_weights();
        self.apply_obstacle_constraints();
    }

    /// Unions a circular signed-distance obstacle into the grid.
    ///
    /// `center` and `radius` are expressed in solver cell coordinates.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    pub fn add_solid_circle(&mut self, center: [f64; 2], radius: f64) {
        validate_point2(center, "MAC fluid solid circle center");
        validate_radius(radius, "MAC fluid solid circle radius");
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let cell = [usize_to_f64(x), usize_to_f64(y)];
                let distance = (cell[0] - center[0]).hypot(cell[1] - center[1]) - radius;
                let index = index_for_dims(self.dims, x, y);
                self.solid_phi[index] = self.solid_phi[index].min(finite_f32(distance));
            }
        }
        self.rebuild_face_weights();
        self.apply_obstacle_constraints();
    }

    /// Replaces all obstacle geometry with one circular signed-distance obstacle.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    pub fn set_solid_circle(&mut self, center: [f64; 2], radius: f64) {
        self.clear_obstacles();
        self.add_solid_circle(center, radius);
    }

    /// Adds density to one center cell.
    ///
    /// Negative amounts remove density but the stored value is clamped to zero.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or `amount` is not finite.
    pub fn add_density(&mut self, cell: [usize; 2], amount: f64) {
        assert!(
            amount.is_finite(),
            "MAC fluid density amount must be finite"
        );
        let index = self.index(cell);
        if self.solid_phi[index] <= 0.0 {
            return;
        }
        self.density[index] = nonnegative_f32(f64::from(self.density[index]) + amount);
    }

    /// Sets density at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or `density` is not finite.
    pub fn set_density(&mut self, cell: [usize; 2], density: f64) {
        assert!(
            density.is_finite(),
            "MAC fluid density value must be finite"
        );
        let index = self.index(cell);
        self.density[index] = if self.solid_phi[index] <= 0.0 {
            0.0
        } else {
            nonnegative_f32(density)
        };
    }

    /// Adds temperature to one center cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or `amount` is not finite.
    pub fn add_temperature(&mut self, cell: [usize; 2], amount: f64) {
        assert!(
            amount.is_finite(),
            "MAC fluid temperature amount must be finite"
        );
        let index = self.index(cell);
        if self.solid_phi[index] <= 0.0 {
            return;
        }
        self.temperature[index] = finite_f32(f64::from(self.temperature[index]) + amount);
    }

    /// Sets temperature at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or `temperature` is not finite.
    pub fn set_temperature(&mut self, cell: [usize; 2], temperature: f64) {
        assert!(
            temperature.is_finite(),
            "MAC fluid temperature value must be finite"
        );
        let index = self.index(cell);
        self.temperature[index] = if self.solid_phi[index] <= 0.0 {
            0.0
        } else {
            finite_f32(temperature)
        };
    }

    /// Sets horizontal velocity at one u face.
    ///
    /// # Panics
    ///
    /// Panics if the face coordinate is outside the u grid dimensions or `velocity` is not finite.
    pub fn set_u(&mut self, face: [usize; 2], velocity: f64) {
        assert!(velocity.is_finite(), "MAC u velocity must be finite");
        let index = self.u_index(face);
        self.u[index] = if self.u_weights[index] <= FACE_WEIGHT_EPSILON {
            0.0
        } else {
            finite_f32(velocity)
        };
    }

    /// Sets vertical velocity at one v face.
    ///
    /// # Panics
    ///
    /// Panics if the face coordinate is outside the v grid dimensions or `velocity` is not finite.
    pub fn set_v(&mut self, face: [usize; 2], velocity: f64) {
        assert!(velocity.is_finite(), "MAC v velocity must be finite");
        let index = self.v_index(face);
        self.v[index] = if self.v_weights[index] <= FACE_WEIGHT_EPSILON {
            0.0
        } else {
            finite_f32(velocity)
        };
    }

    /// Adds velocity to the faces around one center cell.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid dimensions or either velocity component is not
    /// finite.
    pub fn add_velocity_at_cell(&mut self, cell: [usize; 2], velocity: [f64; 2]) {
        validate_point2(velocity, "MAC fluid cell velocity");
        let x = cell[0];
        let y = cell[1];
        assert!(
            x < self.dims[0] && y < self.dims[1],
            "MAC fluid grid index out of bounds"
        );
        if self.solid_phi[index_for_dims(self.dims, x, y)] <= 0.0 {
            return;
        }

        self.add_velocity_to_cell_faces(x, y, velocity);
        self.apply_obstacle_velocity_constraints();
    }

    /// Clears all density and temperature samples.
    pub fn clear_scalars(&mut self) {
        self.density.fill(0.0);
        self.temperature.fill(0.0);
    }

    /// Clears all face velocities and pressure samples.
    pub fn clear_velocity(&mut self) {
        self.u.fill(0.0);
        self.v.fill(0.0);
        self.previous_u.fill(0.0);
        self.previous_v.fill(0.0);
        self.pressure.fill(0.0);
        self.last_projection = MacProjectionStats::default();
    }

    /// Injects a radial density, temperature, and momentum source.
    pub fn apply_emitter(&mut self, emitter: MacFluidEmitter) {
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let cell = [usize_to_f64(x), usize_to_f64(y)];
                let falloff = radial_falloff(cell, emitter.center, emitter.radius, 2.0);
                if falloff <= 0.0 {
                    continue;
                }
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                self.density[index] =
                    nonnegative_f32(f64::from(self.density[index]) + emitter.density * falloff);
                self.temperature[index] =
                    finite_f32(f64::from(self.temperature[index]) + emitter.temperature * falloff);
                self.add_velocity_to_cell_faces(
                    x,
                    y,
                    [emitter.velocity[0] * falloff, emitter.velocity[1] * falloff],
                );
            }
        }
        self.apply_obstacle_constraints();
    }

    /// Adds a uniform wind velocity to all open faces.
    ///
    /// # Panics
    ///
    /// Panics if either velocity component is not finite.
    pub fn add_wind(&mut self, velocity: [f64; 2]) {
        validate_point2(velocity, "MAC fluid wind velocity");
        for u in &mut self.u {
            *u = finite_f32(f64::from(*u) + velocity[0]);
        }
        for v in &mut self.v {
            *v = finite_f32(f64::from(*v) + velocity[1]);
        }
        self.apply_obstacle_velocity_constraints();
    }

    /// Adds buoyant vertical velocity from temperature above an ambient value.
    ///
    /// # Panics
    ///
    /// Panics if either argument is not finite.
    pub fn apply_buoyancy(&mut self, strength: f64, ambient_temperature: f64) {
        assert!(
            strength.is_finite() && ambient_temperature.is_finite(),
            "MAC fluid buoyancy arguments must be finite"
        );
        if strength == 0.0 {
            return;
        }

        let width = self.dims[0];
        let height = self.dims[1];
        for y_face in 1..height {
            for x in 0..width {
                let below = index_for_dims(self.dims, x, y_face - 1);
                let above = index_for_dims(self.dims, x, y_face);
                let temperature =
                    0.5 * (f64::from(self.temperature[below]) + f64::from(self.temperature[above]));
                let impulse = self.dt * strength * (temperature - ambient_temperature);
                let index = v_index_for_dims(self.dims, x, y_face);
                self.v[index] = finite_f32(f64::from(self.v[index]) + impulse);
            }
        }
        self.apply_obstacle_velocity_constraints();
    }

    /// Adds vorticity confinement to preserve small swirling detail.
    ///
    /// This remains an art-directed force, but it is sampled from the face-centered velocity field.
    ///
    /// # Panics
    ///
    /// Panics if `strength` is not finite.
    pub fn apply_vorticity_confinement(&mut self, strength: f64) {
        assert!(
            strength.is_finite(),
            "MAC fluid vorticity strength must be finite"
        );
        if strength == 0.0 {
            return;
        }

        let cell_count = cell_count_for_dims(self.dims);
        let mut curl = vec![0.0_f64; cell_count];
        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let dv_dx = (self.center_velocity_from_faces(&self.u, &self.v, x + 1, y)[1]
                    - self.center_velocity_from_faces(&self.u, &self.v, x - 1, y)[1])
                    / (2.0 * self.cell_size[0]);
                let du_dy = (self.center_velocity_from_faces(&self.u, &self.v, x, y + 1)[0]
                    - self.center_velocity_from_faces(&self.u, &self.v, x, y - 1)[0])
                    / (2.0 * self.cell_size[1]);
                curl[index] = dv_dx - du_dy;
            }
        }

        for y in 1..(self.dims[1] - 1) {
            for x in 1..(self.dims[0] - 1) {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let grad_x = (curl[index_for_dims(self.dims, x + 1, y)].abs()
                    - curl[index_for_dims(self.dims, x - 1, y)].abs())
                    / (2.0 * self.cell_size[0]);
                let grad_y = (curl[index_for_dims(self.dims, x, y + 1)].abs()
                    - curl[index_for_dims(self.dims, x, y - 1)].abs())
                    / (2.0 * self.cell_size[1]);
                let grad_len = grad_x.hypot(grad_y);
                if grad_len <= f64::EPSILON {
                    continue;
                }
                let normal_x = grad_x / grad_len;
                let normal_y = grad_y / grad_len;
                let force = strength * self.dt * curl[index];
                self.add_velocity_to_cell_faces(x, y, [normal_y * force, -normal_x * force]);
            }
        }
        self.apply_obstacle_velocity_constraints();
    }

    /// Returns the root-mean-square divergence of the weighted face velocity field.
    #[must_use]
    pub fn velocity_divergence_l2(&self) -> f64 {
        self.divergence_l2()
    }

    /// Returns a conservative maximum velocity magnitude from open face speeds.
    #[must_use]
    pub fn max_velocity_magnitude(&self) -> f64 {
        let max_u = max_open_face_speed(&self.u, &self.u_weights);
        let max_v = max_open_face_speed(&self.v, &self.v_weights);
        max_u.hypot(max_v)
    }

    /// Returns a CFL-limited timestep no larger than the configured `dt`.
    ///
    /// # Panics
    ///
    /// Panics if `cfl_number` is not positive and finite.
    #[must_use]
    pub fn cfl_timestep(&self, cfl_number: f64) -> f64 {
        assert!(
            cfl_number.is_finite() && cfl_number > 0.0,
            "MAC fluid CFL number must be positive and finite"
        );
        let rate = self.max_cfl_rate();
        if rate <= f64::MIN_POSITIVE {
            self.dt
        } else {
            self.dt.min(cfl_number / rate)
        }
    }

    /// Returns the number of substeps required by a CFL limit.
    ///
    /// # Panics
    ///
    /// Panics if `cfl_number` is not positive and finite.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn cfl_substeps(&self, cfl_number: f64) -> usize {
        let limited_dt = self.cfl_timestep(cfl_number);
        if limited_dt >= self.dt {
            1
        } else {
            (self.dt / limited_dt).ceil().max(1.0) as usize
        }
    }

    /// Projects face velocity with a matrix-free diagonally preconditioned conjugate-gradient solve.
    pub fn project_velocity(&mut self) -> MacProjectionStats {
        self.apply_obstacle_velocity_constraints();
        let divergence_before_l2 = self.divergence_l2();
        let (iterations, pressure_residual_l2) = self.solve_pressure_pcg();
        self.apply_pressure_gradient();
        self.apply_obstacle_velocity_constraints();
        let divergence_after_l2 = self.divergence_l2();
        self.last_projection = MacProjectionStats {
            divergence_before_l2,
            divergence_after_l2,
            pressure_residual_l2,
            iterations,
        };
        self.last_projection
    }

    /// Advances velocity and scalar fields by one solver step.
    pub fn step(&mut self) -> MacStepStats {
        let initial_projection = self.project_velocity();
        self.previous_u.clone_from(&self.u);
        self.previous_v.clone_from(&self.v);
        let previous_u = self.previous_u.clone();
        let previous_v = self.previous_v.clone();
        self.advect_velocity_semi_lagrangian(&previous_u, &previous_v);

        let previous_density = self.density.clone();
        let previous_temperature = self.temperature.clone();
        self.density = self.advect_scalar_field(&previous_density, &previous_u, &previous_v, true);
        self.temperature =
            self.advect_scalar_field(&previous_temperature, &previous_u, &previous_v, false);
        self.apply_obstacle_scalar_constraints();
        let final_projection = self.project_velocity();
        MacStepStats {
            initial_projection,
            final_projection,
        }
    }

    /// Advances with enough substeps to satisfy a CFL limit.
    ///
    /// # Panics
    ///
    /// Panics if `cfl_number` is not positive and finite.
    pub fn step_cfl(&mut self, cfl_number: f64) -> Vec<MacStepStats> {
        let original_dt = self.dt;
        let substeps = self.cfl_substeps(cfl_number);
        self.dt = original_dt / usize_to_f64(substeps);
        let mut stats = Vec::with_capacity(substeps);
        for _ in 0..substeps {
            stats.push(self.step());
        }
        self.dt = original_dt;
        stats
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
        assert!(depth > 0, "MAC fluid volume depth must be non-zero");
        assert!(
            falloff.is_finite() && falloff >= 0.0,
            "MAC fluid volume falloff must be finite and non-negative"
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

    fn rebuild_face_weights(&mut self) {
        let width = self.dims[0];
        let height = self.dims[1];

        for y in 0..height {
            for x_face in 0..=width {
                let index = u_index_for_dims(self.dims, x_face, y);
                self.u_weights[index] = if x_face == 0 || x_face == width {
                    0.0
                } else {
                    let left = self.solid_phi[index_for_dims(self.dims, x_face - 1, y)];
                    let right = self.solid_phi[index_for_dims(self.dims, x_face, y)];
                    sdf_face_open_fraction(left, right)
                };
            }
        }

        for y_face in 0..=height {
            for x in 0..width {
                let index = v_index_for_dims(self.dims, x, y_face);
                self.v_weights[index] = if y_face == 0 || y_face == height {
                    0.0
                } else {
                    let below = self.solid_phi[index_for_dims(self.dims, x, y_face - 1)];
                    let above = self.solid_phi[index_for_dims(self.dims, x, y_face)];
                    sdf_face_open_fraction(below, above)
                };
            }
        }
    }

    fn apply_obstacle_constraints(&mut self) {
        self.apply_obstacle_scalar_constraints();
        self.apply_obstacle_velocity_constraints();
    }

    fn apply_obstacle_scalar_constraints(&mut self) {
        for ((density, temperature), phi) in self
            .density
            .iter_mut()
            .zip(&mut self.temperature)
            .zip(&self.solid_phi)
        {
            if *phi <= 0.0 {
                *density = 0.0;
                *temperature = 0.0;
            } else {
                if !density.is_finite() || *density < 0.0 {
                    *density = 0.0;
                }
                if !temperature.is_finite() {
                    *temperature = 0.0;
                }
            }
        }
    }

    fn apply_obstacle_velocity_constraints(&mut self) {
        for (value, weight) in self.u.iter_mut().zip(&self.u_weights) {
            *value = if *weight <= FACE_WEIGHT_EPSILON || !value.is_finite() {
                0.0
            } else {
                *value
            };
        }
        for (value, weight) in self.v.iter_mut().zip(&self.v_weights) {
            *value = if *weight <= FACE_WEIGHT_EPSILON || !value.is_finite() {
                0.0
            } else {
                *value
            };
        }
    }

    fn add_velocity_to_cell_faces(&mut self, x: usize, y: usize, velocity: [f64; 2]) {
        let left = u_index_for_dims(self.dims, x, y);
        let right = u_index_for_dims(self.dims, x + 1, y);
        let bottom = v_index_for_dims(self.dims, x, y);
        let top = v_index_for_dims(self.dims, x, y + 1);
        self.u[left] = finite_f32(f64::from(self.u[left]) + velocity[0]);
        self.u[right] = finite_f32(f64::from(self.u[right]) + velocity[0]);
        self.v[bottom] = finite_f32(f64::from(self.v[bottom]) + velocity[1]);
        self.v[top] = finite_f32(f64::from(self.v[top]) + velocity[1]);
    }

    fn center_velocity_from_faces(&self, u: &[f32], v: &[f32], x: usize, y: usize) -> [f64; 2] {
        debug_assert!(x < self.dims[0] && y < self.dims[1]);
        let left = u_index_for_dims(self.dims, x, y);
        let right = u_index_for_dims(self.dims, x + 1, y);
        let bottom = v_index_for_dims(self.dims, x, y);
        let top = v_index_for_dims(self.dims, x, y + 1);
        [
            0.5 * (f64::from(u[left]) + f64::from(u[right])),
            0.5 * (f64::from(v[bottom]) + f64::from(v[top])),
        ]
    }

    fn max_cfl_rate(&self) -> f64 {
        max_open_face_speed(&self.u, &self.u_weights) / self.cell_size[0]
            + max_open_face_speed(&self.v, &self.v_weights) / self.cell_size[1]
    }

    fn divergence_l2(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let divergence = self.cell_divergence(x, y);
                sum += divergence * divergence;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / usize_to_f64(count)).sqrt()
        }
    }

    fn cell_divergence(&self, x: usize, y: usize) -> f64 {
        let right = u_index_for_dims(self.dims, x + 1, y);
        let left = u_index_for_dims(self.dims, x, y);
        let top = v_index_for_dims(self.dims, x, y + 1);
        let bottom = v_index_for_dims(self.dims, x, y);
        let du = f64::from(self.u[right] * self.u_weights[right])
            - f64::from(self.u[left] * self.u_weights[left]);
        let dv = f64::from(self.v[top] * self.v_weights[top])
            - f64::from(self.v[bottom] * self.v_weights[bottom]);
        du / self.cell_size[0] + dv / self.cell_size[1]
    }

    fn solve_pressure_pcg(&mut self) -> (usize, f64) {
        let cell_count = cell_count_for_dims(self.dims);
        self.pressure.fill(0.0);

        let mut rhs = vec![0.0_f64; cell_count];
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] > 0.0 {
                    rhs[index] = -self.cell_divergence(x, y) / self.dt;
                }
            }
        }
        subtract_fluid_component_means(
            self.dims,
            &self.u_weights,
            &self.v_weights,
            &mut rhs,
            &self.solid_phi,
        );

        let mut residual = rhs.clone();
        let mut z = vec![0.0_f64; cell_count];
        self.apply_pressure_preconditioner(&residual, &mut z);
        let mut direction = z.clone();
        let mut rz = dot_fluid(&residual, &z, &self.solid_phi);
        let mut residual_l2 = fluid_l2(&residual, &self.solid_phi);
        if residual_l2 <= self.pressure_tolerance || rz.abs() <= f64::MIN_POSITIVE {
            return (0, residual_l2);
        }

        let mut matrix_direction = vec![0.0_f64; cell_count];
        let mut iterations = 0usize;
        for iteration in 0..self.pressure_iterations {
            self.apply_pressure_matrix(&direction, &mut matrix_direction);
            let denom = dot_fluid(&direction, &matrix_direction, &self.solid_phi);
            if denom.abs() <= f64::MIN_POSITIVE {
                break;
            }

            let alpha = rz / denom;
            for i in 0..cell_count {
                if self.solid_phi[i] <= 0.0 {
                    continue;
                }
                self.pressure[i] = finite_f32(f64::from(self.pressure[i]) + alpha * direction[i]);
                residual[i] -= alpha * matrix_direction[i];
            }
            iterations = iteration + 1;
            residual_l2 = fluid_l2(&residual, &self.solid_phi);
            if residual_l2 <= self.pressure_tolerance {
                break;
            }

            self.apply_pressure_preconditioner(&residual, &mut z);
            let rz_next = dot_fluid(&residual, &z, &self.solid_phi);
            if rz.abs() <= f64::MIN_POSITIVE {
                break;
            }
            let beta = rz_next / rz;
            for i in 0..cell_count {
                direction[i] = if self.solid_phi[i] <= 0.0 {
                    0.0
                } else {
                    z[i] + beta * direction[i]
                };
            }
            rz = rz_next;
        }

        (iterations, residual_l2)
    }

    fn apply_pressure_preconditioner(&self, residual: &[f64], out: &mut [f64]) {
        debug_assert_eq!(residual.len(), out.len());
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    out[index] = 0.0;
                    continue;
                }
                let diagonal = self.pressure_diagonal(x, y);
                out[index] = if diagonal > f64::MIN_POSITIVE {
                    residual[index] / diagonal
                } else {
                    0.0
                };
            }
        }
    }

    fn apply_pressure_matrix(&self, pressure: &[f64], out: &mut [f64]) {
        debug_assert_eq!(pressure.len(), out.len());
        out.fill(0.0);
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let mut value = 0.0;
                if x > 0 {
                    let neighbor = index_for_dims(self.dims, x - 1, y);
                    if self.solid_phi[neighbor] > 0.0 {
                        let face = u_index_for_dims(self.dims, x, y);
                        let coeff = f64::from(self.u_weights[face])
                            / (self.cell_size[0] * self.cell_size[0]);
                        value += coeff * (pressure[index] - pressure[neighbor]);
                    }
                }
                if x + 1 < self.dims[0] {
                    let neighbor = index_for_dims(self.dims, x + 1, y);
                    if self.solid_phi[neighbor] > 0.0 {
                        let face = u_index_for_dims(self.dims, x + 1, y);
                        let coeff = f64::from(self.u_weights[face])
                            / (self.cell_size[0] * self.cell_size[0]);
                        value += coeff * (pressure[index] - pressure[neighbor]);
                    }
                }
                if y > 0 {
                    let neighbor = index_for_dims(self.dims, x, y - 1);
                    if self.solid_phi[neighbor] > 0.0 {
                        let face = v_index_for_dims(self.dims, x, y);
                        let coeff = f64::from(self.v_weights[face])
                            / (self.cell_size[1] * self.cell_size[1]);
                        value += coeff * (pressure[index] - pressure[neighbor]);
                    }
                }
                if y + 1 < self.dims[1] {
                    let neighbor = index_for_dims(self.dims, x, y + 1);
                    if self.solid_phi[neighbor] > 0.0 {
                        let face = v_index_for_dims(self.dims, x, y + 1);
                        let coeff = f64::from(self.v_weights[face])
                            / (self.cell_size[1] * self.cell_size[1]);
                        value += coeff * (pressure[index] - pressure[neighbor]);
                    }
                }
                out[index] = value;
            }
        }
    }

    fn pressure_diagonal(&self, x: usize, y: usize) -> f64 {
        let mut diagonal = 0.0;
        if x > 0 && self.solid_phi[index_for_dims(self.dims, x - 1, y)] > 0.0 {
            let face = u_index_for_dims(self.dims, x, y);
            diagonal += f64::from(self.u_weights[face]) / (self.cell_size[0] * self.cell_size[0]);
        }
        if x + 1 < self.dims[0] && self.solid_phi[index_for_dims(self.dims, x + 1, y)] > 0.0 {
            let face = u_index_for_dims(self.dims, x + 1, y);
            diagonal += f64::from(self.u_weights[face]) / (self.cell_size[0] * self.cell_size[0]);
        }
        if y > 0 && self.solid_phi[index_for_dims(self.dims, x, y - 1)] > 0.0 {
            let face = v_index_for_dims(self.dims, x, y);
            diagonal += f64::from(self.v_weights[face]) / (self.cell_size[1] * self.cell_size[1]);
        }
        if y + 1 < self.dims[1] && self.solid_phi[index_for_dims(self.dims, x, y + 1)] > 0.0 {
            let face = v_index_for_dims(self.dims, x, y + 1);
            diagonal += f64::from(self.v_weights[face]) / (self.cell_size[1] * self.cell_size[1]);
        }
        diagonal
    }

    fn apply_pressure_gradient(&mut self) {
        let width = self.dims[0];
        let height = self.dims[1];

        for y in 0..height {
            for x_face in 1..width {
                let face = u_index_for_dims(self.dims, x_face, y);
                let weight = self.u_weights[face];
                if weight <= FACE_WEIGHT_EPSILON {
                    self.u[face] = 0.0;
                    continue;
                }
                let left = index_for_dims(self.dims, x_face - 1, y);
                let right = index_for_dims(self.dims, x_face, y);
                let left_pressure = if self.solid_phi[left] > 0.0 {
                    self.pressure[left]
                } else {
                    self.pressure[right]
                };
                let right_pressure = if self.solid_phi[right] > 0.0 {
                    self.pressure[right]
                } else {
                    self.pressure[left]
                };
                let gradient = f64::from(right_pressure - left_pressure) / self.cell_size[0];
                self.u[face] = finite_f32(f64::from(self.u[face]) - self.dt * gradient);
            }
        }

        for y_face in 1..height {
            for x in 0..width {
                let face = v_index_for_dims(self.dims, x, y_face);
                let weight = self.v_weights[face];
                if weight <= FACE_WEIGHT_EPSILON {
                    self.v[face] = 0.0;
                    continue;
                }
                let below = index_for_dims(self.dims, x, y_face - 1);
                let above = index_for_dims(self.dims, x, y_face);
                let below_pressure = if self.solid_phi[below] > 0.0 {
                    self.pressure[below]
                } else {
                    self.pressure[above]
                };
                let above_pressure = if self.solid_phi[above] > 0.0 {
                    self.pressure[above]
                } else {
                    self.pressure[below]
                };
                let gradient = f64::from(above_pressure - below_pressure) / self.cell_size[1];
                self.v[face] = finite_f32(f64::from(self.v[face]) - self.dt * gradient);
            }
        }
    }

    fn advect_scalar_field(
        &self,
        source: &[f32],
        old_u: &[f32],
        old_v: &[f32],
        nonnegative: bool,
    ) -> Vec<f32> {
        let cell_count = cell_count_for_dims(self.dims);
        let mut forward = vec![0.0_f32; cell_count];
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let position = [usize_to_f64(x), usize_to_f64(y)];
                let velocity = self.velocity_at_position_from_faces(old_u, old_v, position);
                let back = [
                    position[0] - self.dt * velocity[0] / self.cell_size[0],
                    position[1] - self.dt * velocity[1] / self.cell_size[1],
                ];
                forward[index] = self.sample_center_field(source, back);
            }
        }

        if self.scalar_advection == MacScalarAdvection::SemiLagrangian {
            sanitize_advected_scalar(&mut forward, nonnegative);
            return forward;
        }

        let mut backward = vec![0.0_f32; cell_count];
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let position = [usize_to_f64(x), usize_to_f64(y)];
                let velocity = self.velocity_at_position_from_faces(old_u, old_v, position);
                let first_forward = [
                    position[0] + self.dt * velocity[0] / self.cell_size[0],
                    position[1] + self.dt * velocity[1] / self.cell_size[1],
                ];
                let forward_velocity =
                    self.velocity_at_position_from_faces(old_u, old_v, first_forward);
                let traced_forward = [
                    position[0] + self.dt * forward_velocity[0] / self.cell_size[0],
                    position[1] + self.dt * forward_velocity[1] / self.cell_size[1],
                ];
                backward[index] = self.sample_center_field(&forward, traced_forward);
            }
        }

        let mut corrected = vec![0.0_f32; cell_count];
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let index = index_for_dims(self.dims, x, y);
                if self.solid_phi[index] <= 0.0 {
                    continue;
                }
                let position = [usize_to_f64(x), usize_to_f64(y)];
                let velocity = self.velocity_at_position_from_faces(old_u, old_v, position);
                let back = [
                    position[0] - self.dt * velocity[0] / self.cell_size[0],
                    position[1] - self.dt * velocity[1] / self.cell_size[1],
                ];
                let estimate =
                    f64::from(forward[index]) + 0.5 * f64::from(source[index] - backward[index]);
                let [min_value, max_value] = self.source_neighborhood_bounds(source, back);
                corrected[index] = finite_f32(estimate.clamp(min_value, max_value));
            }
        }
        sanitize_advected_scalar(&mut corrected, nonnegative);
        corrected
    }

    fn advect_velocity_semi_lagrangian(&mut self, old_u: &[f32], old_v: &[f32]) {
        let width = self.dims[0];
        let height = self.dims[1];
        let mut next_u = self.u.clone();
        let mut next_v = self.v.clone();

        for y in 0..height {
            for x_face in 0..=width {
                let index = u_index_for_dims(self.dims, x_face, y);
                if self.u_weights[index] <= FACE_WEIGHT_EPSILON {
                    next_u[index] = 0.0;
                    continue;
                }
                let position = [usize_to_f64(x_face) - 0.5, usize_to_f64(y)];
                let velocity = self.velocity_at_position_from_faces(old_u, old_v, position);
                let back = [
                    position[0] - self.dt * velocity[0] / self.cell_size[0],
                    position[1] - self.dt * velocity[1] / self.cell_size[1],
                ];
                next_u[index] = self.sample_u_field(old_u, back);
            }
        }

        for y_face in 0..=height {
            for x in 0..width {
                let index = v_index_for_dims(self.dims, x, y_face);
                if self.v_weights[index] <= FACE_WEIGHT_EPSILON {
                    next_v[index] = 0.0;
                    continue;
                }
                let position = [usize_to_f64(x), usize_to_f64(y_face) - 0.5];
                let velocity = self.velocity_at_position_from_faces(old_u, old_v, position);
                let back = [
                    position[0] - self.dt * velocity[0] / self.cell_size[0],
                    position[1] - self.dt * velocity[1] / self.cell_size[1],
                ];
                next_v[index] = self.sample_v_field(old_v, back);
            }
        }

        self.u = next_u;
        self.v = next_v;
        self.apply_obstacle_velocity_constraints();
    }

    fn velocity_at_position_from_faces(
        &self,
        u: &[f32],
        v: &[f32],
        position: [f64; 2],
    ) -> [f64; 2] {
        [
            f64::from(self.sample_u_field(u, position)),
            f64::from(self.sample_v_field(v, position)),
        ]
    }

    fn sample_center_field(&self, field: &[f32], position: [f64; 2]) -> f32 {
        sample_grid(field, self.dims[0], self.dims[1], position[0], position[1])
    }

    fn sample_u_field(&self, field: &[f32], position: [f64; 2]) -> f32 {
        sample_grid(
            field,
            self.dims[0] + 1,
            self.dims[1],
            position[0] + 0.5,
            position[1],
        )
    }

    fn sample_v_field(&self, field: &[f32], position: [f64; 2]) -> f32 {
        sample_grid(
            field,
            self.dims[0],
            self.dims[1] + 1,
            position[0],
            position[1] + 0.5,
        )
    }

    fn source_neighborhood_bounds(&self, field: &[f32], position: [f64; 2]) -> [f64; 2] {
        let width = self.dims[0];
        let height = self.dims[1];
        let x0 = floor_index(position[0], width);
        let y0 = floor_index(position[1], height);
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let samples = [
            field[index_for_dims(self.dims, x0, y0)],
            field[index_for_dims(self.dims, x1, y0)],
            field[index_for_dims(self.dims, x0, y1)],
            field[index_for_dims(self.dims, x1, y1)],
        ];
        samples
            .into_iter()
            .fold([f64::INFINITY, f64::NEG_INFINITY], |[min, max], sample| {
                [min.min(f64::from(sample)), max.max(f64::from(sample))]
            })
    }
}

fn u_count_for_dims(dims: [usize; 2]) -> usize {
    (dims[0] + 1)
        .checked_mul(dims[1])
        .expect("MAC u grid dimensions overflow")
}

fn v_count_for_dims(dims: [usize; 2]) -> usize {
    dims[0]
        .checked_mul(dims[1] + 1)
        .expect("MAC v grid dimensions overflow")
}

fn u_index_for_dims(dims: [usize; 2], x: usize, y: usize) -> usize {
    x + (dims[0] + 1) * y
}

fn v_index_for_dims(dims: [usize; 2], x: usize, y: usize) -> usize {
    x + dims[0] * y
}

fn sdf_face_open_fraction(a: f32, b: f32) -> f32 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }

    cell_open_fraction(a).min(cell_open_fraction(b))
}

fn cell_open_fraction(phi: f32) -> f32 {
    (phi + 0.5).clamp(0.0, 1.0)
}

fn max_open_face_speed(velocity: &[f32], weights: &[f32]) -> f64 {
    velocity
        .iter()
        .zip(weights)
        .filter_map(|(velocity, weight)| {
            (*weight > FACE_WEIGHT_EPSILON).then_some(f64::from(*velocity).abs())
        })
        .fold(0.0_f64, f64::max)
}

fn subtract_fluid_component_means(
    dims: [usize; 2],
    u_weights: &[f32],
    v_weights: &[f32],
    values: &mut [f64],
    solid_phi: &[f32],
) {
    let mut visited = vec![false; values.len()];
    for y in 0..dims[1] {
        for x in 0..dims[0] {
            let start = index_for_dims(dims, x, y);
            if visited[start] || solid_phi[start] <= 0.0 {
                continue;
            }

            let mut stack = vec![[x, y]];
            let mut component = Vec::new();
            visited[start] = true;
            while let Some([cell_x, cell_y]) = stack.pop() {
                let index = index_for_dims(dims, cell_x, cell_y);
                component.push(index);

                for [next_x, next_y] in
                    fluid_neighbors(dims, u_weights, v_weights, solid_phi, cell_x, cell_y)
                {
                    let next = index_for_dims(dims, next_x, next_y);
                    if !visited[next] {
                        visited[next] = true;
                        stack.push([next_x, next_y]);
                    }
                }
            }

            let mean = component.iter().map(|index| values[*index]).sum::<f64>()
                / usize_to_f64(component.len());
            for index in component {
                values[index] -= mean;
            }
        }
    }
}

fn fluid_neighbors(
    dims: [usize; 2],
    u_weights: &[f32],
    v_weights: &[f32],
    solid_phi: &[f32],
    x: usize,
    y: usize,
) -> Vec<[usize; 2]> {
    let mut neighbors = Vec::with_capacity(4);
    if x > 0 && u_weights[u_index_for_dims(dims, x, y)] > FACE_WEIGHT_EPSILON {
        let next = index_for_dims(dims, x - 1, y);
        if solid_phi[next] > 0.0 {
            neighbors.push([x - 1, y]);
        }
    }
    if x + 1 < dims[0] && u_weights[u_index_for_dims(dims, x + 1, y)] > FACE_WEIGHT_EPSILON {
        let next = index_for_dims(dims, x + 1, y);
        if solid_phi[next] > 0.0 {
            neighbors.push([x + 1, y]);
        }
    }
    if y > 0 && v_weights[v_index_for_dims(dims, x, y)] > FACE_WEIGHT_EPSILON {
        let next = index_for_dims(dims, x, y - 1);
        if solid_phi[next] > 0.0 {
            neighbors.push([x, y - 1]);
        }
    }
    if y + 1 < dims[1] && v_weights[v_index_for_dims(dims, x, y + 1)] > FACE_WEIGHT_EPSILON {
        let next = index_for_dims(dims, x, y + 1);
        if solid_phi[next] > 0.0 {
            neighbors.push([x, y + 1]);
        }
    }
    neighbors
}

fn dot_fluid(lhs: &[f64], rhs: &[f64], solid_phi: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .zip(solid_phi)
        .filter_map(|((lhs, rhs), phi)| (*phi > 0.0).then_some(lhs * rhs))
        .sum()
}

fn fluid_l2(values: &[f64], solid_phi: &[f32]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (value, phi) in values.iter().zip(solid_phi) {
        if *phi <= 0.0 {
            continue;
        }
        sum += value * value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (sum / usize_to_f64(count)).sqrt()
    }
}

fn sanitize_advected_scalar(values: &mut [f32], nonnegative: bool) {
    for value in values {
        if !value.is_finite() || (nonnegative && *value < 0.0) {
            *value = 0.0;
        }
    }
}

fn sample_grid(field: &[f32], width: usize, height: usize, x: f64, y: f64) -> f32 {
    debug_assert_eq!(field.len(), width * height);
    let x = x.clamp(0.0, usize_to_f64(width - 1));
    let y = y.clamp(0.0, usize_to_f64(height - 1));
    let x0 = floor_index(x, width);
    let y0 = floor_index(y, height);
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - usize_to_f64(x0);
    let ty = y - usize_to_f64(y0);

    let sample = |sx: usize, sy: usize| f64::from(field[sx + width * sy]);
    let bottom = sample(x0, y0) * (1.0 - tx) + sample(x1, y0) * tx;
    let top = sample(x0, y1) * (1.0 - tx) + sample(x1, y1) * tx;
    finite_f32(bottom * (1.0 - ty) + top * ty)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn floor_index(value: f64, dim: usize) -> usize {
    value.clamp(0.0, usize_to_f64(dim - 1)).floor() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gmath::vector::Point, graphics::raytracing::volume::field::DensityField};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
    }

    fn density_center_of_mass_x(sim: &MacFluidGrid2) -> f64 {
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
    fn mac_grid_uses_face_velocity_shapes() {
        let sim = MacFluidGrid2::new([8, 6]);

        assert_eq!(sim.densities().len(), 48);
        assert_eq!(sim.temperatures().len(), 48);
        assert_eq!(sim.pressures().len(), 48);
        assert_eq!(sim.u().len(), 54);
        assert_eq!(sim.v().len(), 56);
        assert_eq!(sim.u_weights().len(), 54);
        assert_eq!(sim.v_weights().len(), 56);
    }

    #[test]
    fn mac_projection_reduces_divergence_and_reports_stats() {
        let mut sim = MacFluidGrid2::new([12, 12])
            .with_dt(0.25)
            .with_pressure_iterations(120)
            .with_pressure_tolerance(1.0e-7);
        for y in 1..11 {
            for x in 1..11 {
                sim.set_u([x, y], usize_to_f64(x) - 5.5);
                sim.set_v([x, y], usize_to_f64(y) - 5.5);
            }
        }

        let stats = sim.project_velocity();

        assert!(stats.divergence_before_l2 > 0.0);
        assert!(
            stats.divergence_after_l2 < stats.divergence_before_l2,
            "{stats:?}"
        );
        assert!(stats.pressure_residual_l2.is_finite());
        assert!(stats.iterations > 0);
        assert_eq!(sim.last_projection(), stats);
    }

    #[test]
    fn mac_sdf_circle_creates_fractional_face_weights() {
        let mut sim = MacFluidGrid2::new([16, 16]);
        sim.set_solid_circle([8.0, 8.0], 3.25);

        assert!(sim.solid_phi().iter().any(|phi| *phi <= 0.0));
        assert!(sim.u_weights().contains(&0.0));
        assert!(
            sim.u_weights()
                .iter()
                .chain(sim.v_weights())
                .any(|w| *w > 0.0 && *w < 1.0)
        );
    }

    #[test]
    fn mac_fractional_faces_store_unweighted_velocity() {
        let mut sim = MacFluidGrid2::new([16, 16]);
        sim.set_solid_circle([8.0, 8.0], 3.25);

        let mut partial_face = None;
        for y in 0..sim.dims()[1] {
            for x_face in 1..sim.dims()[0] {
                let index = u_index_for_dims(sim.dims(), x_face, y);
                let weight = sim.u_weights()[index];
                if weight > 0.0 && weight < 1.0 {
                    partial_face = Some([x_face, y]);
                    break;
                }
            }
            if partial_face.is_some() {
                break;
            }
        }

        let face = partial_face.expect("circle SDF should create a partial face");
        sim.set_u(face, 4.0);

        assert_close(sim.u_at(face), 4.0);
    }

    #[test]
    fn mac_accepts_closure_sdf_obstacles() {
        let mut sim = MacFluidGrid2::new([10, 10]);
        sim.set_solid_sdf(|cell| {
            let dx = cell[0] - 5.0;
            let dy = cell[1] - 5.0;
            dx.hypot(dy) - 2.6
        });

        assert!(sim.is_solid([5, 5]));
        assert!(!sim.is_solid([1, 1]));
        assert!(
            sim.u_weights()
                .iter()
                .chain(sim.v_weights())
                .any(|weight| *weight > 0.0 && *weight < 1.0)
        );
    }

    #[test]
    fn mac_maccormack_advects_density_with_velocity() {
        let mut sim = MacFluidGrid2::new([20, 8])
            .with_dt(0.5)
            .with_scalar_advection(MacScalarAdvection::MacCormack);
        sim.add_density([5, 4], 10.0);
        let before = density_center_of_mass_x(&sim);
        for y in 0..8 {
            for x_face in 1..20 {
                sim.set_u([x_face, y], 1.0);
            }
        }

        sim.step();
        let after = density_center_of_mass_x(&sim);

        assert!(after > before, "{after} should be greater than {before}");
        assert!(
            sim.densities()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
    }

    #[test]
    fn mac_maccormack_reverse_trace_uses_forward_position_velocity() {
        let sim = MacFluidGrid2::new([7, 4])
            .with_dt(0.2)
            .with_scalar_advection(MacScalarAdvection::MacCormack);
        let dims = sim.dims();
        let mut source = vec![0.0; cell_count_for_dims(dims)];
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                source[index_for_dims(dims, x, y)] =
                    finite_f32(usize_to_f64(x * x * x) + usize_to_f64(3 * y));
            }
        }
        let mut old_u = vec![0.0; u_count_for_dims(dims)];
        for y in 0..dims[1] {
            for x_face in 0..=dims[0] {
                old_u[u_index_for_dims(dims, x_face, y)] =
                    finite_f32(0.35 * usize_to_f64(x_face * x_face));
            }
        }
        let old_v = vec![0.0; v_count_for_dims(dims)];

        let result = sim.advect_scalar_field(&source, &old_u, &old_v, true);

        let mut forward = vec![0.0_f32; cell_count_for_dims(dims)];
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let position = [usize_to_f64(x), usize_to_f64(y)];
                let velocity = sim.velocity_at_position_from_faces(&old_u, &old_v, position);
                let back = [
                    position[0] - sim.dt() * velocity[0] / sim.cell_size()[0],
                    position[1] - sim.dt() * velocity[1] / sim.cell_size()[1],
                ];
                forward[index_for_dims(dims, x, y)] = sim.sample_center_field(&source, back);
            }
        }

        let cell = [2, 1];
        let index = index_for_dims(dims, cell[0], cell[1]);
        let position = [usize_to_f64(cell[0]), usize_to_f64(cell[1])];
        let velocity = sim.velocity_at_position_from_faces(&old_u, &old_v, position);
        let first_forward = [
            position[0] + sim.dt() * velocity[0] / sim.cell_size()[0],
            position[1] + sim.dt() * velocity[1] / sim.cell_size()[1],
        ];
        let forward_velocity = sim.velocity_at_position_from_faces(&old_u, &old_v, first_forward);
        let traced_forward = [
            position[0] + sim.dt() * forward_velocity[0] / sim.cell_size()[0],
            position[1] + sim.dt() * forward_velocity[1] / sim.cell_size()[1],
        ];
        let backward = sim.sample_center_field(&forward, traced_forward);
        let estimate = f64::from(forward[index]) + 0.5 * f64::from(source[index] - backward);
        let back = [
            position[0] - sim.dt() * velocity[0] / sim.cell_size()[0],
            position[1] - sim.dt() * velocity[1] / sim.cell_size()[1],
        ];
        let [min_value, max_value] = sim.source_neighborhood_bounds(&source, back);
        let expected = estimate.clamp(min_value, max_value);

        let stale_backward = sim.sample_center_field(&forward, first_forward);
        let stale_estimate =
            f64::from(forward[index]) + 0.5 * f64::from(source[index] - stale_backward);
        let stale_expected = stale_estimate.clamp(min_value, max_value);

        assert_close(f64::from(result[index]), expected);
        assert!((expected - stale_expected).abs() > 1.0e-5);
    }

    #[test]
    fn mac_emitter_adds_scalars_and_face_velocity() {
        let mut sim = MacFluidGrid2::new([12, 12]);

        sim.apply_emitter(
            MacFluidEmitter::new([6.0, 6.0], 3.0)
                .with_density(4.0)
                .with_temperature(2.0)
                .with_velocity([1.0, 2.0]),
        );

        assert!(sim.density_at([6, 6]) > 3.9);
        assert!(sim.temperature_at([6, 6]) > 1.9);
        let velocity = sim.velocity_at_cell([6, 6]);
        assert!(velocity[0] > 0.9);
        assert!(velocity[1] > 1.9);
    }

    #[test]
    fn mac_step_reports_initial_and_final_projection_stats() {
        let mut sim = MacFluidGrid2::new([12, 12]).with_pressure_iterations(80);
        sim.add_density([5, 5], 3.0);
        for y in 1..11 {
            for x in 1..11 {
                sim.set_u([x, y], 0.2 * usize_to_f64(x));
            }
        }

        let stats = sim.step();

        assert!(stats.initial_projection.divergence_before_l2 > 0.0);
        assert!(stats.initial_projection.pressure_residual_l2.is_finite());
        assert!(stats.final_projection.divergence_before_l2.is_finite());
        assert_eq!(sim.last_projection(), stats.final_projection);
    }

    #[test]
    fn mac_cfl_substeps_for_large_velocity() {
        let mut sim = MacFluidGrid2::new([8, 8]).with_dt(1.0);
        for y in 0..8 {
            for x in 1..8 {
                sim.set_u([x, y], 4.0);
            }
        }

        assert!(sim.max_velocity_magnitude() > 0.0);
        assert!(sim.cfl_timestep(0.5) < sim.dt());
        assert!(sim.cfl_substeps(0.5) > 1);
        let original_dt = sim.dt();
        let stats = sim.step_cfl(0.5);
        assert!(stats.len() > 1);
        assert_close(sim.dt(), original_dt);
    }

    #[test]
    fn mac_cfl_uses_face_speed_not_center_average() {
        let mut sim = MacFluidGrid2::new([4, 4]).with_dt(1.0);
        for y in 0..4 {
            sim.set_u([1, y], 10.0);
            sim.set_u([2, y], -10.0);
        }

        assert_close(sim.velocity_at_cell([1, 2])[0], 0.0);
        assert_close(sim.max_velocity_magnitude(), 10.0);
        assert_close(sim.cfl_timestep(0.5), 0.05);
    }

    #[test]
    fn mac_pressure_rhs_subtracts_mean_per_fluid_component() {
        let dims = [4, 2];
        let mut solid_phi = vec![1.0; 8];
        solid_phi[index_for_dims(dims, 2, 0)] = -1.0;
        solid_phi[index_for_dims(dims, 2, 1)] = -1.0;
        let u_weights = vec![1.0; (dims[0] + 1) * dims[1]];
        let v_weights = vec![1.0; dims[0] * (dims[1] + 1)];
        let mut values = vec![2.0, 4.0, 99.0, 10.0, 6.0, 8.0, 111.0, 14.0];

        subtract_fluid_component_means(dims, &u_weights, &v_weights, &mut values, &solid_phi);

        let left_sum = values[index_for_dims(dims, 0, 0)]
            + values[index_for_dims(dims, 1, 0)]
            + values[index_for_dims(dims, 0, 1)]
            + values[index_for_dims(dims, 1, 1)];
        let right_sum = values[index_for_dims(dims, 3, 0)] + values[index_for_dims(dims, 3, 1)];
        assert_close(left_sum, 0.0);
        assert_close(right_sum, 0.0);
        assert_close(values[index_for_dims(dims, 2, 0)], 99.0);
        assert_close(values[index_for_dims(dims, 2, 1)], 111.0);
    }

    #[test]
    fn mac_exports_thick_grid_density_volume() {
        let mut sim = MacFluidGrid2::new([4, 3]);
        sim.add_density([2, 1], 7.0);
        let bounds = GridBounds::new(Point::new(-2.0, -0.5, -1.5), Point::new(2.0, 0.5, 1.5));
        let grid = sim.to_density_volume(bounds, 5, 1.0);

        assert_eq!(grid.dims(), [4, 3, 5]);
        assert_eq!(grid.interpolation(), GridInterpolation::Trilinear);
        let center_index = 2 + 4 * (1 + 3 * 2);
        assert_close(f64::from(grid.densities()[center_index]), 7.0);
        assert_close(grid.density(grid.cell_center(2, 1, 2), 0.0), 7.0);
    }
}

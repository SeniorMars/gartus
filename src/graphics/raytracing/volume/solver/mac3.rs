use super::{DEFAULT_DT, finite_f32, nonnegative_f32, usize_to_f64};
use crate::graphics::raytracing::volume::grid::{GridBounds, GridDensityField, GridInterpolation};

use super::mac::{MacProjectionStats, MacStepStats};

const DEFAULT_PRESSURE_ITERATIONS: usize = 240;
const DEFAULT_PRESSURE_TOLERANCE: f64 = 1.0e-5;
const SOLID_PHI_DEFAULT: f32 = 1.0;
const FACE_WEIGHT_EPSILON: f32 = 1.0e-4;

/// Classification flags for a MAC cell.
///
/// Smoke projection and advection treat every non-solid cell as active gas. Liquid projection uses
/// only [`Self::LIQUID`] cells as active liquid and treats [`Self::OPEN`] cells as inactive air.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MacCellFlags(u8);

impl MacCellFlags {
    /// Open gas cell.
    pub const OPEN: Self = Self(0b001);
    /// Solid obstacle cell.
    pub const SOLID: Self = Self(0b010);
    /// Liquid cell classification.
    ///
    /// Liquid projection treats this as active liquid. Smoke solves continue to treat this like an
    /// open non-solid cell.
    pub const LIQUID: Self = Self(0b100);

    /// Returns true when this cell is open gas.
    #[must_use]
    pub const fn is_open(self) -> bool {
        self.0 == Self::OPEN.0
    }

    /// Returns true when this cell is solid.
    #[must_use]
    pub const fn is_solid(self) -> bool {
        self.0 & Self::SOLID.0 != 0
    }

    /// Returns true when this cell is classified as liquid.
    #[must_use]
    pub const fn is_liquid(self) -> bool {
        self.0 & Self::LIQUID.0 != 0
    }
}

/// Signed scalar grid exported from a 3D MAC solver.
#[derive(Clone, Debug, PartialEq)]
pub struct MacScalarGrid3 {
    bounds: GridBounds,
    dims: [usize; 3],
    samples: Vec<f32>,
}

impl MacScalarGrid3 {
    /// Creates a signed scalar grid from row-major samples.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are invalid or if `samples.len()` does not match the grid cell count.
    #[must_use]
    pub fn new(bounds: GridBounds, dims: [usize; 3], samples: Vec<f32>) -> Self {
        validate_dims3(dims);
        assert_eq!(
            samples.len(),
            cell_count_for_dims3(dims),
            "3D MAC scalar grid sample count must match dimensions"
        );
        let samples = samples
            .into_iter()
            .map(|value| if value.is_finite() { value } else { 0.0 })
            .collect();
        Self {
            bounds,
            dims,
            samples,
        }
    }

    /// Returns the grid bounds.
    #[must_use]
    pub const fn bounds(&self) -> GridBounds {
        self.bounds
    }

    /// Returns dimensions as `[width, height, depth]`.
    #[must_use]
    pub const fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Returns row-major signed scalar samples.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Returns a sample by grid coordinate.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is outside the grid.
    #[must_use]
    pub fn sample_at(&self, cell: [usize; 3]) -> f64 {
        assert!(
            cell[0] < self.dims[0] && cell[1] < self.dims[1] && cell[2] < self.dims[2],
            "3D MAC scalar grid index out of bounds"
        );
        f64::from(self.samples[cell_index_for_dims3(self.dims, cell[0], cell[1], cell[2])])
    }
}

/// Three-dimensional smoke/liquid solver using a Marker-and-Cell layout.
///
/// Density, temperature, fuel, and pressure are stored at cell centers. Velocity is stored on cell
/// faces: `u` is x-face velocity, `v` is y-face velocity, and `w` is z-face velocity.
///
/// The default [`Self::step`] path remains a gas/smoke solve over all non-solid cells. Liquid
/// callers opt into single-phase free-surface behavior with [`Self::set_liquid_phi`] and
/// [`Self::step_liquid`].
#[derive(Clone, Debug)]
pub struct MacFluidGrid3 {
    dims: [usize; 3],
    dt: f64,
    cell_size: [f64; 3],
    pressure_iterations: usize,
    pressure_tolerance: f64,
    density: Vec<f32>,
    temperature: Vec<f32>,
    fuel: Vec<f32>,
    pressure: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    solid_phi: Vec<f32>,
    liquid_phi: Vec<f32>,
    flags: Vec<MacCellFlags>,
    u_weights: Vec<f32>,
    v_weights: Vec<f32>,
    w_weights: Vec<f32>,
    liquid_viscosity: f64,
    last_projection: MacProjectionStats,
    last_liquid_projection: MacProjectionStats,
}

impl MacFluidGrid3 {
    /// Creates an empty 3D MAC smoke grid.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is smaller than three or if any allocation would overflow.
    #[must_use]
    pub fn new(dims: [usize; 3]) -> Self {
        validate_dims3(dims);
        let cell_count = cell_count_for_dims3(dims);
        let u_count = u_count_for_dims3(dims);
        let v_count = v_count_for_dims3(dims);
        let w_count = w_count_for_dims3(dims);
        let mut grid = Self {
            dims,
            dt: DEFAULT_DT,
            cell_size: [1.0, 1.0, 1.0],
            pressure_iterations: DEFAULT_PRESSURE_ITERATIONS,
            pressure_tolerance: DEFAULT_PRESSURE_TOLERANCE,
            density: vec![0.0; cell_count],
            temperature: vec![0.0; cell_count],
            fuel: vec![0.0; cell_count],
            pressure: vec![0.0; cell_count],
            u: vec![0.0; u_count],
            v: vec![0.0; v_count],
            w: vec![0.0; w_count],
            solid_phi: vec![SOLID_PHI_DEFAULT; cell_count],
            liquid_phi: vec![SOLID_PHI_DEFAULT; cell_count],
            flags: vec![MacCellFlags::OPEN; cell_count],
            u_weights: vec![1.0; u_count],
            v_weights: vec![1.0; v_count],
            w_weights: vec![1.0; w_count],
            liquid_viscosity: 0.0,
            last_projection: MacProjectionStats::default(),
            last_liquid_projection: MacProjectionStats::default(),
        };
        grid.rebuild_face_weights();
        grid
    }

    /// Returns a copy with a different timestep.
    ///
    /// # Panics
    ///
    /// Panics if `dt` is not positive and finite.
    #[must_use]
    pub fn with_dt(mut self, dt: f64) -> Self {
        assert!(
            dt.is_finite() && dt > 0.0,
            "3D MAC timestep must be positive and finite"
        );
        self.dt = dt;
        self
    }

    /// Returns a copy with a different grid cell size.
    ///
    /// # Panics
    ///
    /// Panics if any component is not positive and finite.
    #[must_use]
    pub fn with_cell_size(mut self, cell_size: [f64; 3]) -> Self {
        assert!(
            cell_size
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "3D MAC cell size must be positive and finite"
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
            "3D MAC pressure iterations must be non-zero"
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
            "3D MAC pressure tolerance must be positive and finite"
        );
        self.pressure_tolerance = pressure_tolerance;
        self
    }

    /// Returns a copy with explicit liquid viscosity used by [`Self::step_liquid`].
    ///
    /// # Panics
    ///
    /// Panics if `viscosity` is negative or not finite.
    #[must_use]
    pub fn with_liquid_viscosity(mut self, viscosity: f64) -> Self {
        assert!(
            viscosity.is_finite() && viscosity >= 0.0,
            "3D MAC liquid viscosity must be non-negative and finite"
        );
        self.liquid_viscosity = viscosity;
        self
    }

    /// Returns grid dimensions as `[width, height, depth]`.
    #[must_use]
    pub const fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Returns the current timestep.
    #[must_use]
    pub const fn dt(&self) -> f64 {
        self.dt
    }

    /// Returns cell size.
    #[must_use]
    pub const fn cell_size(&self) -> [f64; 3] {
        self.cell_size
    }

    /// Returns explicit liquid viscosity used by [`Self::step_liquid`].
    #[must_use]
    pub const fn liquid_viscosity(&self) -> f64 {
        self.liquid_viscosity
    }

    /// Returns all center density samples.
    #[must_use]
    pub fn densities(&self) -> &[f32] {
        &self.density
    }

    /// Returns all center temperature samples.
    #[must_use]
    pub fn temperatures(&self) -> &[f32] {
        &self.temperature
    }

    /// Returns all center fuel samples.
    #[must_use]
    pub fn fuels(&self) -> &[f32] {
        &self.fuel
    }

    /// Returns all center pressure samples.
    #[must_use]
    pub fn pressures(&self) -> &[f32] {
        &self.pressure
    }

    /// Returns x-face velocities with dimensions `[width + 1, height, depth]`.
    #[must_use]
    pub fn u(&self) -> &[f32] {
        &self.u
    }

    /// Returns y-face velocities with dimensions `[width, height + 1, depth]`.
    #[must_use]
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    /// Returns z-face velocities with dimensions `[width, height, depth + 1]`.
    #[must_use]
    pub fn w(&self) -> &[f32] {
        &self.w
    }

    /// Returns center-cell signed distances.
    #[must_use]
    pub fn solid_phi(&self) -> &[f32] {
        &self.solid_phi
    }

    /// Returns center-cell liquid signed distances.
    ///
    /// Negative values are liquid and positive values are air unless the cell is solid.
    #[must_use]
    pub fn liquid_phi(&self) -> &[f32] {
        &self.liquid_phi
    }

    /// Returns center-cell flags.
    #[must_use]
    pub fn flags(&self) -> &[MacCellFlags] {
        &self.flags
    }

    /// Returns x-face open fractions.
    #[must_use]
    pub fn u_weights(&self) -> &[f32] {
        &self.u_weights
    }

    /// Returns y-face open fractions.
    #[must_use]
    pub fn v_weights(&self) -> &[f32] {
        &self.v_weights
    }

    /// Returns z-face open fractions.
    #[must_use]
    pub fn w_weights(&self) -> &[f32] {
        &self.w_weights
    }

    /// Returns projection diagnostics from the most recent projection.
    #[must_use]
    pub const fn last_projection(&self) -> MacProjectionStats {
        self.last_projection
    }

    /// Returns liquid projection diagnostics from the most recent free-surface projection.
    #[must_use]
    pub const fn last_liquid_projection(&self) -> MacProjectionStats {
        self.last_liquid_projection
    }

    /// Returns the center-cell flattened index.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    #[must_use]
    pub fn index(&self, cell: [usize; 3]) -> usize {
        assert!(
            cell[0] < self.dims[0] && cell[1] < self.dims[1] && cell[2] < self.dims[2],
            "3D MAC grid index out of bounds"
        );
        cell_index_for_dims3(self.dims, cell[0], cell[1], cell[2])
    }

    /// Returns the x-face velocity index.
    ///
    /// # Panics
    ///
    /// Panics if `face` is outside the u grid.
    #[must_use]
    pub fn u_index(&self, face: [usize; 3]) -> usize {
        assert!(
            face[0] <= self.dims[0] && face[1] < self.dims[1] && face[2] < self.dims[2],
            "3D MAC u-face index out of bounds"
        );
        u_index_for_dims3(self.dims, face[0], face[1], face[2])
    }

    /// Returns the y-face velocity index.
    ///
    /// # Panics
    ///
    /// Panics if `face` is outside the v grid.
    #[must_use]
    pub fn v_index(&self, face: [usize; 3]) -> usize {
        assert!(
            face[0] < self.dims[0] && face[1] <= self.dims[1] && face[2] < self.dims[2],
            "3D MAC v-face index out of bounds"
        );
        v_index_for_dims3(self.dims, face[0], face[1], face[2])
    }

    /// Returns the z-face velocity index.
    ///
    /// # Panics
    ///
    /// Panics if `face` is outside the w grid.
    #[must_use]
    pub fn w_index(&self, face: [usize; 3]) -> usize {
        assert!(
            face[0] < self.dims[0] && face[1] < self.dims[1] && face[2] <= self.dims[2],
            "3D MAC w-face index out of bounds"
        );
        w_index_for_dims3(self.dims, face[0], face[1], face[2])
    }

    /// Returns density at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    #[must_use]
    pub fn density_at(&self, cell: [usize; 3]) -> f64 {
        f64::from(self.density[self.index(cell)])
    }

    /// Returns the face-averaged center velocity.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    #[must_use]
    pub fn velocity_at_cell(&self, cell: [usize; 3]) -> [f64; 3] {
        let x = cell[0];
        let y = cell[1];
        let z = cell[2];
        assert!(
            x < self.dims[0] && y < self.dims[1] && z < self.dims[2],
            "3D MAC grid index out of bounds"
        );
        self.center_velocity_from_faces(&self.u, &self.v, &self.w, x, y, z)
    }

    /// Returns true when one center cell is solid.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    #[must_use]
    pub fn is_solid(&self, cell: [usize; 3]) -> bool {
        self.flags[self.index(cell)].is_solid()
    }

    /// Returns true when one center cell is active liquid.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    #[must_use]
    pub fn is_liquid(&self, cell: [usize; 3]) -> bool {
        self.flags[self.index(cell)].is_liquid()
    }

    /// Returns true when one center cell is non-solid inactive air for liquid solves.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    #[must_use]
    pub fn is_air(&self, cell: [usize; 3]) -> bool {
        self.flags[self.index(cell)].is_open()
    }

    /// Sets a cell's liquid classification through the liquid level set.
    ///
    /// Solid cells remain solid even when `liquid` is true. Non-solid liquid cells are active in
    /// [`Self::project_liquid_velocity`], while open cells are inactive air for liquid solves.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid.
    pub fn set_liquid(&mut self, cell: [usize; 3], liquid: bool) {
        let index = self.index(cell);
        if self.flags[index].is_solid() {
            self.liquid_phi[index] = SOLID_PHI_DEFAULT;
            return;
        }
        self.liquid_phi[index] = if liquid { -0.5 } else { 0.5 };
        self.rebuild_flags_from_phi();
    }

    /// Adds density to one center cell.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid or if `amount` is not finite.
    pub fn add_density(&mut self, cell: [usize; 3], amount: f64) {
        assert!(amount.is_finite(), "3D MAC density amount must be finite");
        let index = self.index(cell);
        if self.flags[index].is_solid() {
            return;
        }
        self.density[index] = nonnegative_f32(f64::from(self.density[index]) + amount);
    }

    /// Sets density at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid or if `density` is not finite.
    pub fn set_density(&mut self, cell: [usize; 3], density: f64) {
        assert!(density.is_finite(), "3D MAC density must be finite");
        let index = self.index(cell);
        self.density[index] = if self.flags[index].is_solid() {
            0.0
        } else {
            nonnegative_f32(density)
        };
    }

    /// Sets temperature at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid or if `temperature` is not finite.
    pub fn set_temperature(&mut self, cell: [usize; 3], temperature: f64) {
        assert!(temperature.is_finite(), "3D MAC temperature must be finite");
        let index = self.index(cell);
        self.temperature[index] = if self.flags[index].is_solid() {
            0.0
        } else {
            finite_f32(temperature)
        };
    }

    /// Sets fuel at one center cell.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is outside the grid or if `fuel` is not finite.
    pub fn set_fuel(&mut self, cell: [usize; 3], fuel: f64) {
        assert!(fuel.is_finite(), "3D MAC fuel must be finite");
        let index = self.index(cell);
        self.fuel[index] = if self.flags[index].is_solid() {
            0.0
        } else {
            nonnegative_f32(fuel)
        };
    }

    /// Sets x-face velocity.
    ///
    /// # Panics
    ///
    /// Panics if `face` is outside the u grid or if `velocity` is not finite.
    pub fn set_u(&mut self, face: [usize; 3], velocity: f64) {
        assert!(velocity.is_finite(), "3D MAC u velocity must be finite");
        let index = self.u_index(face);
        self.u[index] = if self.u_weights[index] <= FACE_WEIGHT_EPSILON {
            0.0
        } else {
            finite_f32(velocity)
        };
    }

    /// Sets y-face velocity.
    ///
    /// # Panics
    ///
    /// Panics if `face` is outside the v grid or if `velocity` is not finite.
    pub fn set_v(&mut self, face: [usize; 3], velocity: f64) {
        assert!(velocity.is_finite(), "3D MAC v velocity must be finite");
        let index = self.v_index(face);
        self.v[index] = if self.v_weights[index] <= FACE_WEIGHT_EPSILON {
            0.0
        } else {
            finite_f32(velocity)
        };
    }

    /// Sets z-face velocity.
    ///
    /// # Panics
    ///
    /// Panics if `face` is outside the w grid or if `velocity` is not finite.
    pub fn set_w(&mut self, face: [usize; 3], velocity: f64) {
        assert!(velocity.is_finite(), "3D MAC w velocity must be finite");
        let index = self.w_index(face);
        self.w[index] = if self.w_weights[index] <= FACE_WEIGHT_EPSILON {
            0.0
        } else {
            finite_f32(velocity)
        };
    }

    /// Replaces the center-cell signed distance field.
    ///
    /// Negative values are solid, positive values are open fluid cells.
    ///
    /// # Panics
    ///
    /// Panics if `solid_phi` has the wrong length or contains non-finite values.
    pub fn set_solid_phi(&mut self, solid_phi: Vec<f32>) {
        assert_eq!(
            solid_phi.len(),
            self.solid_phi.len(),
            "3D MAC SDF length must match grid dimensions"
        );
        assert!(
            solid_phi.iter().all(|value| value.is_finite()),
            "3D MAC SDF values must be finite"
        );
        self.solid_phi = solid_phi;
        self.rebuild_flags_from_phi();
        self.rebuild_face_weights();
        self.apply_solid_constraints();
    }

    /// Samples a center-cell signed distance field from a closure.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a non-finite value.
    pub fn set_solid_sdf<F>(&mut self, mut sdf: F)
    where
        F: FnMut([f64; 3]) -> f64,
    {
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let value = sdf([usize_to_f64(x), usize_to_f64(y), usize_to_f64(z)]);
                    assert!(value.is_finite(), "3D MAC SDF values must be finite");
                    self.solid_phi[cell_index_for_dims3(self.dims, x, y, z)] = finite_f32(value);
                }
            }
        }
        self.rebuild_flags_from_phi();
        self.rebuild_face_weights();
        self.apply_solid_constraints();
    }

    /// Replaces obstacle geometry with one spherical SDF.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    pub fn set_solid_sphere(&mut self, center: [f64; 3], radius: f64) {
        validate_point3(center, "3D MAC solid sphere center");
        validate_radius3(radius, "3D MAC solid sphere radius");
        self.set_solid_sdf(|cell| {
            let dx = cell[0] - center[0];
            let dy = cell[1] - center[1];
            let dz = cell[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt() - radius
        });
    }

    /// Replaces the center-cell liquid signed distance field.
    ///
    /// Negative values are liquid, positive values are air. Solid SDF classification remains
    /// separate and overrides liquid classification.
    ///
    /// # Panics
    ///
    /// Panics if `liquid_phi` has the wrong length or contains non-finite values.
    pub fn set_liquid_phi(&mut self, liquid_phi: Vec<f32>) {
        assert_eq!(
            liquid_phi.len(),
            self.liquid_phi.len(),
            "3D MAC liquid SDF length must match grid dimensions"
        );
        assert!(
            liquid_phi.iter().all(|value| value.is_finite()),
            "3D MAC liquid SDF values must be finite"
        );
        self.liquid_phi = liquid_phi;
        self.rebuild_flags_from_phi();
        self.apply_solid_constraints();
    }

    /// Samples a center-cell liquid signed distance field from a closure.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a non-finite value.
    pub fn set_liquid_sdf<F>(&mut self, mut sdf: F)
    where
        F: FnMut([f64; 3]) -> f64,
    {
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let value = sdf([usize_to_f64(x), usize_to_f64(y), usize_to_f64(z)]);
                    assert!(value.is_finite(), "3D MAC liquid SDF values must be finite");
                    self.liquid_phi[cell_index_for_dims3(self.dims, x, y, z)] = finite_f32(value);
                }
            }
        }
        self.rebuild_flags_from_phi();
        self.apply_solid_constraints();
    }

    /// Replaces liquid geometry with one spherical level set.
    ///
    /// # Panics
    ///
    /// Panics if `center` is not finite or if `radius` is not positive and finite.
    pub fn set_liquid_sphere(&mut self, center: [f64; 3], radius: f64) {
        validate_point3(center, "3D MAC liquid sphere center");
        validate_radius3(radius, "3D MAC liquid sphere radius");
        self.set_liquid_sdf(|cell| {
            let dx = cell[0] - center[0];
            let dy = cell[1] - center[1];
            let dz = cell[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt() - radius
        });
    }

    /// Returns the root-mean-square weighted divergence.
    #[must_use]
    pub fn velocity_divergence_l2(&self) -> f64 {
        self.divergence_l2()
    }

    /// Returns root-mean-square divergence over active liquid cells only.
    #[must_use]
    pub fn liquid_velocity_divergence_l2(&self) -> f64 {
        self.liquid_divergence_l2()
    }

    /// Returns a conservative maximum velocity magnitude from open face speeds.
    #[must_use]
    pub fn max_velocity_magnitude(&self) -> f64 {
        let max_u = max_open_face_speed(&self.u, &self.u_weights);
        let max_v = max_open_face_speed(&self.v, &self.v_weights);
        let max_w = max_open_face_speed(&self.w, &self.w_weights);
        max_u.hypot(max_v).hypot(max_w)
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
            "3D MAC CFL number must be positive and finite"
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

    /// Projects face velocity with a matrix-free diagonally preconditioned CG solve.
    pub fn project_velocity(&mut self) -> MacProjectionStats {
        self.apply_solid_velocity_constraints();
        let divergence_before_l2 = self.divergence_l2();
        let (iterations, pressure_residual_l2) = self.solve_pressure_pcg();
        self.apply_pressure_gradient();
        self.apply_solid_velocity_constraints();
        let divergence_after_l2 = self.divergence_l2();
        self.last_projection = MacProjectionStats {
            divergence_before_l2,
            divergence_after_l2,
            pressure_residual_l2,
            iterations,
        };
        self.last_projection
    }

    /// Projects liquid velocity with a single-phase free-surface pressure solve.
    ///
    /// Liquid cells are active. Air cells are inactive and impose a zero-pressure Dirichlet
    /// boundary at liquid-air faces. Solid faces keep no-penetration Neumann behavior through the
    /// solid face weights.
    pub fn project_liquid_velocity(&mut self) -> MacProjectionStats {
        self.apply_solid_velocity_constraints();
        let divergence_before_l2 = self.liquid_divergence_l2();
        let (iterations, pressure_residual_l2) = self.solve_liquid_pressure_pcg();
        self.apply_liquid_pressure_gradient();
        self.apply_solid_velocity_constraints();
        let divergence_after_l2 = self.liquid_divergence_l2();
        self.last_liquid_projection = MacProjectionStats {
            divergence_before_l2,
            divergence_after_l2,
            pressure_residual_l2,
            iterations,
        };
        self.last_liquid_projection
    }

    /// Advances scalar fields and projects velocity.
    pub fn step(&mut self) -> MacStepStats {
        let initial_projection = self.project_velocity();
        let previous_u = self.u.clone();
        let previous_v = self.v.clone();
        let previous_w = self.w.clone();
        self.advect_velocity_semi_lagrangian(&previous_u, &previous_v, &previous_w);

        let previous_density = self.density.clone();
        let previous_temperature = self.temperature.clone();
        let previous_fuel = self.fuel.clone();
        self.density = self.advect_center_field(
            &previous_density,
            &previous_u,
            &previous_v,
            &previous_w,
            true,
            true,
        );
        self.temperature = self.advect_center_field(
            &previous_temperature,
            &previous_u,
            &previous_v,
            &previous_w,
            false,
            false,
        );
        self.fuel = self.advect_center_field(
            &previous_fuel,
            &previous_u,
            &previous_v,
            &previous_w,
            true,
            true,
        );
        self.apply_solid_scalar_constraints();
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

    /// Advances liquid level set and velocity, then applies free-surface projection.
    pub fn step_liquid(&mut self) -> MacStepStats {
        let initial_projection = self.project_liquid_velocity();
        let previous_u = self.u.clone();
        let previous_v = self.v.clone();
        let previous_w = self.w.clone();

        self.advect_velocity_semi_lagrangian(&previous_u, &previous_v, &previous_w);
        self.advect_liquid_level_set(&previous_u, &previous_v, &previous_w);
        self.extrapolate_velocity_into_air(3);
        if self.liquid_viscosity > 0.0 {
            self.apply_liquid_viscosity(self.liquid_viscosity);
        }
        let final_projection = self.project_liquid_velocity();
        MacStepStats {
            initial_projection,
            final_projection,
        }
    }

    /// Advances liquid with enough substeps to satisfy a CFL limit.
    ///
    /// # Panics
    ///
    /// Panics if `cfl_number` is not positive and finite.
    pub fn step_liquid_cfl(&mut self, cfl_number: f64) -> Vec<MacStepStats> {
        let original_dt = self.dt;
        let substeps = self.cfl_substeps(cfl_number);
        self.dt = original_dt / usize_to_f64(substeps);
        let mut stats = Vec::with_capacity(substeps);
        for _ in 0..substeps {
            stats.push(self.step_liquid());
        }
        self.dt = original_dt;
        stats
    }

    /// Applies simple explicit viscosity smoothing to liquid-adjacent face velocities.
    ///
    /// This is a conservative explicit smoother intended for stability tests and light damping.
    /// Strong viscosity should eventually move to an implicit solve.
    ///
    /// # Panics
    ///
    /// Panics if `viscosity` is negative or not finite.
    pub fn apply_liquid_viscosity(&mut self, viscosity: f64) {
        assert!(
            viscosity.is_finite() && viscosity >= 0.0,
            "3D MAC liquid viscosity must be non-negative and finite"
        );
        if viscosity <= f64::EPSILON {
            return;
        }
        self.apply_explicit_viscosity(viscosity);
        self.apply_solid_velocity_constraints();
    }

    /// Exports density as a 3D grid field.
    #[must_use]
    pub fn to_density_grid(&self, bounds: GridBounds) -> GridDensityField {
        GridDensityField::new(bounds, self.dims, self.density.clone())
            .with_interpolation(GridInterpolation::Trilinear)
    }

    /// Exports temperature as a signed 3D scalar grid.
    #[must_use]
    pub fn to_temperature_grid(&self, bounds: GridBounds) -> MacScalarGrid3 {
        MacScalarGrid3::new(bounds, self.dims, self.temperature.clone())
    }

    /// Exports fuel as a 3D grid field.
    #[must_use]
    pub fn to_fuel_grid(&self, bounds: GridBounds) -> GridDensityField {
        GridDensityField::new(bounds, self.dims, self.fuel.clone())
            .with_interpolation(GridInterpolation::Trilinear)
    }

    /// Returns cell-centered velocities in row-major order.
    #[must_use]
    pub fn cell_center_velocities(&self) -> Vec<[f32; 3]> {
        let mut velocities = Vec::with_capacity(cell_count_for_dims3(self.dims));
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let velocity =
                        self.center_velocity_from_faces(&self.u, &self.v, &self.w, x, y, z);
                    velocities.push([
                        finite_f32(velocity[0]),
                        finite_f32(velocity[1]),
                        finite_f32(velocity[2]),
                    ]);
                }
            }
        }
        velocities
    }

    fn rebuild_flags_from_phi(&mut self) {
        for ((flag, solid_phi), liquid_phi) in self
            .flags
            .iter_mut()
            .zip(&self.solid_phi)
            .zip(&self.liquid_phi)
        {
            if *solid_phi <= 0.0 {
                *flag = MacCellFlags::SOLID;
            } else if *liquid_phi <= 0.0 {
                *flag = MacCellFlags::LIQUID;
            } else {
                *flag = MacCellFlags::OPEN;
            }
        }
    }

    fn rebuild_face_weights(&mut self) {
        let [width, height, depth] = self.dims;
        for z in 0..depth {
            for y in 0..height {
                for x_face in 0..=width {
                    let index = u_index_for_dims3(self.dims, x_face, y, z);
                    self.u_weights[index] = if x_face == 0 || x_face == width {
                        0.0
                    } else {
                        let left =
                            self.solid_phi[cell_index_for_dims3(self.dims, x_face - 1, y, z)];
                        let right = self.solid_phi[cell_index_for_dims3(self.dims, x_face, y, z)];
                        sdf_face_open_fraction(left, right)
                    };
                }
            }
        }

        for z in 0..depth {
            for y_face in 0..=height {
                for x in 0..width {
                    let index = v_index_for_dims3(self.dims, x, y_face, z);
                    self.v_weights[index] = if y_face == 0 || y_face == height {
                        0.0
                    } else {
                        let below =
                            self.solid_phi[cell_index_for_dims3(self.dims, x, y_face - 1, z)];
                        let above = self.solid_phi[cell_index_for_dims3(self.dims, x, y_face, z)];
                        sdf_face_open_fraction(below, above)
                    };
                }
            }
        }

        for z_face in 0..=depth {
            for y in 0..height {
                for x in 0..width {
                    let index = w_index_for_dims3(self.dims, x, y, z_face);
                    self.w_weights[index] = if z_face == 0 || z_face == depth {
                        0.0
                    } else {
                        let back =
                            self.solid_phi[cell_index_for_dims3(self.dims, x, y, z_face - 1)];
                        let front = self.solid_phi[cell_index_for_dims3(self.dims, x, y, z_face)];
                        sdf_face_open_fraction(back, front)
                    };
                }
            }
        }
    }

    fn apply_solid_constraints(&mut self) {
        self.apply_solid_scalar_constraints();
        self.apply_solid_velocity_constraints();
    }

    fn apply_solid_scalar_constraints(&mut self) {
        for (((density, temperature), fuel), flag) in self
            .density
            .iter_mut()
            .zip(&mut self.temperature)
            .zip(&mut self.fuel)
            .zip(&self.flags)
        {
            if flag.is_solid() {
                *density = 0.0;
                *temperature = 0.0;
                *fuel = 0.0;
            } else {
                if !density.is_finite() || *density < 0.0 {
                    *density = 0.0;
                }
                if !temperature.is_finite() {
                    *temperature = 0.0;
                }
                if !fuel.is_finite() || *fuel < 0.0 {
                    *fuel = 0.0;
                }
            }
        }
    }

    fn apply_solid_velocity_constraints(&mut self) {
        for (velocity, weight) in self.u.iter_mut().zip(&self.u_weights) {
            if *weight <= FACE_WEIGHT_EPSILON || !velocity.is_finite() {
                *velocity = 0.0;
            }
        }
        for (velocity, weight) in self.v.iter_mut().zip(&self.v_weights) {
            if *weight <= FACE_WEIGHT_EPSILON || !velocity.is_finite() {
                *velocity = 0.0;
            }
        }
        for (velocity, weight) in self.w.iter_mut().zip(&self.w_weights) {
            if *weight <= FACE_WEIGHT_EPSILON || !velocity.is_finite() {
                *velocity = 0.0;
            }
        }
    }

    fn center_velocity_from_faces(
        &self,
        u_faces: &[f32],
        v_faces: &[f32],
        w_faces: &[f32],
        cell_x: usize,
        cell_y: usize,
        cell_z: usize,
    ) -> [f64; 3] {
        [
            0.5 * (f64::from(u_faces[u_index_for_dims3(self.dims, cell_x, cell_y, cell_z)])
                + f64::from(u_faces[u_index_for_dims3(self.dims, cell_x + 1, cell_y, cell_z)])),
            0.5 * (f64::from(v_faces[v_index_for_dims3(self.dims, cell_x, cell_y, cell_z)])
                + f64::from(v_faces[v_index_for_dims3(self.dims, cell_x, cell_y + 1, cell_z)])),
            0.5 * (f64::from(w_faces[w_index_for_dims3(self.dims, cell_x, cell_y, cell_z)])
                + f64::from(w_faces[w_index_for_dims3(self.dims, cell_x, cell_y, cell_z + 1)])),
        ]
    }

    fn max_cfl_rate(&self) -> f64 {
        max_open_face_speed(&self.u, &self.u_weights) / self.cell_size[0]
            + max_open_face_speed(&self.v, &self.v_weights) / self.cell_size[1]
            + max_open_face_speed(&self.w, &self.w_weights) / self.cell_size[2]
    }

    fn divergence_l2(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if self.flags[index].is_solid() {
                        continue;
                    }
                    let divergence = self.cell_divergence(x, y, z);
                    sum += divergence * divergence;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / usize_to_f64(count)).sqrt()
        }
    }

    fn liquid_divergence_l2(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if !self.flags[index].is_liquid() {
                        continue;
                    }
                    let divergence = self.cell_divergence(x, y, z);
                    sum += divergence * divergence;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / usize_to_f64(count)).sqrt()
        }
    }

    fn liquid_cell_has_air_neighbor(&self, x: usize, y: usize, z: usize) -> bool {
        let neighbors = [
            x.checked_sub(1).map(|next_x| [next_x, y, z]),
            (x + 1 < self.dims[0]).then_some([x + 1, y, z]),
            y.checked_sub(1).map(|next_y| [x, next_y, z]),
            (y + 1 < self.dims[1]).then_some([x, y + 1, z]),
            z.checked_sub(1).map(|next_z| [x, y, next_z]),
            (z + 1 < self.dims[2]).then_some([x, y, z + 1]),
        ];
        neighbors.into_iter().flatten().any(|[nx, ny, nz]| {
            let index = cell_index_for_dims3(self.dims, nx, ny, nz);
            self.flags[index].is_open()
        })
    }

    fn cell_divergence(&self, x: usize, y: usize, z: usize) -> f64 {
        let right = u_index_for_dims3(self.dims, x + 1, y, z);
        let left = u_index_for_dims3(self.dims, x, y, z);
        let top = v_index_for_dims3(self.dims, x, y + 1, z);
        let bottom = v_index_for_dims3(self.dims, x, y, z);
        let front = w_index_for_dims3(self.dims, x, y, z + 1);
        let back = w_index_for_dims3(self.dims, x, y, z);
        let du = f64::from(self.u[right] * self.u_weights[right])
            - f64::from(self.u[left] * self.u_weights[left]);
        let dv = f64::from(self.v[top] * self.v_weights[top])
            - f64::from(self.v[bottom] * self.v_weights[bottom]);
        let dw = f64::from(self.w[front] * self.w_weights[front])
            - f64::from(self.w[back] * self.w_weights[back]);
        du / self.cell_size[0] + dv / self.cell_size[1] + dw / self.cell_size[2]
    }

    fn solve_pressure_pcg(&mut self) -> (usize, f64) {
        let cell_count = cell_count_for_dims3(self.dims);
        self.pressure.fill(0.0);
        let mut rhs = vec![0.0_f64; cell_count];
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if !self.flags[index].is_solid() {
                        rhs[index] = -self.cell_divergence(x, y, z) / self.dt;
                    }
                }
            }
        }
        subtract_active_component_means(
            self.dims,
            &self.u_weights,
            &self.v_weights,
            &self.w_weights,
            &mut rhs,
            &self.flags,
        );

        let mut residual = rhs.clone();
        let mut z_preconditioned = vec![0.0_f64; cell_count];
        self.apply_pressure_preconditioner(&residual, &mut z_preconditioned);
        let mut direction = z_preconditioned.clone();
        let mut rz = dot_active(&residual, &z_preconditioned, &self.flags);
        let mut residual_l2 = active_l2(&residual, &self.flags);
        if residual_l2 <= self.pressure_tolerance || rz.abs() <= f64::MIN_POSITIVE {
            return (0, residual_l2);
        }

        let mut matrix_direction = vec![0.0_f64; cell_count];
        let mut iterations = 0usize;
        for iteration in 0..self.pressure_iterations {
            self.apply_pressure_matrix(&direction, &mut matrix_direction);
            let denom = dot_active(&direction, &matrix_direction, &self.flags);
            if denom.abs() <= f64::MIN_POSITIVE {
                break;
            }
            let alpha = rz / denom;
            for i in 0..cell_count {
                if self.flags[i].is_solid() {
                    continue;
                }
                self.pressure[i] = finite_f32(f64::from(self.pressure[i]) + alpha * direction[i]);
                residual[i] -= alpha * matrix_direction[i];
            }
            iterations = iteration + 1;
            residual_l2 = active_l2(&residual, &self.flags);
            if residual_l2 <= self.pressure_tolerance {
                break;
            }

            self.apply_pressure_preconditioner(&residual, &mut z_preconditioned);
            let rz_next = dot_active(&residual, &z_preconditioned, &self.flags);
            if rz.abs() <= f64::MIN_POSITIVE {
                break;
            }
            let beta = rz_next / rz;
            for i in 0..cell_count {
                direction[i] = if self.flags[i].is_solid() {
                    0.0
                } else {
                    z_preconditioned[i] + beta * direction[i]
                };
            }
            rz = rz_next;
        }
        (iterations, residual_l2)
    }

    fn solve_liquid_pressure_pcg(&mut self) -> (usize, f64) {
        let cell_count = cell_count_for_dims3(self.dims);
        self.pressure.fill(0.0);
        let mut rhs = vec![0.0_f64; cell_count];
        let mut has_liquid = false;
        let mut has_free_surface = false;
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if self.flags[index].is_liquid() {
                        has_liquid = true;
                        has_free_surface |= self.liquid_cell_has_air_neighbor(x, y, z);
                        rhs[index] = -self.cell_divergence(x, y, z) / self.dt;
                    }
                }
            }
        }
        if !has_liquid {
            return (0, 0.0);
        }
        if !has_free_surface {
            subtract_liquid_component_means(
                self.dims,
                &self.u_weights,
                &self.v_weights,
                &self.w_weights,
                &mut rhs,
                &self.flags,
            );
        }

        let mut residual = rhs.clone();
        let mut z_preconditioned = vec![0.0_f64; cell_count];
        self.apply_liquid_pressure_preconditioner(&residual, &mut z_preconditioned);
        let mut direction = z_preconditioned.clone();
        let mut rz = dot_liquid(&residual, &z_preconditioned, &self.flags);
        let mut residual_l2 = liquid_l2(&residual, &self.flags);
        if residual_l2 <= self.pressure_tolerance || rz.abs() <= f64::MIN_POSITIVE {
            return (0, residual_l2);
        }

        let mut matrix_direction = vec![0.0_f64; cell_count];
        let mut iterations = 0usize;
        for iteration in 0..self.pressure_iterations {
            self.apply_liquid_pressure_matrix(&direction, &mut matrix_direction);
            let denom = dot_liquid(&direction, &matrix_direction, &self.flags);
            if denom.abs() <= f64::MIN_POSITIVE {
                break;
            }
            let alpha = rz / denom;
            for i in 0..cell_count {
                if !self.flags[i].is_liquid() {
                    continue;
                }
                self.pressure[i] = finite_f32(f64::from(self.pressure[i]) + alpha * direction[i]);
                residual[i] -= alpha * matrix_direction[i];
            }
            iterations = iteration + 1;
            residual_l2 = liquid_l2(&residual, &self.flags);
            if residual_l2 <= self.pressure_tolerance {
                break;
            }

            self.apply_liquid_pressure_preconditioner(&residual, &mut z_preconditioned);
            let rz_next = dot_liquid(&residual, &z_preconditioned, &self.flags);
            if rz.abs() <= f64::MIN_POSITIVE {
                break;
            }
            let beta = rz_next / rz;
            for i in 0..cell_count {
                direction[i] = if self.flags[i].is_liquid() {
                    z_preconditioned[i] + beta * direction[i]
                } else {
                    0.0
                };
            }
            rz = rz_next;
        }
        (iterations, residual_l2)
    }

    fn apply_pressure_preconditioner(&self, residual: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if self.flags[index].is_solid() {
                        continue;
                    }
                    let diagonal = self.pressure_diagonal(x, y, z);
                    if diagonal > f64::MIN_POSITIVE {
                        out[index] = residual[index] / diagonal;
                    }
                }
            }
        }
    }

    fn apply_liquid_pressure_preconditioner(&self, residual: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if !self.flags[index].is_liquid() {
                        continue;
                    }
                    let diagonal = self.liquid_pressure_diagonal(x, y, z);
                    if diagonal > f64::MIN_POSITIVE {
                        out[index] = residual[index] / diagonal;
                    }
                }
            }
        }
    }

    fn apply_pressure_matrix(&self, pressure: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if self.flags[index].is_solid() {
                        continue;
                    }
                    let mut value = 0.0;
                    for neighbor in self.pressure_neighbors(x, y, z) {
                        if self.flags[neighbor.cell].is_solid() {
                            continue;
                        }
                        value += neighbor.coeff * (pressure[index] - pressure[neighbor.cell]);
                    }
                    out[index] = value;
                }
            }
        }
    }

    fn apply_liquid_pressure_matrix(&self, pressure: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if !self.flags[index].is_liquid() {
                        continue;
                    }
                    let mut value = 0.0;
                    for neighbor in self.liquid_pressure_neighbors(x, y, z) {
                        value += neighbor.coeff
                            * (pressure[index] - neighbor.cell.map_or(0.0, |cell| pressure[cell]));
                    }
                    out[index] = value;
                }
            }
        }
    }

    fn pressure_diagonal(&self, x: usize, y: usize, z: usize) -> f64 {
        self.pressure_neighbors(x, y, z)
            .into_iter()
            .filter(|neighbor| !self.flags[neighbor.cell].is_solid())
            .map(|neighbor| neighbor.coeff)
            .sum()
    }

    fn liquid_pressure_diagonal(&self, x: usize, y: usize, z: usize) -> f64 {
        self.liquid_pressure_neighbors(x, y, z)
            .into_iter()
            .map(|neighbor| neighbor.coeff)
            .sum()
    }

    fn pressure_neighbors(&self, x: usize, y: usize, z: usize) -> Vec<PressureNeighbor> {
        let mut neighbors = Vec::with_capacity(6);
        if x > 0 {
            let face = u_index_for_dims3(self.dims, x, y, z);
            neighbors.push(PressureNeighbor {
                cell: cell_index_for_dims3(self.dims, x - 1, y, z),
                coeff: f64::from(self.u_weights[face]) / (self.cell_size[0] * self.cell_size[0]),
            });
        }
        if x + 1 < self.dims[0] {
            let face = u_index_for_dims3(self.dims, x + 1, y, z);
            neighbors.push(PressureNeighbor {
                cell: cell_index_for_dims3(self.dims, x + 1, y, z),
                coeff: f64::from(self.u_weights[face]) / (self.cell_size[0] * self.cell_size[0]),
            });
        }
        if y > 0 {
            let face = v_index_for_dims3(self.dims, x, y, z);
            neighbors.push(PressureNeighbor {
                cell: cell_index_for_dims3(self.dims, x, y - 1, z),
                coeff: f64::from(self.v_weights[face]) / (self.cell_size[1] * self.cell_size[1]),
            });
        }
        if y + 1 < self.dims[1] {
            let face = v_index_for_dims3(self.dims, x, y + 1, z);
            neighbors.push(PressureNeighbor {
                cell: cell_index_for_dims3(self.dims, x, y + 1, z),
                coeff: f64::from(self.v_weights[face]) / (self.cell_size[1] * self.cell_size[1]),
            });
        }
        if z > 0 {
            let face = w_index_for_dims3(self.dims, x, y, z);
            neighbors.push(PressureNeighbor {
                cell: cell_index_for_dims3(self.dims, x, y, z - 1),
                coeff: f64::from(self.w_weights[face]) / (self.cell_size[2] * self.cell_size[2]),
            });
        }
        if z + 1 < self.dims[2] {
            let face = w_index_for_dims3(self.dims, x, y, z + 1);
            neighbors.push(PressureNeighbor {
                cell: cell_index_for_dims3(self.dims, x, y, z + 1),
                coeff: f64::from(self.w_weights[face]) / (self.cell_size[2] * self.cell_size[2]),
            });
        }
        neighbors
    }

    fn liquid_pressure_neighbors(
        &self,
        x: usize,
        y: usize,
        z: usize,
    ) -> Vec<LiquidPressureNeighbor> {
        let mut neighbors = Vec::with_capacity(6);
        self.push_liquid_pressure_neighbor(
            &mut neighbors,
            x.checked_sub(1).map(|next_x| [next_x, y, z]),
            self.u_weights[u_index_for_dims3(self.dims, x, y, z)],
            self.cell_size[0],
        );
        self.push_liquid_pressure_neighbor(
            &mut neighbors,
            (x + 1 < self.dims[0]).then_some([x + 1, y, z]),
            self.u_weights[u_index_for_dims3(self.dims, x + 1, y, z)],
            self.cell_size[0],
        );
        self.push_liquid_pressure_neighbor(
            &mut neighbors,
            y.checked_sub(1).map(|next_y| [x, next_y, z]),
            self.v_weights[v_index_for_dims3(self.dims, x, y, z)],
            self.cell_size[1],
        );
        self.push_liquid_pressure_neighbor(
            &mut neighbors,
            (y + 1 < self.dims[1]).then_some([x, y + 1, z]),
            self.v_weights[v_index_for_dims3(self.dims, x, y + 1, z)],
            self.cell_size[1],
        );
        self.push_liquid_pressure_neighbor(
            &mut neighbors,
            z.checked_sub(1).map(|next_z| [x, y, next_z]),
            self.w_weights[w_index_for_dims3(self.dims, x, y, z)],
            self.cell_size[2],
        );
        self.push_liquid_pressure_neighbor(
            &mut neighbors,
            (z + 1 < self.dims[2]).then_some([x, y, z + 1]),
            self.w_weights[w_index_for_dims3(self.dims, x, y, z + 1)],
            self.cell_size[2],
        );
        neighbors
    }

    fn push_liquid_pressure_neighbor(
        &self,
        neighbors: &mut Vec<LiquidPressureNeighbor>,
        neighbor: Option<[usize; 3]>,
        face_weight: f32,
        cell_size: f64,
    ) {
        if face_weight <= FACE_WEIGHT_EPSILON {
            return;
        }
        let Some(neighbor) = neighbor else {
            return;
        };
        let neighbor_index = cell_index_for_dims3(self.dims, neighbor[0], neighbor[1], neighbor[2]);
        if self.flags[neighbor_index].is_solid() {
            return;
        }
        let coeff = f64::from(face_weight) / (cell_size * cell_size);
        neighbors.push(LiquidPressureNeighbor {
            cell: self.flags[neighbor_index]
                .is_liquid()
                .then_some(neighbor_index),
            coeff,
        });
    }

    fn apply_pressure_gradient(&mut self) {
        let [width, height, depth] = self.dims;
        for z in 0..depth {
            for y in 0..height {
                for x_face in 1..width {
                    let face = u_index_for_dims3(self.dims, x_face, y, z);
                    if self.u_weights[face] <= FACE_WEIGHT_EPSILON {
                        self.u[face] = 0.0;
                        continue;
                    }
                    let left = cell_index_for_dims3(self.dims, x_face - 1, y, z);
                    let right = cell_index_for_dims3(self.dims, x_face, y, z);
                    let gradient =
                        f64::from(self.pressure[right] - self.pressure[left]) / self.cell_size[0];
                    self.u[face] = finite_f32(f64::from(self.u[face]) - self.dt * gradient);
                }
            }
        }

        for z in 0..depth {
            for y_face in 1..height {
                for x in 0..width {
                    let face = v_index_for_dims3(self.dims, x, y_face, z);
                    if self.v_weights[face] <= FACE_WEIGHT_EPSILON {
                        self.v[face] = 0.0;
                        continue;
                    }
                    let below = cell_index_for_dims3(self.dims, x, y_face - 1, z);
                    let above = cell_index_for_dims3(self.dims, x, y_face, z);
                    let gradient =
                        f64::from(self.pressure[above] - self.pressure[below]) / self.cell_size[1];
                    self.v[face] = finite_f32(f64::from(self.v[face]) - self.dt * gradient);
                }
            }
        }

        for z_face in 1..depth {
            for y in 0..height {
                for x in 0..width {
                    let face = w_index_for_dims3(self.dims, x, y, z_face);
                    if self.w_weights[face] <= FACE_WEIGHT_EPSILON {
                        self.w[face] = 0.0;
                        continue;
                    }
                    let back = cell_index_for_dims3(self.dims, x, y, z_face - 1);
                    let front = cell_index_for_dims3(self.dims, x, y, z_face);
                    let gradient =
                        f64::from(self.pressure[front] - self.pressure[back]) / self.cell_size[2];
                    self.w[face] = finite_f32(f64::from(self.w[face]) - self.dt * gradient);
                }
            }
        }
    }

    fn apply_liquid_pressure_gradient(&mut self) {
        let [width, height, depth] = self.dims;
        for z in 0..depth {
            for y in 0..height {
                for x_face in 1..width {
                    let face = u_index_for_dims3(self.dims, x_face, y, z);
                    if self.u_weights[face] <= FACE_WEIGHT_EPSILON {
                        self.u[face] = 0.0;
                        continue;
                    }
                    let left = cell_index_for_dims3(self.dims, x_face - 1, y, z);
                    let right = cell_index_for_dims3(self.dims, x_face, y, z);
                    if !self.flags[left].is_liquid() && !self.flags[right].is_liquid() {
                        continue;
                    }
                    let left_pressure = if self.flags[left].is_liquid() {
                        f64::from(self.pressure[left])
                    } else {
                        0.0
                    };
                    let right_pressure = if self.flags[right].is_liquid() {
                        f64::from(self.pressure[right])
                    } else {
                        0.0
                    };
                    let gradient = (right_pressure - left_pressure) / self.cell_size[0];
                    self.u[face] = finite_f32(f64::from(self.u[face]) - self.dt * gradient);
                }
            }
        }

        for z in 0..depth {
            for y_face in 1..height {
                for x in 0..width {
                    let face = v_index_for_dims3(self.dims, x, y_face, z);
                    if self.v_weights[face] <= FACE_WEIGHT_EPSILON {
                        self.v[face] = 0.0;
                        continue;
                    }
                    let below = cell_index_for_dims3(self.dims, x, y_face - 1, z);
                    let above = cell_index_for_dims3(self.dims, x, y_face, z);
                    if !self.flags[below].is_liquid() && !self.flags[above].is_liquid() {
                        continue;
                    }
                    let below_pressure = if self.flags[below].is_liquid() {
                        f64::from(self.pressure[below])
                    } else {
                        0.0
                    };
                    let above_pressure = if self.flags[above].is_liquid() {
                        f64::from(self.pressure[above])
                    } else {
                        0.0
                    };
                    let gradient = (above_pressure - below_pressure) / self.cell_size[1];
                    self.v[face] = finite_f32(f64::from(self.v[face]) - self.dt * gradient);
                }
            }
        }

        for z_face in 1..depth {
            for y in 0..height {
                for x in 0..width {
                    let face = w_index_for_dims3(self.dims, x, y, z_face);
                    if self.w_weights[face] <= FACE_WEIGHT_EPSILON {
                        self.w[face] = 0.0;
                        continue;
                    }
                    let back = cell_index_for_dims3(self.dims, x, y, z_face - 1);
                    let front = cell_index_for_dims3(self.dims, x, y, z_face);
                    if !self.flags[back].is_liquid() && !self.flags[front].is_liquid() {
                        continue;
                    }
                    let back_pressure = if self.flags[back].is_liquid() {
                        f64::from(self.pressure[back])
                    } else {
                        0.0
                    };
                    let front_pressure = if self.flags[front].is_liquid() {
                        f64::from(self.pressure[front])
                    } else {
                        0.0
                    };
                    let gradient = (front_pressure - back_pressure) / self.cell_size[2];
                    self.w[face] = finite_f32(f64::from(self.w[face]) - self.dt * gradient);
                }
            }
        }
    }

    fn advect_center_field(
        &self,
        source: &[f32],
        u_faces: &[f32],
        v_faces: &[f32],
        w_faces: &[f32],
        nonnegative: bool,
        preserve_mass: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; cell_count_for_dims3(self.dims)];
        let source_mass = scalar_mass(source, &self.flags);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let index = cell_index_for_dims3(self.dims, x, y, z);
                    if self.flags[index].is_solid() {
                        continue;
                    }
                    let position = [usize_to_f64(x), usize_to_f64(y), usize_to_f64(z)];
                    let velocity =
                        self.velocity_at_position_from_faces(u_faces, v_faces, w_faces, position);
                    let back = [
                        position[0] - self.dt * velocity[0] / self.cell_size[0],
                        position[1] - self.dt * velocity[1] / self.cell_size[1],
                        position[2] - self.dt * velocity[2] / self.cell_size[2],
                    ];
                    out[index] = sample_grid3(source, self.dims, back);
                }
            }
        }
        sanitize_advected_scalar(&mut out, nonnegative);
        if preserve_mass {
            rescale_scalar_mass(&mut out, &self.flags, source_mass);
        }
        out
    }

    fn advect_liquid_level_set(&mut self, old_u: &[f32], old_v: &[f32], old_w: &[f32]) {
        let previous_phi = self.liquid_phi.clone();
        self.liquid_phi =
            self.advect_center_field(&previous_phi, old_u, old_v, old_w, false, false);
        self.rebuild_flags_from_phi();
        self.apply_solid_constraints();
    }

    fn advect_velocity_semi_lagrangian(&mut self, old_u: &[f32], old_v: &[f32], old_w: &[f32]) {
        let [width, height, depth] = self.dims;
        let mut next_u = self.u.clone();
        let mut next_v = self.v.clone();
        let mut next_w = self.w.clone();

        for z in 0..depth {
            for y in 0..height {
                for x_face in 0..=width {
                    let index = u_index_for_dims3(self.dims, x_face, y, z);
                    if self.u_weights[index] <= FACE_WEIGHT_EPSILON {
                        next_u[index] = 0.0;
                        continue;
                    }
                    let position = [usize_to_f64(x_face) - 0.5, usize_to_f64(y), usize_to_f64(z)];
                    let velocity =
                        self.velocity_at_position_from_faces(old_u, old_v, old_w, position);
                    let back = [
                        position[0] - self.dt * velocity[0] / self.cell_size[0],
                        position[1] - self.dt * velocity[1] / self.cell_size[1],
                        position[2] - self.dt * velocity[2] / self.cell_size[2],
                    ];
                    next_u[index] = self.sample_u_field(old_u, back);
                }
            }
        }

        for z in 0..depth {
            for y_face in 0..=height {
                for x in 0..width {
                    let index = v_index_for_dims3(self.dims, x, y_face, z);
                    if self.v_weights[index] <= FACE_WEIGHT_EPSILON {
                        next_v[index] = 0.0;
                        continue;
                    }
                    let position = [usize_to_f64(x), usize_to_f64(y_face) - 0.5, usize_to_f64(z)];
                    let velocity =
                        self.velocity_at_position_from_faces(old_u, old_v, old_w, position);
                    let back = [
                        position[0] - self.dt * velocity[0] / self.cell_size[0],
                        position[1] - self.dt * velocity[1] / self.cell_size[1],
                        position[2] - self.dt * velocity[2] / self.cell_size[2],
                    ];
                    next_v[index] = self.sample_v_field(old_v, back);
                }
            }
        }

        for z_face in 0..=depth {
            for y in 0..height {
                for x in 0..width {
                    let index = w_index_for_dims3(self.dims, x, y, z_face);
                    if self.w_weights[index] <= FACE_WEIGHT_EPSILON {
                        next_w[index] = 0.0;
                        continue;
                    }
                    let position = [usize_to_f64(x), usize_to_f64(y), usize_to_f64(z_face) - 0.5];
                    let velocity =
                        self.velocity_at_position_from_faces(old_u, old_v, old_w, position);
                    let back = [
                        position[0] - self.dt * velocity[0] / self.cell_size[0],
                        position[1] - self.dt * velocity[1] / self.cell_size[1],
                        position[2] - self.dt * velocity[2] / self.cell_size[2],
                    ];
                    next_w[index] = self.sample_w_field(old_w, back);
                }
            }
        }

        self.u = next_u;
        self.v = next_v;
        self.w = next_w;
        self.apply_solid_velocity_constraints();
    }

    fn extrapolate_velocity_into_air(&mut self, iterations: usize) {
        let [width, height, depth] = self.dims;
        let mut u_known = vec![false; self.u.len()];
        for z in 0..depth {
            for y in 0..height {
                for x_face in 0..=width {
                    let index = u_index_for_dims3(self.dims, x_face, y, z);
                    u_known[index] = self.u_weights[index] > FACE_WEIGHT_EPSILON
                        && ((x_face > 0
                            && self.flags[cell_index_for_dims3(self.dims, x_face - 1, y, z)]
                                .is_liquid())
                            || (x_face < width
                                && self.flags[cell_index_for_dims3(self.dims, x_face, y, z)]
                                    .is_liquid()));
                }
            }
        }
        extrapolate_face_grid(
            &mut self.u,
            &self.u_weights,
            [width + 1, height, depth],
            &mut u_known,
            iterations,
        );

        let mut v_known = vec![false; self.v.len()];
        for z in 0..depth {
            for y_face in 0..=height {
                for x in 0..width {
                    let index = v_index_for_dims3(self.dims, x, y_face, z);
                    v_known[index] = self.v_weights[index] > FACE_WEIGHT_EPSILON
                        && ((y_face > 0
                            && self.flags[cell_index_for_dims3(self.dims, x, y_face - 1, z)]
                                .is_liquid())
                            || (y_face < height
                                && self.flags[cell_index_for_dims3(self.dims, x, y_face, z)]
                                    .is_liquid()));
                }
            }
        }
        extrapolate_face_grid(
            &mut self.v,
            &self.v_weights,
            [width, height + 1, depth],
            &mut v_known,
            iterations,
        );

        let mut w_known = vec![false; self.w.len()];
        for z_face in 0..=depth {
            for y in 0..height {
                for x in 0..width {
                    let index = w_index_for_dims3(self.dims, x, y, z_face);
                    w_known[index] = self.w_weights[index] > FACE_WEIGHT_EPSILON
                        && ((z_face > 0
                            && self.flags[cell_index_for_dims3(self.dims, x, y, z_face - 1)]
                                .is_liquid())
                            || (z_face < depth
                                && self.flags[cell_index_for_dims3(self.dims, x, y, z_face)]
                                    .is_liquid()));
                }
            }
        }
        extrapolate_face_grid(
            &mut self.w,
            &self.w_weights,
            [width, height, depth + 1],
            &mut w_known,
            iterations,
        );
        self.apply_solid_velocity_constraints();
    }

    fn apply_explicit_viscosity(&mut self, viscosity: f64) {
        let min_cell = self.cell_size[0]
            .min(self.cell_size[1])
            .min(self.cell_size[2]);
        let alpha = (viscosity * self.dt / (min_cell * min_cell)).clamp(0.0, 0.2);
        if alpha <= f64::EPSILON {
            return;
        }
        let [width, height, depth] = self.dims;
        let mut u_active = vec![false; self.u.len()];
        for z in 0..depth {
            for y in 0..height {
                for x_face in 0..=width {
                    let index = u_index_for_dims3(self.dims, x_face, y, z);
                    u_active[index] = self.u_weights[index] > FACE_WEIGHT_EPSILON
                        && ((x_face > 0
                            && self.flags[cell_index_for_dims3(self.dims, x_face - 1, y, z)]
                                .is_liquid())
                            || (x_face < width
                                && self.flags[cell_index_for_dims3(self.dims, x_face, y, z)]
                                    .is_liquid()));
                }
            }
        }
        smooth_face_grid(
            &mut self.u,
            &self.u_weights,
            [width + 1, height, depth],
            &u_active,
            alpha,
        );

        let mut v_active = vec![false; self.v.len()];
        for z in 0..depth {
            for y_face in 0..=height {
                for x in 0..width {
                    let index = v_index_for_dims3(self.dims, x, y_face, z);
                    v_active[index] = self.v_weights[index] > FACE_WEIGHT_EPSILON
                        && ((y_face > 0
                            && self.flags[cell_index_for_dims3(self.dims, x, y_face - 1, z)]
                                .is_liquid())
                            || (y_face < height
                                && self.flags[cell_index_for_dims3(self.dims, x, y_face, z)]
                                    .is_liquid()));
                }
            }
        }
        smooth_face_grid(
            &mut self.v,
            &self.v_weights,
            [width, height + 1, depth],
            &v_active,
            alpha,
        );

        let mut w_active = vec![false; self.w.len()];
        for z_face in 0..=depth {
            for y in 0..height {
                for x in 0..width {
                    let index = w_index_for_dims3(self.dims, x, y, z_face);
                    w_active[index] = self.w_weights[index] > FACE_WEIGHT_EPSILON
                        && ((z_face > 0
                            && self.flags[cell_index_for_dims3(self.dims, x, y, z_face - 1)]
                                .is_liquid())
                            || (z_face < depth
                                && self.flags[cell_index_for_dims3(self.dims, x, y, z_face)]
                                    .is_liquid()));
                }
            }
        }
        smooth_face_grid(
            &mut self.w,
            &self.w_weights,
            [width, height, depth + 1],
            &w_active,
            alpha,
        );
    }

    fn velocity_at_position_from_faces(
        &self,
        u_faces: &[f32],
        v_faces: &[f32],
        w_faces: &[f32],
        position: [f64; 3],
    ) -> [f64; 3] {
        [
            f64::from(self.sample_u_field(u_faces, position)),
            f64::from(self.sample_v_field(v_faces, position)),
            f64::from(self.sample_w_field(w_faces, position)),
        ]
    }

    fn sample_u_field(&self, field: &[f32], position: [f64; 3]) -> f32 {
        sample_grid3(
            field,
            [self.dims[0] + 1, self.dims[1], self.dims[2]],
            [position[0] + 0.5, position[1], position[2]],
        )
    }

    fn sample_v_field(&self, field: &[f32], position: [f64; 3]) -> f32 {
        sample_grid3(
            field,
            [self.dims[0], self.dims[1] + 1, self.dims[2]],
            [position[0], position[1] + 0.5, position[2]],
        )
    }

    fn sample_w_field(&self, field: &[f32], position: [f64; 3]) -> f32 {
        sample_grid3(
            field,
            [self.dims[0], self.dims[1], self.dims[2] + 1],
            [position[0], position[1], position[2] + 0.5],
        )
    }
}

#[derive(Clone, Copy)]
struct PressureNeighbor {
    cell: usize,
    coeff: f64,
}

#[derive(Clone, Copy)]
struct LiquidPressureNeighbor {
    cell: Option<usize>,
    coeff: f64,
}

fn validate_dims3(dims: [usize; 3]) {
    assert!(
        dims.into_iter().all(|dim| dim >= 3),
        "3D MAC grid dimensions must each be at least 3"
    );
    let _ = cell_count_for_dims3(dims);
}

fn validate_point3(value: [f64; 3], label: &str) {
    assert!(
        value.into_iter().all(f64::is_finite),
        "{label} must be finite"
    );
}

fn validate_radius3(radius: f64, label: &str) {
    assert!(
        radius.is_finite() && radius > 0.0,
        "{label} must be positive and finite"
    );
}

fn cell_count_for_dims3(dims: [usize; 3]) -> usize {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .expect("3D MAC grid dimensions overflow")
}

fn u_count_for_dims3(dims: [usize; 3]) -> usize {
    (dims[0] + 1)
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .expect("3D MAC u grid dimensions overflow")
}

fn v_count_for_dims3(dims: [usize; 3]) -> usize {
    dims[0]
        .checked_mul(dims[1] + 1)
        .and_then(|count| count.checked_mul(dims[2]))
        .expect("3D MAC v grid dimensions overflow")
}

fn w_count_for_dims3(dims: [usize; 3]) -> usize {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2] + 1))
        .expect("3D MAC w grid dimensions overflow")
}

fn cell_index_for_dims3(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + dims[0] * (y + dims[1] * z)
}

fn u_index_for_dims3(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + (dims[0] + 1) * (y + dims[1] * z)
}

fn v_index_for_dims3(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + dims[0] * (y + (dims[1] + 1) * z)
}

fn w_index_for_dims3(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + dims[0] * (y + dims[1] * z)
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

fn subtract_active_component_means(
    dims: [usize; 3],
    u_weights: &[f32],
    v_weights: &[f32],
    w_weights: &[f32],
    values: &mut [f64],
    flags: &[MacCellFlags],
) {
    let mut visited = vec![false; values.len()];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let start = cell_index_for_dims3(dims, x, y, z);
                if visited[start] || flags[start].is_solid() {
                    continue;
                }

                let mut stack = vec![[x, y, z]];
                let mut component = Vec::new();
                visited[start] = true;
                while let Some([cell_x, cell_y, cell_z]) = stack.pop() {
                    let index = cell_index_for_dims3(dims, cell_x, cell_y, cell_z);
                    component.push(index);

                    for [next_x, next_y, next_z] in active_neighbors(
                        dims,
                        u_weights,
                        v_weights,
                        w_weights,
                        flags,
                        [cell_x, cell_y, cell_z],
                    ) {
                        let next = cell_index_for_dims3(dims, next_x, next_y, next_z);
                        if !visited[next] {
                            visited[next] = true;
                            stack.push([next_x, next_y, next_z]);
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
}

fn active_neighbors(
    dims: [usize; 3],
    u_weights: &[f32],
    v_weights: &[f32],
    w_weights: &[f32],
    flags: &[MacCellFlags],
    cell: [usize; 3],
) -> Vec<[usize; 3]> {
    let [x, y, z] = cell;
    let mut neighbors = Vec::with_capacity(6);
    if x > 0 && u_weights[u_index_for_dims3(dims, x, y, z)] > FACE_WEIGHT_EPSILON {
        let next = cell_index_for_dims3(dims, x - 1, y, z);
        if !flags[next].is_solid() {
            neighbors.push([x - 1, y, z]);
        }
    }
    if x + 1 < dims[0] && u_weights[u_index_for_dims3(dims, x + 1, y, z)] > FACE_WEIGHT_EPSILON {
        let next = cell_index_for_dims3(dims, x + 1, y, z);
        if !flags[next].is_solid() {
            neighbors.push([x + 1, y, z]);
        }
    }
    if y > 0 && v_weights[v_index_for_dims3(dims, x, y, z)] > FACE_WEIGHT_EPSILON {
        let next = cell_index_for_dims3(dims, x, y - 1, z);
        if !flags[next].is_solid() {
            neighbors.push([x, y - 1, z]);
        }
    }
    if y + 1 < dims[1] && v_weights[v_index_for_dims3(dims, x, y + 1, z)] > FACE_WEIGHT_EPSILON {
        let next = cell_index_for_dims3(dims, x, y + 1, z);
        if !flags[next].is_solid() {
            neighbors.push([x, y + 1, z]);
        }
    }
    if z > 0 && w_weights[w_index_for_dims3(dims, x, y, z)] > FACE_WEIGHT_EPSILON {
        let next = cell_index_for_dims3(dims, x, y, z - 1);
        if !flags[next].is_solid() {
            neighbors.push([x, y, z - 1]);
        }
    }
    if z + 1 < dims[2] && w_weights[w_index_for_dims3(dims, x, y, z + 1)] > FACE_WEIGHT_EPSILON {
        let next = cell_index_for_dims3(dims, x, y, z + 1);
        if !flags[next].is_solid() {
            neighbors.push([x, y, z + 1]);
        }
    }
    neighbors
}

fn dot_active(lhs: &[f64], rhs: &[f64], flags: &[MacCellFlags]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .zip(flags)
        .filter_map(|((lhs, rhs), flag)| (!flag.is_solid()).then_some(lhs * rhs))
        .sum()
}

fn active_l2(values: &[f64], flags: &[MacCellFlags]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (value, flag) in values.iter().zip(flags) {
        if flag.is_solid() {
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

fn subtract_liquid_component_means(
    dims: [usize; 3],
    u_weights: &[f32],
    v_weights: &[f32],
    w_weights: &[f32],
    values: &mut [f64],
    flags: &[MacCellFlags],
) {
    let mut visited = vec![false; values.len()];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let start = cell_index_for_dims3(dims, x, y, z);
                if visited[start] || !flags[start].is_liquid() {
                    continue;
                }

                let mut stack = vec![[x, y, z]];
                let mut component = Vec::new();
                visited[start] = true;
                while let Some([cell_x, cell_y, cell_z]) = stack.pop() {
                    let index = cell_index_for_dims3(dims, cell_x, cell_y, cell_z);
                    component.push(index);

                    for [next_x, next_y, next_z] in active_liquid_neighbors(
                        dims,
                        u_weights,
                        v_weights,
                        w_weights,
                        flags,
                        [cell_x, cell_y, cell_z],
                    ) {
                        let next = cell_index_for_dims3(dims, next_x, next_y, next_z);
                        if !visited[next] {
                            visited[next] = true;
                            stack.push([next_x, next_y, next_z]);
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
}

fn active_liquid_neighbors(
    dims: [usize; 3],
    u_weights: &[f32],
    v_weights: &[f32],
    w_weights: &[f32],
    flags: &[MacCellFlags],
    cell: [usize; 3],
) -> Vec<[usize; 3]> {
    active_neighbors(dims, u_weights, v_weights, w_weights, flags, cell)
        .into_iter()
        .filter(|[x, y, z]| flags[cell_index_for_dims3(dims, *x, *y, *z)].is_liquid())
        .collect()
}

fn dot_liquid(lhs: &[f64], rhs: &[f64], flags: &[MacCellFlags]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .zip(flags)
        .filter_map(|((lhs, rhs), flag)| flag.is_liquid().then_some(lhs * rhs))
        .sum()
}

fn liquid_l2(values: &[f64], flags: &[MacCellFlags]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (value, flag) in values.iter().zip(flags) {
        if !flag.is_liquid() {
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

fn scalar_mass(samples: &[f32], flags: &[MacCellFlags]) -> f64 {
    samples
        .iter()
        .zip(flags)
        .filter_map(|(sample, flag)| (!flag.is_solid()).then_some(f64::from(*sample)))
        .sum()
}

fn rescale_scalar_mass(samples: &mut [f32], flags: &[MacCellFlags], target_mass: f64) {
    if target_mass <= f64::MIN_POSITIVE {
        return;
    }
    let current_mass = scalar_mass(samples, flags);
    if current_mass <= f64::MIN_POSITIVE {
        return;
    }
    let scale = target_mass / current_mass;
    for (sample, flag) in samples.iter_mut().zip(flags) {
        if !flag.is_solid() {
            *sample = finite_f32(f64::from(*sample) * scale);
        }
    }
}

fn sample_grid3(field: &[f32], dims: [usize; 3], position: [f64; 3]) -> f32 {
    debug_assert_eq!(field.len(), cell_count_for_dims3(dims));
    let x = position[0].clamp(0.0, usize_to_f64(dims[0] - 1));
    let y = position[1].clamp(0.0, usize_to_f64(dims[1] - 1));
    let z = position[2].clamp(0.0, usize_to_f64(dims[2] - 1));
    let x0 = floor_index(x, dims[0]);
    let y0 = floor_index(y, dims[1]);
    let z0 = floor_index(z, dims[2]);
    let x1 = (x0 + 1).min(dims[0] - 1);
    let y1 = (y0 + 1).min(dims[1] - 1);
    let z1 = (z0 + 1).min(dims[2] - 1);
    let tx = x - usize_to_f64(x0);
    let ty = y - usize_to_f64(y0);
    let tz = z - usize_to_f64(z0);

    let sample =
        |sx: usize, sy: usize, sz: usize| f64::from(field[cell_index_for_dims3(dims, sx, sy, sz)]);
    let c00 = sample(x0, y0, z0) * (1.0 - tx) + sample(x1, y0, z0) * tx;
    let c10 = sample(x0, y1, z0) * (1.0 - tx) + sample(x1, y1, z0) * tx;
    let c01 = sample(x0, y0, z1) * (1.0 - tx) + sample(x1, y0, z1) * tx;
    let c11 = sample(x0, y1, z1) * (1.0 - tx) + sample(x1, y1, z1) * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    finite_f32(c0 * (1.0 - tz) + c1 * tz)
}

fn extrapolate_face_grid(
    field: &mut [f32],
    weights: &[f32],
    dims: [usize; 3],
    known: &mut [bool],
    iterations: usize,
) {
    debug_assert_eq!(field.len(), cell_count_for_dims3(dims));
    debug_assert_eq!(weights.len(), field.len());
    debug_assert_eq!(known.len(), field.len());

    for _ in 0..iterations {
        let previous = field.to_vec();
        let previous_known = known.to_vec();
        let mut changed = false;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let index = cell_index_for_dims3(dims, x, y, z);
                    if previous_known[index] || weights[index] <= FACE_WEIGHT_EPSILON {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for [nx, ny, nz] in grid_neighbors3(dims, [x, y, z]) {
                        let neighbor = cell_index_for_dims3(dims, nx, ny, nz);
                        if previous_known[neighbor] {
                            sum += f64::from(previous[neighbor]);
                            count += 1;
                        }
                    }
                    if count > 0 {
                        field[index] = finite_f32(sum / usize_to_f64(count));
                        known[index] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn smooth_face_grid(
    field: &mut [f32],
    weights: &[f32],
    dims: [usize; 3],
    active: &[bool],
    alpha: f64,
) {
    debug_assert_eq!(field.len(), cell_count_for_dims3(dims));
    debug_assert_eq!(weights.len(), field.len());
    debug_assert_eq!(active.len(), field.len());
    let previous = field.to_vec();
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let index = cell_index_for_dims3(dims, x, y, z);
                if !active[index] || weights[index] <= FACE_WEIGHT_EPSILON {
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0usize;
                for [nx, ny, nz] in grid_neighbors3(dims, [x, y, z]) {
                    let neighbor = cell_index_for_dims3(dims, nx, ny, nz);
                    if active[neighbor] && weights[neighbor] > FACE_WEIGHT_EPSILON {
                        sum += f64::from(previous[neighbor]);
                        count += 1;
                    }
                }
                if count > 0 {
                    let average = sum / usize_to_f64(count);
                    field[index] = finite_f32(
                        f64::from(previous[index]) + alpha * (average - f64::from(previous[index])),
                    );
                }
            }
        }
    }
}

fn grid_neighbors3(dims: [usize; 3], cell: [usize; 3]) -> Vec<[usize; 3]> {
    let [x, y, z] = cell;
    let mut neighbors = Vec::with_capacity(6);
    if x > 0 {
        neighbors.push([x - 1, y, z]);
    }
    if x + 1 < dims[0] {
        neighbors.push([x + 1, y, z]);
    }
    if y > 0 {
        neighbors.push([x, y - 1, z]);
    }
    if y + 1 < dims[1] {
        neighbors.push([x, y + 1, z]);
    }
    if z > 0 {
        neighbors.push([x, y, z - 1]);
    }
    if z + 1 < dims[2] {
        neighbors.push([x, y, z + 1]);
    }
    neighbors
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
        assert!((actual - expected).abs() < 1.0e-8, "{actual} != {expected}");
    }

    fn density_mass(sim: &MacFluidGrid3) -> f64 {
        sim.densities().iter().map(|value| f64::from(*value)).sum()
    }

    #[test]
    fn mac3_grid_uses_3d_face_velocity_shapes_and_fields() {
        let sim = MacFluidGrid3::new([4, 5, 6]);

        assert_eq!(sim.densities().len(), 120);
        assert_eq!(sim.temperatures().len(), 120);
        assert_eq!(sim.fuels().len(), 120);
        assert_eq!(sim.pressures().len(), 120);
        assert_eq!(sim.u().len(), 150);
        assert_eq!(sim.v().len(), 144);
        assert_eq!(sim.w().len(), 140);
        assert!(sim.flags().iter().all(|flag| flag.is_open()));
    }

    #[test]
    fn mac3_projection_reduces_divergence() {
        let mut sim = MacFluidGrid3::new([8, 8, 8])
            .with_dt(0.2)
            .with_pressure_iterations(180)
            .with_pressure_tolerance(1.0e-7);
        for z in 1..7 {
            for y in 1..7 {
                for x in 1..7 {
                    sim.set_u([x, y, z], usize_to_f64(x) - 3.5);
                    sim.set_v([x, y, z], usize_to_f64(y) - 3.5);
                    sim.set_w([x, y, z], usize_to_f64(z) - 3.5);
                }
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
    }

    #[test]
    fn mac3_sdf_sphere_sets_flags_and_blocks_faces() {
        let mut sim = MacFluidGrid3::new([10, 10, 10]);
        sim.set_solid_sphere([5.0, 5.0, 5.0], 2.5);

        assert!(sim.is_solid([5, 5, 5]));
        assert!(!sim.is_solid([1, 1, 1]));
        assert!(sim.flags().iter().any(|flag| flag.is_solid()));
        assert!(
            sim.u_weights()
                .iter()
                .chain(sim.v_weights())
                .chain(sim.w_weights())
                .any(|weight| *weight > 0.0 && *weight < 1.0)
        );
    }

    #[test]
    fn mac3_cfl_substeps_for_large_velocity() {
        let mut sim = MacFluidGrid3::new([5, 5, 5]).with_dt(1.0);
        for z in 0..5 {
            for y in 0..5 {
                for x in 1..5 {
                    sim.set_u([x, y, z], 4.0);
                }
            }
        }

        assert!(sim.max_velocity_magnitude() > 0.0);
        assert!(sim.cfl_timestep(0.5) < sim.dt());
        assert!(sim.cfl_substeps(0.5) > 1);
    }

    #[test]
    fn mac3_cfl_uses_face_speed_not_center_average() {
        let mut sim = MacFluidGrid3::new([4, 4, 4]).with_dt(1.0);
        for z in 0..4 {
            for y in 0..4 {
                sim.set_u([1, y, z], 12.0);
                sim.set_u([2, y, z], -12.0);
            }
        }

        assert_close(sim.velocity_at_cell([1, 2, 2])[0], 0.0);
        assert_close(sim.max_velocity_magnitude(), 12.0);
        assert_close(sim.cfl_timestep(0.5), 0.5 / 12.0);
    }

    #[test]
    fn mac3_pressure_rhs_subtracts_mean_per_active_component() {
        let dims = [4, 2, 2];
        let mut flags = vec![MacCellFlags::OPEN; cell_count_for_dims3(dims)];
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                flags[cell_index_for_dims3(dims, 2, y, z)] = MacCellFlags::SOLID;
            }
        }
        let u_weights = vec![1.0; u_count_for_dims3(dims)];
        let v_weights = vec![1.0; v_count_for_dims3(dims)];
        let w_weights = vec![1.0; w_count_for_dims3(dims)];
        let mut values = vec![0.0; cell_count_for_dims3(dims)];
        let mut left_value = 2.0;
        let mut right_value = 20.0;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let index = cell_index_for_dims3(dims, x, y, z);
                    values[index] = match x {
                        0 | 1 => {
                            let value = left_value;
                            left_value += 2.0;
                            value
                        }
                        2 => 123.0,
                        _ => {
                            let value = right_value;
                            right_value += 4.0;
                            value
                        }
                    };
                }
            }
        }

        subtract_active_component_means(
            dims,
            &u_weights,
            &v_weights,
            &w_weights,
            &mut values,
            &flags,
        );

        let mut left_sum = 0.0;
        let mut right_sum = 0.0;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                left_sum += values[cell_index_for_dims3(dims, 0, y, z)]
                    + values[cell_index_for_dims3(dims, 1, y, z)];
                right_sum += values[cell_index_for_dims3(dims, 3, y, z)];
                assert_close(values[cell_index_for_dims3(dims, 2, y, z)], 123.0);
            }
        }
        assert_close(left_sum, 0.0);
        assert_close(right_sum, 0.0);
    }

    #[test]
    fn mac3_velocity_advection_moves_face_velocities() {
        let mut sim = MacFluidGrid3::new([5, 4, 4]).with_dt(0.25);
        for z in 0..4 {
            for y in 0..4 {
                for x in 1..5 {
                    sim.set_u([x, y, z], usize_to_f64(x));
                }
            }
        }
        let old_u = sim.u.clone();
        let old_v = sim.v.clone();
        let old_w = sim.w.clone();

        sim.advect_velocity_semi_lagrangian(&old_u, &old_v, &old_w);

        assert!(
            sim.u
                .iter()
                .zip(old_u)
                .any(|(current, previous)| (*current - previous).abs() > 1.0e-5)
        );
    }

    #[test]
    fn mac3_mass_is_preserved_under_nonzero_velocity_advection() {
        let mut sim = MacFluidGrid3::new([6, 6, 6]).with_dt(0.5);
        sim.add_density([3, 3, 3], 5.0);
        sim.add_density([2, 3, 3], 2.0);
        for z in 1..5 {
            for y in 1..5 {
                for x in 1..6 {
                    sim.set_u([x, y, z], 0.75);
                }
            }
        }
        let before = density_mass(&sim);

        sim.step();

        assert!((density_mass(&sim) - before).abs() < 1.0e-5);
    }

    #[test]
    fn mac3_boundary_no_penetration_around_obstacle() {
        let mut sim = MacFluidGrid3::new([10, 10, 10]);
        sim.set_solid_sphere([5.0, 5.0, 5.0], 2.5);
        for z in 0..10 {
            for y in 0..10 {
                for x in 1..10 {
                    sim.set_u([x, y, z], 1.0);
                }
            }
        }

        sim.project_velocity();

        for (velocity, weight) in sim.u().iter().zip(sim.u_weights()) {
            if *weight <= FACE_WEIGHT_EPSILON {
                assert_close(f64::from(*velocity), 0.0);
            }
        }
    }

    #[test]
    fn mac3_liquid_level_set_classifies_liquid_air_and_solid_separately() {
        let mut sim = MacFluidGrid3::new([6, 6, 6]);
        sim.set_liquid_sphere([3.0, 3.0, 3.0], 1.75);
        sim.set_solid_sphere([3.0, 3.0, 3.0], 0.75);

        assert!(sim.is_solid([3, 3, 3]));
        assert!(!sim.is_liquid([3, 3, 3]));
        assert!(sim.is_liquid([4, 3, 3]));
        assert!(sim.is_air([0, 0, 0]));
        assert!(sim.liquid_phi()[sim.index([4, 3, 3])] <= 0.0);

        sim.set_liquid([0, 0, 0], true);

        assert!(sim.is_liquid([0, 0, 0]));
        assert!(!sim.is_solid([0, 0, 0]));
    }

    #[test]
    fn mac3_free_surface_projection_reduces_liquid_divergence_and_leaves_air_pressure_zero() {
        let mut sim = MacFluidGrid3::new([8, 8, 8])
            .with_dt(0.2)
            .with_pressure_iterations(180)
            .with_pressure_tolerance(1.0e-7);
        sim.set_liquid_sphere([4.0, 4.0, 4.0], 2.3);
        for z in 2..6 {
            for y in 2..6 {
                for x in 2..7 {
                    sim.set_u([x, y, z], usize_to_f64(x) - 4.0);
                }
            }
        }

        let smoke_divergence_before = sim.velocity_divergence_l2();
        let stats = sim.project_liquid_velocity();

        assert!(stats.divergence_before_l2 > 0.0);
        assert!(
            stats.divergence_after_l2 < stats.divergence_before_l2,
            "{stats:?}"
        );
        assert_eq!(sim.last_liquid_projection(), stats);
        assert!(sim.velocity_divergence_l2() <= smoke_divergence_before);
        for (pressure, flag) in sim.pressures().iter().zip(sim.flags()) {
            if flag.is_open() {
                assert_close(f64::from(*pressure), 0.0);
            }
        }
    }

    #[test]
    fn mac3_liquid_projection_keeps_no_penetration_around_obstacle() {
        let mut sim = MacFluidGrid3::new([10, 10, 10]).with_pressure_iterations(180);
        sim.set_liquid_sphere([5.0, 5.0, 5.0], 3.5);
        sim.set_solid_sphere([5.0, 5.0, 5.0], 1.8);
        for z in 0..10 {
            for y in 0..10 {
                for x in 1..10 {
                    sim.set_u([x, y, z], 1.0);
                }
            }
        }

        let stats = sim.project_liquid_velocity();

        assert!(
            stats.divergence_after_l2 < stats.divergence_before_l2,
            "{stats:?}"
        );
        for (velocity, weight) in sim.u().iter().zip(sim.u_weights()) {
            if *weight <= FACE_WEIGHT_EPSILON {
                assert_close(f64::from(*velocity), 0.0);
            }
        }
    }

    #[test]
    fn mac3_liquid_level_set_advects_with_velocity() {
        let mut sim = MacFluidGrid3::new([6, 4, 4]).with_dt(0.5);
        sim.set_liquid_sdf(|cell| cell[0] - 2.5);
        for z in 0..4 {
            for y in 0..4 {
                for x in 1..6 {
                    sim.set_u([x, y, z], 1.0);
                }
            }
        }
        let old_phi = sim.liquid_phi().to_vec();
        let old_u = sim.u.clone();
        let old_v = sim.v.clone();
        let old_w = sim.w.clone();

        sim.advect_liquid_level_set(&old_u, &old_v, &old_w);

        assert!(
            sim.liquid_phi()
                .iter()
                .zip(old_phi)
                .any(|(current, previous)| (*current - previous).abs() > 1.0e-5)
        );
        assert!(sim.flags().iter().any(|flag| flag.is_liquid()));
        assert!(sim.flags().iter().any(|flag| flag.is_open()));
    }

    #[test]
    fn mac3_extrapolates_liquid_velocity_into_nearby_air() {
        let mut sim = MacFluidGrid3::new([6, 4, 4]);
        sim.set_liquid([2, 2, 2], true);
        sim.set_u([2, 2, 2], 2.0);
        let air_face = u_index_for_dims3(sim.dims(), 1, 2, 2);
        assert_close(f64::from(sim.u()[air_face]), 0.0);

        sim.extrapolate_velocity_into_air(3);

        assert!(f64::from(sim.u()[air_face]).abs() > 0.0);
    }

    #[test]
    fn mac3_explicit_liquid_viscosity_damps_velocity_peak() {
        let mut sim = MacFluidGrid3::new([6, 6, 6]).with_dt(0.1);
        sim.set_liquid_sphere([3.0, 3.0, 3.0], 2.0);
        sim.set_u([3, 3, 3], 5.0);
        let peak = f64::from(sim.u()[u_index_for_dims3(sim.dims(), 3, 3, 3)]);

        sim.apply_liquid_viscosity(1.0);

        assert!(f64::from(sim.u()[u_index_for_dims3(sim.dims(), 3, 3, 3)]) < peak);
    }

    #[test]
    fn mac3_poiseuille_profile_is_divergence_free() {
        let mut sim = MacFluidGrid3::new([12, 8, 6]);
        let height_center = 3.5;
        for z in 0..6 {
            for y in 0..8 {
                let dy = (usize_to_f64(y) - height_center) / height_center;
                let velocity = (1.0 - dy * dy).max(0.0);
                for x in 1..12 {
                    sim.set_u([x, y, z], velocity);
                }
            }
        }

        for z in 0..6 {
            for y in 0..8 {
                for x in 1..11 {
                    assert!(sim.cell_divergence(x, y, z).abs() < 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn mac3_lid_driven_cavity_projection_keeps_no_penetration() {
        let mut sim = MacFluidGrid3::new([8, 8, 8]).with_pressure_iterations(160);
        for z in 1..7 {
            for x in 1..8 {
                sim.set_u([x, 7, z], 1.0);
            }
        }

        let stats = sim.project_velocity();

        assert!(stats.divergence_after_l2 < stats.divergence_before_l2);
        for z in 0..8 {
            for y in 0..8 {
                assert_close(
                    f64::from(sim.u()[u_index_for_dims3(sim.dims(), 0, y, z)]),
                    0.0,
                );
                assert_close(
                    f64::from(sim.u()[u_index_for_dims3(sim.dims(), 8, y, z)]),
                    0.0,
                );
            }
        }
    }

    #[test]
    fn mac3_cylinder_obstacle_vortex_shedding_setup_projects() {
        let mut sim = MacFluidGrid3::new([14, 10, 6]).with_pressure_iterations(160);
        sim.set_solid_sdf(|cell| {
            let dx = cell[0] - 5.0;
            let dy = cell[1] - 5.0;
            dx.hypot(dy) - 1.8
        });
        for z in 0..6 {
            for y in 0..10 {
                for x in 1..14 {
                    sim.set_u([x, y, z], 1.2);
                }
            }
        }

        let stats = sim.project_velocity();

        assert!(stats.divergence_after_l2 < stats.divergence_before_l2);
        assert!(sim.flags().iter().any(|flag| flag.is_solid()));
    }

    #[test]
    fn mac3_exports_density_temperature_fuel_and_velocity_grids() {
        let mut sim = MacFluidGrid3::new([4, 3, 5]);
        sim.set_density([2, 1, 3], 7.0);
        sim.set_temperature([2, 1, 3], 2.0);
        sim.set_temperature([1, 1, 2], -3.0);
        sim.set_fuel([2, 1, 3], 4.0);
        sim.set_u([2, 1, 3], 1.0);
        let bounds = GridBounds::new(Point::new(-1.0, -1.0, -1.0), Point::new(1.0, 1.0, 1.0));

        let density = sim.to_density_grid(bounds);
        let temperature = sim.to_temperature_grid(bounds);
        let fuel = sim.to_fuel_grid(bounds);
        let velocities = sim.cell_center_velocities();

        assert_eq!(density.dims(), [4, 3, 5]);
        assert_eq!(density.interpolation(), GridInterpolation::Trilinear);
        assert_close(density.density(density.cell_center(2, 1, 3), 0.0), 7.0);
        assert_eq!(temperature.dims(), [4, 3, 5]);
        assert_eq!(temperature.bounds(), bounds);
        assert_eq!(temperature.samples().len(), sim.temperatures().len());
        assert_close(temperature.sample_at([2, 1, 3]), 2.0);
        assert_close(temperature.sample_at([1, 1, 2]), -3.0);
        assert_close(fuel.density(fuel.cell_center(2, 1, 3), 0.0), 4.0);
        assert_eq!(velocities.len(), sim.densities().len());
    }
}

use super::{
    BoundaryCondition, apply_solid_scalar_boundaries, apply_solid_velocity_boundaries, finite_f32,
    index_for_dims, is_solid_cell, set_boundary, set_velocity_boundary,
};

pub(super) struct VelocityFieldMut<'a> {
    pub x: &'a mut [f32],
    pub y: &'a mut [f32],
}

pub(super) struct VelocityField<'a> {
    pub x: &'a [f32],
    pub y: &'a [f32],
}

#[derive(Clone, Copy)]
struct LinearSolveParams {
    a: f64,
    c: f64,
    iterations: usize,
    boundary: BoundaryCondition,
}

pub(super) fn diffuse_scalar_2d(
    dims: [usize; 2],
    solid: &[bool],
    dt: f64,
    diffusion: f64,
    iterations: usize,
    density: &mut [f32],
    previous_density: &[f32],
) {
    debug_assert_eq!(density.len(), previous_density.len());
    debug_assert_eq!(density.len(), solid.len());

    if diffusion <= 0.0 {
        density.clone_from_slice(previous_density);
        set_boundary(dims, density, BoundaryCondition::Scalar);
        apply_solid_scalar_boundaries(dims, solid, density);
        return;
    }

    let a = dt * diffusion;
    density.clone_from_slice(previous_density);
    linear_solve_2d(
        dims,
        solid,
        density,
        previous_density,
        LinearSolveParams {
            a,
            c: 1.0 + 4.0 * a,
            iterations,
            boundary: BoundaryCondition::Scalar,
        },
    );
}

#[allow(clippy::needless_pass_by_value)]
pub(super) fn diffuse_velocity_2d(
    dims: [usize; 2],
    solid: &[bool],
    dt: f64,
    viscosity: f64,
    iterations: usize,
    velocity: VelocityFieldMut<'_>,
    previous_velocity: VelocityField<'_>,
) {
    debug_assert_eq!(velocity.x.len(), velocity.y.len());
    debug_assert_eq!(velocity.x.len(), previous_velocity.x.len());
    debug_assert_eq!(velocity.x.len(), previous_velocity.y.len());
    debug_assert_eq!(velocity.x.len(), solid.len());

    if viscosity <= 0.0 {
        velocity.x.clone_from_slice(previous_velocity.x);
        velocity.y.clone_from_slice(previous_velocity.y);
        set_velocity_boundary(dims, velocity.x, velocity.y);
        apply_solid_velocity_boundaries(dims, solid, velocity.x, velocity.y);
        return;
    }

    let a = dt * viscosity;
    velocity.x.clone_from_slice(previous_velocity.x);
    velocity.y.clone_from_slice(previous_velocity.y);
    linear_solve_2d(
        dims,
        solid,
        velocity.x,
        previous_velocity.x,
        LinearSolveParams {
            a,
            c: 1.0 + 4.0 * a,
            iterations,
            boundary: BoundaryCondition::HorizontalVelocity,
        },
    );
    linear_solve_2d(
        dims,
        solid,
        velocity.y,
        previous_velocity.y,
        LinearSolveParams {
            a,
            c: 1.0 + 4.0 * a,
            iterations,
            boundary: BoundaryCondition::VerticalVelocity,
        },
    );
    set_velocity_boundary(dims, velocity.x, velocity.y);
    apply_solid_velocity_boundaries(dims, solid, velocity.x, velocity.y);
}

pub(super) fn solve_poisson_2d(
    dims: [usize; 2],
    solid: &[bool],
    pressure: &mut [f32],
    divergence: &[f32],
    iterations: usize,
) {
    linear_solve_2d(
        dims,
        solid,
        pressure,
        divergence,
        LinearSolveParams {
            a: 1.0,
            c: 4.0,
            iterations,
            boundary: BoundaryCondition::Scalar,
        },
    );
}

fn linear_solve_2d(
    dims: [usize; 2],
    solid: &[bool],
    current: &mut [f32],
    previous: &[f32],
    params: LinearSolveParams,
) {
    debug_assert_eq!(current.len(), previous.len());
    debug_assert_eq!(current.len(), solid.len());

    for _ in 0..params.iterations {
        for y in 1..(dims[1] - 1) {
            for x in 1..(dims[0] - 1) {
                let index = index_for_dims(dims, x, y);
                if solid[index] {
                    current[index] = 0.0;
                    continue;
                }
                let neighbors = sample_neighbor(dims, solid, current, index, x - 1, y)
                    + sample_neighbor(dims, solid, current, index, x + 1, y)
                    + sample_neighbor(dims, solid, current, index, x, y - 1)
                    + sample_neighbor(dims, solid, current, index, x, y + 1);
                current[index] =
                    finite_f32((f64::from(previous[index]) + params.a * neighbors) / params.c);
            }
        }
        set_boundary(dims, current, params.boundary);
        apply_solid_scalar_boundaries(dims, solid, current);
    }
}

fn sample_neighbor(
    dims: [usize; 2],
    solid: &[bool],
    field: &[f32],
    current_index: usize,
    x: usize,
    y: usize,
) -> f64 {
    if is_solid_cell(dims, solid, x, y) {
        f64::from(field[current_index])
    } else {
        f64::from(field[index_for_dims(dims, x, y)])
    }
}

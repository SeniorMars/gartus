use super::{
    BoundaryCondition, apply_solid_scalar_boundaries, apply_solid_velocity_boundaries, finite_f32,
    index_for_dims, is_solid_cell, linear_solve, set_boundary, set_velocity_boundary,
};

pub(super) fn project_2d(
    dims: [usize; 2],
    solid: &[bool],
    velocity_x: &mut [f32],
    velocity_y: &mut [f32],
    pressure: &mut [f32],
    divergence: &mut [f32],
    iterations: usize,
) {
    debug_assert_eq!(velocity_x.len(), velocity_y.len());
    debug_assert_eq!(velocity_x.len(), pressure.len());
    debug_assert_eq!(velocity_x.len(), divergence.len());
    debug_assert_eq!(velocity_x.len(), solid.len());

    apply_solid_velocity_boundaries(dims, solid, velocity_x, velocity_y);
    pressure.fill(0.0);
    divergence.fill(0.0);

    for y in 1..(dims[1] - 1) {
        for x in 1..(dims[0] - 1) {
            let index = index_for_dims(dims, x, y);
            if solid[index] {
                continue;
            }
            let div = -0.5
                * (sample_velocity(dims, solid, velocity_x, x + 1, y)
                    - sample_velocity(dims, solid, velocity_x, x - 1, y)
                    + sample_velocity(dims, solid, velocity_y, x, y + 1)
                    - sample_velocity(dims, solid, velocity_y, x, y - 1));
            divergence[index] = finite_f32(div);
        }
    }

    set_boundary(dims, divergence, BoundaryCondition::Scalar);
    set_boundary(dims, pressure, BoundaryCondition::Scalar);
    apply_solid_scalar_boundaries(dims, solid, divergence);
    apply_solid_scalar_boundaries(dims, solid, pressure);
    linear_solve::solve_poisson_2d(dims, solid, pressure, divergence, iterations);

    for y in 1..(dims[1] - 1) {
        for x in 1..(dims[0] - 1) {
            let index = index_for_dims(dims, x, y);
            if solid[index] {
                velocity_x[index] = 0.0;
                velocity_y[index] = 0.0;
                continue;
            }
            let horizontal_pressure_delta = sample_pressure(dims, solid, pressure, index, x + 1, y)
                - sample_pressure(dims, solid, pressure, index, x - 1, y);
            let vertical_pressure_delta = sample_pressure(dims, solid, pressure, index, x, y + 1)
                - sample_pressure(dims, solid, pressure, index, x, y - 1);
            velocity_x[index] =
                finite_f32(f64::from(velocity_x[index]) - 0.5 * horizontal_pressure_delta);
            velocity_y[index] =
                finite_f32(f64::from(velocity_y[index]) - 0.5 * vertical_pressure_delta);
        }
    }

    set_velocity_boundary(dims, velocity_x, velocity_y);
    apply_solid_velocity_boundaries(dims, solid, velocity_x, velocity_y);
}

pub(super) fn divergence_l2_2d(
    dims: [usize; 2],
    solid: &[bool],
    velocity_x: &[f32],
    velocity_y: &[f32],
) -> f64 {
    debug_assert_eq!(velocity_x.len(), velocity_y.len());
    debug_assert_eq!(velocity_x.len(), solid.len());

    let mut sum = 0.0;
    let mut count = 0_usize;
    for y in 1..(dims[1] - 1) {
        for x in 1..(dims[0] - 1) {
            if is_solid_cell(dims, solid, x, y) {
                continue;
            }
            let div = 0.5
                * (sample_velocity(dims, solid, velocity_x, x + 1, y)
                    - sample_velocity(dims, solid, velocity_x, x - 1, y)
                    + sample_velocity(dims, solid, velocity_y, x, y + 1)
                    - sample_velocity(dims, solid, velocity_y, x, y - 1));
            sum += div * div;
            count += 1;
        }
    }

    (sum / super::usize_to_f64(count).max(1.0)).sqrt()
}

fn sample_velocity(dims: [usize; 2], solid: &[bool], velocity: &[f32], x: usize, y: usize) -> f64 {
    if is_solid_cell(dims, solid, x, y) {
        0.0
    } else {
        f64::from(velocity[index_for_dims(dims, x, y)])
    }
}

fn sample_pressure(
    dims: [usize; 2],
    solid: &[bool],
    pressure: &[f32],
    current_index: usize,
    x: usize,
    y: usize,
) -> f64 {
    if is_solid_cell(dims, solid, x, y) {
        f64::from(pressure[current_index])
    } else {
        f64::from(pressure[index_for_dims(dims, x, y)])
    }
}

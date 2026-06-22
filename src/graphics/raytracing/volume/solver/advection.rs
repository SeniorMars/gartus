use super::{
    BoundaryCondition, apply_solid_scalar_boundaries, apply_solid_velocity_boundaries, finite_f32,
    index_for_dims, is_solid_cell, set_boundary, set_velocity_boundary, usize_to_f64,
};

pub(super) fn advect_scalar_2d(
    dims: [usize; 2],
    solid: &[bool],
    dt: f64,
    density: &mut [f32],
    previous_density: &[f32],
    velocity_x: &[f32],
    velocity_y: &[f32],
) {
    debug_assert_eq!(density.len(), previous_density.len());
    debug_assert_eq!(density.len(), velocity_x.len());
    debug_assert_eq!(density.len(), velocity_y.len());
    debug_assert_eq!(density.len(), solid.len());

    for y in 1..(dims[1] - 1) {
        for x in 1..(dims[0] - 1) {
            let index = index_for_dims(dims, x, y);
            if solid[index] {
                density[index] = 0.0;
                continue;
            }
            let back_x = usize_to_f64(x) - dt * f64::from(velocity_x[index]);
            let back_y = usize_to_f64(y) - dt * f64::from(velocity_y[index]);
            density[index] =
                finite_f32(sample_bilinear(dims, solid, previous_density, back_x, back_y).max(0.0));
        }
    }

    set_boundary(dims, density, BoundaryCondition::Scalar);
    apply_solid_scalar_boundaries(dims, solid, density);
}

pub(super) fn advect_velocity_2d(
    dims: [usize; 2],
    solid: &[bool],
    dt: f64,
    velocity_x: &mut [f32],
    velocity_y: &mut [f32],
    previous_velocity_x: &[f32],
    previous_velocity_y: &[f32],
) {
    debug_assert_eq!(velocity_x.len(), velocity_y.len());
    debug_assert_eq!(velocity_x.len(), previous_velocity_x.len());
    debug_assert_eq!(velocity_x.len(), previous_velocity_y.len());
    debug_assert_eq!(velocity_x.len(), solid.len());

    for y in 1..(dims[1] - 1) {
        for x in 1..(dims[0] - 1) {
            let index = index_for_dims(dims, x, y);
            if solid[index] {
                velocity_x[index] = 0.0;
                velocity_y[index] = 0.0;
                continue;
            }
            let back_x = usize_to_f64(x) - dt * f64::from(previous_velocity_x[index]);
            let back_y = usize_to_f64(y) - dt * f64::from(previous_velocity_y[index]);
            velocity_x[index] = finite_f32(sample_bilinear(
                dims,
                solid,
                previous_velocity_x,
                back_x,
                back_y,
            ));
            velocity_y[index] = finite_f32(sample_bilinear(
                dims,
                solid,
                previous_velocity_y,
                back_x,
                back_y,
            ));
        }
    }

    set_velocity_boundary(dims, velocity_x, velocity_y);
    apply_solid_velocity_boundaries(dims, solid, velocity_x, velocity_y);
}

fn sample_bilinear(dims: [usize; 2], solid: &[bool], field: &[f32], x: f64, y: f64) -> f64 {
    let x = x.clamp(0.0, usize_to_f64(dims[0] - 1));
    let y = y.clamp(0.0, usize_to_f64(dims[1] - 1));

    let x0 = floor_to_usize(x);
    let y0 = floor_to_usize(y);
    let x1 = (x0 + 1).min(dims[0] - 1);
    let y1 = (y0 + 1).min(dims[1] - 1);
    let tx = x - usize_to_f64(x0);
    let ty = y - usize_to_f64(y0);

    let c00 = sample_cell(dims, solid, field, x0, y0);
    let c10 = sample_cell(dims, solid, field, x1, y0);
    let c01 = sample_cell(dims, solid, field, x0, y1);
    let c11 = sample_cell(dims, solid, field, x1, y1);

    let bottom = c00.mul_add(1.0 - tx, c10 * tx);
    let top = c01.mul_add(1.0 - tx, c11 * tx);
    bottom.mul_add(1.0 - ty, top * ty)
}

fn sample_cell(dims: [usize; 2], solid: &[bool], field: &[f32], x: usize, y: usize) -> f64 {
    if is_solid_cell(dims, solid, x, y) {
        0.0
    } else {
        f64::from(field[index_for_dims(dims, x, y)])
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn floor_to_usize(value: f64) -> usize {
    value.floor() as usize
}

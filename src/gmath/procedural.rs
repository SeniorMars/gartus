//! Small deterministic helpers for procedural examples and generated scenes.

/// Full turn in radians.
pub const TAU: f64 = std::f64::consts::TAU;

/// Linearly interpolates from `start` to `end`.
#[must_use]
pub fn lerp(start: f64, end: f64, t: f64) -> f64 {
    start + (end - start) * t
}

/// Hermite smoothstep interpolation over `[edge0, edge1]`.
#[must_use]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = if (edge1 - edge0).abs() <= f64::EPSILON {
        0.0
    } else {
        ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

/// Quintic smootherstep interpolation over `[edge0, edge1]`.
#[must_use]
pub fn smootherstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = if (edge1 - edge0).abs() <= f64::EPSILON {
        0.0
    } else {
        ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    };
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Deterministically hashes one signed integer and a salt to a value in `[0, 1]`.
#[must_use]
pub fn hash01(index: i32, salt: u32) -> f64 {
    let mut x = u32::from_ne_bytes(index.to_ne_bytes());
    mix_hash(&mut x, salt.wrapping_mul(0x9e37_79b9));
    f64::from(x) / f64::from(u32::MAX)
}

/// Deterministically hashes two signed integers and a salt to a value in `[0, 1]`.
#[must_use]
pub fn hash01_2d(a: i32, b: i32, salt: u32) -> f64 {
    let mut x = u32::from_ne_bytes(a.to_ne_bytes());
    mix_hash(
        &mut x,
        u32::from_ne_bytes(b.to_ne_bytes()).wrapping_mul(0x9e37_79b9),
    );
    mix_hash(&mut x, salt.wrapping_mul(0x85eb_ca6b));
    f64::from(x) / f64::from(u32::MAX)
}

fn mix_hash(x: &mut u32, salt: u32) {
    *x ^= salt;
    *x = x.wrapping_mul(0x85eb_ca6b);
    *x ^= *x >> 16;
    *x = x.wrapping_mul(0xc2b2_ae35);
    *x ^= *x >> 16;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn procedural_hashes_are_stable_and_unit_interval() {
        assert_close(hash01(7, 11), hash01(7, 11));
        assert!((hash01(7, 11) - hash01(8, 11)).abs() > f64::EPSILON);
        assert!((0.0..=1.0).contains(&hash01_2d(-3, 5, 19)));
    }

    #[test]
    fn procedural_easing_clamps_to_range() {
        assert_close(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_close(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert_close(smootherstep(0.0, 1.0, -1.0), 0.0);
        assert_close(smootherstep(0.0, 1.0, 2.0), 1.0);
        assert_close(lerp(2.0, 4.0, 0.25), 2.5);
    }
}

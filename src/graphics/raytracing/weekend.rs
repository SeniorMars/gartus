//! "Ray Tracing in One Weekend" educational namespace.

pub use super::scenes::{
    checkered_spheres_world, cornell_box_world, cornell_smoke_world, dielectric_sphere_world,
    final_scene_bvh_world, final_scene_ray_scene, final_scene_sphere_list, final_scene_world,
    first_sphere_color, metal_sphere_world, next_week_final_scene_world, normal_scene_color,
    normal_sphere_world, perlin_spheres_world, quads_world, simple_light_world, sky_gradient,
    wide_angle_sphere_world,
};
pub use super::{
    WIDESCREEN_ASPECT_RATIO, degrees_to_radians, hit_sphere, hit_sphere_in_interval,
    linear_color_to_rgb, linear_to_gamma, rgb_bytes_to_unit_color, rgb_to_linear_color,
};

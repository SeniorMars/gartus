//! Contains common types that can be glob-imported (`*`) for convience
pub use crate::{
    gmath::{
        edge_matrix::EdgeMatrix,
        geometry::SphereGeometry,
        matrix::Matrix,
        polygon_matrix::{HeightMapOptions, PolygonMatrix},
        stack::MatrixStack,
        vector::{Point, Vector},
    },
    graphics::{
        animation::{AnimationRenderOptions, FrameRecorder},
        camera::{Camera3D, ProjectedSegment, ScreenPoint, sort_segments_back_to_front},
        colors::{ColorRamp, ColorSpace, Hsl, Hsv, Rgb},
        display::{Canvas, Domain2D, PolygonColorMode, ShadingMode},
        draw::TexturedVertex,
        lighting::{
            LightAttenuation, Lighting, PhongMaterial, PointLight, ReflectionConstants,
            RefractiveIndex,
        },
        texture::{Texture, TextureFilter, TextureWrap},
    },
    mdl::ast::VaryInterpolation,
};

#[cfg(feature = "fancy_math")]
pub use crate::{
    gmath::ray::Ray,
    graphics::camera::RayCamera,
    graphics::raytracing::{
        Dielectric, HitRecord, Hittable, HittableList, INFINITY, Interval, Lambertian, LinearColor,
        Material, MaterialRef, Metal, PI, SHADOW_ACNE_EPSILON, SampleRng, ScatterRecord, Sphere,
        WIDESCREEN_ASPECT_RATIO, degrees_to_radians, dielectric_sphere_world, final_scene_world,
        first_sphere_color, hit_sphere, linear_color_to_rgb, linear_to_gamma, metal_sphere_world,
        normal_scene_color, normal_sphere_world, render_defocus_sphere_scene,
        render_dielectric_sphere_scene, render_diffuse_sphere_scene, render_final_scene,
        render_final_scene_with_samples, render_first_sphere, render_metal_sphere_scene,
        render_normal_sphere_scene, render_unit_gradient, render_wide_angle_sphere_scene,
        rgb_to_linear_color, sky_gradient, wide_angle_sphere_world,
    },
};

#[cfg(feature = "external")]
pub use crate::external::{
    MaterialMesh, MaterialMeshGroup, MeshMaterial, MeshStats, MeshUpAxis, TexturedMeshTriangle,
    TexturedMeshVertex, normalize_material_mesh_transform,
};

#[cfg(feature = "turtle")]
pub use crate::graphics::turtle::{Turtle, TurtleState};

//! Contains common types that can be glob-imported (`*`) for convience
pub use crate::{
    gmath::{
        edge_matrix::EdgeMatrix,
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
        LinearColor, WIDESCREEN_ASPECT_RATIO, first_sphere_color, hit_sphere, render_first_sphere,
        render_unit_gradient, sky_gradient,
    },
};

#[cfg(feature = "external")]
pub use crate::external::{
    MaterialMesh, MaterialMeshGroup, MeshMaterial, MeshStats, MeshUpAxis, TexturedMeshTriangle,
    TexturedMeshVertex, normalize_material_mesh_transform,
};

#[cfg(feature = "turtle")]
pub use crate::graphics::turtle::{Turtle, TurtleState};

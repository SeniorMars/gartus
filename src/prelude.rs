//! Contains common types that can be glob-imported (`*`) for convience
pub use crate::{
    gmath::{
        edge_matrix::EdgeMatrix,
        geometry::{CameraBasis, CameraFrame, CameraPose, SphereGeometry, TriangleGeometry},
        matrix::Matrix,
        polygon_matrix::{Bounds3, HeightMapOptions, PolygonMatrix},
        random::SampleRng,
        stack::MatrixStack,
        vector::{Point, Vector},
    },
    graphics::{
        animation::{AnimationRenderOptions, FrameRecorder},
        camera::{Camera3D, ProjectedSegment, ScreenPoint, sort_segments_back_to_front},
        colors::{ColorRamp, ColorSpace, Hsl, Hsv, LinearRgb, Rgb},
        display::{Canvas, Domain2D, PolygonColorMode, ShadingMode},
        draw::TexturedVertex,
        lighting::{
            LightAttenuation, Lighting, PhongMaterial, PointLight, ReflectionConstants,
            RefractiveIndex, SurfaceMaterial,
        },
        texture::{Texture, TextureFilter, TextureWrap},
    },
    mdl::ast::VaryInterpolation,
};

pub use crate::{
    gmath::ray::Ray,
    graphics::camera::RayCamera,
    graphics::raytracing::{
        Aabb, BvhNode, Dielectric, HitRecord, Hittable, HittableList, INFINITY, Intersect,
        Interval, Lambertian, LinearColor, Material, MaterialId, MaterialRef, Metal, PI,
        PathTracer, RayGeometry, RayMaterial, RayPrimitive, RayScene, SHADOW_ACNE_EPSILON,
        ScatterRecord, SceneObject, Sphere, SphereList, SurfaceHit, TriangleMesh,
    },
};

#[cfg(feature = "external")]
pub use crate::external::{
    MaterialMesh, MaterialMeshGroup, MeshMaterial, MeshStats, MeshUpAxis, TexturedMeshTriangle,
    TexturedMeshVertex, normalize_material_mesh_transform,
};

#[cfg(feature = "turtle")]
pub use crate::graphics::turtle::{Turtle, TurtleState};

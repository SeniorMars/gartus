//! Contains common types that can be glob-imported (`*`) for convience
pub use crate::{
    gmath::{
        edge_matrix::EdgeMatrix,
        geometry::{
            CameraBasis, CameraFrame, CameraPose, MovingSphereGeometry, OrthonormalBasis,
            QuadGeometry, SphereGeometry, TriangleGeometry,
        },
        matrix::Matrix,
        perlin::{Perlin, scale_point},
        polygon_matrix::{Bounds3, HeightMapOptions, PolygonMatrix},
        random::SampleRng,
        stack::MatrixStack,
        vector::{Point, Vector},
    },
    graphics::{
        animation::{AnimationRenderOptions, FrameRecorder},
        camera::{
            Camera3D, PixelSampleMode, ProjectedSegment, ScreenPoint, sort_segments_back_to_front,
        },
        colors::{ColorRamp, ColorSpace, Hsl, Hsv, LinearRgb, Rgb},
        display::{Canvas, Domain2D, PolygonColorMode, ShadingMode},
        draw::TexturedVertex,
        lighting::{
            LightAttenuation, Lighting, PhongMaterial, PointLight, ReflectionConstants,
            RefractiveIndex,
        },
        material::SurfaceMaterial,
        scene::{SurfaceMesh, SurfaceScene},
        texture::{
            SurfaceTexture, SurfaceTextureRef, Texture, TextureFilter, TextureSample, TextureWrap,
        },
    },
    mdl::ast::VaryInterpolation,
};

pub use crate::{
    gmath::ray::Ray,
    graphics::camera::RayCamera,
    graphics::raytracing::{
        Dielectric, DiffuseLight, Lambertian, LinearColor, Metal, PathTracer, Quad, RayGeometry,
        RayMaterial, RayScene, SamplingTargetList, Sphere,
    },
};

#[cfg(feature = "external")]
pub use crate::external::{
    MaterialMesh, MaterialMeshGroup, MeshMaterial, MeshStats, MeshUpAxis, TexturedMeshTriangle,
    TexturedMeshVertex, normalize_material_mesh_transform,
};

#[cfg(feature = "turtle")]
pub use crate::graphics::turtle::{Turtle, TurtleState};

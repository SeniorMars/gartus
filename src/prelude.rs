//! Common types that can be glob-imported for examples and small programs.
//!
//! The root prelude exports renderer-neutral scene/material/texture types plus the high-level
//! path-tracing entry points. Low-level PDF internals intentionally stay out of the prelude; import
//! them from `gartus::graphics::raytracing::pdf` or `gartus::gmath::sampling` when you are working
//! on sampling code directly. For larger programs, prefer a narrower prelude such as
//! [`crate::prelude::math`], [`crate::prelude::raster`], [`crate::prelude::ray`], or
//! [`crate::prelude::mdl`].
//!
//! ```no_run
//! use gartus::prelude::*;
//!
//! let mut scene = SurfaceScene::new();
//! scene.clear();
//! let ray_scene = scene.to_ray_scene();
//! let image = PathTracer::new(RayCamera::new(100, 1.0)).render(&ray_scene);
//! assert_eq!(image.width(), 100);
//! ```
pub use crate::{
    gmath::{
        edge_matrix::EdgeMatrix,
        geometry::{
            CameraBasis, CameraFrame, CameraPose, MovingSphereGeometry, OrthonormalBasis,
            QuadGeometry, SphereGeometry, TriangleGeometry,
        },
        matrix::{Matrix, MatrixShapeError},
        perlin::{Perlin, scale_point},
        polygon_matrix::{Bounds3, HeightMapOptions, PolygonMatrix},
        procedural::{TAU, hash01, hash01_2d, lerp, smootherstep, smoothstep},
        random::SampleRng,
        stack::MatrixStack,
        vector::{Point, Vector},
    },
    graphics::{
        animation::{AnimationRenderOptions, FrameRecorder},
        camera::{
            AdaptiveSampling, Camera3D, PixelSampleMode, ProjectedSegment, ScreenPoint,
            sort_segments_back_to_front,
        },
        colors::{ColorRamp, ColorSpace, Hsl, Hsv, LinearRgb, Rgb},
        display::{Canvas, CanvasBuildError, Domain2D, PolygonColorMode, RgbImage, ShadingMode},
        draw::TexturedVertex,
        lighting::{
            LightAttenuation, Lighting, PhongMaterial, PointLight, ReflectionConstants,
            RefractiveIndex,
        },
        material::SurfaceMaterial,
        scene::{SurfaceMesh, SurfaceScene},
        texture::{
            SurfaceTexture, SurfaceTextureRef, Texture, TextureCache, TextureFilter, TextureSample,
            TextureWrap,
        },
    },
    mdl::ast::VaryInterpolation,
};

pub use crate::{
    gmath::ray::Ray,
    graphics::camera::RayCamera,
    graphics::raytracing::{
        ConstantDensity, ConstantMedium, DensityField, DensityFieldRef, Dielectric, DiffuseLight,
        DistanceField, DistanceFieldRef, FnDensityField, FnDistanceField, Hittable, HittableLayers,
        HittableList, Lambertian, LinearColor, MaterialRef, MatrixInstance, Metal,
        NonUniformMedium, PathTracer, Quad, RayGeometry, RayMaterial, RayScene, RaySceneBuilder,
        RenderOptions, RotateY, SamplingTargetList, SdfObject, Sphere, SurfaceRayMaterialMapper,
        SurfaceRayMaterialMode, Translate, TriangleMesh, WeightedSamplingTargetList, box_object,
    },
};

#[cfg(feature = "external")]
pub use crate::external::{
    MaterialMesh, MaterialMeshGroup, MeshMaterial, MeshStats, MeshUpAxis, TexturedMeshTriangle,
    TexturedMeshVertex, normalize_material_mesh_transform, normalize_mesh_transform,
    try_normalize_material_mesh_transform, try_normalize_mesh_transform,
};

#[cfg(feature = "turtle")]
pub use crate::graphics::turtle::{Turtle, TurtleState};

/// Math types commonly used by raster and ray renderers.
pub mod math {
    pub use super::{
        Bounds3, CameraBasis, CameraFrame, CameraPose, EdgeMatrix, HeightMapOptions, Matrix,
        MatrixShapeError, MatrixStack, MovingSphereGeometry, OrthonormalBasis, Perlin, Point,
        PolygonMatrix, QuadGeometry, Ray, SampleRng, SphereGeometry, TAU, TriangleGeometry, Vector,
        hash01, hash01_2d, lerp, scale_point, smootherstep, smoothstep,
    };
}

/// Raster drawing, camera projection, lighting, and renderer-neutral surface scene types.
pub mod raster {
    pub use super::{
        AnimationRenderOptions, Bounds3, Camera3D, Canvas, CanvasBuildError, ColorRamp, ColorSpace,
        Domain2D, EdgeMatrix, FrameRecorder, HeightMapOptions, Hsl, Hsv, Lighting, LinearRgb,
        Matrix, MatrixShapeError, MatrixStack, PhongMaterial, PixelSampleMode, Point, PointLight,
        PolygonColorMode, PolygonMatrix, ProjectedSegment, ReflectionConstants, RefractiveIndex,
        Rgb, RgbImage, ScreenPoint, ShadingMode, SurfaceMaterial, SurfaceMesh, SurfaceScene,
        SurfaceTexture, SurfaceTextureRef, Texture, TextureCache, TextureFilter, TextureSample,
        TextureWrap, TexturedVertex, Vector, sort_segments_back_to_front,
    };
}

/// Path-tracing cameras, materials, primitives, scenes, volumes, SDFs, and sampling targets.
pub mod ray {
    pub use super::{
        AdaptiveSampling, ConstantDensity, ConstantMedium, DensityField, DensityFieldRef,
        Dielectric, DiffuseLight, DistanceField, DistanceFieldRef, FnDensityField, FnDistanceField,
        Hittable, HittableLayers, HittableList, Lambertian, LinearColor, MaterialRef,
        MatrixInstance, Metal, NonUniformMedium, PathTracer, PixelSampleMode, Quad, Ray, RayCamera,
        RayGeometry, RayMaterial, RayScene, RaySceneBuilder, RenderOptions, RotateY, SampleRng,
        SamplingTargetList, SdfObject, Sphere, SurfaceRayMaterialMapper, SurfaceRayMaterialMode,
        Translate, TriangleMesh, WeightedSamplingTargetList, box_object,
    };
}

/// Motion Description Language front-end and runtime entry points.
pub mod mdl {
    pub use crate::mdl::{
        Command, CompiledProgram, Diagnostic, MdlError, Program, RenderConfig, compile_file,
        compile_source, parse_file, parse_source, run_file, run_file_streaming, run_source,
        run_source_streaming,
    };

    pub use super::VaryInterpolation;
}

/// External asset loader types and mesh normalization helpers.
#[cfg(feature = "external")]
pub mod external {
    pub use crate::external::{
        MaterialMesh, MaterialMeshGroup, MaterialMeshTriangle, MeshError, MeshMaterial, MeshStats,
        MeshUpAxis, TexturedMeshTriangle, TexturedMeshVertex, add_mesh, meshify,
        meshify_with_materials, normalize_material_mesh_transform, normalize_mesh_transform,
        ppmify, try_normalize_material_mesh_transform, try_normalize_mesh_transform,
    };
}

/// Turtle graphics types.
#[cfg(feature = "turtle")]
pub mod turtle {
    pub use super::{Turtle, TurtleState};
}

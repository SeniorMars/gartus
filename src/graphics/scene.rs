//! Renderer-neutral scene data shared by raster and ray renderers.

use crate::{
    gmath::{matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::{
        camera::Camera3D,
        colors::Rgb,
        display::{Canvas, PolygonColorMode, ShadingMode},
        lighting::Lighting,
        material::SurfaceMaterial,
    },
};

/// One polygon mesh paired with renderer-neutral material data.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMesh {
    /// Mesh triangles in world coordinates.
    pub polygons: PolygonMatrix,
    /// Surface material shared by raster and ray renderers.
    pub material: SurfaceMaterial,
}

impl SurfaceMesh {
    /// Creates a surface mesh from polygons and material data.
    #[must_use]
    pub fn new(polygons: PolygonMatrix, material: impl Into<SurfaceMaterial>) -> Self {
        Self {
            polygons,
            material: material.into(),
        }
    }
}

/// Shared surface scene description.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceScene {
    meshes: Vec<SurfaceMesh>,
}

impl SurfaceScene {
    /// Creates an empty surface scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty surface scene with reserved mesh capacity.
    #[must_use]
    pub fn with_capacity(meshes: usize) -> Self {
        Self {
            meshes: Vec::with_capacity(meshes),
        }
    }

    /// Adds a polygon mesh with shared material data.
    pub fn add_mesh(&mut self, polygons: PolygonMatrix, material: impl Into<SurfaceMaterial>) {
        self.meshes.push(SurfaceMesh::new(polygons, material));
    }

    /// Returns all scene meshes.
    #[must_use]
    pub fn meshes(&self) -> &[SurfaceMesh] {
        &self.meshes
    }

    /// Returns the number of meshes in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    /// Returns true if the scene contains no meshes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// Rasterizes this scene with the given camera using default canvas options.
    pub fn rasterize(&self, camera: &Camera3D) -> Canvas {
        self.rasterize_with_options(camera, Rgb::BLACK, None)
    }

    /// Rasterizes this scene with explicit background and optional lighting.
    ///
    /// Mesh vertices are projected through `camera`, then drawn as filled screen-space triangles.
    /// This keeps the scene description shared while preserving the existing canvas rasterizer.
    pub fn rasterize_with_options(
        &self,
        camera: &Camera3D,
        background: Rgb,
        lighting: Option<Lighting>,
    ) -> Canvas {
        let mut canvas = Canvas::builder(camera.width(), camera.height())
            .background(background)
            .upper_left_origin(true)
            .wrapped(false)
            .shading_mode(ShadingMode::Flat)
            .polygon_color_mode(if lighting.is_some() {
                PolygonColorMode::PhongReflection
            } else {
                PolygonColorMode::LineColor
            })
            .build();
        if let Some(lighting) = lighting {
            canvas.set_lighting(lighting);
        }

        for mesh in &self.meshes {
            let projected = project_mesh(camera, &mesh.polygons);
            if projected.is_empty() {
                continue;
            }
            canvas.line = mesh.material.base_color.gamma_encode();
            canvas.draw_polygons(&projected);
        }

        canvas
    }
}

fn project_mesh(camera: &Camera3D, mesh: &PolygonMatrix) -> PolygonMatrix {
    let identity = Matrix::identity_matrix(4);
    let mut projected = PolygonMatrix::with_capacity(mesh.cols());
    for (p0, p1, p2) in mesh.transformed_triangles(&identity) {
        let Some(p0) = camera.project(&p0) else {
            continue;
        };
        let Some(p1) = camera.project(&p1) else {
            continue;
        };
        let Some(p2) = camera.project(&p2) else {
            continue;
        };
        projected.add_polygon(
            (p0.x, p0.y, -p0.depth),
            (p1.x, p1.y, -p1.depth),
            (p2.x, p2.y, -p2.depth),
        );
    }
    projected
}

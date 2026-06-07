//! Renderer-neutral scene data shared by raster and ray renderers.

use std::borrow::Borrow;

use crate::gmath::{matrix::Matrix, polygon_matrix::PolygonMatrix};
use crate::graphics::{
    camera::Camera3D,
    colors::Rgb,
    display::{Canvas, PolygonColorMode, ShadingMode},
    lighting::{Lighting, PhongMaterial},
    material::SurfaceMaterial,
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

/// Shared surface scene description for raster and ray renderers.
///
/// This is the preferred user-facing scene container for mesh/material content that should render
/// through both pipelines. Use [`Self::rasterize`] for the canvas raster path, or pass a
/// `SurfaceScene` to [`crate::graphics::raytracing::PathTracer::render_scene`] for path tracing.
/// Rasterization maps [`SurfaceMaterial`] into Phong-style lighting data; path tracing currently
/// compiles surface materials to Lambertian ray materials unless you build a low-level
/// [`crate::graphics::raytracing::RayScene`] yourself. Diffuse texture paths stored on
/// [`SurfaceMaterial`] are metadata for lower-level textured raster and ray APIs; `SurfaceScene`
/// render helpers do not load or sample them.
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

    /// Applies `transform` to `polygons`, then adds the transformed mesh.
    ///
    /// This bakes the transform into copied triangle data. Use this for simple shared scene setup;
    /// a future instance layer can avoid copying large meshes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_mesh_transformed<T>(
        &mut self,
        polygons: PolygonMatrix,
        material: impl Into<SurfaceMaterial>,
        transform: T,
    ) where
        T: Borrow<Matrix>,
    {
        self.add_mesh(polygons.apply(transform.borrow()), material);
    }

    /// Removes all meshes.
    pub fn clear(&mut self) {
        self.meshes.clear();
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
    /// Mesh triangles are clipped against `camera`'s near plane, projected, then drawn as filled
    /// screen-space triangles. This keeps the scene description shared while preserving the
    /// existing canvas rasterizer.
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
        let lighting = lighting.inspect(|lighting| {
            canvas.set_lighting(lighting.clone());
        });

        for mesh in &self.meshes {
            let projected = project_mesh(camera, &mesh.polygons);
            if projected.is_empty() {
                continue;
            }
            canvas.set_line_color(mesh.material.base_color.gamma_encode());
            if let Some(lighting) = &lighting {
                let mut mesh_lighting = lighting.clone();
                mesh_lighting.set_material(PhongMaterial::from(&mesh.material));
                canvas.set_lighting(mesh_lighting);
            }
            canvas.draw_polygons(&projected);
        }

        canvas
    }
}

fn project_mesh(camera: &Camera3D, mesh: &PolygonMatrix) -> PolygonMatrix {
    let mut projected = PolygonMatrix::with_capacity(mesh.cols());
    for (p0, p1, p2) in mesh.triangles() {
        for [p0, p1, p2] in camera.project_clipped_triangle([p0, p1, p2]) {
            projected.add_polygon(
                (p0.x, p0.y, -p0.depth),
                (p1.x, p1.y, -p1.depth),
                (p2.x, p2.y, -p2.depth),
            );
        }
    }
    projected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmath::{matrix::Matrix, vector::Point};
    use crate::graphics::colors::LinearRgb;

    fn test_camera() -> Camera3D {
        Camera3D::new(100, 100)
            .with_look_at(Point::new(0.0, 0.0, 0.0), Point::new(0.0, 0.0, 1.0))
            .with_focal_length(10.0)
            .with_near_depth(1.0)
    }

    fn test_material(base_color: LinearRgb) -> SurfaceMaterial {
        SurfaceMaterial::new(LinearRgb::default(), base_color, LinearRgb::default(), 1.0)
    }

    fn assert_projected_depths_are_clipped(projected: &PolygonMatrix) {
        for point in projected.iter_points() {
            assert!(point[2] <= -1.0);
        }
    }

    #[test]
    fn project_mesh_keeps_triangle_with_one_vertex_behind_near_plane() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.5), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0));

        let projected = project_mesh(&test_camera(), &mesh);

        assert_eq!(projected.triangle_count(), 2);
        assert_projected_depths_are_clipped(&projected);
    }

    #[test]
    fn project_mesh_keeps_triangle_with_two_vertices_behind_near_plane() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (0.0, 1.0, 2.0));

        let projected = project_mesh(&test_camera(), &mesh);

        assert_eq!(projected.triangle_count(), 1);
        assert_projected_depths_are_clipped(&projected);
    }

    #[test]
    fn project_mesh_does_not_duplicate_vertex_on_near_plane() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 1.0), (1.0, 0.0, 0.5), (0.0, 1.0, 2.0));

        let projected = project_mesh(&test_camera(), &mesh);

        assert_eq!(projected.triangle_count(), 1);
        assert_projected_depths_are_clipped(&projected);
    }

    #[test]
    fn project_mesh_drops_triangle_behind_near_plane() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5));

        let projected = project_mesh(&test_camera(), &mesh);

        assert!(projected.is_empty());
    }

    #[test]
    fn rasterize_draws_front_and_near_plane_crossing_triangles() {
        let mut front = PolygonMatrix::new();
        front.add_polygon((-2.0, 1.0, 2.0), (-1.0, -1.0, 2.0), (-3.0, -1.0, 2.0));
        let mut crossing = PolygonMatrix::new();
        crossing.add_polygon((1.0, 1.0, 0.5), (2.0, -1.0, 2.0), (0.0, -1.0, 2.0));
        let mut scene = SurfaceScene::new();

        scene.add_mesh(front, test_material(LinearRgb::new(0.0, 1.0, 0.0)));
        scene.add_mesh(crossing, test_material(LinearRgb::new(1.0, 0.0, 0.0)));
        let canvas = scene.rasterize(&test_camera());

        assert!(canvas.pixels().contains(&Rgb::GREEN));
        assert!(canvas.pixels().contains(&Rgb::RED));
    }

    #[test]
    fn add_mesh_transformed_bakes_transform_into_scene_mesh() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        let mut scene = SurfaceScene::new();

        scene.add_mesh_transformed(
            mesh,
            SurfaceMaterial::default(),
            Matrix::translate(2.0, 3.0, 4.0),
        );

        let first = scene.meshes()[0].polygons.as_matrix().data()[0];
        assert_eq!(scene.len(), 1);
        assert!((first - 2.0).abs() < f64::EPSILON);
    }
}

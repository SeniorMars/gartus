//! Triangle mesh primitives and mesh-local acceleration.

use super::{
    Aabb, HitRecord, Hittable, Interval, Material, MaterialRef, MatrixInstance, SampleRng,
    SurfaceHit,
    bvh::{BvhBuildOptions, BvhPrimitiveInfo, FlatBvh, RayTraversal},
};
#[cfg(feature = "external")]
use super::{Lambertian, material::default_material, texture::ImageTexture};
use crate::gmath::{
    geometry::TriangleGeometry, matrix::Matrix, polygon_matrix::PolygonMatrix, ray::Ray,
    vector::Point, vector::Vector,
};
#[cfg(feature = "external")]
use crate::graphics::material::SurfaceMaterial;
use std::{fmt, sync::Arc};

/// One triangle in a ray-traced mesh, with optional imported shading metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTriangle {
    /// Triangle geometry.
    pub geometry: TriangleGeometry,
    /// Optional per-vertex texture coordinates.
    pub texcoords: Option<[(f64, f64); 3]>,
    /// Optional per-vertex smooth normals.
    pub vertex_normals: Option<[Vector; 3]>,
}

impl MeshTriangle {
    /// Creates a mesh triangle with only geometry data.
    #[must_use]
    pub const fn new(geometry: TriangleGeometry) -> Self {
        Self {
            geometry,
            texcoords: None,
            vertex_normals: None,
        }
    }

    /// Adds per-vertex texture coordinates.
    #[must_use]
    pub const fn with_texcoords(mut self, texcoords: [(f64, f64); 3]) -> Self {
        self.texcoords = Some(texcoords);
        self
    }

    /// Adds per-vertex shading normals.
    #[must_use]
    pub const fn with_vertex_normals(mut self, vertex_normals: [Vector; 3]) -> Self {
        self.vertex_normals = Some(vertex_normals);
        self
    }

    fn hit<'a>(
        self,
        ray: &Ray,
        ray_t: Interval,
        material: &'a dyn Material,
    ) -> Option<HitRecord<'a>> {
        let triangle_hit = self.geometry.hit_ray(ray, ray_t.min, ray_t.max)?;
        let geometric_normal = self.geometry.geometric_normal();
        if geometric_normal.length_squared() <= f64::EPSILON {
            return None;
        }

        let barycentric_u = triangle_hit.u;
        let barycentric_v = triangle_hit.v;
        let barycentric_w = 1.0 - barycentric_u - barycentric_v;
        let (surface_u, surface_v) =
            self.texcoords
                .map_or((barycentric_u, barycentric_v), |texcoords| {
                    (
                        barycentric_w * texcoords[0].0
                            + barycentric_u * texcoords[1].0
                            + barycentric_v * texcoords[2].0,
                        barycentric_w * texcoords[0].1
                            + barycentric_u * texcoords[1].1
                            + barycentric_v * texcoords[2].1,
                    )
                });

        let surface = SurfaceHit::with_uv(
            ray,
            ray.at(triangle_hit.t),
            geometric_normal,
            triangle_hit.t,
            surface_u,
            surface_v,
        );
        let mut record = HitRecord::from_surface(surface, material);
        if let Some(shading_normal) =
            self.shading_normal(barycentric_w, barycentric_u, barycentric_v)
        {
            record.set_shading_normal(shading_normal);
        }
        Some(record)
    }

    fn shading_normal(
        self,
        barycentric_w: f64,
        barycentric_u: f64,
        barycentric_v: f64,
    ) -> Option<Vector> {
        let normals = self.vertex_normals?;
        let normal = (barycentric_w * normals[0])
            + (barycentric_u * normals[1])
            + (barycentric_v * normals[2]);
        (normal.length_squared() > f64::EPSILON).then(|| normal.normalized())
    }
}

impl From<TriangleGeometry> for MeshTriangle {
    fn from(geometry: TriangleGeometry) -> Self {
        Self::new(geometry)
    }
}

/// Triangle mesh with a monomorphic internal BVH.
#[derive(Clone)]
pub struct TriangleMesh {
    triangles: Vec<MeshTriangle>,
    material: MaterialRef,
    bounds: Option<Aabb>,
    bvh: Option<TriangleBvh>,
}

impl fmt::Debug for TriangleMesh {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TriangleMesh")
            .field("triangles", &self.triangles.len())
            .field("bounds", &self.bounds)
            .field("has_bvh", &self.bvh.is_some())
            .finish_non_exhaustive()
    }
}

impl TriangleMesh {
    /// Creates a triangle mesh from geometry and a concrete material.
    #[must_use]
    pub fn new(triangles: Vec<TriangleGeometry>, material: impl Material + 'static) -> Self {
        Self::with_shared_material(triangles, Arc::new(material))
    }

    /// Creates a triangle mesh from geometry, material, and explicit BVH build options.
    #[must_use]
    pub fn new_with_bvh_options(
        triangles: Vec<TriangleGeometry>,
        material: impl Material + 'static,
        bvh_options: BvhBuildOptions,
    ) -> Self {
        Self::with_shared_material_and_bvh_options(triangles, Arc::new(material), bvh_options)
    }

    /// Creates a triangle mesh from geometry and a shared material handle.
    #[must_use]
    pub fn with_shared_material(triangles: Vec<TriangleGeometry>, material: MaterialRef) -> Self {
        Self::with_shared_material_and_bvh_options(triangles, material, BvhBuildOptions::default())
    }

    /// Creates a triangle mesh from geometry, shared material, and explicit BVH build options.
    #[must_use]
    pub fn with_shared_material_and_bvh_options(
        triangles: Vec<TriangleGeometry>,
        material: MaterialRef,
        bvh_options: BvhBuildOptions,
    ) -> Self {
        let triangles = triangles.into_iter().map(MeshTriangle::new).collect();
        Self::with_mesh_triangles_and_shared_material_and_bvh_options(
            triangles,
            material,
            bvh_options,
        )
    }

    /// Creates a triangle mesh from mesh triangles and a shared material handle.
    #[must_use]
    pub fn with_mesh_triangles_and_shared_material(
        triangles: Vec<MeshTriangle>,
        material: MaterialRef,
    ) -> Self {
        Self::with_mesh_triangles_and_shared_material_and_bvh_options(
            triangles,
            material,
            BvhBuildOptions::default(),
        )
    }

    /// Creates a triangle mesh from mesh triangles, material, and explicit BVH build options.
    #[must_use]
    pub fn with_mesh_triangles_and_shared_material_and_bvh_options(
        triangles: Vec<MeshTriangle>,
        material: MaterialRef,
        bvh_options: BvhBuildOptions,
    ) -> Self {
        let bounds = triangle_bounds_for_slice(&triangles);
        let bvh = TriangleBvh::build(&triangles, bvh_options);
        Self {
            triangles,
            material,
            bounds,
            bvh,
        }
    }

    /// Creates a triangle mesh from a polygon matrix and a concrete material.
    #[must_use]
    pub fn from_polygon_matrix(mesh: &PolygonMatrix, material: impl Material + 'static) -> Self {
        Self::from_shared_polygon_matrix(mesh, Arc::new(material))
    }

    /// Creates a triangle mesh from a polygon matrix, material, and explicit BVH build options.
    #[must_use]
    pub fn from_polygon_matrix_with_bvh_options(
        mesh: &PolygonMatrix,
        material: impl Material + 'static,
        bvh_options: BvhBuildOptions,
    ) -> Self {
        Self::from_shared_polygon_matrix_with_bvh_options(mesh, Arc::new(material), bvh_options)
    }

    /// Creates a triangle mesh from a polygon matrix and shared material handle.
    #[must_use]
    pub fn from_shared_polygon_matrix(mesh: &PolygonMatrix, material: MaterialRef) -> Self {
        Self::from_shared_polygon_matrix_with_bvh_options(
            mesh,
            material,
            BvhBuildOptions::default(),
        )
    }

    /// Creates a triangle mesh from a polygon matrix, material, and explicit BVH build options.
    #[must_use]
    pub fn from_shared_polygon_matrix_with_bvh_options(
        mesh: &PolygonMatrix,
        material: MaterialRef,
        bvh_options: BvhBuildOptions,
    ) -> Self {
        let triangles = mesh
            .iter_triangles()
            .map(|(p0, p1, p2)| {
                MeshTriangle::new(TriangleGeometry::new(
                    Point::new(p0[0], p0[1], p0[2]),
                    Point::new(p1[0], p1[1], p1[2]),
                    Point::new(p2[0], p2[1], p2[2]),
                ))
            })
            .collect();
        Self::with_mesh_triangles_and_shared_material_and_bvh_options(
            triangles,
            material,
            bvh_options,
        )
    }

    /// Creates a triangle mesh from one material mesh group and a shared material handle.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn from_material_mesh_group_with_shared_material(
        group: &crate::external::MaterialMeshGroup,
        material: MaterialRef,
    ) -> Self {
        let vertex_normals = group
            .normal_plan
            .normals_for_polygon_data(group.polygons.as_matrix().data());
        let triangles = group
            .triangles
            .iter()
            .enumerate()
            .map(|(index, triangle)| {
                let geometry = TriangleGeometry::new(
                    Point::new(
                        triangle.positions[0].0,
                        triangle.positions[0].1,
                        triangle.positions[0].2,
                    ),
                    Point::new(
                        triangle.positions[1].0,
                        triangle.positions[1].1,
                        triangle.positions[1].2,
                    ),
                    Point::new(
                        triangle.positions[2].0,
                        triangle.positions[2].1,
                        triangle.positions[2].2,
                    ),
                );
                let mut mesh_triangle = MeshTriangle::new(geometry);
                if let Some(texcoords) = triangle.texcoords {
                    mesh_triangle = mesh_triangle.with_texcoords(texcoords);
                }
                let normal_base = index * 3;
                if let Some(vertex_normals) = vertex_normals.as_ref()
                    && normal_base + 2 < vertex_normals.len()
                {
                    mesh_triangle = mesh_triangle.with_vertex_normals([
                        vertex_normals[normal_base],
                        vertex_normals[normal_base + 1],
                        vertex_normals[normal_base + 2],
                    ]);
                }
                mesh_triangle
            })
            .collect();
        Self::with_mesh_triangles_and_shared_material(triangles, material)
    }

    /// Creates one triangle mesh per material group using a caller-supplied material policy.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn from_material_mesh_with_policy<F>(
        mesh: &crate::external::MaterialMesh,
        mut policy: F,
    ) -> Vec<Self>
    where
        F: FnMut(&crate::external::MaterialMeshGroup) -> MaterialRef,
    {
        mesh.groups
            .iter()
            .filter(|group| !group.polygons.is_empty())
            .map(|group| Self::from_material_mesh_group_with_shared_material(group, policy(group)))
            .collect()
    }

    /// Creates one Lambertian triangle mesh per material group.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn from_material_mesh_lambertian(mesh: &crate::external::MaterialMesh) -> Vec<Self> {
        Self::from_material_mesh_with_policy(mesh, default_material_for_mesh_group)
    }

    /// Creates one Lambertian triangle mesh per material group, resolving `map_Kd` textures.
    ///
    /// Groups with loadable diffuse texture maps use [`ImageTexture`]. Other groups fall back to
    /// the material's diffuse/base color policy.
    ///
    /// # Errors
    ///
    /// Returns an error if a group has a diffuse texture path that cannot be loaded.
    #[cfg(feature = "external")]
    pub fn from_material_mesh_lambertian_textured(
        mesh: &crate::external::MaterialMesh,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        mesh.groups
            .iter()
            .filter(|group| !group.polygons.is_empty())
            .map(|group| {
                let material = material_for_mesh_group_with_texture(group)?;
                Ok(Self::from_material_mesh_group_with_shared_material(
                    group, material,
                ))
            })
            .collect()
    }

    /// Returns mesh triangles with imported metadata.
    #[must_use]
    pub fn triangles(&self) -> &[MeshTriangle] {
        &self.triangles
    }

    /// Returns the number of triangles in this mesh.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    /// Returns true if this mesh has no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Returns the material associated with this mesh.
    #[must_use]
    pub fn material(&self) -> &dyn Material {
        self.material.as_ref()
    }

    /// Brute-force hit path used for testing and diagnostics.
    #[must_use]
    pub fn hit_bruteforce(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        hit_triangle_range(
            &self.triangles,
            self.material(),
            0..self.triangles.len(),
            ray,
            ray_t,
        )
    }

    /// Returns the number of flat BVH nodes when this mesh has a built triangle BVH.
    #[must_use]
    pub fn bvh_node_count(&self) -> Option<usize> {
        self.bvh.as_ref().map(TriangleBvh::node_count)
    }

    /// Creates a matrix instance that shares this mesh through an [`Arc`].
    ///
    /// This is useful when placing many transformed copies of the same imported mesh without
    /// cloning its triangle storage, material handle, or mesh-local BVH.
    #[must_use]
    pub fn shared_instance(shared: Arc<Self>, transform: Matrix) -> Option<MatrixInstance> {
        MatrixInstance::new(shared, transform)
    }
}

#[cfg(feature = "external")]
fn default_material_for_mesh_group(group: &crate::external::MaterialMeshGroup) -> MaterialRef {
    if let Some(material) = group.material.clone() {
        Arc::new(Lambertian::from(SurfaceMaterial::from(material)))
    } else if let Some(diffuse_color) = group.diffuse_color {
        Arc::new(Lambertian::from(diffuse_color))
    } else {
        default_material()
    }
}

#[cfg(feature = "external")]
fn material_for_mesh_group_with_texture(
    group: &crate::external::MaterialMeshGroup,
) -> Result<MaterialRef, Box<dyn std::error::Error>> {
    if let Some(texture_path) = group
        .material
        .as_ref()
        .and_then(|material| material.diffuse_texture.as_ref())
    {
        return Ok(Arc::new(Lambertian::from_texture(ImageTexture::from_file(
            texture_path.to_string_lossy(),
        )?)));
    }

    Ok(default_material_for_mesh_group(group))
}

impl Hittable for TriangleMesh {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.bvh.as_ref().map_or_else(
            || self.hit_bruteforce(ray, ray_t),
            |bvh| bvh.hit(&self.triangles, self.material(), ray, ray_t),
        )
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.bounds
    }
}

#[derive(Clone, Debug)]
struct TriangleBvh {
    bvh: FlatBvh,
}

impl TriangleBvh {
    fn build(triangles: &[MeshTriangle], options: BvhBuildOptions) -> Option<Self> {
        let primitive_info = triangles
            .iter()
            .enumerate()
            .map(|(index, triangle)| BvhPrimitiveInfo::new(index, triangle.geometry.bounds()))
            .collect::<Vec<_>>();
        FlatBvh::build(&primitive_info, options).map(|bvh| Self { bvh })
    }

    fn node_count(&self) -> usize {
        self.bvh.node_count()
    }

    fn hit<'a>(
        &'a self,
        triangles: &'a [MeshTriangle],
        material: &'a dyn Material,
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        self.bvh
            .hit_with(ray_t, RayTraversal::new(ray), |indices, ray_t| {
                hit_triangle_indices(triangles, material, indices.iter().copied(), ray, ray_t)
            })
    }
}

fn triangle_bounds_for_slice(triangles: &[MeshTriangle]) -> Option<Aabb> {
    triangle_bounds_for_indices(triangles, 0..triangles.len())
}

fn triangle_bounds_for_indices(
    triangles: &[MeshTriangle],
    indices: impl IntoIterator<Item = usize>,
) -> Option<Aabb> {
    indices
        .into_iter()
        .map(|index| triangles[index].geometry.bounds())
        .reduce(Aabb::union)
}

fn hit_triangle_range<'a>(
    triangles: &'a [MeshTriangle],
    material: &'a dyn Material,
    range: std::ops::Range<usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    hit_triangle_indices(triangles, material, range, ray, ray_t)
}

fn hit_triangle_indices<'a>(
    triangles: &'a [MeshTriangle],
    material: &'a dyn Material,
    indices: impl IntoIterator<Item = usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        if let Some(record) =
            triangles[index].hit(ray, Interval::new(ray_t.min, closest_so_far), material)
        {
            closest_so_far = record.t;
            closest_hit = Some(record);
        }
    }

    closest_hit
}

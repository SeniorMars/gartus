//! Path-tracing scene containers and compatibility adapters.

use super::{
    Aabb, HitRecord, Hittable, Intersect, Interval, RayGeometry, RayMaterial, SampleRng, Sphere,
    bvh::{BvhPrimitiveInfo, FlatBvh, RayTraversal},
};
use crate::{
    gmath::{matrix::Matrix, ray::Ray, vector::Point, vector::Vector},
    graphics::scene::SurfaceScene,
};
use std::{fmt, sync::OnceLock};

/// Compatibility collection of boxed hittable scene objects.
///
/// [`RayScene`] is the primary path-tracing scene container for built-in geometry because it
/// stores compact primitives and material ids. Use `HittableList` when following the book steps,
/// mixing custom [`Hittable`] implementations, or building adapter scenes for examples/tests.
#[derive(Default)]
pub struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
    bounds: Option<Aabb>,
    has_unbounded: bool,
}

impl fmt::Debug for HittableList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HittableList")
            .field("len", &self.objects.len())
            .field("bounds", &self.bounds)
            .field("has_unbounded", &self.has_unbounded)
            .finish()
    }
}

impl HittableList {
    /// Creates an empty hittable list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty hittable list with space for at least `capacity` objects.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
            bounds: None,
            has_unbounded: false,
        }
    }

    /// Creates a hittable list containing one object.
    #[must_use]
    pub fn with_object(object: impl Hittable + 'static) -> Self {
        let mut list = Self::new();
        list.add(object);
        list
    }

    /// Removes all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.bounds = None;
        self.has_unbounded = false;
    }

    /// Adds an object to the scene.
    pub fn add(&mut self, object: impl Hittable + 'static) {
        self.add_box(Box::new(object));
    }

    /// Adds a boxed hittable object to the scene.
    pub fn add_box(&mut self, object: Box<dyn Hittable>) {
        if !self.has_unbounded {
            if let Some(object_bounds) = object.bounding_box() {
                self.bounds = Some(
                    self.bounds
                        .map_or(object_bounds, |bounds| bounds.surrounding(object_bounds)),
                );
            } else {
                self.bounds = None;
                self.has_unbounded = true;
            }
        }
        self.objects.push(object);
    }

    /// Builds a BVH from this list, returning `None` if any object lacks bounds.
    #[must_use]
    pub fn into_bvh(self) -> Option<BvhNode> {
        BvhNode::from_hittables(self.objects)
    }

    /// Returns the number of objects in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns true when the scene has no objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl Hittable for HittableList {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for object in &self.objects {
            if let Some(record) =
                object.hit_with_rng(ray, Interval::new(ray_t.min, closest_so_far), rng)
            {
                closest_so_far = record.t;
                closest_hit = Some(record);
            }
        }

        closest_hit
    }

    fn bounding_box(&self) -> Option<Aabb> {
        if self.has_unbounded {
            None
        } else {
            self.bounds
        }
    }
}

/// Bounding-volume hierarchy over arbitrary bounded boxed hittables.
///
/// This accelerates [`HittableList`] and custom-object scenes. Data-oriented scenes should prefer
/// [`RayScene`], whose primitive BVH is built from compact geometry/material tables.
pub struct BvhNode {
    objects: Vec<Box<dyn Hittable>>,
    bvh: ObjectBvh,
    bounds: Aabb,
}

impl fmt::Debug for BvhNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BvhNode")
            .field("objects", &self.objects.len())
            .field("bounds", &self.bounds)
            .field("nodes", &self.bvh.node_count())
            .finish_non_exhaustive()
    }
}

impl BvhNode {
    /// Builds a BVH from bounded hittable objects.
    #[must_use]
    pub fn from_hittables(objects: Vec<Box<dyn Hittable>>) -> Option<Self> {
        let bvh = ObjectBvh::build(&objects)?;
        let bounds = bvh.bounds();
        Some(Self {
            objects,
            bvh,
            bounds,
        })
    }

    /// Brute-force hit path used as a correctness oracle for the object BVH.
    #[must_use]
    pub fn hit_bruteforce(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut rng = SampleRng::default();
        hit_object_indices(&self.objects, 0..self.objects.len(), ray, ray_t, &mut rng)
    }
}

impl Hittable for BvhNode {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.bvh.hit(&self.objects, ray, ray_t, rng)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        Some(self.bounds)
    }
}

#[derive(Clone, Debug)]
struct ObjectBvh {
    bvh: FlatBvh,
}

impl ObjectBvh {
    const LEAF_SIZE: usize = 4;

    fn build(objects: &[Box<dyn Hittable>]) -> Option<Self> {
        let primitive_info = objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                object
                    .bounding_box()
                    .map(|bounds| BvhPrimitiveInfo::new(index, bounds))
            })
            .collect::<Option<Vec<_>>>()?;
        FlatBvh::build(&primitive_info, Self::LEAF_SIZE).map(|bvh| Self { bvh })
    }

    fn build_ray_primitives(primitives: &[RayPrimitive]) -> Option<Self> {
        let primitive_info = primitives
            .iter()
            .map(|primitive| primitive.geometry.bounding_box())
            .enumerate()
            .map(|(index, bounds)| bounds.map(|bounds| BvhPrimitiveInfo::new(index, bounds)))
            .collect::<Option<Vec<_>>>()?;
        FlatBvh::build(&primitive_info, Self::LEAF_SIZE).map(|bvh| Self { bvh })
    }

    fn bounds(&self) -> Aabb {
        self.bvh.bounds()
    }

    fn node_count(&self) -> usize {
        self.bvh.node_count()
    }

    fn hit<'a>(
        &'a self,
        objects: &'a [Box<dyn Hittable>],
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'a>> {
        self.bvh
            .hit_with(ray_t, RayTraversal::new(ray), |indices, ray_t| {
                hit_object_indices(objects, indices.iter().copied(), ray, ray_t, rng)
            })
    }

    fn hit_ray_scene<'a>(
        &'a self,
        primitives: &'a [RayPrimitive],
        materials: &'a [RayMaterial],
        ray: &Ray,
        ray_t: Interval,
    ) -> Option<HitRecord<'a>> {
        self.bvh
            .hit_with(ray_t, RayTraversal::new(ray), |indices, ray_t| {
                hit_ray_scene_indices(primitives, materials, indices.iter().copied(), ray, ray_t)
            })
    }
}

fn hit_object_indices<'a>(
    objects: &'a [Box<dyn Hittable>],
    indices: impl IntoIterator<Item = usize>,
    ray: &Ray,
    ray_t: Interval,
    rng: &mut SampleRng,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        if let Some(record) =
            objects[index].hit_with_rng(ray, Interval::new(ray_t.min, closest_so_far), rng)
        {
            closest_so_far = record.t;
            closest_hit = Some(record);
        }
    }

    closest_hit
}

fn hit_ray_scene_indices<'a>(
    primitives: &'a [RayPrimitive],
    materials: &'a [RayMaterial],
    indices: impl IntoIterator<Item = usize>,
    ray: &Ray,
    ray_t: Interval,
) -> Option<HitRecord<'a>> {
    let mut closest_so_far = ray_t.max;
    let mut closest_hit = None;

    for index in indices {
        let primitive = primitives[index];
        if let Some(surface) = primitive
            .geometry
            .intersect(ray, Interval::new(ray_t.min, closest_so_far))
        {
            closest_so_far = surface.t;
            closest_hit = Some(HitRecord::from_surface(
                surface,
                &materials[primitive.material],
            ));
        }
    }

    closest_hit
}

/// Index into a [`RayScene`] material table.
pub type MaterialId = usize;

/// One data-oriented scene primitive: compact geometry plus a material table index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayPrimitive {
    /// Primitive geometry.
    pub geometry: RayGeometry,
    /// Material table index.
    pub material: MaterialId,
}

/// Primary data-oriented path-tracing scene.
///
/// `RayScene` is the canonical internal scene for built-in path-traced primitives. It stores
/// compact [`RayPrimitive`] values plus a material table, and caches a scene-level BVH over those
/// primitives. [`HittableList`], [`BvhNode`], and [`SphereList`] remain useful compatibility or
/// educational adapters for boxed/custom hittables and book-style examples.
#[derive(Debug, Default)]
pub struct RayScene {
    materials: Vec<RayMaterial>,
    primitives: Vec<RayPrimitive>,
    bvh: OnceLock<Option<ObjectBvh>>,
}

impl Clone for RayScene {
    fn clone(&self) -> Self {
        let bvh = self
            .bvh
            .get()
            .map_or_else(OnceLock::new, |cached| OnceLock::from(cached.clone()));
        Self {
            materials: self.materials.clone(),
            primitives: self.primitives.clone(),
            bvh,
        }
    }
}

impl RayScene {
    /// Creates an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty scene with reserved material and primitive capacity.
    #[must_use]
    pub fn with_capacity(materials: usize, primitives: usize) -> Self {
        Self {
            materials: Vec::with_capacity(materials),
            primitives: Vec::with_capacity(primitives),
            bvh: OnceLock::new(),
        }
    }

    /// Adds a material and returns its table index.
    pub fn add_material(&mut self, material: impl Into<RayMaterial>) -> MaterialId {
        let id = self.materials.len();
        self.materials.push(material.into());
        id
    }

    /// Adds a primitive using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_primitive(&mut self, geometry: impl Into<RayGeometry>, material: MaterialId) {
        assert!(
            material < self.materials.len(),
            "ray scene material id out of bounds"
        );
        self.primitives.push(RayPrimitive {
            geometry: geometry.into(),
            material,
        });
        self.invalidate_bvh();
    }

    /// Adds multiple primitives and invalidates the cached BVH once.
    ///
    /// # Panics
    ///
    /// Panics if any primitive references an invalid material id for this scene.
    pub fn add_primitives<I>(&mut self, primitives: I)
    where
        I: IntoIterator<Item = RayPrimitive>,
    {
        let primitives = primitives.into_iter();
        let (lower_bound, _) = primitives.size_hint();
        self.primitives.reserve(lower_bound);

        for primitive in primitives {
            assert!(
                primitive.material < self.materials.len(),
                "ray scene material id out of bounds"
            );
            self.primitives.push(primitive);
        }

        self.invalidate_bvh();
    }

    /// Builds the primitive BVH and returns this scene.
    #[must_use]
    pub fn with_bvh(mut self) -> Self {
        self.build_bvh();
        self
    }

    /// Adds a sphere using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_sphere(&mut self, center: Point, radius: f64, material: MaterialId) {
        self.add_primitive(RayGeometry::sphere(center, radius), material);
    }

    /// Adds a moving sphere using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_moving_sphere(
        &mut self,
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: MaterialId,
    ) {
        self.add_primitive(
            RayGeometry::moving_sphere(center_start, center_end, radius),
            material,
        );
    }

    /// Adds a quad using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_quad(&mut self, corner: Point, u: Vector, v: Vector, material: MaterialId) {
        self.add_primitive(RayGeometry::quad(corner, u, v), material);
    }

    /// Adds a triangle using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_triangle(&mut self, p0: Point, p1: Point, p2: Point, material: MaterialId) {
        self.add_primitive(RayGeometry::triangle(p0, p1, p2), material);
    }

    /// Adds a material and a sphere that references it.
    pub fn add_sphere_with_material(
        &mut self,
        center: Point,
        radius: f64,
        material: impl Into<RayMaterial>,
    ) -> MaterialId {
        let material = self.add_material(material);
        self.add_sphere(center, radius, material);
        material
    }

    /// Adds a material and a moving sphere that references it.
    pub fn add_moving_sphere_with_material(
        &mut self,
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: impl Into<RayMaterial>,
    ) -> MaterialId {
        let material = self.add_material(material);
        self.add_moving_sphere(center_start, center_end, radius, material);
        material
    }

    /// Adds a material and a quad that references it.
    pub fn add_quad_with_material(
        &mut self,
        corner: Point,
        u: Vector,
        v: Vector,
        material: impl Into<RayMaterial>,
    ) -> MaterialId {
        let material = self.add_material(material);
        self.add_quad(corner, u, v, material);
        material
    }

    /// Returns a material by id.
    #[must_use]
    pub fn material(&self, id: MaterialId) -> Option<&RayMaterial> {
        self.materials.get(id)
    }

    /// Returns the material table.
    #[must_use]
    pub fn materials(&self) -> &[RayMaterial] {
        &self.materials
    }

    /// Returns scene primitives.
    #[must_use]
    pub fn primitives(&self) -> &[RayPrimitive] {
        &self.primitives
    }

    /// Returns true when this scene has a built primitive BVH.
    #[must_use]
    pub fn has_bvh(&self) -> bool {
        self.bvh.get().is_some_and(Option::is_some)
    }

    /// Returns the number of primitives in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Returns true when the scene has no primitives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Returns the number of stored materials.
    #[must_use]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Brute-force hit path used as a correctness oracle for the scene BVH.
    #[must_use]
    pub fn hit_bruteforce(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        hit_ray_scene_indices(
            &self.primitives,
            &self.materials,
            0..self.primitives.len(),
            ray,
            ray_t,
        )
    }

    /// Builds and caches the primitive BVH.
    ///
    /// Calling this after bulk scene construction avoids paying BVH build cost on the first ray.
    /// Empty scenes, or scenes containing unbounded primitives, cache a `None` BVH and use the
    /// linear fallback path.
    pub fn build_bvh(&mut self) {
        let bvh = ObjectBvh::build_ray_primitives(&self.primitives);
        self.bvh = OnceLock::from(bvh);
    }

    fn cached_bvh(&self) -> Option<&ObjectBvh> {
        self.bvh
            .get_or_init(|| ObjectBvh::build_ray_primitives(&self.primitives))
            .as_ref()
    }

    fn invalidate_bvh(&mut self) {
        self.bvh = OnceLock::new();
    }
}

impl From<&SurfaceScene> for RayScene {
    fn from(scene: &SurfaceScene) -> Self {
        let primitive_count = scene
            .meshes()
            .iter()
            .map(|mesh| mesh.polygons.triangle_count())
            .sum();
        let mut ray_scene = Self::with_capacity(scene.len(), primitive_count);
        let mut primitives = Vec::with_capacity(primitive_count);
        let identity = Matrix::identity_matrix(4);

        for mesh in scene.meshes() {
            let material = ray_scene.add_material(RayMaterial::from(mesh.material.as_lambertian()));
            for (p0, p1, p2) in mesh.polygons.transformed_triangles(&identity) {
                primitives.push(RayPrimitive {
                    geometry: RayGeometry::triangle(
                        Point::new(p0[0], p0[1], p0[2]),
                        Point::new(p1[0], p1[1], p1[2]),
                        Point::new(p2[0], p2[1], p2[2]),
                    ),
                    material,
                });
            }
        }

        ray_scene.add_primitives(primitives);
        ray_scene.build_bvh();
        ray_scene
    }
}

impl SurfaceScene {
    /// Converts this shared scene into a data-oriented ray scene.
    ///
    /// Surface materials are mapped to Lambertian ray materials. Use [`RayScene`] directly when a
    /// scene needs ray-specific material choices such as metal, dielectric, or emissive surfaces.
    #[must_use]
    pub fn to_ray_scene(&self) -> RayScene {
        self.into()
    }
}

impl Hittable for RayScene {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        self.cached_bvh().map_or_else(
            || {
                hit_ray_scene_indices(
                    &self.primitives,
                    &self.materials,
                    0..self.primitives.len(),
                    ray,
                    ray_t,
                )
            },
            |bvh| bvh.hit_ray_scene(&self.primitives, &self.materials, ray, ray_t),
        )
    }

    fn bounding_box(&self) -> Option<Aabb> {
        if let Some(bvh) = self.cached_bvh() {
            return Some(bvh.bounds());
        }

        let mut primitives = self.primitives.iter();
        let first = primitives.next()?.geometry.bounding_box()?;
        primitives.try_fold(first, |bounds, primitive| {
            primitive
                .geometry
                .bounding_box()
                .map(|other| bounds.surrounding(other))
        })
    }
}

/// Compatibility sphere-only hittable list that avoids boxed geometry dispatch in hit loops.
///
/// Prefer [`RayScene`] for new built-in path-traced scenes. `SphereList` remains as a small
/// specialized adapter for book-style random-sphere scenes and profiling comparisons.
#[derive(Clone, Debug, Default)]
pub struct SphereList {
    spheres: Vec<Sphere>,
}

impl SphereList {
    /// Creates an empty sphere list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty sphere list with space for `capacity` spheres.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            spheres: Vec::with_capacity(capacity),
        }
    }

    /// Adds a sphere.
    pub fn add(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    /// Returns the number of spheres.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spheres.len()
    }

    /// Returns true when there are no spheres.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spheres.is_empty()
    }
}

impl Hittable for SphereList {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for sphere in &self.spheres {
            if let Some(record) =
                sphere.hit_with_rng(ray, Interval::new(ray_t.min, closest_so_far), rng)
            {
                closest_so_far = record.t;
                closest_hit = Some(record);
            }
        }

        closest_hit
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let mut spheres = self.spheres.iter();
        let first = spheres.next()?.bounding_box()?;
        spheres.try_fold(first, |bounds, sphere| {
            sphere.bounding_box().map(|other| bounds.surrounding(other))
        })
    }
}

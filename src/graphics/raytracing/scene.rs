//! Path-tracing scene containers and compatibility adapters.
//!
//! The main scene tiers are:
//!
//! - [`SurfaceScene`]: renderer-neutral mesh/material data shared by raster and ray renderers.
//! - [`RayScene`]: compiled, data-oriented path-tracing scene with a cached BVH.
//! - [`HittableList`]: book-style boxed hittable collection for custom objects and examples.
//! - [`HittableLayers`]: borrowed scene layers for composing cached static and dynamic worlds.
//! - [`SamplingTargetList`]: importance-sampling targets such as lights, windows, or caustic
//!   objects.

use super::{
    Aabb, HitRecord, Hittable, Intersect, Interval, MovingSphere, PdfContext, Quad, RayGeometry,
    RayMaterial, SampleRng, Sphere,
    bvh::{BvhBuildOptions, BvhPrimitiveInfo, FlatBvh, RayTraversal},
};
use crate::{
    gmath::{
        geometry::{MovingSphereGeometry, QuadGeometry, SphereGeometry, TriangleGeometry},
        ray::Ray,
        vector::Point,
        vector::Vector,
    },
    graphics::{material::SurfaceMaterial, scene::SurfaceScene},
};
use std::{collections::HashMap, fmt, sync::OnceLock};

/// Custom [`SurfaceMaterial`] to [`RayMaterial`] mapper used by [`SurfaceRayMaterialMode::Custom`].
pub type SurfaceRayMaterialMapper<'a> = dyn Fn(&SurfaceMaterial) -> RayMaterial + Send + Sync + 'a;

/// Policy for converting renderer-neutral [`SurfaceMaterial`] values into ray materials.
#[derive(Default)]
pub enum SurfaceRayMaterialMode<'a> {
    /// Convert every surface material to Lambertian using its base color.
    #[default]
    Lambertian,
    /// Use dielectric conversion when the surface has a refractive index; otherwise Lambertian.
    PreferDielectric,
    /// Convert every surface material to metal using its specular color and the supplied fuzz.
    PreferMetal {
        /// Metal fuzz value used for every converted material.
        fuzz: f64,
    },
    /// Use a caller-supplied conversion function.
    Custom(Box<SurfaceRayMaterialMapper<'a>>),
}

impl fmt::Debug for SurfaceRayMaterialMode<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lambertian => formatter.write_str("Lambertian"),
            Self::PreferDielectric => formatter.write_str("PreferDielectric"),
            Self::PreferMetal { fuzz } => formatter
                .debug_struct("PreferMetal")
                .field("fuzz", fuzz)
                .finish(),
            Self::Custom(_) => formatter.write_str("Custom(..)"),
        }
    }
}

impl SurfaceRayMaterialMode<'_> {
    fn convert(&self, material: &SurfaceMaterial) -> RayMaterial {
        match self {
            Self::Lambertian => RayMaterial::from_surface_lambertian(material),
            Self::PreferDielectric => RayMaterial::from_surface_dielectric(material)
                .unwrap_or_else(|| RayMaterial::from_surface_lambertian(material)),
            Self::PreferMetal { fuzz } => RayMaterial::from_surface_metal(material, *fuzz),
            Self::Custom(convert) => convert(material),
        }
    }
}

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
        self.into_bvh_with_options(BvhBuildOptions::default())
    }

    /// Builds a BVH from this list using explicit build options.
    ///
    /// Returns `None` if any object lacks bounds.
    #[must_use]
    pub fn into_bvh_with_options(self, options: BvhBuildOptions) -> Option<BvhNode> {
        BvhNode::from_hittables_with_options(self.objects, options)
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

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        if self.objects.is_empty() {
            return 0.0;
        }

        let weight = reciprocal_count(self.objects.len());
        self.objects
            .iter()
            .map(|object| weight * object.pdf_value(context, direction))
            .sum()
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        rng.random_index(self.objects.len()).map_or_else(
            || Vector::new(1.0, 0.0, 0.0),
            |index| self.objects[index].random_direction(context, rng),
        )
    }
}

/// Borrowed hittable layers for composing prebuilt scene pieces.
///
/// This is useful for animated path-traced scenes where a large static BVH can be built once and a
/// smaller dynamic BVH can be rebuilt per frame. The layer container itself owns no geometry; it
/// only forwards hit and sampling queries to borrowed layers while preserving closest-hit pruning.
#[derive(Default)]
pub struct HittableLayers<'a> {
    layers: Vec<&'a dyn Hittable>,
    bounds: Option<Aabb>,
    has_unbounded: bool,
}

impl fmt::Debug for HittableLayers<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HittableLayers")
            .field("len", &self.layers.len())
            .field("bounds", &self.bounds)
            .field("has_unbounded", &self.has_unbounded)
            .finish()
    }
}

impl<'a> HittableLayers<'a> {
    /// Creates an empty layer set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty layer set with space for at least `capacity` layers.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            layers: Vec::with_capacity(capacity),
            bounds: None,
            has_unbounded: false,
        }
    }

    /// Adds a borrowed hittable layer.
    pub fn add(&mut self, layer: &'a dyn Hittable) {
        if !self.has_unbounded {
            if let Some(layer_bounds) = layer.bounding_box() {
                self.bounds = Some(
                    self.bounds
                        .map_or(layer_bounds, |bounds| bounds.surrounding(layer_bounds)),
                );
            } else {
                self.bounds = None;
                self.has_unbounded = true;
            }
        }
        self.layers.push(layer);
    }

    /// Removes all borrowed layers.
    pub fn clear(&mut self) {
        self.layers.clear();
        self.bounds = None;
        self.has_unbounded = false;
    }

    /// Returns the number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true when there are no layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Hittable for HittableLayers<'_> {
    fn hit_with_rng(
        &self,
        ray: &Ray,
        ray_t: Interval,
        rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        let mut closest_so_far = ray_t.max;
        let mut closest_hit = None;

        for layer in &self.layers {
            if let Some(record) =
                layer.hit_with_rng(ray, Interval::new(ray_t.min, closest_so_far), rng)
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

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }

        let weight = reciprocal_count(self.layers.len());
        self.layers
            .iter()
            .map(|layer| weight * layer.pdf_value(context, direction))
            .sum()
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        rng.random_index(self.layers.len()).map_or_else(
            || Vector::new(1.0, 0.0, 0.0),
            |index| self.layers[index].random_direction(context, rng),
        )
    }
}

/// Dedicated importance-sampling target list.
///
/// This is meant for lights, glass caustic targets, windows, or other geometry that should drive
/// path-sampling PDFs. Unlike [`HittableList`], it is not a scene container and does not report
/// ray intersections; use it only as the `lights` / sampling-target argument to path tracing.
///
/// Keep this list small and intentional. Passing the full world as a sampling target is unbiased
/// when every object's PDF methods are valid, but it usually wastes samples on non-emissive or
/// low-importance geometry.
#[derive(Default)]
pub struct SamplingTargetList {
    targets: WeightedSamplingTargetList,
}

impl fmt::Debug for SamplingTargetList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SamplingTargetList")
            .field("len", &self.len())
            .finish()
    }
}

impl SamplingTargetList {
    /// Creates an empty sampling target list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty sampling target list with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            targets: WeightedSamplingTargetList::with_capacity(capacity),
        }
    }

    /// Adds a sphere sampling target.
    pub fn add_sphere(&mut self, center: Point, radius: f64) {
        self.targets.add_sphere_weighted(center, radius, 1.0);
    }

    /// Adds a moving sphere sampling target.
    pub fn add_moving_sphere(&mut self, center_start: Point, center_end: Point, radius: f64) {
        self.targets
            .add_moving_sphere_weighted(center_start, center_end, radius, 1.0);
    }

    /// Adds a quad sampling target.
    pub fn add_quad(&mut self, corner: Point, u: Vector, v: Vector) {
        self.targets.add_quad_weighted(corner, u, v, 1.0);
    }

    /// Adds a custom sampling target.
    ///
    /// The object should implement meaningful [`Hittable::pdf_value`] and
    /// [`Hittable::random_direction`] methods.
    pub fn add_target(&mut self, object: impl Hittable + 'static) {
        self.targets.add_target_weighted(object, 1.0);
    }

    /// Removes all sampling targets.
    pub fn clear(&mut self) {
        self.targets.clear();
    }

    /// Returns the number of sampling targets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Returns true when there are no sampling targets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl Hittable for SamplingTargetList {
    fn hit_with_rng(
        &self,
        _ray: &Ray,
        _ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        None
    }

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        self.targets.pdf_value(context, direction)
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        self.targets.random_direction(context, rng)
    }
}

/// Importance-sampling target list with per-target selection weights.
///
/// This is useful when a scene mixes large area lights with many tiny lights. Each selected
/// target still evaluates its own geometric PDF; the list only changes how often each target is
/// chosen.
#[derive(Default)]
pub struct WeightedSamplingTargetList {
    objects: Vec<WeightedSamplingTarget>,
    total_weight: f64,
}

struct WeightedSamplingTarget {
    object: Box<dyn Hittable>,
    weight: f64,
}

impl fmt::Debug for WeightedSamplingTargetList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightedSamplingTargetList")
            .field("len", &self.objects.len())
            .field("total_weight", &self.total_weight)
            .finish()
    }
}

impl WeightedSamplingTargetList {
    /// Creates an empty weighted sampling target list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty weighted sampling target list with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
            total_weight: 0.0,
        }
    }

    /// Adds a sphere sampling target with a selection weight.
    ///
    /// # Panics
    /// Panics if `weight` is not finite and positive.
    pub fn add_sphere_weighted(&mut self, center: Point, radius: f64, weight: f64) {
        self.add_target_weighted(Sphere::new(center, radius), weight);
    }

    /// Adds a moving sphere sampling target with a selection weight.
    ///
    /// # Panics
    /// Panics if `weight` is not finite and positive.
    pub fn add_moving_sphere_weighted(
        &mut self,
        center_start: Point,
        center_end: Point,
        radius: f64,
        weight: f64,
    ) {
        self.add_target_weighted(MovingSphere::new(center_start, center_end, radius), weight);
    }

    /// Adds a quad sampling target with a selection weight.
    ///
    /// # Panics
    /// Panics if `weight` is not finite and positive.
    pub fn add_quad_weighted(&mut self, corner: Point, u: Vector, v: Vector, weight: f64) {
        self.add_target_weighted(Quad::new(corner, u, v), weight);
    }

    /// Adds a custom sampling target with a selection weight.
    ///
    /// # Panics
    /// Panics if `weight` is not finite and positive.
    pub fn add_target_weighted(&mut self, object: impl Hittable + 'static, weight: f64) {
        assert!(
            weight.is_finite() && weight > 0.0,
            "sampling target weight must be finite and positive"
        );
        let total_weight = self.total_weight + weight;
        assert!(
            total_weight.is_finite(),
            "sampling target total weight must remain finite"
        );
        self.total_weight = total_weight;
        self.objects.push(WeightedSamplingTarget {
            object: Box::new(object),
            weight,
        });
    }

    /// Removes all weighted sampling targets.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.total_weight = 0.0;
    }

    /// Returns the number of weighted sampling targets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns true when there are no weighted sampling targets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Returns the sum of all target weights.
    #[must_use]
    pub const fn total_weight(&self) -> f64 {
        self.total_weight
    }
}

impl Hittable for WeightedSamplingTargetList {
    fn hit_with_rng(
        &self,
        _ray: &Ray,
        _ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        None
    }

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        if self.objects.is_empty() || self.total_weight <= 0.0 {
            return 0.0;
        }

        self.objects
            .iter()
            .map(|target| {
                target.weight / self.total_weight * target.object.pdf_value(context, direction)
            })
            .sum()
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        if self.objects.is_empty() || self.total_weight <= 0.0 {
            return Vector::new(1.0, 0.0, 0.0);
        }

        let mut pick = rng.random_range(0.0, self.total_weight);
        for target in &self.objects {
            pick -= target.weight;
            if pick <= 0.0 {
                return target.object.random_direction(context, rng);
            }
        }
        self.objects[self.objects.len() - 1]
            .object
            .random_direction(context, rng)
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
        Self::from_hittables_with_options(objects, BvhBuildOptions::default())
    }

    /// Builds a BVH from bounded hittable objects using explicit build options.
    #[must_use]
    pub fn from_hittables_with_options(
        objects: Vec<Box<dyn Hittable>>,
        options: BvhBuildOptions,
    ) -> Option<Self> {
        let bvh = ObjectBvh::build(&objects, options)?;
        let bounds = bvh.bounds();
        Some(Self {
            objects,
            bvh,
            bounds,
        })
    }

    /// Returns the number of flat BVH nodes in this object hierarchy.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.bvh.node_count()
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

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        if self.objects.is_empty() {
            return 0.0;
        }

        let weight = reciprocal_count(self.objects.len());
        self.objects
            .iter()
            .map(|object| weight * object.pdf_value(context, direction))
            .sum()
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        rng.random_index(self.objects.len()).map_or_else(
            || Vector::new(1.0, 0.0, 0.0),
            |index| self.objects[index].random_direction(context, rng),
        )
    }
}

#[derive(Clone, Debug)]
struct ObjectBvh {
    bvh: FlatBvh,
}

impl ObjectBvh {
    fn build(objects: &[Box<dyn Hittable>], options: BvhBuildOptions) -> Option<Self> {
        let primitive_info = objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                object
                    .bounding_box()
                    .map(|bounds| BvhPrimitiveInfo::new(index, bounds))
            })
            .collect::<Option<Vec<_>>>()?;
        FlatBvh::build(&primitive_info, options).map(|bvh| Self { bvh })
    }

    fn build_ray_primitives(primitives: &[RayPrimitive], options: BvhBuildOptions) -> Option<Self> {
        let primitive_info = primitives
            .iter()
            .map(|primitive| primitive.geometry.bounding_box())
            .enumerate()
            .map(|(index, bounds)| bounds.map(|bounds| BvhPrimitiveInfo::new(index, bounds)))
            .collect::<Option<Vec<_>>>()?;
        FlatBvh::build(&primitive_info, options).map(|bvh| Self { bvh })
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct GeometrySamplingTarget {
    geometry: RayGeometry,
}

impl Hittable for GeometrySamplingTarget {
    fn hit_with_rng(
        &self,
        _ray: &Ray,
        _ray_t: Interval,
        _rng: &mut SampleRng,
    ) -> Option<HitRecord<'_>> {
        None
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.geometry.bounding_box()
    }

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        self.geometry.pdf_value(context, direction)
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        self.geometry.random_direction(context, rng)
    }
}

/// Ergonomic builder for [`RayScene`] that resolves primitives through named materials.
#[derive(Debug, Default)]
pub struct RaySceneBuilder {
    scene: RayScene,
    material_names: HashMap<String, MaterialId>,
}

impl RaySceneBuilder {
    /// Creates an empty scene builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty scene builder with reserved material and primitive capacity.
    #[must_use]
    pub fn with_capacity(materials: usize, primitives: usize) -> Self {
        Self {
            scene: RayScene::with_capacity(materials, primitives),
            material_names: HashMap::with_capacity(materials),
        }
    }

    /// Adds a named material.
    ///
    /// # Panics
    ///
    /// Panics if `name` has already been registered.
    #[must_use]
    pub fn material(mut self, name: impl Into<String>, material: impl Into<RayMaterial>) -> Self {
        let name = name.into();
        assert!(
            !self.material_names.contains_key(&name),
            "ray scene material name already exists"
        );
        let id = self.scene.add_material(material);
        self.material_names.insert(name, id);
        self
    }

    /// Adds a primitive with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn primitive(
        mut self,
        geometry: impl Into<RayGeometry>,
        material: impl AsRef<str>,
    ) -> Self {
        let material = self.material_id(material.as_ref());
        self.scene.add_primitive(geometry, material);
        self
    }

    /// Adds multiple geometry descriptors with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn geometries<I, G>(mut self, geometries: I, material: impl AsRef<str>) -> Self
    where
        I: IntoIterator<Item = G>,
        G: Into<RayGeometry>,
    {
        let material = self.material_id(material.as_ref());
        self.scene.add_geometries(material, geometries);
        self
    }

    /// Adds a sphere with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn sphere(mut self, center: Point, radius: f64, material: impl AsRef<str>) -> Self {
        let material = self.material_id(material.as_ref());
        self.scene.add_sphere(center, radius, material);
        self
    }

    /// Adds a moving sphere with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn moving_sphere(
        mut self,
        center_start: Point,
        center_end: Point,
        radius: f64,
        material: impl AsRef<str>,
    ) -> Self {
        let material = self.material_id(material.as_ref());
        self.scene
            .add_moving_sphere(center_start, center_end, radius, material);
        self
    }

    /// Adds a quad with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn quad(mut self, corner: Point, u: Vector, v: Vector, material: impl AsRef<str>) -> Self {
        let material = self.material_id(material.as_ref());
        self.scene.add_quad(corner, u, v, material);
        self
    }

    /// Adds a triangle with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn triangle(mut self, p0: Point, p1: Point, p2: Point, material: impl AsRef<str>) -> Self {
        let material = self.material_id(material.as_ref());
        self.scene.add_triangle(p0, p1, p2, material);
        self
    }

    /// Adds multiple triangles with a named material.
    ///
    /// # Panics
    ///
    /// Panics if `material` has not been registered.
    #[must_use]
    pub fn triangles<I>(mut self, triangles: I, material: impl AsRef<str>) -> Self
    where
        I: IntoIterator<Item = TriangleGeometry>,
    {
        let material = self.material_id(material.as_ref());
        self.scene.add_triangles(material, triangles);
        self
    }

    /// Finishes the builder without prebuilding the BVH.
    #[must_use]
    pub fn build(self) -> RayScene {
        self.scene
    }

    /// Finishes the builder after prebuilding the primitive BVH.
    #[must_use]
    pub fn build_bvh(mut self) -> RayScene {
        self.scene.build_bvh();
        self.scene
    }

    /// Finishes the builder after prebuilding the primitive BVH with explicit build options.
    #[must_use]
    pub fn build_bvh_with_options(mut self, options: BvhBuildOptions) -> RayScene {
        self.scene.build_bvh_with_options(options);
        self.scene
    }

    fn material_id(&self, name: &str) -> MaterialId {
        *self
            .material_names
            .get(name)
            .unwrap_or_else(|| panic!("unknown ray scene material name: {name}"))
    }
}

/// Primary data-oriented path-tracing scene.
///
/// `RayScene` is the canonical compiled scene for built-in path-traced primitives. It stores
/// compact [`RayPrimitive`] values plus a material table, and caches a scene-level BVH over those
/// primitives. General mesh/material application code can build a [`SurfaceScene`] and call its
/// [`to_ray_scene`](SurfaceScene::to_ray_scene) method once when repeated renders should reuse the compiled
/// primitive table and cached BVH. [`PathTracer::render_scene`](crate::graphics::raytracing::PathTracer::render_scene)
/// remains available as a one-shot convenience helper. Use `RayScene` directly for low-level
/// ray-specific materials, emissive primitives, and custom path-tracing scenes. [`HittableList`],
/// [`BvhNode`], and [`SphereList`] remain useful compatibility or educational adapters for
/// boxed/custom hittables and book-style examples.
///
/// For large procedural scenes, reserve capacity with [`Self::with_capacity`], insert geometry in
/// bulk with [`Self::add_geometries`] or the typed bulk helpers, and call [`Self::build_bvh`]
/// before rendering when you want to pay BVH construction cost up front.
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

    /// Returns a named-material builder for ergonomic scene construction.
    #[must_use]
    pub fn builder() -> RaySceneBuilder {
        RaySceneBuilder::new()
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

    /// Adds multiple geometry descriptors using one existing material table index.
    ///
    /// This is the ergonomic bulk path for procedural scenes. It reserves from the iterator size
    /// hint and invalidates the cached BVH once after all geometry has been appended.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_geometries<I, G>(&mut self, material: MaterialId, geometries: I)
    where
        I: IntoIterator<Item = G>,
        G: Into<RayGeometry>,
    {
        assert!(
            material < self.materials.len(),
            "ray scene material id out of bounds"
        );

        let geometries = geometries.into_iter();
        let (lower_bound, _) = geometries.size_hint();
        self.primitives.reserve(lower_bound);

        for geometry in geometries {
            self.primitives.push(RayPrimitive {
                geometry: geometry.into(),
                material,
            });
        }

        self.invalidate_bvh();
    }

    /// Builds the primitive BVH and returns this scene.
    #[must_use]
    pub fn with_bvh(mut self) -> Self {
        self.build_bvh();
        self
    }

    /// Builds the primitive BVH with explicit build options and returns this scene.
    #[must_use]
    pub fn with_bvh_options(mut self, options: BvhBuildOptions) -> Self {
        self.build_bvh_with_options(options);
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

    /// Adds multiple spheres using one existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_spheres<I>(&mut self, material: MaterialId, spheres: I)
    where
        I: IntoIterator<Item = SphereGeometry>,
    {
        self.add_geometries(material, spheres);
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

    /// Adds multiple moving spheres using one existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_moving_spheres<I>(&mut self, material: MaterialId, spheres: I)
    where
        I: IntoIterator<Item = MovingSphereGeometry>,
    {
        self.add_geometries(material, spheres);
    }

    /// Adds a quad using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_quad(&mut self, corner: Point, u: Vector, v: Vector, material: MaterialId) {
        self.add_primitive(RayGeometry::quad(corner, u, v), material);
    }

    /// Adds multiple quads using one existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_quads<I>(&mut self, material: MaterialId, quads: I)
    where
        I: IntoIterator<Item = QuadGeometry>,
    {
        self.add_geometries(material, quads);
    }

    /// Adds a triangle using an existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_triangle(&mut self, p0: Point, p1: Point, p2: Point, material: MaterialId) {
        self.add_primitive(RayGeometry::triangle(p0, p1, p2), material);
    }

    /// Adds multiple triangles using one existing material table index.
    ///
    /// # Panics
    ///
    /// Panics if `material` is not a valid material id for this scene.
    pub fn add_triangles<I>(&mut self, material: MaterialId, triangles: I)
    where
        I: IntoIterator<Item = TriangleGeometry>,
    {
        self.add_geometries(material, triangles);
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

    /// Builds an unweighted sampling target list from primitives whose material matches `include`.
    ///
    /// This is useful for deriving a compact light list from a mixed scene:
    ///
    /// ```
    /// # use gartus::prelude::*;
    /// # let scene = RayScene::new();
    /// let lights = scene.sampling_targets_by_material(RayMaterial::is_emissive);
    /// ```
    ///
    /// The returned list is independent of scene intersections and contains only the matched
    /// geometry PDF/random-direction behavior.
    #[must_use]
    pub fn sampling_targets_by_material<F>(&self, mut include: F) -> SamplingTargetList
    where
        F: FnMut(&RayMaterial) -> bool,
    {
        let mut targets = SamplingTargetList::with_capacity(self.primitives.len());
        for primitive in &self.primitives {
            if include(&self.materials[primitive.material]) {
                targets.add_target(GeometrySamplingTarget {
                    geometry: primitive.geometry,
                });
            }
        }
        targets
    }

    /// Builds a sampling target list containing primitives whose material is emissive.
    ///
    /// Pass this to [`PathTracer::render_with_lights`](crate::graphics::raytracing::PathTracer::render_with_lights)
    /// instead of using the whole scene as a light sampler.
    #[must_use]
    pub fn emissive_targets(&self) -> SamplingTargetList {
        self.sampling_targets_by_material(RayMaterial::is_emissive)
    }

    /// Returns true when this scene has a built primitive BVH.
    #[must_use]
    pub fn has_bvh(&self) -> bool {
        self.bvh.get().is_some_and(Option::is_some)
    }

    /// Returns the number of flat BVH nodes when this scene has a built primitive BVH.
    #[must_use]
    pub fn bvh_node_count(&self) -> Option<usize> {
        self.bvh
            .get()
            .and_then(|cached| cached.as_ref().map(ObjectBvh::node_count))
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

    /// Converts a renderer-neutral surface scene using an explicit material conversion policy.
    ///
    /// The returned scene has its primitive BVH built before return. Diffuse texture paths remain
    /// source-scene metadata unless the selected custom material mapper resolves them.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_surface_scene_with_material_mode(
        scene: &SurfaceScene,
        material_mode: SurfaceRayMaterialMode<'_>,
    ) -> Self {
        Self::from_surface_scene_with_material_mode_and_bvh_options(
            scene,
            material_mode,
            BvhBuildOptions::default(),
        )
    }

    /// Converts a renderer-neutral surface scene with explicit material and BVH build policies.
    ///
    /// The returned scene has its primitive BVH built before return. Diffuse texture paths remain
    /// source-scene metadata unless the selected custom material mapper resolves them.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_surface_scene_with_material_mode_and_bvh_options(
        scene: &SurfaceScene,
        material_mode: SurfaceRayMaterialMode<'_>,
        bvh_options: BvhBuildOptions,
    ) -> Self {
        let primitive_count = scene
            .meshes()
            .iter()
            .map(|mesh| mesh.polygons.triangle_count())
            .sum();
        let mut ray_scene = Self::with_capacity(scene.len(), primitive_count);
        let mut primitives = Vec::with_capacity(primitive_count);

        for mesh in scene.meshes() {
            let material = ray_scene.add_material(material_mode.convert(&mesh.material));
            for (p0, p1, p2) in mesh.polygons.triangles() {
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
        ray_scene.build_bvh_with_options(bvh_options);
        ray_scene
    }

    /// Builds and caches the primitive BVH.
    ///
    /// Calling this after bulk scene construction avoids paying BVH build cost on the first ray.
    /// Empty scenes, or scenes containing unbounded primitives, cache a `None` BVH and use the
    /// linear fallback path.
    pub fn build_bvh(&mut self) {
        self.build_bvh_with_options(BvhBuildOptions::default());
    }

    /// Builds and caches the primitive BVH using explicit build options.
    ///
    /// Empty scenes, or scenes containing unbounded primitives, cache a `None` BVH and use the
    /// linear fallback path.
    pub fn build_bvh_with_options(&mut self, options: BvhBuildOptions) {
        let bvh = ObjectBvh::build_ray_primitives(&self.primitives, options);
        self.bvh = OnceLock::from(bvh);
    }

    fn cached_bvh(&self) -> Option<&ObjectBvh> {
        self.bvh
            .get_or_init(|| {
                ObjectBvh::build_ray_primitives(&self.primitives, BvhBuildOptions::default())
            })
            .as_ref()
    }

    fn invalidate_bvh(&mut self) {
        self.bvh = OnceLock::new();
    }
}

impl From<&SurfaceScene> for RayScene {
    fn from(scene: &SurfaceScene) -> Self {
        Self::from_surface_scene_with_material_mode(scene, SurfaceRayMaterialMode::Lambertian)
    }
}

impl SurfaceScene {
    /// Converts this shared scene into a data-oriented ray scene.
    ///
    /// Surface materials are mapped to Lambertian ray materials using base colors, and the
    /// resulting [`RayScene`] has its BVH built before it is returned. Diffuse texture paths are
    /// retained only as source-scene metadata and are not loaded by this conversion.
    ///
    /// Use this method instead of [`PathTracer::render_scene`](crate::graphics::raytracing::PathTracer::render_scene)
    /// for repeated renders of the same surface content. Use [`RayScene`] directly when a scene
    /// needs ray-specific material choices such as textured, metal, dielectric, or emissive
    /// surfaces.
    #[must_use]
    pub fn to_ray_scene(&self) -> RayScene {
        self.into()
    }

    /// Converts this shared scene into a data-oriented ray scene with a material policy.
    ///
    /// Use this when surface materials should preserve ray-specific hints such as refractive
    /// indices or specular-metal conversion. The returned [`RayScene`] has its BVH built before it
    /// is returned.
    #[must_use]
    pub fn to_ray_scene_with_material_mode(
        &self,
        material_mode: SurfaceRayMaterialMode<'_>,
    ) -> RayScene {
        RayScene::from_surface_scene_with_material_mode(self, material_mode)
    }

    /// Converts this shared scene into a data-oriented ray scene with material and BVH policies.
    #[must_use]
    pub fn to_ray_scene_with_material_mode_and_bvh_options(
        &self,
        material_mode: SurfaceRayMaterialMode<'_>,
        bvh_options: BvhBuildOptions,
    ) -> RayScene {
        RayScene::from_surface_scene_with_material_mode_and_bvh_options(
            self,
            material_mode,
            bvh_options,
        )
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

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        if self.primitives.is_empty() {
            return 0.0;
        }

        let weight = reciprocal_count(self.primitives.len());
        self.primitives
            .iter()
            .map(|primitive| weight * primitive.geometry.pdf_value(context, direction))
            .sum()
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        rng.random_index(self.primitives.len()).map_or_else(
            || Vector::new(1.0, 0.0, 0.0),
            |index| {
                self.primitives[index]
                    .geometry
                    .random_direction(context, rng)
            },
        )
    }
}

/// Compatibility sphere-only hittable list that avoids boxed geometry dispatch in hit loops.
///
/// Prefer [`RayScene`] for new built-in path-traced scenes. `SphereList` remains as a small
/// specialized adapter for book-style random-sphere scenes and profiling comparisons.
#[doc(hidden)]
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

    fn pdf_value(&self, context: PdfContext, direction: Vector) -> f64 {
        if self.spheres.is_empty() {
            return 0.0;
        }

        let weight = reciprocal_count(self.spheres.len());
        self.spheres
            .iter()
            .map(|sphere| weight * sphere.pdf_value(context, direction))
            .sum()
    }

    fn random_direction(&self, context: PdfContext, rng: &mut SampleRng) -> Vector {
        rng.random_index(self.spheres.len()).map_or_else(
            || Vector::new(1.0, 0.0, 0.0),
            |index| self.spheres[index].random_direction(context, rng),
        )
    }
}

fn reciprocal_count(count: usize) -> f64 {
    1.0 / f64::from(u32::try_from(count).expect("scene object count should fit in u32"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmath::ray::Ray;

    #[test]
    fn ray_scene_build_bvh_with_options_uses_leaf_size() {
        let mut scene = RayScene::new();
        let material = scene.add_material(RayMaterial::lambertian(
            crate::graphics::raytracing::LinearColor::new(0.5, 0.5, 0.5),
        ));
        for index in 0..8 {
            let x = f64::from(u32::try_from(index).expect("test index fits u32")) * 3.0;
            scene.add_sphere(Point::new(x, 0.0, -5.0), 0.5, material);
        }

        let mut single_primitive_leaves = scene.clone();
        single_primitive_leaves.build_bvh_with_options(BvhBuildOptions::new().with_leaf_size(1));
        let mut all_primitives_in_one_leaf = scene;
        all_primitives_in_one_leaf.build_bvh_with_options(BvhBuildOptions::new().with_leaf_size(8));

        assert!(single_primitive_leaves.has_bvh());
        assert!(all_primitives_in_one_leaf.has_bvh());
        assert!(
            single_primitive_leaves.bvh_node_count() > all_primitives_in_one_leaf.bvh_node_count()
        );
        assert_eq!(all_primitives_in_one_leaf.bvh_node_count(), Some(1));
    }

    #[test]
    fn hittable_layers_return_nearest_hit_across_prebuilt_bvhs() {
        let mut far = HittableList::new();
        far.add(Sphere::new(Point::new(0.0, 0.0, -5.0), 0.5));
        let far = far.into_bvh().expect("far sphere should be bounded");

        let mut near = HittableList::new();
        near.add(Sphere::new(Point::new(0.0, 0.0, -2.0), 0.5));
        let near = near.into_bvh().expect("near sphere should be bounded");

        let mut layers = HittableLayers::with_capacity(2);
        layers.add(&far);
        layers.add(&near);

        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let hit = layers
            .hit(&ray, Interval::new(0.001, f64::INFINITY))
            .expect("ray should hit a layer");

        assert!((hit.t - 1.5).abs() < 1.0e-10);
        assert!(layers.bounding_box().is_some());
    }
}

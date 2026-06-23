use super::{Aabb, HitRecord, Interval};
use crate::gmath::{ray::Ray, vector::Point};

const BVH_BUCKETS: usize = 12;
const BVH_BUCKETS_F64: f64 = 12.0;
const DEFAULT_BVH_LEAF_SIZE: usize = 4;

/// Aggregate counters collected during flat BVH traversal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvhTraversalStats {
    /// Number of rays traced through the BVH.
    pub rays: u64,
    /// Number of node bounding boxes tested, including the root node.
    pub node_bounds_tests: u64,
    /// Number of node bounding-box tests that accepted the ray interval.
    pub node_bounds_hits: u64,
    /// Number of leaf nodes visited.
    pub leaf_visits: u64,
    /// Number of primitive indices handed to leaf intersection routines.
    pub primitive_candidates: u64,
    /// Number of leaf callbacks that returned a hit.
    pub leaf_hit_results: u64,
    /// Largest pending traversal stack depth observed for any ray.
    pub max_stack_depth: usize,
}

impl BvhTraversalStats {
    /// Adds another traversal sample into this aggregate.
    pub fn merge(&mut self, other: Self) {
        self.rays = self.rays.saturating_add(other.rays);
        self.node_bounds_tests = self
            .node_bounds_tests
            .saturating_add(other.node_bounds_tests);
        self.node_bounds_hits = self.node_bounds_hits.saturating_add(other.node_bounds_hits);
        self.leaf_visits = self.leaf_visits.saturating_add(other.leaf_visits);
        self.primitive_candidates = self
            .primitive_candidates
            .saturating_add(other.primitive_candidates);
        self.leaf_hit_results = self.leaf_hit_results.saturating_add(other.leaf_hit_results);
        self.max_stack_depth = self.max_stack_depth.max(other.max_stack_depth);
    }

    /// Returns true when no rays have contributed to this aggregate.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.rays == 0
    }

    fn record_stack_depth(&mut self, depth: usize) {
        self.max_stack_depth = self.max_stack_depth.max(depth);
    }
}

/// Build-time options for flat BVH acceleration structures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BvhBuildOptions {
    leaf_size: usize,
}

impl BvhBuildOptions {
    /// Creates options using the default BVH leaf size.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            leaf_size: DEFAULT_BVH_LEAF_SIZE,
        }
    }

    /// Sets the maximum number of primitives stored in a leaf node.
    ///
    /// A zero value is clamped to one.
    #[must_use]
    pub const fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = if leaf_size == 0 { 1 } else { leaf_size };
        self
    }

    /// Returns the configured maximum number of primitives per leaf.
    #[must_use]
    pub const fn leaf_size(self) -> usize {
        self.leaf_size
    }
}

impl Default for BvhBuildOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RayTraversal {
    origin: [f64; 3],
    inv_direction: [f64; 3],
    parallel: [bool; 3],
}

impl RayTraversal {
    pub(super) fn new(ray: &Ray) -> Self {
        let origin = [ray.origin().x(), ray.origin().y(), ray.origin().z()];
        let direction = [
            ray.direction().x(),
            ray.direction().y(),
            ray.direction().z(),
        ];
        let parallel = direction.map(|component| component.abs() <= f64::EPSILON);
        let inv_direction = [
            if parallel[0] { 0.0 } else { 1.0 / direction[0] },
            if parallel[1] { 0.0 } else { 1.0 / direction[1] },
            if parallel[2] { 0.0 } else { 1.0 / direction[2] },
        ];
        Self {
            origin,
            inv_direction,
            parallel,
        }
    }

    fn hit_bounds(self, bounds: Aabb, t_min: f64, t_max: f64) -> Option<f64> {
        let min = [bounds.min.0, bounds.min.1, bounds.min.2];
        let max = [bounds.max.0, bounds.max.1, bounds.max.2];
        let mut entry = t_min;
        let mut exit = t_max;

        for axis in 0..3 {
            if self.parallel[axis] {
                if self.origin[axis] < min[axis] || self.origin[axis] > max[axis] {
                    return None;
                }
                continue;
            }

            let mut t0 = (min[axis] - self.origin[axis]) * self.inv_direction[axis];
            let mut t1 = (max[axis] - self.origin[axis]) * self.inv_direction[axis];
            if self.inv_direction[axis] < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            entry = entry.max(t0);
            exit = exit.min(t1);
            if exit <= entry {
                return None;
            }
        }

        Some(entry)
    }
}

#[derive(Clone, Debug)]
pub(super) struct FlatBvh {
    nodes: Vec<FlatBvhNode>,
    indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct FlatBvhNode {
    bounds: Aabb,
    kind: FlatBvhNodeKind,
}

#[derive(Clone, Copy, Debug)]
enum FlatBvhNodeKind {
    Leaf { first: usize, count: usize },
    Internal { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug)]
struct StackEntry {
    node: usize,
    entry_t: f64,
}

#[derive(Debug)]
struct TraversalStack {
    stack: [StackEntry; 64],
    stack_len: usize,
    overflow: Vec<StackEntry>,
}

impl TraversalStack {
    fn new(root_entry: f64) -> Self {
        let mut stack = Self {
            stack: [StackEntry {
                node: 0,
                entry_t: 0.0,
            }; 64],
            stack_len: 0,
            overflow: Vec::new(),
        };
        stack.push(StackEntry {
            node: 0,
            entry_t: root_entry,
        });
        stack
    }

    fn push(&mut self, entry: StackEntry) {
        if self.overflow.is_empty() && self.stack_len < self.stack.len() {
            self.stack[self.stack_len] = entry;
            self.stack_len += 1;
        } else {
            self.overflow.push(entry);
        }
    }

    fn pop(&mut self) -> Option<StackEntry> {
        if let Some(entry) = self.overflow.pop() {
            Some(entry)
        } else if self.stack_len > 0 {
            self.stack_len -= 1;
            Some(self.stack[self.stack_len])
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.stack_len + self.overflow.len()
    }
}

pub(super) trait BvhHit {
    fn hit_t(&self) -> f64;
}

impl BvhHit for HitRecord<'_> {
    fn hit_t(&self) -> f64 {
        self.t
    }
}

impl FlatBvh {
    pub(super) fn build(
        primitive_info: &[BvhPrimitiveInfo],
        options: BvhBuildOptions,
    ) -> Option<Self> {
        if primitive_info.is_empty() {
            return None;
        }

        let mut bvh = Self {
            nodes: Vec::with_capacity(primitive_info.len().saturating_mul(2).saturating_sub(1)),
            indices: (0..primitive_info.len()).collect(),
        };
        bvh.build_range(primitive_info, options.leaf_size(), 0, primitive_info.len());
        Some(bvh)
    }

    pub(super) fn bounds(&self) -> Aabb {
        self.nodes[0].bounds
    }

    pub(super) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn build_range(
        &mut self,
        primitive_info: &[BvhPrimitiveInfo],
        leaf_size: usize,
        first: usize,
        count: usize,
    ) -> usize {
        let bounds = bounds_for_primitive_indices(
            primitive_info,
            self.indices[first..first + count].iter().copied(),
        )
        .expect("BVH node has at least one object");
        let node_index = self.nodes.len();
        self.nodes.push(FlatBvhNode {
            bounds,
            kind: FlatBvhNodeKind::Leaf { first, count },
        });

        if let Some(left_count) = split_bvh_indices(
            &mut self.indices[first..first + count],
            primitive_info,
            leaf_size,
        ) {
            let right_count = count - left_count;
            let left = self.build_range(primitive_info, leaf_size, first, left_count);
            let right =
                self.build_range(primitive_info, leaf_size, first + left_count, right_count);
            self.nodes[node_index].kind = FlatBvhNodeKind::Internal { left, right };
        }

        node_index
    }

    pub(super) fn hit_with<H, F>(
        &self,
        ray_t: Interval,
        traversal: RayTraversal,
        mut hit_leaf: F,
    ) -> Option<H>
    where
        H: BvhHit,
        F: FnMut(&[usize], Interval) -> Option<H>,
    {
        let mut stats = BvhTraversalStats::default();
        self.hit_with_stats(ray_t, traversal, &mut stats, |indices, ray_t| {
            hit_leaf(indices, ray_t)
        })
    }

    pub(super) fn hit_with_stats<H, F>(
        &self,
        ray_t: Interval,
        traversal: RayTraversal,
        stats: &mut BvhTraversalStats,
        mut hit_leaf: F,
    ) -> Option<H>
    where
        H: BvhHit,
        F: FnMut(&[usize], Interval) -> Option<H>,
    {
        stats.rays = stats.rays.saturating_add(1);
        stats.node_bounds_tests = stats.node_bounds_tests.saturating_add(1);
        let root_entry = traversal.hit_bounds(self.nodes[0].bounds, ray_t.min, ray_t.max)?;
        stats.node_bounds_hits = stats.node_bounds_hits.saturating_add(1);
        let mut stack = TraversalStack::new(root_entry);
        stats.record_stack_depth(stack.len());

        let mut closest = ray_t.max;
        let mut closest_hit = None;

        while let Some(entry) = stack.pop() {
            if entry.entry_t >= closest {
                continue;
            }

            match self.nodes[entry.node].kind {
                FlatBvhNodeKind::Leaf { first, count } => {
                    stats.leaf_visits = stats.leaf_visits.saturating_add(1);
                    stats.primitive_candidates = stats
                        .primitive_candidates
                        .saturating_add(usize_to_u64(count));
                    if let Some(hit) = hit_leaf(
                        &self.indices[first..first + count],
                        Interval::new(ray_t.min, closest),
                    ) {
                        stats.leaf_hit_results = stats.leaf_hit_results.saturating_add(1);
                        closest = hit.hit_t();
                        closest_hit = Some(hit);
                    }
                }
                FlatBvhNodeKind::Internal { left, right } => {
                    stats.node_bounds_tests = stats.node_bounds_tests.saturating_add(2);
                    let left_entry =
                        traversal.hit_bounds(self.nodes[left].bounds, ray_t.min, closest);
                    let right_entry =
                        traversal.hit_bounds(self.nodes[right].bounds, ray_t.min, closest);
                    stats.node_bounds_hits = stats
                        .node_bounds_hits
                        .saturating_add(u64::from(left_entry.is_some()))
                        .saturating_add(u64::from(right_entry.is_some()));

                    match (left_entry, right_entry) {
                        (Some(left_entry), Some(right_entry)) if right_entry < left_entry => {
                            stack.push(StackEntry {
                                node: left,
                                entry_t: left_entry,
                            });
                            stack.push(StackEntry {
                                node: right,
                                entry_t: right_entry,
                            });
                        }
                        (Some(left_entry), Some(right_entry)) => {
                            stack.push(StackEntry {
                                node: right,
                                entry_t: right_entry,
                            });
                            stack.push(StackEntry {
                                node: left,
                                entry_t: left_entry,
                            });
                        }
                        (Some(left_entry), None) => {
                            stack.push(StackEntry {
                                node: left,
                                entry_t: left_entry,
                            });
                        }
                        (None, Some(right_entry)) => {
                            stack.push(StackEntry {
                                node: right,
                                entry_t: right_entry,
                            });
                        }
                        (None, None) => {}
                    }
                    stats.record_stack_depth(stack.len());
                }
            }
        }

        closest_hit
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BvhPrimitiveInfo {
    bounds: Aabb,
    centroid: Point,
}

impl BvhPrimitiveInfo {
    pub(super) fn new(_index: usize, bounds: Aabb) -> Self {
        Self {
            bounds,
            centroid: bounds.centroid(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BvhBucket {
    count: usize,
    bounds: Option<Aabb>,
}

impl BvhBucket {
    fn add(&mut self, bounds: Aabb) {
        self.count += 1;
        self.bounds = Some(self.bounds.map_or(bounds, |current| current.union(bounds)));
    }
}

fn split_bvh_indices(
    indices: &mut [usize],
    primitive_info: &[BvhPrimitiveInfo],
    leaf_size: usize,
) -> Option<usize> {
    if indices.len() <= leaf_size {
        return None;
    }

    let centroid_bounds =
        centroid_bounds_for_primitive_indices(primitive_info, indices.iter().copied())
            .expect("BVH split range has centroid bounds");
    let axis = centroid_bounds.largest_axis();
    let centroid_extent = centroid_bounds.axis_max(axis) - centroid_bounds.axis_min(axis);
    if centroid_extent <= f64::EPSILON {
        return Some(midpoint_split(indices, primitive_info, axis));
    }

    let mut buckets = [BvhBucket::default(); BVH_BUCKETS];
    for &index in indices.iter() {
        let offset = (point_axis(primitive_info[index].centroid, axis)
            - centroid_bounds.axis_min(axis))
            / centroid_extent;
        let bucket = bucket_index(offset);
        buckets[bucket].add(primitive_info[index].bounds);
    }

    let mut best_split = 0;
    let mut best_cost = f64::INFINITY;
    for split in 0..BVH_BUCKETS - 1 {
        let (left_count, left_bounds) = merge_buckets(&buckets[..=split]);
        let (right_count, right_bounds) = merge_buckets(&buckets[split + 1..]);
        if left_count == 0 || right_count == 0 {
            continue;
        }

        let cost = left_bounds.expect("left bounds").surface_area() * count_as_f64(left_count)
            + right_bounds.expect("right bounds").surface_area() * count_as_f64(right_count);
        if cost < best_cost {
            best_cost = cost;
            best_split = split;
        }
    }

    if !best_cost.is_finite() {
        return Some(midpoint_split(indices, primitive_info, axis));
    }

    let min_axis = centroid_bounds.axis_min(axis);
    let mut left_count = 0;
    for next in 0..indices.len() {
        let index = indices[next];
        let offset =
            (point_axis(primitive_info[index].centroid, axis) - min_axis) / centroid_extent;
        let bucket = bucket_index(offset);
        if bucket <= best_split {
            indices.swap(left_count, next);
            left_count += 1;
        }
    }

    if left_count == 0 || left_count == indices.len() {
        Some(midpoint_split(indices, primitive_info, axis))
    } else {
        Some(left_count)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bucket_index(offset: f64) -> usize {
    ((offset * BVH_BUCKETS_F64) as usize).min(BVH_BUCKETS - 1)
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}

fn midpoint_split(
    indices: &mut [usize],
    primitive_info: &[BvhPrimitiveInfo],
    axis: usize,
) -> usize {
    indices.sort_by(|left, right| {
        let left_axis = point_axis(primitive_info[*left].centroid, axis);
        let right_axis = point_axis(primitive_info[*right].centroid, axis);
        left_axis
            .partial_cmp(&right_axis)
            .expect("BVH centroids should be finite")
            .then_with(|| left.cmp(right))
    });
    indices.len() / 2
}

fn merge_buckets(buckets: &[BvhBucket]) -> (usize, Option<Aabb>) {
    buckets.iter().fold((0, None), |(count, bounds), bucket| {
        let count = count + bucket.count;
        let bounds = match (bounds, bucket.bounds) {
            (Some(left), Some(right)) => Some(left.union(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        (count, bounds)
    })
}

fn bounds_for_primitive_indices(
    primitive_info: &[BvhPrimitiveInfo],
    indices: impl IntoIterator<Item = usize>,
) -> Option<Aabb> {
    indices
        .into_iter()
        .map(|index| primitive_info[index].bounds)
        .reduce(Aabb::union)
}

fn centroid_bounds_for_primitive_indices(
    primitive_info: &[BvhPrimitiveInfo],
    indices: impl IntoIterator<Item = usize>,
) -> Option<Aabb> {
    let mut centroids = indices
        .into_iter()
        .map(|index| primitive_info[index].centroid);
    let first = centroids.next()?;
    Some(centroids.fold(Aabb::from_points(first, first), Aabb::union_point))
}

fn point_axis(point: Point, axis: usize) -> f64 {
    match axis {
        0 => point.x(),
        1 => point.y(),
        2 => point.z(),
        _ => panic!("point axis index out of bounds"),
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmath::{ray::Ray, vector::Vector};

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TestHit {
        t: f64,
    }

    impl BvhHit for TestHit {
        fn hit_t(&self) -> f64 {
            self.t
        }
    }

    fn z_bounds(min_z: f64, max_z: f64) -> Aabb {
        Aabb::new((-1.0, -1.0, min_z), (1.0, 1.0, max_z))
    }

    fn unit_bounds() -> Aabb {
        Aabb::new((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    }

    #[test]
    fn bvh_build_options_clamps_zero_leaf_size() {
        assert_eq!(BvhBuildOptions::new().leaf_size(), 4);
        assert_eq!(BvhBuildOptions::new().with_leaf_size(0).leaf_size(), 1);
        assert_eq!(BvhBuildOptions::new().with_leaf_size(8).leaf_size(), 8);
    }

    fn bounds_entry(ray: Ray, t_min: f64, t_max: f64) -> Option<f64> {
        RayTraversal::new(&ray).hit_bounds(unit_bounds(), t_min, t_max)
    }

    #[test]
    fn traversal_stack_preserves_entries_past_inline_capacity() {
        let mut stack = TraversalStack::new(0.0);
        for node in 1..=70 {
            stack.push(StackEntry {
                node,
                entry_t: f64::from(u32::try_from(node).expect("test node index fits u32")),
            });
        }

        let mut popped = Vec::new();
        while let Some(entry) = stack.pop() {
            popped.push(entry.node);
        }

        assert_eq!(popped.len(), 71);
        assert_eq!(popped[0], 70);
        assert_eq!(popped[69], 1);
        assert_eq!(popped[70], 0);
    }

    #[test]
    fn flat_bvh_prunes_far_child_after_near_hit() {
        let left = z_bounds(1.0, 2.0);
        let right = z_bounds(10.0, 11.0);
        let bvh = FlatBvh {
            nodes: vec![
                FlatBvhNode {
                    bounds: left.union(right),
                    kind: FlatBvhNodeKind::Internal { left: 1, right: 2 },
                },
                FlatBvhNode {
                    bounds: left,
                    kind: FlatBvhNodeKind::Leaf { first: 0, count: 1 },
                },
                FlatBvhNode {
                    bounds: right,
                    kind: FlatBvhNodeKind::Leaf { first: 1, count: 1 },
                },
            ],
            indices: vec![0, 1],
        };
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, 1.0));
        let traversal = RayTraversal::new(&ray);
        let mut visited = Vec::new();

        let hit = bvh.hit_with(Interval::new(0.0, 20.0), traversal, |indices, _| {
            visited.extend_from_slice(indices);
            match indices[0] {
                0 => Some(TestHit { t: 1.5 }),
                1 => Some(TestHit { t: 10.5 }),
                _ => None,
            }
        });

        assert_eq!(hit, Some(TestHit { t: 1.5 }));
        assert_eq!(visited, vec![0]);
    }

    #[test]
    fn flat_bvh_reports_traversal_stats() {
        let left = z_bounds(1.0, 2.0);
        let right = z_bounds(10.0, 11.0);
        let bvh = FlatBvh {
            nodes: vec![
                FlatBvhNode {
                    bounds: left.union(right),
                    kind: FlatBvhNodeKind::Internal { left: 1, right: 2 },
                },
                FlatBvhNode {
                    bounds: left,
                    kind: FlatBvhNodeKind::Leaf { first: 0, count: 1 },
                },
                FlatBvhNode {
                    bounds: right,
                    kind: FlatBvhNodeKind::Leaf { first: 1, count: 1 },
                },
            ],
            indices: vec![0, 1],
        };
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, 1.0));
        let traversal = RayTraversal::new(&ray);
        let mut stats = BvhTraversalStats::default();

        let hit = bvh.hit_with_stats(
            Interval::new(0.0, 20.0),
            traversal,
            &mut stats,
            |indices, _| match indices[0] {
                0 => Some(TestHit { t: 1.5 }),
                1 => Some(TestHit { t: 10.5 }),
                _ => None,
            },
        );

        assert_eq!(hit, Some(TestHit { t: 1.5 }));
        assert_eq!(stats.rays, 1);
        assert_eq!(stats.node_bounds_tests, 3);
        assert_eq!(stats.node_bounds_hits, 3);
        assert_eq!(stats.leaf_visits, 1);
        assert_eq!(stats.primitive_candidates, 1);
        assert_eq!(stats.leaf_hit_results, 1);
        assert!(stats.max_stack_depth >= 1);
    }

    #[test]
    fn ray_traversal_handles_axis_parallel_rays_inside_bounds() {
        for direction in [
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
            Vector::new(-1.0, 0.0, 0.0),
            Vector::new(0.0, -1.0, 0.0),
            Vector::new(0.0, 0.0, -1.0),
        ] {
            let ray = Ray::new(Point::new(0.0, 0.0, 0.0), direction);

            assert_eq!(bounds_entry(ray, 0.0, 10.0), Some(0.0));
        }
    }

    #[test]
    fn ray_traversal_rejects_parallel_rays_outside_slab() {
        let cases = [
            (Point::new(0.0, 2.0, 0.0), Vector::new(1.0, 0.0, 0.0)),
            (Point::new(0.0, 0.0, 2.0), Vector::new(0.0, 1.0, 0.0)),
            (Point::new(2.0, 0.0, 0.0), Vector::new(0.0, 0.0, 1.0)),
        ];

        for (origin, direction) in cases {
            let ray = Ray::new(origin, direction);

            assert_eq!(bounds_entry(ray, 0.0, 10.0), None);
        }
    }

    #[test]
    fn ray_traversal_handles_rays_starting_inside_bounds() {
        let ray = Ray::new(Point::new(0.25, -0.25, 0.5), Vector::new(0.0, 0.0, 1.0));

        assert_eq!(bounds_entry(ray, 0.0, 10.0), Some(0.0));
    }

    #[test]
    fn ray_traversal_accepts_boundary_parallel_rays() {
        let ray = Ray::new(Point::new(1.0, -2.0, 0.0), Vector::new(0.0, 1.0, 0.0));

        assert_eq!(bounds_entry(ray, 0.0, 10.0), Some(1.0));
    }

    #[test]
    fn ray_traversal_rejects_single_point_boundary_touch() {
        let ray = Ray::new(Point::new(2.0, 2.0, 0.0), Vector::new(-1.0, -1.0, 1.0));

        assert_eq!(bounds_entry(ray, 0.0, 10.0), None);
    }

    #[test]
    fn ray_traversal_handles_zero_direction_components() {
        let inside = Ray::new(Point::new(0.0, 0.0, -2.0), Vector::new(0.0, 0.0, 1.0));
        let outside = Ray::new(Point::new(2.0, 0.0, -2.0), Vector::new(0.0, 0.0, 1.0));

        assert_eq!(bounds_entry(inside, 0.0, 10.0), Some(1.0));
        assert_eq!(bounds_entry(outside, 0.0, 10.0), None);
    }
}

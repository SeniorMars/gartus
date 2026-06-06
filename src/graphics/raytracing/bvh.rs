use super::{Aabb, HitRecord, Interval};
use crate::gmath::{ray::Ray, vector::Point};

const BVH_BUCKETS: usize = 12;
const BVH_BUCKETS_F64: f64 = 12.0;

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

pub(super) trait BvhHit {
    fn hit_t(&self) -> f64;
}

impl BvhHit for HitRecord<'_> {
    fn hit_t(&self) -> f64 {
        self.t
    }
}

impl FlatBvh {
    pub(super) fn build(primitive_info: &[BvhPrimitiveInfo], leaf_size: usize) -> Option<Self> {
        if primitive_info.is_empty() {
            return None;
        }

        let mut bvh = Self {
            nodes: Vec::with_capacity(primitive_info.len().saturating_mul(2).saturating_sub(1)),
            indices: (0..primitive_info.len()).collect(),
        };
        bvh.build_range(primitive_info, leaf_size, 0, primitive_info.len());
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
        self.hit_node_with(0, ray_t, traversal, &mut hit_leaf)
    }

    fn hit_node_with<H, F>(
        &self,
        node_index: usize,
        ray_t: Interval,
        traversal: RayTraversal,
        hit_leaf: &mut F,
    ) -> Option<H>
    where
        H: BvhHit,
        F: FnMut(&[usize], Interval) -> Option<H>,
    {
        let node = self.nodes[node_index];
        traversal.hit_bounds(node.bounds, ray_t.min, ray_t.max)?;
        self.hit_node_unchecked_with(node, ray_t, traversal, hit_leaf)
    }

    fn hit_node_unchecked_with<H, F>(
        &self,
        node: FlatBvhNode,
        ray_t: Interval,
        traversal: RayTraversal,
        hit_leaf: &mut F,
    ) -> Option<H>
    where
        H: BvhHit,
        F: FnMut(&[usize], Interval) -> Option<H>,
    {
        match node.kind {
            FlatBvhNodeKind::Leaf { first, count } => {
                hit_leaf(&self.indices[first..first + count], ray_t)
            }
            FlatBvhNodeKind::Internal { left, right } => {
                let left_entry =
                    traversal.hit_bounds(self.nodes[left].bounds, ray_t.min, ray_t.max);
                let right_entry =
                    traversal.hit_bounds(self.nodes[right].bounds, ray_t.min, ray_t.max);
                let (first, second, second_entry) = match (left_entry, right_entry) {
                    (Some(left_entry), Some(right_entry)) if right_entry < left_entry => {
                        (right, left, Some(left_entry))
                    }
                    (Some(_), Some(right_entry)) => (left, right, Some(right_entry)),
                    (Some(_), None) => (left, right, None),
                    (None, Some(_)) => (right, left, None),
                    (None, None) => return None,
                };
                let first_hit =
                    self.hit_node_unchecked_with(self.nodes[first], ray_t, traversal, hit_leaf);
                let closest = first_hit.as_ref().map_or(ray_t.max, BvhHit::hit_t);
                let second_hit = second_entry.filter(|entry| *entry < closest).and_then(|_| {
                    self.hit_node_unchecked_with(
                        self.nodes[second],
                        Interval::new(ray_t.min, closest),
                        traversal,
                        hit_leaf,
                    )
                });
                second_hit.or(first_hit)
            }
        }
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

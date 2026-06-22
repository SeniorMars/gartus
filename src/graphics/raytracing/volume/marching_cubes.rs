use super::{
    field::DensityField,
    grid::{GridBounds, GridDensityField},
    particles::{FluidParticle, ParticleSplatField, SplatKernel},
};
use crate::gmath::{
    geometry::TriangleGeometry,
    vector::{Point, Vector},
};

const TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
];

const TETRA_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const DEFAULT_LIQUID_RESOLUTION: [usize; 3] = [64, 64, 64];

/// Extracts iso-surfaces from voxel density grids.
///
/// This first implementation walks every grid cube and decomposes it into tetrahedra before
/// polygonizing the iso-surface. That keeps the public API aligned with marching-cubes workflows
/// while avoiding a large case table in the initial Stage 6A extractor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchingCubes {
    iso_value: f64,
}

impl MarchingCubes {
    /// Creates an extractor with iso value `0.5`.
    #[must_use]
    pub const fn new() -> Self {
        Self { iso_value: 0.5 }
    }

    /// Returns a copy with a different iso value.
    ///
    /// # Panics
    ///
    /// Panics if `iso_value` is not finite.
    #[must_use]
    pub fn with_iso_value(mut self, iso_value: f64) -> Self {
        assert!(
            iso_value.is_finite(),
            "marching cubes iso value must be finite"
        );
        self.iso_value = iso_value;
        self
    }

    /// Returns the active iso value.
    #[must_use]
    pub const fn iso_value(self) -> f64 {
        self.iso_value
    }

    /// Extracts an iso-surface from `grid`.
    #[must_use]
    pub fn extract(self, grid: &GridDensityField) -> ExtractedSurface {
        let dims = grid.dims();
        if dims.into_iter().any(|dim| dim < 2) {
            return ExtractedSurface::default();
        }

        let mut surface = ExtractedSurface::new();
        for z in 0..(dims[2] - 1) {
            for y in 0..(dims[1] - 1) {
                for x in 0..(dims[0] - 1) {
                    self.extract_cell(grid, [x, y, z], &mut surface);
                }
            }
        }
        surface
    }

    fn extract_cell(
        self,
        grid: &GridDensityField,
        cell: [usize; 3],
        surface: &mut ExtractedSurface,
    ) {
        let corners = cube_corners(grid, cell);
        for tetrahedron in TETRAHEDRA {
            self.extract_tetrahedron(grid, &corners, tetrahedron, surface);
        }
    }

    fn extract_tetrahedron(
        self,
        grid: &GridDensityField,
        corners: &[GridVertex; 8],
        tetrahedron: [usize; 4],
        surface: &mut ExtractedSurface,
    ) {
        let mut intersections = Vec::with_capacity(4);
        for (a, b) in TETRA_EDGES {
            let first = corners[tetrahedron[a]];
            let second = corners[tetrahedron[b]];
            if crosses_iso(first.value, second.value, self.iso_value) {
                intersections.push(interpolate_vertex(first, second, self.iso_value));
            }
        }

        match intersections.len() {
            3 => Self::push_oriented_triangle(
                grid,
                surface,
                intersections[0],
                intersections[1],
                intersections[2],
            ),
            4 => {
                Self::push_oriented_triangle(
                    grid,
                    surface,
                    intersections[0],
                    intersections[1],
                    intersections[2],
                );
                Self::push_oriented_triangle(
                    grid,
                    surface,
                    intersections[0],
                    intersections[2],
                    intersections[3],
                );
            }
            _ => {}
        }
    }

    fn push_oriented_triangle(
        grid: &GridDensityField,
        surface: &mut ExtractedSurface,
        p0: Point,
        p1: Point,
        p2: Point,
    ) {
        let triangle = TriangleGeometry::new(p0, p1, p2);
        if triangle.area_squared() <= f64::EPSILON {
            return;
        }

        let centroid = triangle.centroid();
        let outward = -density_gradient(grid, centroid);
        let oriented = if outward.length_squared() > f64::EPSILON
            && triangle.area_weighted_normal().dot(outward) < 0.0
        {
            TriangleGeometry::new(p0, p2, p1)
        } else {
            triangle
        };
        surface.push(oriented);
    }
}

impl Default for MarchingCubes {
    fn default() -> Self {
        Self::new()
    }
}

/// Triangle output from [`MarchingCubes`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtractedSurface {
    triangles: Vec<TriangleGeometry>,
    normals: Vec<Vector>,
}

impl ExtractedSurface {
    /// Creates an empty extracted surface.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            triangles: Vec::new(),
            normals: Vec::new(),
        }
    }

    /// Returns true when no triangles were extracted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Returns the number of extracted triangles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    /// Returns extracted triangles.
    #[must_use]
    pub fn triangles(&self) -> &[TriangleGeometry] {
        &self.triangles
    }

    /// Returns one flat normal per extracted triangle.
    #[must_use]
    pub fn normals(&self) -> &[Vector] {
        &self.normals
    }

    /// Consumes this surface and returns extracted triangles.
    #[must_use]
    pub fn into_triangles(self) -> Vec<TriangleGeometry> {
        self.triangles
    }

    fn push(&mut self, triangle: TriangleGeometry) {
        let normal = triangle.geometric_normal();
        if normal.length_squared() <= f64::EPSILON {
            return;
        }
        self.triangles.push(triangle);
        self.normals.push(normal);
    }
}

/// Convenience builder that turns particle splats into a liquid-like triangle surface.
#[derive(Clone, Debug)]
pub struct LiquidSurface {
    particles: Vec<FluidParticle>,
    resolution: [usize; 3],
    iso_value: f64,
    bounds: Option<GridBounds>,
    kernel: SplatKernel,
    max_density: Option<f64>,
    cell_size: Option<f64>,
}

impl LiquidSurface {
    /// Creates a liquid-surface builder from density particles.
    #[must_use]
    pub fn from_particles(particles: impl Into<Vec<FluidParticle>>) -> Self {
        Self {
            particles: particles.into(),
            resolution: DEFAULT_LIQUID_RESOLUTION,
            iso_value: 0.45,
            bounds: None,
            kernel: SplatKernel::Poly6,
            max_density: None,
            cell_size: None,
        }
    }

    /// Returns a copy with a different voxel resolution.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is smaller than two.
    #[must_use]
    pub fn with_resolution(mut self, resolution: [usize; 3]) -> Self {
        assert!(
            resolution.into_iter().all(|dim| dim >= 2),
            "liquid surface resolution must be at least 2 on every axis"
        );
        self.resolution = resolution;
        self
    }

    /// Returns a copy with a different iso value.
    ///
    /// # Panics
    ///
    /// Panics if `iso_value` is not finite.
    #[must_use]
    pub fn with_iso_value(mut self, iso_value: f64) -> Self {
        assert!(
            iso_value.is_finite(),
            "liquid surface iso value must be finite"
        );
        self.iso_value = iso_value;
        self
    }

    /// Returns a copy with explicit bake bounds.
    #[must_use]
    pub const fn with_bounds(mut self, bounds: GridBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Returns a copy with a different particle splat kernel.
    #[must_use]
    pub const fn with_kernel(mut self, kernel: SplatKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Returns a copy with a particle-density majorant.
    ///
    /// # Panics
    ///
    /// Panics if `max_density` is not positive and finite.
    #[must_use]
    pub fn with_max_density(mut self, max_density: f64) -> Self {
        assert!(
            max_density.is_finite() && max_density > 0.0,
            "liquid surface max density must be positive and finite"
        );
        self.max_density = Some(max_density);
        self
    }

    /// Returns a copy with a particle spatial-hash cell size.
    ///
    /// # Panics
    ///
    /// Panics if `cell_size` is not positive and finite.
    #[must_use]
    pub fn with_cell_size(mut self, cell_size: f64) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "liquid surface cell size must be positive and finite"
        );
        self.cell_size = Some(cell_size);
        self
    }

    /// Bakes particles into a density grid and extracts a surface.
    #[must_use]
    pub fn build_triangle_mesh(self) -> ExtractedSurface {
        let iso_value = self.iso_value;
        MarchingCubes::new()
            .with_iso_value(iso_value)
            .extract(&self.build_density_grid())
    }

    /// Bakes particles into a density grid without extracting geometry.
    #[must_use]
    pub fn build_density_grid(&self) -> GridDensityField {
        let mut splats = ParticleSplatField::new(self.particles.clone()).with_kernel(self.kernel);
        if let Some(max_density) = self.max_density {
            splats = splats.with_max_density(max_density);
        }
        if let Some(cell_size) = self.cell_size {
            splats = splats.with_cell_size(cell_size);
        }

        GridDensityField::from_density_field(
            self.bounds
                .unwrap_or_else(|| particle_bounds(&self.particles)),
            self.resolution,
            &splats,
            0.0,
        )
    }
}

#[derive(Clone, Copy)]
struct GridVertex {
    point: Point,
    value: f64,
}

fn cube_corners(grid: &GridDensityField, cell: [usize; 3]) -> [GridVertex; 8] {
    let x = cell[0];
    let y = cell[1];
    let z = cell[2];
    [
        grid_vertex(grid, x, y, z),
        grid_vertex(grid, x + 1, y, z),
        grid_vertex(grid, x + 1, y + 1, z),
        grid_vertex(grid, x, y + 1, z),
        grid_vertex(grid, x, y, z + 1),
        grid_vertex(grid, x + 1, y, z + 1),
        grid_vertex(grid, x + 1, y + 1, z + 1),
        grid_vertex(grid, x, y + 1, z + 1),
    ]
}

fn grid_vertex(grid: &GridDensityField, x: usize, y: usize, z: usize) -> GridVertex {
    GridVertex {
        point: grid.cell_center(x, y, z),
        value: f64::from(grid.densities()[grid.index(x, y, z)]),
    }
}

fn crosses_iso(first: f64, second: f64, iso_value: f64) -> bool {
    (first < iso_value && second >= iso_value) || (second < iso_value && first >= iso_value)
}

fn interpolate_vertex(first: GridVertex, second: GridVertex, iso_value: f64) -> Point {
    let denom = second.value - first.value;
    let t = if denom.abs() <= f64::EPSILON {
        0.5
    } else {
        ((iso_value - first.value) / denom).clamp(0.0, 1.0)
    };
    first.point + t * (second.point - first.point)
}

fn density_gradient(grid: &GridDensityField, point: Point) -> Vector {
    let bounds = grid.bounds();
    let extent = bounds.extent();
    let dims = grid.dims();
    let hx = gradient_step(extent.x(), dims[0]);
    let hy = gradient_step(extent.y(), dims[1]);
    let hz = gradient_step(extent.z(), dims[2]);
    Vector::new(
        grid.density(point + Vector::new(hx, 0.0, 0.0), 0.0)
            - grid.density(point - Vector::new(hx, 0.0, 0.0), 0.0),
        grid.density(point + Vector::new(0.0, hy, 0.0), 0.0)
            - grid.density(point - Vector::new(0.0, hy, 0.0), 0.0),
        grid.density(point + Vector::new(0.0, 0.0, hz), 0.0)
            - grid.density(point - Vector::new(0.0, 0.0, hz), 0.0),
    )
}

fn gradient_step(extent: f64, dim: usize) -> f64 {
    extent / f64::from(u32::try_from(dim.max(2)).expect("grid dimension should fit in u32"))
}

fn particle_bounds(particles: &[FluidParticle]) -> GridBounds {
    let Some(first) = particles.first() else {
        return GridBounds::new(Point::new(-1.0, -1.0, -1.0), Point::new(1.0, 1.0, 1.0));
    };

    let mut min = first.position - Vector::new(first.radius, first.radius, first.radius);
    let mut max = first.position + Vector::new(first.radius, first.radius, first.radius);
    for particle in particles.iter().skip(1) {
        let radius = Vector::new(particle.radius, particle.radius, particle.radius);
        let pmin = particle.position - radius;
        let pmax = particle.position + radius;
        min = Point::new(
            min.x().min(pmin.x()),
            min.y().min(pmin.y()),
            min.z().min(pmin.z()),
        );
        max = Point::new(
            max.x().max(pmax.x()),
            max.y().max(pmax.y()),
            max.z().max(pmax.z()),
        );
    }

    let extent = max - min;
    let pad = Vector::new(
        (0.05 * extent.x()).max(0.05),
        (0.05 * extent.y()).max(0.05),
        (0.05 * extent.z()).max(0.05),
    );
    GridBounds::new(min - pad, max + pad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_grid() -> GridDensityField {
        let bounds = GridBounds::new(Point::new(-1.5, -1.5, -1.5), Point::new(1.5, 1.5, 1.5));
        GridDensityField::from_fn(bounds, [18, 18, 18], |point| {
            let radius = (point - Point::new(0.0, 0.0, 0.0)).length();
            (1.0 - radius / 1.2).clamp(0.0, 1.0)
        })
    }

    #[test]
    fn marching_cubes_empty_grid_outputs_no_triangles() {
        let grid = GridDensityField::new(
            GridBounds::new(Point::new(0.0, 0.0, 0.0), Point::new(1.0, 1.0, 1.0)),
            [3, 3, 3],
            vec![0.0; 27],
        );

        assert!(
            MarchingCubes::new()
                .with_iso_value(0.5)
                .extract(&grid)
                .is_empty()
        );
    }

    #[test]
    fn marching_cubes_full_grid_outputs_no_boundary_triangles_by_default() {
        let grid = GridDensityField::new(
            GridBounds::new(Point::new(0.0, 0.0, 0.0), Point::new(1.0, 1.0, 1.0)),
            [3, 3, 3],
            vec![1.0; 27],
        );

        assert!(
            MarchingCubes::new()
                .with_iso_value(0.5)
                .extract(&grid)
                .is_empty()
        );
    }

    #[test]
    fn marching_cubes_sphere_field_outputs_non_empty_mesh() {
        let surface = MarchingCubes::new()
            .with_iso_value(0.5)
            .extract(&sphere_grid());

        assert!(!surface.is_empty());
        assert_eq!(surface.triangles().len(), surface.normals().len());
        assert!(
            surface
                .normals()
                .iter()
                .all(|normal| normal.length_squared() > 0.0)
        );
    }

    #[test]
    fn marching_cubes_vertices_are_inside_grid_bounds() {
        let grid = sphere_grid();
        let surface = MarchingCubes::new().with_iso_value(0.5).extract(&grid);

        assert!(surface.triangles().iter().all(|triangle| {
            triangle
                .vertices()
                .into_iter()
                .all(|vertex| grid.bounds().contains(vertex))
        }));
    }

    #[test]
    fn liquid_surface_from_particles_builds_mesh() {
        let particles = vec![
            FluidParticle::new(Point::new(-0.25, 0.0, 0.0), 0.5, 1.0),
            FluidParticle::new(Point::new(0.25, 0.0, 0.0), 0.5, 1.0),
        ];

        let surface = LiquidSurface::from_particles(particles)
            .with_resolution([12, 12, 12])
            .with_iso_value(0.35)
            .build_triangle_mesh();

        assert!(!surface.is_empty());
    }
}

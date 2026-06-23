use std::{
    collections::HashMap,
    error::Error,
    fmt,
    fs::File,
    io::{BufRead, BufReader, Read, Seek},
    path::{Path, PathBuf},
};

use crate::gmath::{
    matrix::Matrix,
    polygon_matrix::{Bounds3, PolygonMatrix},
};
use crate::graphics::{
    colors::{LinearRgb, Rgb},
    draw::VertexNormalPlan,
    lighting::{RefractiveIndex, SurfaceMaterial},
};

type MeshResult<T> = Result<T, MeshError>;
type Point3 = (f64, f64, f64);
type TexCoord = (f64, f64);
type Triangle = [Point3; 3];

const MESH_TRIANGLE_BATCH: usize = 4096;

/// One mesh vertex with an OBJ texture coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TexturedMeshVertex {
    /// Vertex position in object space.
    pub position: (f64, f64, f64),
    /// Normalized texture coordinate `(s, t)`.
    pub texcoord: (f64, f64),
}

/// One textured mesh triangle.
pub type TexturedMeshTriangle = [TexturedMeshVertex; 3];

/// One material mesh triangle, optionally carrying per-vertex texture coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialMeshTriangle {
    /// Triangle vertex positions in object space.
    pub positions: [(f64, f64, f64); 3],
    /// Optional normalized texture coordinates `(s, t)` for each vertex.
    pub texcoords: Option<[(f64, f64); 3]>,
}

/// Summary of triangles imported from a mesh file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStats {
    /// Number of triangles imported by one mesh load.
    pub triangles: usize,
    /// Axis-aligned bounds of the imported triangles, or `None` for an empty mesh.
    pub bounds: Option<Bounds3>,
}

/// Mesh triangles grouped by OBJ material.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialMesh {
    /// Triangle groups in source order.
    pub groups: Vec<MaterialMeshGroup>,
    /// Axis-aligned bounds of all imported triangles, or `None` for an empty mesh.
    pub bounds: Option<Bounds3>,
}

impl MaterialMesh {
    /// Returns the total number of imported triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.polygons.triangle_count())
            .sum()
    }

    /// Returns true if at least one group has a material diffuse color.
    #[must_use]
    pub fn has_material_colors(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.diffuse_color.is_some())
    }

    /// Returns true if at least one group has UV triangles and a diffuse texture map.
    #[must_use]
    pub fn has_textures(&self) -> bool {
        self.groups.iter().any(|group| {
            !group.textured_triangles.is_empty()
                && group
                    .material
                    .as_ref()
                    .and_then(|material| material.diffuse_texture.as_ref())
                    .is_some()
        })
    }
}

/// One material-colored chunk of a mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialMeshGroup {
    /// OBJ material name from `usemtl`, if present.
    pub material_name: Option<String>,
    /// Material coefficients from the referenced MTL file, if available.
    pub material: Option<MeshMaterial>,
    /// Diffuse `Kd` color from the referenced MTL file, if available.
    pub diffuse_color: Option<Rgb>,
    /// Triangles assigned to this material group.
    pub polygons: PolygonMatrix,
    /// Cached vertex-normal adjacency for `polygons`.
    pub normal_plan: VertexNormalPlan,
    /// Triangles assigned to this material group, preserving whether each triangle had UVs.
    pub triangles: Vec<MaterialMeshTriangle>,
    /// Triangles with per-vertex texture coordinates assigned to this material group.
    pub textured_triangles: Vec<TexturedMeshTriangle>,
}

/// Material coefficients parsed from an MTL file.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshMaterial {
    /// Ambient `Ka` coefficients.
    pub ambient: Option<[f64; 3]>,
    /// Diffuse `Kd` coefficients.
    pub diffuse: Option<[f64; 3]>,
    /// Specular `Ks` coefficients.
    pub specular: Option<[f64; 3]>,
    /// Specular exponent from `Ns`.
    pub shininess: Option<f64>,
    /// Optical density / index of refraction from `Ni`.
    pub optical_density: Option<f64>,
    /// Opacity from `d`, or `1 - Tr`.
    pub alpha: Option<f64>,
    /// Illumination model from `illum`.
    pub illumination_model: Option<u32>,
    /// Diffuse texture map from `map_Kd`, resolved relative to the MTL file.
    pub diffuse_texture: Option<PathBuf>,
    /// Tangent-space normal map from `map_Bump`, `bump`, or `norm`, resolved relative to the MTL file.
    pub normal_texture: Option<PathBuf>,
}

impl MeshMaterial {
    fn diffuse_color(&self) -> Option<Rgb> {
        self.diffuse.map(rgb_from_unit_color)
    }
}

impl From<MeshMaterial> for SurfaceMaterial {
    fn from(material: MeshMaterial) -> Self {
        let to_color = |coefficients: Option<[f64; 3]>, fallback: LinearRgb| {
            coefficients.map_or(fallback, |[red, green, blue]| {
                LinearRgb::new(red, green, blue)
            })
        };

        let mut surface = SurfaceMaterial::new(
            to_color(material.ambient, LinearRgb::new(0.0, 0.0, 0.0)),
            to_color(material.diffuse, LinearRgb::new(0.5, 0.5, 0.5)),
            to_color(material.specular, LinearRgb::new(0.0, 0.0, 0.0)),
            material.shininess.unwrap_or(1.0),
        );
        surface.refractive_index = material.optical_density.and_then(RefractiveIndex::try_new);
        surface.diffuse_texture = material.diffuse_texture;
        surface.normal_texture = material.normal_texture;
        surface
    }
}

/// Source up-axis convention for imported mesh files.
///
/// Gartus examples treat `Y` as the vertical axis and `Z` as depth. Many OBJ
/// files downloaded from modeling tools are `Z`-up, so they need an explicit
/// axis conversion before they are projected or drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshUpAxis {
    /// Mesh vertices already use Gartus' `Y`-up convention.
    Y,
    /// Mesh vertices use `Z` as vertical; convert `Z` into Gartus' `Y`.
    Z,
}

impl MeshUpAxis {
    /// Returns the transform that converts this source convention to Gartus `Y`-up space.
    pub fn to_y_up_transform(self) -> Matrix {
        match self {
            Self::Y => Matrix::identity_matrix(4),
            Self::Z => Matrix::rotate_x(-90.0),
        }
    }
}

/// Builds a transform that centers, orients, and uniformly scales an imported mesh.
///
/// This leaves [`meshify`] raw and predictable while giving examples and callers
/// one place to handle common OBJ/STL up-axis differences.
///
/// Use [`try_normalize_mesh_transform`] when an empty mesh should be handled without panicking.
///
/// # Panics
/// Panics if `mesh` is empty.
pub fn normalize_mesh_transform(
    mesh: &PolygonMatrix,
    target_size: f64,
    source_up_axis: MeshUpAxis,
) -> Matrix {
    try_normalize_mesh_transform(mesh, target_size, source_up_axis)
        .expect("mesh should have bounds")
}

/// Builds a transform that centers, orients, and uniformly scales a non-empty imported mesh.
///
/// Returns `None` when `mesh` has no triangles and therefore no bounds.
#[must_use]
pub fn try_normalize_mesh_transform(
    mesh: &PolygonMatrix,
    target_size: f64,
    source_up_axis: MeshUpAxis,
) -> Option<Matrix> {
    mesh.bounds()
        .map(|bounds| normalize_bounds_transform(bounds, target_size, source_up_axis))
}

/// Builds a transform that centers, orients, and uniformly scales a material-grouped mesh.
///
/// Use [`try_normalize_material_mesh_transform`] when an empty mesh should be handled without
/// panicking.
///
/// # Panics
/// Panics if `mesh` is empty.
pub fn normalize_material_mesh_transform(
    mesh: &MaterialMesh,
    target_size: f64,
    source_up_axis: MeshUpAxis,
) -> Matrix {
    try_normalize_material_mesh_transform(mesh, target_size, source_up_axis)
        .expect("mesh should have bounds")
}

/// Builds a transform that centers, orients, and uniformly scales a non-empty material mesh.
///
/// Returns `None` when `mesh` has no triangles and therefore no bounds.
#[must_use]
pub fn try_normalize_material_mesh_transform(
    mesh: &MaterialMesh,
    target_size: f64,
    source_up_axis: MeshUpAxis,
) -> Option<Matrix> {
    mesh.bounds
        .map(|bounds| normalize_bounds_transform(bounds, target_size, source_up_axis))
}

fn normalize_bounds_transform(
    bounds: Bounds3,
    target_size: f64,
    source_up_axis: MeshUpAxis,
) -> Matrix {
    let center = bounds_center(bounds);
    let span = bounds_span(bounds);
    let scale = target_size / span.max(1e-9);
    Matrix::scale(scale, scale, scale)
        * source_up_axis.to_y_up_transform()
        * Matrix::translate(-center.0, -center.1, -center.2)
}

/// Error returned while loading OBJ or ASCII STL meshes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshError {
    path: Option<PathBuf>,
    line: Option<usize>,
    message: String,
}

impl MeshError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            path: None,
            line: None,
            message: message.into(),
        }
    }

    fn at_path(path: &Path, message: impl Into<String>) -> Self {
        Self {
            path: Some(path.to_path_buf()),
            line: None,
            message: message.into(),
        }
    }

    fn at_line(path: &Path, line: usize, message: impl Into<String>) -> Self {
        Self {
            path: Some(path.to_path_buf()),
            line: Some(line),
            message: message.into(),
        }
    }
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.path, self.line) {
            (Some(path), Some(line)) => write!(f, "{}:{line}: {}", path.display(), self.message),
            (Some(path), None) => write!(f, "{}: {}", path.display(), self.message),
            (None, Some(line)) => write!(f, "line {line}: {}", self.message),
            (None, None) => f.write_str(&self.message),
        }
    }
}

impl Error for MeshError {}

/// Converts an OBJ or STL mesh file into a [`PolygonMatrix`].
///
/// Supported mesh formats are Wavefront OBJ polygon meshes, ASCII STL, and binary STL. OBJ faces
/// with three or more vertices are triangulated with a fan, so quadrilateral faces are accepted.
/// OBJ vertex texture coordinates, vertex normals, materials, groups, objects, and smoothing
/// directives are ignored by this geometry-only loader. Use [`meshify_with_materials`] when
/// material groups, diffuse colors, texture coordinates, and texture paths should be preserved.
///
/// # Arguments
/// * `file_name` - The mesh file to load. Supported extensions are `.obj` and `.stl`.
///
/// # Errors
/// Returns an error if the file cannot be read, has an unsupported extension, or contains
/// malformed mesh data.
pub fn meshify(file_name: &str) -> MeshResult<PolygonMatrix> {
    let mut polygons = PolygonMatrix::new();
    add_mesh(file_name, &mut polygons)?;
    Ok(polygons)
}

/// Converts an OBJ or STL mesh file into material-grouped triangles.
///
/// OBJ `mtllib`, `usemtl`, MTL material coefficients, `map_Kd` diffuse texture paths, and per-face
/// vertex texture coordinates are preserved. OBJ faces with three or more vertices are triangulated
/// with a fan, so quadrilateral faces are accepted. STL files are returned as a single uncolored
/// group without texture data.
///
/// # Errors
/// Returns an error if the mesh cannot be read, has an unsupported extension, or contains malformed
/// mesh or material data.
pub fn meshify_with_materials(file_name: &str) -> MeshResult<MaterialMesh> {
    let path = Path::new(file_name);
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| MeshError::at_path(path, "invalid file extension"))?;

    match ext.as_str() {
        "obj" => {
            let file = File::open(path)
                .map_err(|err| MeshError::at_path(path, format!("could not open file: {err}")))?;
            parse_obj_with_materials(BufReader::new(file), path)
        }
        "stl" => {
            let polygons = meshify(file_name)?;
            let bounds = polygons.bounds();
            Ok(MaterialMesh {
                groups: vec![MaterialMeshGroup {
                    material_name: None,
                    material: None,
                    diffuse_color: None,
                    normal_plan: VertexNormalPlan::from_polygon_data(polygons.as_matrix().data()),
                    polygons,
                    triangles: Vec::new(),
                    textured_triangles: Vec::new(),
                }],
                bounds,
            })
        }
        _ => Err(MeshError::at_path(
            path,
            format!("unsupported mesh extension {ext}"),
        )),
    }
}

fn bounds_center(bounds: Bounds3) -> Point3 {
    (
        (bounds.min.0 + bounds.max.0) * 0.5,
        (bounds.min.1 + bounds.max.1) * 0.5,
        (bounds.min.2 + bounds.max.2) * 0.5,
    )
}

fn bounds_span(bounds: Bounds3) -> f64 {
    let x = bounds.max.0 - bounds.min.0;
    let y = bounds.max.1 - bounds.min.1;
    let z = bounds.max.2 - bounds.min.2;
    x.max(y).max(z)
}

/// Appends an OBJ or STL mesh file to an existing [`PolygonMatrix`].
///
/// Supported mesh formats are Wavefront OBJ polygon meshes, ASCII STL, and binary STL. OBJ faces
/// with three or more vertices are triangulated with a fan, so quadrilateral faces are accepted.
/// OBJ vertex texture coordinates, vertex normals, materials, groups, objects, and smoothing
/// directives are ignored by this geometry-only loader. Use [`meshify_with_materials`] when
/// material groups, diffuse colors, texture coordinates, and texture paths should be preserved.
///
/// # Arguments
/// * `file_name` - The mesh file to load. Supported extensions are `.obj` and `.stl`.
/// * `polygons` - The polygon matrix that receives the parsed triangles.
///
/// # Errors
/// Returns an error if the file cannot be read, has an unsupported extension, or contains
/// malformed mesh data.
///
/// # Atomicity
/// Meshes are streamed directly into `polygons` in batches. If parsing fails after some batches
/// have been appended, the matrix is truncated back to its pre-import column count before the
/// error is returned.
pub fn add_mesh(file_name: &str, polygons: &mut PolygonMatrix) -> MeshResult<MeshStats> {
    let path = Path::new(file_name);
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| MeshError::at_path(path, "invalid file extension"))?;

    let start_col = polygons.cols();

    let result = match ext.as_str() {
        "obj" => {
            let file = File::open(path)
                .map_err(|err| MeshError::at_path(path, format!("could not open file: {err}")))?;
            parse_obj(BufReader::new(file), path, polygons)
        }
        "stl" => {
            let (file, is_binary) = open_stl_file(path)?;
            if is_binary {
                parse_binary_stl(file, path, polygons)
            } else {
                parse_stl(BufReader::new(file), path, polygons)
            }
        }
        _ => Err(MeshError::at_path(
            path,
            format!("unsupported mesh extension {ext}"),
        )),
    };

    if let Err(err) = result {
        polygons.truncate_cols(start_col);
        return Err(err);
    }

    Ok(MeshStats {
        triangles: (polygons.cols() - start_col) / 3,
        bounds: polygons.bounds_from_col(start_col),
    })
}

fn parse_obj<R: BufRead>(reader: R, source: &Path, polygons: &mut PolygonMatrix) -> MeshResult<()> {
    let mut vertices = Vec::new();
    let mut triangle_batch = Vec::with_capacity(MESH_TRIANGLE_BATCH);

    for_each_text_line(reader, source, |line_num, line| {
        let line = strip_obj_comment(line).trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let x = parse_f64_arg(parts.next(), source, line_num, "x")?;
                let y = parse_f64_arg(parts.next(), source, line_num, "y")?;
                let z = parse_f64_arg(parts.next(), source, line_num, "z")?;
                let w = parts
                    .next()
                    .map(|token| parse_f64_arg(Some(token), source, line_num, "w"))
                    .transpose()?
                    .unwrap_or(1.0);
                if w == 0.0 {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "OBJ vertex weight cannot be zero",
                    ));
                }
                vertices.push((x / w, y / w, z / w));
            }
            Some("f") => {
                triangulate_obj_face(parts, &vertices, source, line_num, |triangle| {
                    triangle_batch.push(triangle);
                    flush_triangle_batch(polygons, &mut triangle_batch);
                })?;
            }
            _ => {}
        }
        Ok(())
    })?;

    polygons.push_polygons(triangle_batch.as_slice());
    Ok(())
}

fn for_each_text_line<R, F>(mut reader: R, source: &Path, mut visit: F) -> MeshResult<()>
where
    R: BufRead,
    F: FnMut(usize, &str) -> MeshResult<()>,
{
    let mut line = String::new();
    let mut line_num = 0;
    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).map_err(|err| {
            MeshError::at_line(source, line_num + 1, format!("could not read line: {err}"))
        })?;
        if bytes_read == 0 {
            break;
        }
        line_num += 1;
        visit(line_num, &line)?;
    }
    Ok(())
}

#[derive(Debug)]
struct MaterialGroupBuilder {
    material_name: Option<String>,
    polygons: PolygonMatrix,
    triangle_batch: Vec<Triangle>,
    triangles: Vec<MaterialMeshTriangle>,
    textured_triangles: Vec<TexturedMeshTriangle>,
}

impl MaterialGroupBuilder {
    fn new(material_name: Option<String>) -> Self {
        Self {
            material_name,
            polygons: PolygonMatrix::new(),
            triangle_batch: Vec::with_capacity(MESH_TRIANGLE_BATCH),
            triangles: Vec::new(),
            textured_triangles: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.polygons.cols() == 0 && self.triangle_batch.is_empty()
    }

    fn push_triangle(
        &mut self,
        triangle: Triangle,
        textured_triangle: Option<TexturedMeshTriangle>,
    ) {
        self.triangle_batch.push(triangle);
        self.triangles.push(MaterialMeshTriangle {
            positions: triangle,
            texcoords: textured_triangle.map(|triangle| triangle.map(|vertex| vertex.texcoord)),
        });
        if let Some(textured_triangle) = textured_triangle {
            self.textured_triangles.push(textured_triangle);
        }
        if self.triangle_batch.len() >= MESH_TRIANGLE_BATCH {
            self.flush();
        }
    }

    fn flush(&mut self) {
        self.polygons.push_polygons(self.triangle_batch.as_slice());
        self.triangle_batch.clear();
    }

    fn finish(mut self, materials: &HashMap<String, MeshMaterial>) -> MaterialMeshGroup {
        self.flush();
        let material = self
            .material_name
            .as_ref()
            .and_then(|name| materials.get(name).cloned());
        let diffuse_color = material.as_ref().and_then(MeshMaterial::diffuse_color);
        let normal_plan = VertexNormalPlan::from_polygon_data(self.polygons.as_matrix().data());
        MaterialMeshGroup {
            material_name: self.material_name,
            material,
            diffuse_color,
            polygons: self.polygons,
            normal_plan,
            triangles: self.triangles,
            textured_triangles: self.textured_triangles,
        }
    }
}

fn parse_obj_with_materials<R: BufRead>(reader: R, source: &Path) -> MeshResult<MaterialMesh> {
    let mut vertices = Vec::new();
    let mut texcoords = Vec::new();
    let mut materials = HashMap::new();
    let mut groups = Vec::new();
    let mut current_group = MaterialGroupBuilder::new(None);

    for_each_text_line(reader, source, |line_num, line| {
        let line = strip_obj_comment(line).trim();
        if line.is_empty() {
            return Ok(());
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("mtllib") => {
                let filename = parts.collect::<Vec<_>>().join(" ");
                if filename.is_empty() {
                    return Err(MeshError::at_line(source, line_num, "missing MTL filename"));
                }
                let path = resolve_sibling_path(source, &filename);
                materials.extend(load_mtl_materials(&path)?);
            }
            Some("usemtl") => {
                let name = parts.collect::<Vec<_>>().join(" ");
                if name.is_empty() {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "missing OBJ material name",
                    ));
                }
                let finished_group =
                    std::mem::replace(&mut current_group, MaterialGroupBuilder::new(Some(name)));
                if !finished_group.is_empty() {
                    groups.push(finished_group.finish(&materials));
                }
            }
            Some("v") => {
                let x = parse_f64_arg(parts.next(), source, line_num, "x")?;
                let y = parse_f64_arg(parts.next(), source, line_num, "y")?;
                let z = parse_f64_arg(parts.next(), source, line_num, "z")?;
                let w = parts
                    .next()
                    .map(|token| parse_f64_arg(Some(token), source, line_num, "w"))
                    .transpose()?
                    .unwrap_or(1.0);
                if w == 0.0 {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "OBJ vertex weight cannot be zero",
                    ));
                }
                vertices.push((x / w, y / w, z / w));
            }
            Some("vt") => {
                let s = parse_f64_arg(parts.next(), source, line_num, "texture s")?;
                let t = parts
                    .next()
                    .map(|token| parse_f64_arg(Some(token), source, line_num, "texture t"))
                    .transpose()?
                    .unwrap_or(0.0);
                if let Some(token) = parts.next() {
                    let _ = parse_f64_arg(Some(token), source, line_num, "texture w")?;
                }
                texcoords.push((s, t));
            }
            Some("f") => {
                triangulate_obj_face_with_texcoords(
                    parts,
                    &vertices,
                    &texcoords,
                    source,
                    line_num,
                    |triangle, textured_triangle| {
                        current_group.push_triangle(triangle, textured_triangle);
                    },
                )?;
            }
            _ => {}
        }
        Ok(())
    })?;

    if !current_group.is_empty() {
        groups.push(current_group.finish(&materials));
    }

    let bounds = bounds_for_material_groups(&groups);
    Ok(MaterialMesh { groups, bounds })
}

fn bounds_for_material_groups(groups: &[MaterialMeshGroup]) -> Option<Bounds3> {
    groups
        .iter()
        .filter_map(|group| group.polygons.bounds())
        .reduce(Bounds3::union)
}

fn resolve_sibling_path(source: &Path, filename: &str) -> PathBuf {
    let path = Path::new(filename);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        source
            .parent()
            .map_or_else(|| path.to_path_buf(), |dir| dir.join(path))
    }
}

fn load_mtl_materials(path: &Path) -> MeshResult<HashMap<String, MeshMaterial>> {
    let file = File::open(path)
        .map_err(|err| MeshError::at_path(path, format!("could not open material file: {err}")))?;
    parse_mtl(BufReader::new(file), path)
}

#[allow(clippy::too_many_lines)]
fn parse_mtl<R: BufRead>(reader: R, source: &Path) -> MeshResult<HashMap<String, MeshMaterial>> {
    let mut materials = HashMap::new();
    let mut current_material = None::<String>;
    for_each_text_line(reader, source, |line_num, line| {
        let line = strip_obj_comment(line).trim();
        if line.is_empty() {
            return Ok(());
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("newmtl") => {
                current_material = Some(parse_mtl_name(parts, source, line_num)?);
            }
            Some("Kd") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.diffuse = Some(parse_mtl_color(parts, source, line_num)?);
                }
            }
            Some("Ka") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.ambient = Some(parse_mtl_color(parts, source, line_num)?);
                }
            }
            Some("Ks") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.specular = Some(parse_mtl_color(parts, source, line_num)?);
                }
            }
            Some("Ns") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.shininess =
                        Some(parse_f64_arg(parts.next(), source, line_num, "shininess")?);
                }
            }
            Some("Ni") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.optical_density = Some(parse_f64_arg(
                        parts.next(),
                        source,
                        line_num,
                        "optical density",
                    )?);
                }
            }
            Some("d") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.alpha = Some(parse_mtl_alpha(parts, source, line_num)?);
                }
            }
            Some("Tr") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    let transparency = parse_f64_arg(parts.next(), source, line_num, "alpha")?;
                    material.alpha = Some((1.0 - transparency).clamp(0.0, 1.0));
                }
            }
            Some("illum") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    material.illumination_model = Some(parse_u32_arg(
                        parts.next(),
                        source,
                        line_num,
                        "illumination model",
                    )?);
                }
            }
            Some("map_Kd") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    let filename = parse_mtl_texture_filename(parts, source, line_num, "map_Kd")?;
                    material.diffuse_texture = Some(resolve_sibling_path(source, &filename));
                }
            }
            Some("map_Bump" | "map_bump" | "bump" | "norm") => {
                if let Some(material) =
                    current_mtl_material(&mut materials, current_material.as_ref())
                {
                    let filename =
                        parse_mtl_texture_filename(parts, source, line_num, "normal map")?;
                    material.normal_texture = Some(resolve_sibling_path(source, &filename));
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(materials)
}

fn parse_mtl_name<'a>(
    parts: impl Iterator<Item = &'a str>,
    source: &Path,
    line_num: usize,
) -> MeshResult<String> {
    let name = parts.collect::<Vec<_>>().join(" ");
    if name.is_empty() {
        Err(MeshError::at_line(
            source,
            line_num,
            "missing MTL material name",
        ))
    } else {
        Ok(name)
    }
}

fn current_mtl_material<'a>(
    materials: &'a mut HashMap<String, MeshMaterial>,
    current_material: Option<&String>,
) -> Option<&'a mut MeshMaterial> {
    current_material.map(|name| {
        materials
            .entry(name.clone())
            .or_insert_with(empty_mesh_material)
    })
}

fn parse_mtl_texture_filename<'a>(
    parts: impl Iterator<Item = &'a str>,
    source: &Path,
    line_num: usize,
    label: &str,
) -> MeshResult<String> {
    let tokens = parts.collect::<Vec<_>>();
    let mut index = 0;

    while index < tokens.len() && is_mtl_option_token(tokens[index]) {
        let option = tokens[index];
        index += 1;

        match option {
            "-blendu" | "-blendv" | "-boost" | "-bm" | "-cc" | "-clamp" | "-imfchan"
            | "-texres" | "-type" => {
                require_mtl_option_values(&tokens, &mut index, 1, option, source, line_num)?;
            }
            "-mm" => {
                require_mtl_option_values(&tokens, &mut index, 2, option, source, line_num)?;
            }
            "-o" | "-s" | "-t" => {
                let start = index;
                while index < tokens.len()
                    && index - start < 3
                    && !is_mtl_option_token(tokens[index])
                    && tokens[index].parse::<f64>().is_ok()
                {
                    index += 1;
                }
                if index == start {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        format!("{label} option `{option}` requires at least one numeric value"),
                    ));
                }
            }
            _ => {
                return Err(MeshError::at_line(
                    source,
                    line_num,
                    format!("unsupported {label} option `{option}`"),
                ));
            }
        }
    }

    let filename = tokens[index..].join(" ");
    if filename.is_empty() {
        return Err(MeshError::at_line(
            source,
            line_num,
            format!("missing {label} texture filename"),
        ));
    }
    Ok(filename)
}

fn require_mtl_option_values(
    tokens: &[&str],
    index: &mut usize,
    count: usize,
    option: &str,
    source: &Path,
    line_num: usize,
) -> MeshResult<()> {
    for _ in 0..count {
        if *index >= tokens.len() || is_mtl_option_token(tokens[*index]) {
            return Err(MeshError::at_line(
                source,
                line_num,
                format!("map_Kd option `{option}` expects {count} value(s)"),
            ));
        }
        *index += 1;
    }
    Ok(())
}

fn is_mtl_option_token(token: &str) -> bool {
    token.starts_with('-') && token.parse::<f64>().is_err()
}

fn empty_mesh_material() -> MeshMaterial {
    MeshMaterial {
        ambient: None,
        diffuse: None,
        specular: None,
        shininess: None,
        optical_density: None,
        alpha: None,
        illumination_model: None,
        diffuse_texture: None,
        normal_texture: None,
    }
}

fn parse_mtl_alpha<'a>(
    parts: impl Iterator<Item = &'a str>,
    source: &Path,
    line_num: usize,
) -> MeshResult<f64> {
    for part in parts {
        if part == "-halo" {
            continue;
        }
        return part.parse::<f64>().map_or_else(
            |_| {
                Err(MeshError::at_line(
                    source,
                    line_num,
                    format!("invalid alpha value `{part}`"),
                ))
            },
            |alpha| Ok(alpha.clamp(0.0, 1.0)),
        );
    }

    Err(MeshError::at_line(source, line_num, "missing alpha value"))
}

fn parse_mtl_color<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    source: &Path,
    line_num: usize,
) -> MeshResult<[f64; 3]> {
    Ok([
        parse_f64_arg(parts.next(), source, line_num, "red")?.clamp(0.0, 1.0),
        parse_f64_arg(parts.next(), source, line_num, "green")?.clamp(0.0, 1.0),
        parse_f64_arg(parts.next(), source, line_num, "blue")?.clamp(0.0, 1.0),
    ])
}

fn rgb_from_unit_color(color: [f64; 3]) -> Rgb {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Rgb::new(
        (color[0] * 255.0).round() as u8,
        (color[1] * 255.0).round() as u8,
        (color[2] * 255.0).round() as u8,
    )
}

fn strip_obj_comment(line: &str) -> &str {
    line.split_once('#')
        .map_or(line, |(before_comment, _comment)| before_comment)
}

#[derive(Clone, Copy, Debug)]
struct ObjFaceRef {
    vertex: usize,
    texcoord: Option<usize>,
}

fn parse_obj_index(
    raw: &str,
    count: usize,
    source: &Path,
    line_num: usize,
    label: &str,
) -> MeshResult<usize> {
    let index = raw.parse::<i64>().map_err(|_| {
        MeshError::at_line(
            source,
            line_num,
            format!("OBJ {label} index is not an integer"),
        )
    })?;

    if index == 0 {
        return Err(MeshError::at_line(
            source,
            line_num,
            format!("OBJ {label} index cannot be zero"),
        ));
    }

    let count = i64::try_from(count).map_err(|_| MeshError::new("OBJ has too many elements"))?;
    let resolved = if index > 0 { index - 1 } else { count + index };

    if !(0..count).contains(&resolved) {
        return Err(MeshError::at_line(
            source,
            line_num,
            format!("OBJ {label} index is out of bounds"),
        ));
    }

    usize::try_from(resolved).map_err(|_| MeshError::new("OBJ face index overflowed usize"))
}

fn parse_obj_vertex_index(
    token: &str,
    vertex_count: usize,
    source: &Path,
    line_num: usize,
) -> MeshResult<usize> {
    let raw = token
        .split('/')
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| {
            MeshError::at_line(source, line_num, "OBJ face index is missing a vertex")
        })?;
    parse_obj_index(raw, vertex_count, source, line_num, "face")
}

fn parse_obj_face_ref(
    token: &str,
    vertex_count: usize,
    texcoord_count: usize,
    source: &Path,
    line_num: usize,
) -> MeshResult<ObjFaceRef> {
    let mut parts = token.split('/');
    let vertex_raw = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| {
            MeshError::at_line(source, line_num, "OBJ face index is missing a vertex")
        })?;
    let texcoord_raw = parts.next().filter(|part| !part.is_empty());

    Ok(ObjFaceRef {
        vertex: parse_obj_index(vertex_raw, vertex_count, source, line_num, "face")?,
        texcoord: texcoord_raw
            .map(|raw| parse_obj_index(raw, texcoord_count, source, line_num, "texture"))
            .transpose()?,
    })
}

fn triangulate_obj_face<'a, I, F>(
    mut parts: I,
    vertices: &[Point3],
    source: &Path,
    line_num: usize,
    mut push_triangle: F,
) -> MeshResult<()>
where
    I: Iterator<Item = &'a str>,
    F: FnMut(Triangle),
{
    let Some(first_token) = parts.next() else {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face has fewer than 3 vertices",
        ));
    };
    let Some(second_token) = parts.next() else {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face has fewer than 3 vertices",
        ));
    };
    let Some(third_token) = parts.next() else {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face has fewer than 3 vertices",
        ));
    };

    let first = parse_obj_vertex_index(first_token, vertices.len(), source, line_num)?;
    let mut previous = parse_obj_vertex_index(second_token, vertices.len(), source, line_num)?;
    let mut current = parse_obj_vertex_index(third_token, vertices.len(), source, line_num)?;

    push_triangle([vertices[first], vertices[previous], vertices[current]]);
    previous = current;

    for token in parts {
        current = parse_obj_vertex_index(token, vertices.len(), source, line_num)?;
        push_triangle([vertices[first], vertices[previous], vertices[current]]);
        previous = current;
    }

    Ok(())
}

fn triangulate_obj_face_with_texcoords<'a, I, F>(
    mut parts: I,
    vertices: &[Point3],
    texcoords: &[TexCoord],
    source: &Path,
    line_num: usize,
    mut push_triangle: F,
) -> MeshResult<()>
where
    I: Iterator<Item = &'a str>,
    F: FnMut(Triangle, Option<TexturedMeshTriangle>),
{
    let Some(first_token) = parts.next() else {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face has fewer than 3 vertices",
        ));
    };
    let Some(second_token) = parts.next() else {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face has fewer than 3 vertices",
        ));
    };
    let Some(third_token) = parts.next() else {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face has fewer than 3 vertices",
        ));
    };

    let first = parse_obj_face_ref(
        first_token,
        vertices.len(),
        texcoords.len(),
        source,
        line_num,
    )?;
    let mut previous = parse_obj_face_ref(
        second_token,
        vertices.len(),
        texcoords.len(),
        source,
        line_num,
    )?;
    let mut current = parse_obj_face_ref(
        third_token,
        vertices.len(),
        texcoords.len(),
        source,
        line_num,
    )?;

    push_obj_ref_triangle(
        first,
        previous,
        current,
        vertices,
        texcoords,
        &mut push_triangle,
    );
    previous = current;

    for token in parts {
        current = parse_obj_face_ref(token, vertices.len(), texcoords.len(), source, line_num)?;
        push_obj_ref_triangle(
            first,
            previous,
            current,
            vertices,
            texcoords,
            &mut push_triangle,
        );
        previous = current;
    }

    Ok(())
}

fn push_obj_ref_triangle<F>(
    first: ObjFaceRef,
    second: ObjFaceRef,
    third: ObjFaceRef,
    vertices: &[Point3],
    texcoords: &[TexCoord],
    push_triangle: &mut F,
) where
    F: FnMut(Triangle, Option<TexturedMeshTriangle>),
{
    let triangle = [
        vertices[first.vertex],
        vertices[second.vertex],
        vertices[third.vertex],
    ];
    let textured_triangle = match (first.texcoord, second.texcoord, third.texcoord) {
        (Some(first_texcoord), Some(second_texcoord), Some(third_texcoord)) => Some([
            TexturedMeshVertex {
                position: vertices[first.vertex],
                texcoord: texcoords[first_texcoord],
            },
            TexturedMeshVertex {
                position: vertices[second.vertex],
                texcoord: texcoords[second_texcoord],
            },
            TexturedMeshVertex {
                position: vertices[third.vertex],
                texcoord: texcoords[third_texcoord],
            },
        ]),
        _ => None,
    };

    push_triangle(triangle, textured_triangle);
}

fn open_stl_file(path: &Path) -> MeshResult<(File, bool)> {
    let mut file = File::open(path)
        .map_err(|err| MeshError::at_path(path, format!("could not open file: {err}")))?;
    let metadata = file
        .metadata()
        .map_err(|err| MeshError::at_path(path, format!("could not stat file: {err}")))?;
    let mut header = [0_u8; 84];
    let bytes_read = file
        .read(&mut header)
        .map_err(|err| MeshError::at_path(path, format!("could not read STL header: {err}")))?;

    let is_binary = classify_stl_header(&header, bytes_read, metadata.len());

    file.rewind()
        .map_err(|err| MeshError::at_path(path, format!("could not rewind STL file: {err}")))?;
    Ok((file, is_binary))
}

fn classify_stl_header(header: &[u8; 84], bytes_read: usize, metadata_len: u64) -> bool {
    if header[..bytes_read].contains(&0) {
        return true;
    }

    if bytes_read == header.len() {
        let triangle_count = u32::from_le_bytes([header[80], header[81], header[82], header[83]]);
        let expected_len = 84_u64 + u64::from(triangle_count) * 50;
        return expected_len == metadata_len;
    }

    false
}

fn parse_binary_stl(file: File, path: &Path, polygons: &mut PolygonMatrix) -> MeshResult<()> {
    let metadata = file
        .metadata()
        .map_err(|err| MeshError::at_path(path, format!("could not stat file: {err}")))?;
    let mut reader = BufReader::new(file);

    let mut header = [0_u8; 80];
    reader.read_exact(&mut header).map_err(|err| {
        MeshError::at_path(path, format!("could not read binary STL header: {err}"))
    })?;

    let triangle_count = read_binary_stl_u32(&mut reader, path)?;
    let expected_len = 84_u64 + u64::from(triangle_count) * 50;
    if metadata.len() != expected_len {
        return Err(MeshError::at_path(
            path,
            format!(
                "binary STL length mismatch: expected {expected_len} bytes for {triangle_count} triangles, found {}",
                metadata.len()
            ),
        ));
    }

    let mut triangle_batch = Vec::with_capacity(MESH_TRIANGLE_BATCH);
    for triangle_index in 0..triangle_count {
        let mut record = [0_u8; 50];
        reader.read_exact(&mut record).map_err(|err| {
            MeshError::at_path(
                path,
                format!("could not read binary STL triangle {triangle_index}: {err}"),
            )
        })?;
        let _normal = read_binary_stl_record_point(&record, path, triangle_index, "normal", 0)?;
        let p0 = read_binary_stl_record_point(&record, path, triangle_index, "vertex", 12)?;
        let p1 = read_binary_stl_record_point(&record, path, triangle_index, "vertex", 24)?;
        let p2 = read_binary_stl_record_point(&record, path, triangle_index, "vertex", 36)?;

        triangle_batch.push([p0, p1, p2]);
        flush_triangle_batch(polygons, &mut triangle_batch);
    }

    polygons.push_polygons(triangle_batch.as_slice());
    Ok(())
}

fn read_binary_stl_u32(reader: &mut impl Read, source: &Path) -> MeshResult<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(|err| {
        MeshError::at_path(
            source,
            format!("could not read binary STL triangle count: {err}"),
        )
    })?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_binary_stl_record_point(
    record: &[u8; 50],
    source: &Path,
    triangle_index: u32,
    label: &str,
    offset: usize,
) -> MeshResult<Point3> {
    let x = read_binary_stl_record_f32(record, source, triangle_index, label, "x", offset)?;
    let y = read_binary_stl_record_f32(record, source, triangle_index, label, "y", offset + 4)?;
    let z = read_binary_stl_record_f32(record, source, triangle_index, label, "z", offset + 8)?;
    Ok((f64::from(x), f64::from(y), f64::from(z)))
}

fn read_binary_stl_record_f32(
    record: &[u8; 50],
    source: &Path,
    triangle_index: u32,
    label: &str,
    axis: &str,
    offset: usize,
) -> MeshResult<f32> {
    let bytes = [
        record[offset],
        record[offset + 1],
        record[offset + 2],
        record[offset + 3],
    ];
    let value = f32::from_le_bytes(bytes);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MeshError::at_path(
            source,
            format!("binary STL triangle {triangle_index} {label} {axis} is not finite"),
        ))
    }
}

fn parse_stl<R: BufRead>(reader: R, source: &Path, polygons: &mut PolygonMatrix) -> MeshResult<()> {
    let mut triangle = Vec::with_capacity(3);
    let mut triangle_batch = Vec::with_capacity(MESH_TRIANGLE_BATCH);

    for_each_text_line(reader, source, |line_num, line| {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some(keyword) if keyword.eq_ignore_ascii_case("facet") && triangle.is_empty() => {}
            Some(keyword) if keyword.eq_ignore_ascii_case("facet") => {
                return Err(MeshError::at_line(
                    source,
                    line_num,
                    "ASCII STL previous facet is incomplete",
                ));
            }
            Some(keyword) if keyword.eq_ignore_ascii_case("vertex") => {
                let x = parse_f64_arg(parts.next(), source, line_num, "x")?;
                let y = parse_f64_arg(parts.next(), source, line_num, "y")?;
                let z = parse_f64_arg(parts.next(), source, line_num, "z")?;
                triangle.push((x, y, z));
                if triangle.len() > 3 {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "ASCII STL facet has too many vertices",
                    ));
                }
            }
            Some(keyword) if keyword.eq_ignore_ascii_case("endfacet") => {
                if triangle.len() != 3 {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "ASCII STL facet does not have 3 vertices",
                    ));
                }
                triangle_batch.push([triangle[0], triangle[1], triangle[2]]);
                flush_triangle_batch(polygons, &mut triangle_batch);
                triangle.clear();
            }
            _ => {}
        }
        Ok(())
    })?;

    if !triangle.is_empty() {
        return Err(MeshError::at_path(
            source,
            "ASCII STL ended with an incomplete triangle",
        ));
    }
    polygons.push_polygons(triangle_batch.as_slice());
    Ok(())
}

fn flush_triangle_batch(polygons: &mut PolygonMatrix, triangle_batch: &mut Vec<Triangle>) {
    if triangle_batch.len() >= MESH_TRIANGLE_BATCH {
        polygons.push_polygons(triangle_batch.as_slice());
        triangle_batch.clear();
    }
}

fn parse_f64_arg(
    token: Option<&str>,
    source: &Path,
    line_num: usize,
    name: &str,
) -> MeshResult<f64> {
    let value = token
        .ok_or_else(|| MeshError::at_line(source, line_num, format!("missing {name} coordinate")))?
        .parse::<f64>()
        .map_err(|_| MeshError::at_line(source, line_num, format!("invalid {name} coordinate")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MeshError::at_line(
            source,
            line_num,
            format!("{name} coordinate is not finite"),
        ))
    }
}

fn parse_u32_arg(
    token: Option<&str>,
    source: &Path,
    line_num: usize,
    name: &str,
) -> MeshResult<u32> {
    token
        .ok_or_else(|| MeshError::at_line(source, line_num, format!("missing {name}")))?
        .parse::<u32>()
        .map_err(|_| MeshError::at_line(source, line_num, format!("invalid {name}")))
}

#[cfg(test)]
mod tests {
    use super::{
        MaterialMesh, MeshUpAxis, add_mesh, meshify, meshify_with_materials,
        normalize_mesh_transform, parse_obj, parse_stl, try_normalize_material_mesh_transform,
        try_normalize_mesh_transform,
    };
    use crate::gmath::polygon_matrix::{Bounds3, PolygonMatrix};
    use crate::graphics::colors::Rgb;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn temp_file(name: &str, extension: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gartus-external-{name}-{}.{}",
            std::process::id(),
            extension
        ))
    }

    #[test]
    fn parses_obj_faces_into_polygon_triangles() {
        let mut polygons = PolygonMatrix::new();

        parse_obj(
            "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"
            .as_bytes(),
            Path::new("inline.obj"),
            &mut polygons,
        )
        .expect("parse obj");

        assert_eq!(polygons.cols(), 6);
        assert_eq!(
            polygons.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn parses_obj_slash_and_relative_indices() {
        let mut polygons = PolygonMatrix::new();

        parse_obj(
            "\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vn 0 0 1
f -3/1/1 -2//1 -1/1
"
            .as_bytes(),
            Path::new("inline.obj"),
            &mut polygons,
        )
        .expect("parse obj");

        assert_eq!(polygons.cols(), 3);
        assert_eq!(
            polygons.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn parses_obj_inline_comments_and_vertex_weights() {
        let mut polygons = PolygonMatrix::new();

        parse_obj(
            "\
# comment before vertices

v 0 0 0 # origin
v 2 0 0 2
v 0 2 0 2
f 1 2 3 # weighted triangle
"
            .as_bytes(),
            Path::new("inline.obj"),
            &mut polygons,
        )
        .expect("parse obj");

        assert_eq!(
            polygons.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn rejects_invalid_obj_indices() {
        let mut polygons = PolygonMatrix::new();

        let error = parse_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 4 3\n".as_bytes(),
            Path::new("inline.obj"),
            &mut polygons,
        )
        .expect_err("obj should fail");

        assert!(error.to_string().contains("out of bounds"));
        assert!(error.to_string().contains("inline.obj:4"));
    }

    #[test]
    fn rejects_zero_obj_vertex_weight() {
        let mut polygons = PolygonMatrix::new();

        let error = parse_obj(
            "v 1 2 3 0\n".as_bytes(),
            Path::new("inline.obj"),
            &mut polygons,
        )
        .expect_err("obj should fail");

        assert!(error.to_string().contains("cannot be zero"));
        assert!(error.to_string().contains("inline.obj:1"));
    }

    #[test]
    fn parses_ascii_stl_vertices_into_triangles() {
        let mut polygons = PolygonMatrix::new();

        parse_stl(
            "\
solid tri
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid tri
"
            .as_bytes(),
            Path::new("inline.stl"),
            &mut polygons,
        )
        .expect("parse stl");

        assert_eq!(polygons.cols(), 3);
        assert_eq!(
            polygons.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn rejects_incomplete_ascii_stl_facets() {
        let mut polygons = PolygonMatrix::new();

        let error = parse_stl(
            "\
solid bad
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
  endloop
endfacet
endsolid bad
"
            .as_bytes(),
            Path::new("inline.stl"),
            &mut polygons,
        )
        .expect_err("stl should fail");

        assert!(error.to_string().contains("does not have 3 vertices"));
        assert!(error.to_string().contains("inline.stl:7"));
    }

    #[test]
    fn rejects_ascii_stl_facets_with_too_many_vertices() {
        let mut polygons = PolygonMatrix::new();

        let error = parse_stl(
            "\
solid bad
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
    vertex 1 1 0
  endloop
endfacet
endsolid bad
"
            .as_bytes(),
            Path::new("inline.stl"),
            &mut polygons,
        )
        .expect_err("stl should fail");

        assert!(error.to_string().contains("too many vertices"));
        assert!(error.to_string().contains("inline.stl:7"));
    }

    #[test]
    fn loads_fixture_obj_ngons() {
        let polygons = meshify("examples/data/meshes/fixture_ngon.obj").expect("load ngon fixture");

        assert_eq!(polygons.cols(), 15);
        assert_eq!(polygons.triangle_count(), 5);
        assert_eq!(
            polygons.bounds(),
            Some(Bounds3 {
                min: (0.0, 0.0, 0.0),
                max: (3.0, 1.5, 0.0),
            })
        );
    }

    #[test]
    fn loads_fixture_obj_negative_and_slash_indices() {
        let polygons =
            meshify("examples/data/meshes/fixture_obj_indices.obj").expect("load index fixture");

        assert_eq!(polygons.cols(), 6);
        assert_eq!(polygons.triangle_count(), 2);
    }

    #[test]
    fn loads_obj_material_groups_with_diffuse_colors() {
        let obj_path = temp_file("materials", "obj");
        let mtl_path = obj_path.with_extension("mtl");
        let mtl_name = mtl_path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf8 mtl filename");

        fs::write(
            &mtl_path,
            b"newmtl red\nKa 0.1 0.2 0.3\nKd 1 0 0\nKs 0.4 0.5 0.6\nNs 42\nNi 1.5\nd -halo 0.75\nillum 4\nmap_Kd textures/red.ppm\nmap_Bump -bm 0.5 textures/red-normal.ppm\nnewmtl green\nKd 0 0.5 0\nTr 0.25\n",
        )
        .expect("write temp mtl");
        fs::write(
            &obj_path,
            format!(
                "\
mtllib {mtl_name}
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl red
f 1/1 2/2 3/3
usemtl green
f 1/1 2/2 4/4 3/3
"
            ),
        )
        .expect("write temp obj");

        let mesh = meshify_with_materials(obj_path.to_str().expect("utf8 path"))
            .expect("load material mesh");

        assert_eq!(mesh.triangle_count(), 3);
        assert_eq!(mesh.groups.len(), 2);
        assert_eq!(mesh.groups[1].polygons.triangle_count(), 2);
        assert_eq!(mesh.groups[0].textured_triangles.len(), 1);
        assert_eq!(mesh.groups[1].textured_triangles.len(), 2);
        assert_eq!(mesh.groups[0].material_name.as_deref(), Some("red"));
        assert_eq!(mesh.groups[0].diffuse_color, Some(Rgb::RED));
        let red_material = mesh.groups[0].material.as_ref().expect("red material");
        assert_eq!(red_material.ambient, Some([0.1, 0.2, 0.3]));
        assert_eq!(red_material.diffuse, Some([1.0, 0.0, 0.0]));
        assert_eq!(red_material.specular, Some([0.4, 0.5, 0.6]));
        assert_eq!(red_material.shininess, Some(42.0));
        assert_eq!(red_material.optical_density, Some(1.5));
        assert_eq!(red_material.alpha, Some(0.75));
        assert_eq!(red_material.illumination_model, Some(4));
        assert_eq!(
            red_material.diffuse_texture,
            Some(
                mtl_path
                    .parent()
                    .expect("mtl parent")
                    .join("textures/red.ppm")
            )
        );
        assert_eq!(
            red_material.normal_texture,
            Some(
                mtl_path
                    .parent()
                    .expect("mtl parent")
                    .join("textures/red-normal.ppm")
            )
        );
        assert_eq!(mesh.groups[0].textured_triangles[0][2].texcoord, (1.0, 1.0));
        assert!(mesh.has_textures());
        assert_eq!(mesh.groups[1].material_name.as_deref(), Some("green"));
        assert_eq!(mesh.groups[1].diffuse_color, Some(Rgb::new(0, 128, 0)));
        assert_eq!(
            mesh.groups[1].material.as_ref().and_then(|m| m.alpha),
            Some(0.75)
        );
        assert!(mesh.has_material_colors());
        let _ = fs::remove_file(obj_path);
        let _ = fs::remove_file(mtl_path);
    }

    #[test]
    fn material_group_preserves_mixed_uv_and_non_uv_triangles() {
        let obj_path = temp_file("mixed-uv", "obj");
        let mtl_path = obj_path.with_extension("mtl");
        let mtl_name = mtl_path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf8 mtl filename");

        fs::write(&mtl_path, b"newmtl tex\nKd 1 1 1\nmap_Kd texture.ppm\n")
            .expect("write temp mtl");
        fs::write(
            &obj_path,
            format!(
                "\
mtllib {mtl_name}
v 0 0 0
v 1 0 0
v 0 1 0
v 2 0 0
v 3 0 0
v 2 1 0
vt 0 0
vt 1 0
vt 0 1
usemtl tex
f 1/1 2/2 3/3
f 4 5 6
"
            ),
        )
        .expect("write temp obj");

        let mesh = meshify_with_materials(obj_path.to_str().expect("utf8 path"))
            .expect("load material mesh");
        let group = &mesh.groups[0];

        assert_eq!(group.polygons.triangle_count(), 2);
        assert_eq!(group.textured_triangles.len(), 1);
        assert_eq!(group.triangles.len(), 2);
        assert!(group.triangles[0].texcoords.is_some());
        assert!(group.triangles[1].texcoords.is_none());

        let _ = fs::remove_file(obj_path);
        let _ = fs::remove_file(mtl_path);
    }

    #[test]
    fn parses_mtl_map_kd_options_before_texture_filename() {
        let obj_path = temp_file("map-options", "obj");
        let mtl_path = obj_path.with_extension("mtl");
        let mtl_name = mtl_path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf8 mtl filename");

        fs::write(
            &mtl_path,
            b"newmtl tex\nKd 1 1 1\nmap_Kd -s 1 1 1 -o 0 0 0 -clamp on textures/blue tile.ppm\n",
        )
        .expect("write temp mtl");
        fs::write(
            &obj_path,
            format!(
                "\
mtllib {mtl_name}
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
usemtl tex
f 1/1 2/2 3/3
"
            ),
        )
        .expect("write temp obj");

        let mesh = meshify_with_materials(obj_path.to_str().expect("utf8 path"))
            .expect("load material mesh");

        let material = mesh.groups[0].material.as_ref().expect("material");
        assert_eq!(
            material.diffuse_texture,
            Some(
                mtl_path
                    .parent()
                    .expect("mtl parent")
                    .join("textures/blue tile.ppm")
            )
        );
        let _ = fs::remove_file(obj_path);
        let _ = fs::remove_file(mtl_path);
    }

    #[test]
    fn loads_fixture_ascii_stl_with_case_and_spacing_variants() {
        let polygons = meshify("examples/data/meshes/fixture_weird_ascii.stl")
            .expect("load weird stl fixture");

        assert_eq!(polygons.cols(), 6);
        assert_eq!(polygons.triangle_count(), 2);
        assert_eq!(
            polygons.bounds(),
            Some(Bounds3 {
                min: (0.0, 0.0, 0.0),
                max: (1.0, 1.0, 1.0),
            })
        );
    }

    #[test]
    fn mesh_up_axis_z_converts_source_z_height_to_gartus_y() {
        let mut polygons = PolygonMatrix::new();
        polygons.add_polygon((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (1.0, 0.0, 0.0));

        let transformed = polygons.apply(&MeshUpAxis::Z.to_y_up_transform());
        let bounds = transformed.bounds().expect("converted mesh bounds");

        assert!((bounds.min.0 - 0.0).abs() < 1e-9);
        assert!((bounds.min.1 - 0.0).abs() < 1e-9);
        assert!(bounds.min.2.abs() < 1e-9);
        assert!((bounds.max.0 - 1.0).abs() < 1e-9);
        assert!((bounds.max.1 - 2.0).abs() < 1e-9);
        assert!(bounds.max.2.abs() < 1e-9);
    }

    #[test]
    fn normalize_mesh_transform_centers_scales_and_orients_z_up_mesh() {
        let mut polygons = PolygonMatrix::new();
        polygons.add_polygon((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (4.0, 0.0, 0.0));

        let transformed =
            polygons.apply(&normalize_mesh_transform(&polygons, 200.0, MeshUpAxis::Z));
        let bounds = transformed.bounds().expect("normalized mesh bounds");

        assert!((bounds.min.0 + 100.0).abs() < 1e-9);
        assert!((bounds.max.0 - 100.0).abs() < 1e-9);
        assert!((bounds.min.1 + 50.0).abs() < 1e-9);
        assert!((bounds.max.1 - 50.0).abs() < 1e-9);
    }

    #[test]
    fn try_normalize_mesh_transform_returns_none_for_empty_mesh() {
        let polygons = PolygonMatrix::new();

        assert_eq!(
            try_normalize_mesh_transform(&polygons, 200.0, MeshUpAxis::Y),
            None
        );
    }

    #[test]
    fn try_normalize_material_mesh_transform_returns_none_for_empty_mesh() {
        let mesh = MaterialMesh {
            groups: Vec::new(),
            bounds: None,
        };

        assert_eq!(
            try_normalize_material_mesh_transform(&mesh, 200.0, MeshUpAxis::Y),
            None
        );
    }

    #[test]
    fn meshify_dispatches_by_extension() {
        let path = temp_file("triangle", "obj");
        fs::write(&path, b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write temp obj");

        let polygons = meshify(path.to_str().expect("utf8 path")).expect("load mesh");

        assert_eq!(polygons.cols(), 3);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn add_mesh_rolls_back_on_parse_error() {
        let path = temp_file("rollback", "obj");
        fs::write(&path, b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 4 3\n").expect("write temp obj");

        let mut polygons = PolygonMatrix::new();
        polygons.add_polygon((9.0, 9.0, 9.0), (8.0, 9.0, 9.0), (9.0, 8.0, 9.0));
        let original = polygons.clone();

        let error = add_mesh(path.to_str().expect("utf8 path"), &mut polygons)
            .expect_err("mesh should fail");

        assert!(error.to_string().contains("out of bounds"));
        assert_eq!(polygons, original);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_unsupported_mesh_extensions() {
        let path = temp_file("triangle", "mesh");
        fs::write(&path, b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write temp mesh");

        let error = meshify(path.to_str().expect("utf8 path")).expect_err("mesh should fail");

        assert!(error.to_string().contains("unsupported mesh extension"));
        let _ = fs::remove_file(path);
    }

    fn binary_stl_fixture(triangles: &[[(f32, f32, f32); 3]]) -> Vec<u8> {
        let mut bytes = vec![b' '; 80];
        bytes.extend_from_slice(
            &u32::try_from(triangles.len())
                .expect("fixture triangle count fits u32")
                .to_le_bytes(),
        );
        for triangle in triangles {
            for value in [0.0_f32, 0.0, 1.0] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for point in triangle {
                for value in [point.0, point.1, point.2] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            bytes.extend_from_slice(&0_u16.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn loads_binary_stl_vertices_into_triangles() {
        let path = temp_file("binary", "stl");
        let bytes = binary_stl_fixture(&[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]]);
        fs::write(&path, bytes).expect("write binary stl");

        let polygons = meshify(path.to_str().expect("utf8 path")).expect("load binary stl");

        assert_eq!(polygons.cols(), 3);
        assert_eq!(
            polygons.as_matrix().data(),
            &[
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0,
            ]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_binary_stl_length_mismatch() {
        let path = temp_file("binary-short", "stl");
        let mut bytes = binary_stl_fixture(&[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]]);
        bytes.pop();
        fs::write(&path, bytes).expect("write short binary stl");

        let error = meshify(path.to_str().expect("utf8 path")).expect_err("mesh should fail");

        assert!(error.to_string().contains("binary STL length mismatch"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_example_teapot_meshes() {
        let obj = meshify("examples/data/meshes/teapot.obj").expect("load obj teapot");
        let stl = meshify("examples/data/meshes/teapot_ascii.stl").expect("load stl teapot");
        let mut appended = PolygonMatrix::new();
        let stats = add_mesh("examples/data/meshes/teapot_ascii.stl", &mut appended)
            .expect("append stl teapot");

        assert!(obj.cols() > 0);
        assert_eq!(stl.cols(), 52_146);
        assert_eq!(stats.triangles, 17_382);
        assert_eq!(appended.triangle_count(), 17_382);
        assert!(stats.bounds.is_some());
        assert!(obj.cols().is_multiple_of(3));
        assert!(stl.cols().is_multiple_of(3));
    }
}

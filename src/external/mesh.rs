use std::{
    collections::HashMap,
    error::Error,
    fmt,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

use crate::gmath::{
    matrix::Matrix,
    polygon_matrix::{Bounds3, PolygonMatrix},
};
use crate::graphics::colors::Rgb;

type MeshResult<T> = Result<T, MeshError>;
type Point3 = (f64, f64, f64);
type Triangle = [Point3; 3];

const MESH_TRIANGLE_BATCH: usize = 4096;

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
}

/// Material coefficients parsed from an MTL file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshMaterial {
    /// Ambient `Ka` coefficients.
    pub ambient: Option<[f64; 3]>,
    /// Diffuse `Kd` coefficients.
    pub diffuse: Option<[f64; 3]>,
    /// Specular `Ks` coefficients.
    pub specular: Option<[f64; 3]>,
}

impl MeshMaterial {
    fn diffuse_color(self) -> Option<Rgb> {
        self.diffuse.map(rgb_from_unit_color)
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
/// # Panics
/// Panics if `mesh` is empty.
pub fn normalize_mesh_transform(
    mesh: &PolygonMatrix,
    target_size: f64,
    source_up_axis: MeshUpAxis,
) -> Matrix {
    let bounds = mesh.bounds().expect("mesh should have bounds");
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
/// Supported mesh formats are Wavefront OBJ polygon meshes, ASCII STL, and binary STL. OBJ vertex
/// texture coordinates, vertex normals, materials, groups, objects, and smoothing directives are
/// ignored because Gartus renders imported meshes as wireframe triangles.
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
/// OBJ `mtllib`, `usemtl`, and MTL `Kd` diffuse colors are preserved. STL files are returned as a
/// single uncolored group.
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
                    polygons,
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
/// Supported mesh formats are Wavefront OBJ polygon meshes, ASCII STL, and binary STL. OBJ vertex
/// texture coordinates, vertex normals, materials, groups, objects, and smoothing directives are
/// ignored because Gartus renders imported meshes as wireframe triangles.
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
        "stl" if is_binary_stl(path)? => parse_binary_stl(path, polygons),
        "stl" => {
            let file = File::open(path)
                .map_err(|err| MeshError::at_path(path, format!("could not open file: {err}")))?;
            parse_stl(BufReader::new(file), path, polygons)
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

    for (line_idx, line) in reader.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.map_err(|err| {
            MeshError::at_line(source, line_num, format!("could not read line: {err}"))
        })?;
        let line = strip_obj_comment(&line).trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
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
                let face = parts
                    .map(|part| parse_obj_vertex_index(part, vertices.len(), source, line_num))
                    .collect::<MeshResult<Vec<_>>>()?;
                if face.len() < 3 {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "OBJ face has fewer than 3 vertices",
                    ));
                }

                let first = vertices[face[0]];
                for indices in face[1..].windows(2) {
                    triangle_batch.push([first, vertices[indices[0]], vertices[indices[1]]]);
                    flush_triangle_batch(polygons, &mut triangle_batch);
                }
            }
            _ => {}
        }
    }

    polygons.push_polygons(triangle_batch.as_slice());
    Ok(())
}

#[derive(Debug)]
struct MaterialGroupBuilder {
    material_name: Option<String>,
    polygons: PolygonMatrix,
    triangle_batch: Vec<Triangle>,
}

impl MaterialGroupBuilder {
    fn new(material_name: Option<String>) -> Self {
        Self {
            material_name,
            polygons: PolygonMatrix::new(),
            triangle_batch: Vec::with_capacity(MESH_TRIANGLE_BATCH),
        }
    }

    fn is_empty(&self) -> bool {
        self.polygons.cols() == 0 && self.triangle_batch.is_empty()
    }

    fn push_triangle(&mut self, triangle: Triangle) {
        self.triangle_batch.push(triangle);
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
            .and_then(|name| materials.get(name).copied());
        let diffuse_color = material.and_then(MeshMaterial::diffuse_color);
        MaterialMeshGroup {
            material_name: self.material_name,
            material,
            diffuse_color,
            polygons: self.polygons,
        }
    }
}

fn parse_obj_with_materials<R: BufRead>(reader: R, source: &Path) -> MeshResult<MaterialMesh> {
    let mut vertices = Vec::new();
    let mut materials = HashMap::new();
    let mut groups = Vec::new();
    let mut current_group = MaterialGroupBuilder::new(None);

    for (line_idx, line) in reader.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.map_err(|err| {
            MeshError::at_line(source, line_num, format!("could not read line: {err}"))
        })?;
        let line = strip_obj_comment(&line).trim();
        if line.is_empty() {
            continue;
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
                if !current_group.is_empty() {
                    groups.push(current_group.finish(&materials));
                }
                current_group = MaterialGroupBuilder::new(Some(name));
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
            Some("f") => {
                let face = parts
                    .map(|part| parse_obj_vertex_index(part, vertices.len(), source, line_num))
                    .collect::<MeshResult<Vec<_>>>()?;
                if face.len() < 3 {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "OBJ face has fewer than 3 vertices",
                    ));
                }

                let first = vertices[face[0]];
                for indices in face[1..].windows(2) {
                    current_group.push_triangle([
                        first,
                        vertices[indices[0]],
                        vertices[indices[1]],
                    ]);
                }
            }
            _ => {}
        }
    }

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
        .reduce(|mut acc, bounds| {
            acc.min.0 = acc.min.0.min(bounds.min.0);
            acc.min.1 = acc.min.1.min(bounds.min.1);
            acc.min.2 = acc.min.2.min(bounds.min.2);
            acc.max.0 = acc.max.0.max(bounds.max.0);
            acc.max.1 = acc.max.1.max(bounds.max.1);
            acc.max.2 = acc.max.2.max(bounds.max.2);
            acc
        })
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

fn parse_mtl<R: BufRead>(reader: R, source: &Path) -> MeshResult<HashMap<String, MeshMaterial>> {
    let mut materials = HashMap::new();
    let mut current_material = None::<String>;

    for (line_idx, line) in reader.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.map_err(|err| {
            MeshError::at_line(source, line_num, format!("could not read line: {err}"))
        })?;
        let line = strip_obj_comment(&line).trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("newmtl") => {
                let name = parts.collect::<Vec<_>>().join(" ");
                if name.is_empty() {
                    return Err(MeshError::at_line(
                        source,
                        line_num,
                        "missing MTL material name",
                    ));
                }
                current_material = Some(name);
            }
            Some("Kd") => {
                if let Some(name) = current_material.as_ref() {
                    materials
                        .entry(name.clone())
                        .or_insert(empty_mesh_material())
                        .diffuse = Some(parse_mtl_color(parts, source, line_num)?);
                }
            }
            Some("Ka") => {
                if let Some(name) = current_material.as_ref() {
                    materials
                        .entry(name.clone())
                        .or_insert(empty_mesh_material())
                        .ambient = Some(parse_mtl_color(parts, source, line_num)?);
                }
            }
            Some("Ks") => {
                if let Some(name) = current_material.as_ref() {
                    materials
                        .entry(name.clone())
                        .or_insert(empty_mesh_material())
                        .specular = Some(parse_mtl_color(parts, source, line_num)?);
                }
            }
            _ => {}
        }
    }

    Ok(materials)
}

fn empty_mesh_material() -> MeshMaterial {
    MeshMaterial {
        ambient: None,
        diffuse: None,
        specular: None,
    }
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
    let index = raw
        .parse::<i64>()
        .map_err(|_| MeshError::at_line(source, line_num, "OBJ face index is not an integer"))?;

    if index == 0 {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face index cannot be zero",
        ));
    }

    let vertex_count =
        i64::try_from(vertex_count).map_err(|_| MeshError::new("OBJ has too many vertices"))?;
    let resolved = if index > 0 {
        index - 1
    } else {
        vertex_count + index
    };

    if !(0..vertex_count).contains(&resolved) {
        return Err(MeshError::at_line(
            source,
            line_num,
            "OBJ face index is out of bounds",
        ));
    }

    usize::try_from(resolved).map_err(|_| MeshError::new("OBJ face index overflowed usize"))
}

fn is_binary_stl(path: &Path) -> MeshResult<bool> {
    let mut file = File::open(path)
        .map_err(|err| MeshError::at_path(path, format!("could not open file: {err}")))?;
    let metadata = file
        .metadata()
        .map_err(|err| MeshError::at_path(path, format!("could not stat file: {err}")))?;
    let mut header = [0_u8; 84];
    let bytes_read = file
        .read(&mut header)
        .map_err(|err| MeshError::at_path(path, format!("could not read STL header: {err}")))?;

    if header[..bytes_read].contains(&0) {
        return Ok(true);
    }

    if bytes_read == header.len() {
        let triangle_count = u32::from_le_bytes([header[80], header[81], header[82], header[83]]);
        let expected_len = 84_u64 + u64::from(triangle_count) * 50;
        if expected_len == metadata.len() {
            return Ok(true);
        }
    }

    Ok(false)
}

fn parse_binary_stl(path: &Path, polygons: &mut PolygonMatrix) -> MeshResult<()> {
    let mut file = File::open(path)
        .map_err(|err| MeshError::at_path(path, format!("could not open file: {err}")))?;
    let metadata = file
        .metadata()
        .map_err(|err| MeshError::at_path(path, format!("could not stat file: {err}")))?;

    let mut header = [0_u8; 80];
    file.read_exact(&mut header).map_err(|err| {
        MeshError::at_path(path, format!("could not read binary STL header: {err}"))
    })?;

    let triangle_count = read_binary_stl_u32(&mut file, path)?;
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
        let _normal = read_binary_stl_point(&mut file, path, triangle_index, "normal")?;
        let p0 = read_binary_stl_point(&mut file, path, triangle_index, "vertex")?;
        let p1 = read_binary_stl_point(&mut file, path, triangle_index, "vertex")?;
        let p2 = read_binary_stl_point(&mut file, path, triangle_index, "vertex")?;
        let mut attribute = [0_u8; 2];
        file.read_exact(&mut attribute).map_err(|err| {
            MeshError::at_path(
                path,
                format!("could not read binary STL attribute byte count: {err}"),
            )
        })?;

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

fn read_binary_stl_point(
    reader: &mut impl Read,
    source: &Path,
    triangle_index: u32,
    label: &str,
) -> MeshResult<Point3> {
    let x = read_binary_stl_f32(reader, source, triangle_index, label, "x")?;
    let y = read_binary_stl_f32(reader, source, triangle_index, label, "y")?;
    let z = read_binary_stl_f32(reader, source, triangle_index, label, "z")?;
    Ok((f64::from(x), f64::from(y), f64::from(z)))
}

fn read_binary_stl_f32(
    reader: &mut impl Read,
    source: &Path,
    triangle_index: u32,
    label: &str,
    axis: &str,
) -> MeshResult<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(|err| {
        MeshError::at_path(
            source,
            format!("could not read binary STL triangle {triangle_index} {label} {axis}: {err}"),
        )
    })?;
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

    for (line_idx, line) in reader.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.map_err(|err| {
            MeshError::at_line(source, line_num, format!("could not read line: {err}"))
        })?;
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
    }

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

#[cfg(test)]
mod tests {
    use super::{
        MeshUpAxis, add_mesh, meshify, meshify_with_materials, normalize_mesh_transform, parse_obj,
        parse_stl,
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
            b"newmtl red\nKa 0.1 0.2 0.3\nKd 1 0 0\nKs 0.4 0.5 0.6\nnewmtl green\nKd 0 0.5 0\n",
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
usemtl red
f 1 2 3
usemtl green
f 2 4 3
"
            ),
        )
        .expect("write temp obj");

        let mesh = meshify_with_materials(obj_path.to_str().expect("utf8 path"))
            .expect("load material mesh");

        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.groups.len(), 2);
        assert_eq!(mesh.groups[0].material_name.as_deref(), Some("red"));
        assert_eq!(mesh.groups[0].diffuse_color, Some(Rgb::RED));
        let red_material = mesh.groups[0].material.expect("red material");
        assert_eq!(red_material.ambient, Some([0.1, 0.2, 0.3]));
        assert_eq!(red_material.diffuse, Some([1.0, 0.0, 0.0]));
        assert_eq!(red_material.specular, Some([0.4, 0.5, 0.6]));
        assert_eq!(mesh.groups[1].material_name.as_deref(), Some("green"));
        assert_eq!(mesh.groups[1].diffuse_color, Some(Rgb::new(0, 128, 0)));
        assert!(mesh.has_material_colors());
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

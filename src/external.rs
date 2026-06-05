use std::{
    error::Error,
    fmt, fs,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::gmath::{
    matrix::Matrix,
    polygon_matrix::{Bounds3, PolygonMatrix},
};
use crate::graphics::{colors::Rgb, display::Canvas};

type ExternalResult<T> = Result<T, Box<dyn Error>>;
type MeshResult<T> = Result<T, MeshError>;
type Point3 = (f64, f64, f64);
type Triangle = [Point3; 3];

const MESH_TRIANGLE_BATCH: usize = 4096;
static TEMP_PPM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Summary of triangles imported from a mesh file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStats {
    /// Number of triangles imported by one mesh load.
    pub triangles: usize,
    /// Axis-aligned bounds of the imported triangles, or `None` for an empty mesh.
    pub bounds: Option<Bounds3>,
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

/// Converts an image to a [`Canvas`], converting non-PPM images to a sibling `.ppm` file first.
///
/// # Arguments
/// * `file_name` - The name of the file to load.
/// * `pos_glitch` - Whether to swap the parsed canvas dimensions after loading.
///
/// # Note
/// Non-PPM inputs are converted through a temporary `.ppm` file that is removed after parsing.
///
/// # Errors
/// todo!()
///
/// # Examples
///
/// Basic usage:
///```no_run
/// use crate::gartus::prelude::{Canvas, Rgb};
/// use crate::gartus::external;
/// let colors = vec![
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
/// ];
/// let mut canvas = Canvas::new(3, 3, Rgb::BLACK);
/// canvas.fill_canvas(colors);
/// canvas.save_binary("./works.ppm").expect("Works");
/// let other = external::ppmify("./works.ppm", false).expect("Life is wrong");
/// assert_eq!(canvas.pixels(), other.pixels());
/// ```
pub fn ppmify(file_name: &str, pos_glitch: bool) -> ExternalResult<Canvas> {
    let path = Path::new(file_name);
    if !path.exists() {
        return Err(format!("File does not exist: {file_name}").into());
    }

    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or("Invalid file extension")?;

    let canvas = if ext == "ppm" {
        parse_ppm(path)?
    } else {
        let converted = temp_ppm_path(path)?;
        let status = Command::new("magick").arg(path).arg(&converted).status()?;
        if !status.success() {
            let _ = fs::remove_file(&converted);
            return Err("ImageMagick `magick` failed to convert image to ppm".into());
        }

        let parsed = parse_ppm(&converted);
        let _ = fs::remove_file(&converted);
        parsed?
    };

    Ok(if pos_glitch {
        dimension_glitch(&canvas)
    } else {
        canvas
    })
}

fn temp_ppm_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or("Invalid file name")?;
    let counter = TEMP_PPM_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(std::env::temp_dir().join(format!(
        "gartus-ppmify-{stem}-{}-{counter}.ppm",
        std::process::id()
    )))
}

fn dimension_glitch(canvas: &Canvas) -> Canvas {
    let mut glitched = Canvas::new(canvas.height(), canvas.width(), canvas.line);
    glitched.fill_canvas(canvas.pixels().to_vec());
    glitched
}

fn next_token(buffer: &[u8], cursor: &mut usize) -> Option<String> {
    loop {
        while *cursor < buffer.len() && buffer[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }

        if *cursor < buffer.len() && buffer[*cursor] == b'#' {
            while *cursor < buffer.len() && buffer[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }

        break;
    }

    if *cursor >= buffer.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < buffer.len()
        && !buffer[*cursor].is_ascii_whitespace()
        && buffer[*cursor] != b'#'
    {
        *cursor += 1;
    }

    Some(String::from_utf8_lossy(&buffer[start..*cursor]).into_owned())
}

fn scale_channel(value: u16, maxval: u16) -> Result<u8, Box<dyn Error>> {
    if value > maxval {
        return Err(format!("PPM channel value {value} exceeds maxval {maxval}").into());
    }

    Ok(
        u8::try_from((u32::from(value) * 255 + u32::from(maxval) / 2) / u32::from(maxval))
            .unwrap_or(255),
    )
}

fn consume_p6_separator(buffer: &[u8], cursor: &mut usize) -> Result<(), Box<dyn Error>> {
    if *cursor >= buffer.len() || !buffer[*cursor].is_ascii_whitespace() {
        return Err("Invalid PPM file: missing binary data separator".into());
    }

    let separator = buffer[*cursor];
    *cursor += 1;
    if separator == b'\r' && *cursor < buffer.len() && buffer[*cursor] == b'\n' {
        *cursor += 1;
    }
    Ok(())
}

fn parse_ppm(path: &Path) -> Result<Canvas, Box<dyn Error>> {
    let buffer = fs::read(path)?;
    let mut cursor = 0;

    let magic = next_token(&buffer, &mut cursor).ok_or("Invalid PPM file: missing magic")?;
    let width = next_token(&buffer, &mut cursor)
        .ok_or("Invalid PPM file: missing width")?
        .parse::<u32>()?;
    let height = next_token(&buffer, &mut cursor)
        .ok_or("Invalid PPM file: missing height")?
        .parse::<u32>()?;
    let maxval = next_token(&buffer, &mut cursor)
        .ok_or("Invalid PPM file: missing maxval")?
        .parse::<u16>()?;

    if maxval == 0 {
        return Err("unsupported PPM maxval 0; maxval must be 1..=65535".into());
    }

    let pixel_count = u64::from(width) * u64::from(height);
    let pixel_count = usize::try_from(pixel_count).map_err(|_| "PPM image too large")?;
    let mut pixels = Vec::with_capacity(pixel_count);

    match magic.as_str() {
        "P3" => {
            for _ in 0..pixel_count {
                let red = next_token(&buffer, &mut cursor)
                    .ok_or("Invalid PPM file: missing red channel")?
                    .parse::<u16>()?;
                let green = next_token(&buffer, &mut cursor)
                    .ok_or("Invalid PPM file: missing green channel")?
                    .parse::<u16>()?;
                let blue = next_token(&buffer, &mut cursor)
                    .ok_or("Invalid PPM file: missing blue channel")?
                    .parse::<u16>()?;

                pixels.push(Rgb::new(
                    scale_channel(red, maxval)?,
                    scale_channel(green, maxval)?,
                    scale_channel(blue, maxval)?,
                ));
            }
        }
        "P6" => {
            consume_p6_separator(&buffer, &mut cursor)?;

            let bytes_per_sample = if maxval < 256 { 1 } else { 2 };
            let needed = pixel_count
                .checked_mul(3)
                .and_then(|count| count.checked_mul(bytes_per_sample))
                .ok_or("PPM image data is too large")?;
            if buffer.len().saturating_sub(cursor) < needed {
                return Err(format!(
                    "Invalid PPM file: expected {needed} bytes of pixel data, found {}",
                    buffer.len().saturating_sub(cursor)
                )
                .into());
            }

            if bytes_per_sample == 1 {
                for chunk in buffer[cursor..cursor + needed].chunks_exact(3) {
                    pixels.push(Rgb::new(
                        scale_channel(u16::from(chunk[0]), maxval)?,
                        scale_channel(u16::from(chunk[1]), maxval)?,
                        scale_channel(u16::from(chunk[2]), maxval)?,
                    ));
                }
            } else {
                for chunk in buffer[cursor..cursor + needed].chunks_exact(6) {
                    let red = u16::from_be_bytes([chunk[0], chunk[1]]);
                    let green = u16::from_be_bytes([chunk[2], chunk[3]]);
                    let blue = u16::from_be_bytes([chunk[4], chunk[5]]);
                    pixels.push(Rgb::new(
                        scale_channel(red, maxval)?,
                        scale_channel(green, maxval)?,
                        scale_channel(blue, maxval)?,
                    ));
                }
            }
        }
        _ => return Err(format!("Invalid PPM file: unsupported magic {magic}").into()),
    }

    let mut canvas = Canvas::new(width, height, Rgb::default());
    canvas.fill_canvas(pixels);
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::{
        MeshUpAxis, add_mesh, meshify, normalize_mesh_transform, parse_obj, parse_stl, ppmify,
        temp_ppm_path,
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
    fn parses_p3_comments_whitespace_and_scaled_maxval() {
        let path = temp_file("comments", "ppm");
        fs::write(
            &path,
            b"P3
# exported in 2026
2   1
# max value
100
100 0 50   0 100 25
",
        )
        .expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), false).expect("parse ppm");

        assert_eq!(canvas.width(), 2);
        assert_eq!(canvas.height(), 1);
        assert_eq!(canvas.pixels()[0], Rgb::new(255, 0, 128));
        assert_eq!(canvas.pixels()[1], Rgb::new(0, 255, 64));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_uppercase_ppm_extension_without_conversion() {
        let path = temp_file("uppercase", "PPM");
        fs::write(&path, b"P6\n1 1\n255\n\x01\x02\x03").expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), false).expect("parse ppm");

        assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3)]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_p6_with_crlf_header_separator() {
        let path = temp_file("crlf", "ppm");
        fs::write(&path, b"P6\r\n1 1\r\n255\r\n\x01\x02\x03").expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), false).expect("parse ppm");

        assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3)]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_p6_sixteen_bit_samples() {
        let path = temp_file("sixteen-bit", "ppm");
        fs::write(&path, b"P6\n1 1\n1023\n\x03\xff\x02\x00\x00\x00").expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), false).expect("parse ppm");

        assert_eq!(canvas.pixels(), &[Rgb::new(255, 128, 0)]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn temp_ppm_paths_are_unique_per_call() {
        let path = Path::new("/tmp/source.png");

        let first = temp_ppm_path(path).expect("temp path");
        let second = temp_ppm_path(path).expect("temp path");

        assert_ne!(first, second);
    }

    #[test]
    fn truncated_p6_returns_error() {
        let path = temp_file("truncated", "ppm");
        fs::write(&path, b"P6\n2 1\n255\n\x01\x02\x03").expect("write temp ppm");

        let error = ppmify(path.to_str().expect("utf8 path"), false).expect_err("should fail");

        assert!(error.to_string().contains("expected 6 bytes"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn pos_glitch_is_applied_after_parsing() {
        let path = temp_file("glitch", "ppm");
        fs::write(&path, b"P3\n2 1\n255\n1 2 3 4 5 6\n").expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), true).expect("parse ppm");

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 2);
        assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3), Rgb::new(4, 5, 6)]);
        let _ = fs::remove_file(path);
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

#[test]
#[ignore = "requires external files and a display"]
fn external_fun() {
    let pos_glitch = true;
    let canvas = ppmify("./corro.png", pos_glitch).expect("Implmentation is wrong");
    canvas.display().expect("Could not display image");
    let sobel = canvas.sobel();
    sobel.display().expect("Could not display image");
    sobel
        .save_extension("pics/corro.png")
        .expect("Could not save image");
}

#[test]
#[ignore = "requires external files and a display"]
fn command_block() {
    let pos_glitch = true;
    let canvas = ppmify("./CAR.png", pos_glitch).expect("Implmentation is wrong");
    canvas.display().expect("Could not display image");
    let sobel = canvas.sobel();
    sobel.display().expect("Could not display image");
    sobel
        .save_extension("pics/corro.png")
        .expect("Could not save image");
}

#[test]
#[ignore = "requires external files and a display"]
fn parse_and_display() {
    let canvas = ppmify("./stop_1.ppm", false).expect("Implmentation is wrong");
    // let blur = canvas.blur();
    // let sobel = canvas.sobel();
    let edge = canvas.laplacian_edge_detection();
    // blur.display().expect("Could not display image");
    // sobel.display().expect("Could not display image");
    edge.display().expect("Could not display image");
}

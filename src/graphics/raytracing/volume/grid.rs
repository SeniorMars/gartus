use super::field::DensityField;
use crate::gmath::vector::{Point, Vector};
use std::{
    fs,
    io::{self, BufRead, BufReader, ErrorKind, Write},
    path::Path,
};

const GRID_FILE_MAGIC: &str = "gartus-grid-density-v1";
const GRID_FILE_DATA_MARKER: &str = "data little-endian-f32\n";

/// Axis-aligned world-space bounds for a voxel density grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridBounds {
    /// Minimum world-space corner.
    pub min: Point,
    /// Maximum world-space corner.
    pub max: Point,
}

impl GridBounds {
    /// Creates grid bounds from two finite corners.
    ///
    /// # Panics
    ///
    /// Panics if either point is not finite or if any axis has non-positive extent.
    #[must_use]
    pub fn new(min: Point, max: Point) -> Self {
        assert!(
            min.is_finite() && max.is_finite(),
            "grid bounds must be finite"
        );
        assert!(
            max.x() > min.x() && max.y() > min.y() && max.z() > min.z(),
            "grid bounds must have positive extent on every axis"
        );
        Self { min, max }
    }

    /// Returns the size of the bounds on each axis.
    #[must_use]
    pub fn extent(self) -> Vector {
        self.max - self.min
    }

    /// Returns true when `point` lies inside or on the bounds.
    #[must_use]
    pub fn contains(self, point: Point) -> bool {
        point.is_finite()
            && point.x() >= self.min.x()
            && point.x() <= self.max.x()
            && point.y() >= self.min.y()
            && point.y() <= self.max.y()
            && point.z() >= self.min.z()
            && point.z() <= self.max.z()
    }
}

/// Density interpolation mode for [`GridDensityField`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridInterpolation {
    /// Sample the nearest voxel center.
    Nearest,
    /// Trilinearly interpolate between neighboring voxel centers.
    Trilinear,
}

impl GridInterpolation {
    fn as_file_str(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Trilinear => "trilinear",
        }
    }

    fn from_file_str(value: &str) -> io::Result<Self> {
        match value {
            "nearest" => Ok(Self::Nearest),
            "trilinear" => Ok(Self::Trilinear),
            _ => Err(invalid_data("unknown grid interpolation mode")),
        }
    }
}

/// Metadata stored in the self-describing grid-density file format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridDensityMetadata {
    /// Grid dimensions as `[width, height, depth]`.
    pub dims: [usize; 3],
    /// World-space voxel bounds.
    pub bounds: GridBounds,
    /// Density interpolation mode.
    pub interpolation: GridInterpolation,
    /// Optional frame index for cached simulations.
    pub frame_index: Option<usize>,
}

impl GridDensityMetadata {
    /// Creates grid metadata.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero or if the dimensions overflow.
    #[must_use]
    pub fn new(
        dims: [usize; 3],
        bounds: GridBounds,
        interpolation: GridInterpolation,
        frame_index: Option<usize>,
    ) -> Self {
        validate_dims(dims);
        Self {
            dims,
            bounds,
            interpolation,
            frame_index,
        }
    }
}

/// Static voxel density field stored as `f32` samples.
///
/// Grid samples are interpreted as voxel-center densities inside [`GridBounds`]. Sampling outside
/// the bounds returns zero. The grid is static; `DensityField::density` ignores ray time.
#[derive(Clone, Debug)]
pub struct GridDensityField {
    bounds: GridBounds,
    dims: [usize; 3],
    density: Vec<f32>,
    max_density: f64,
    interpolation: GridInterpolation,
}

impl GridDensityField {
    /// Creates a grid from raw voxel-center density samples.
    ///
    /// Negative and non-finite samples are stored as zero. The density majorant is computed from
    /// the sanitized samples.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero, if the dimensions overflow, or if `density.len()` does not
    /// equal `dims[0] * dims[1] * dims[2]`.
    #[must_use]
    pub fn new(bounds: GridBounds, dims: [usize; 3], density: Vec<f32>) -> Self {
        validate_dims(dims);
        let cell_count = cell_count_for_dims(dims);
        assert_eq!(
            density.len(),
            cell_count,
            "grid density length must match dimensions"
        );

        let mut max_density = 0.0_f64;
        let density = density
            .into_iter()
            .map(|value| {
                let value = sanitize_density_f32(value);
                max_density = max_density.max(f64::from(value));
                value
            })
            .collect();

        Self {
            bounds,
            dims,
            density,
            max_density: positive_majorant(max_density),
            interpolation: GridInterpolation::Trilinear,
        }
    }

    /// Samples `density_fn` at every voxel center and stores the result in a grid.
    ///
    /// Negative and non-finite closure results are stored as zero.
    #[must_use]
    pub fn from_fn<F>(bounds: GridBounds, dims: [usize; 3], mut density_fn: F) -> Self
    where
        F: FnMut(Point) -> f64,
    {
        validate_dims(dims);
        let mut density = Vec::with_capacity(cell_count_for_dims(dims));
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    density.push(sanitize_density_f64(density_fn(cell_center(
                        bounds, dims, x, y, z,
                    ))));
                }
            }
        }
        Self::new(bounds, dims, density)
    }

    /// Bakes another density field into a static grid at `time`.
    #[must_use]
    pub fn from_density_field<D>(
        bounds: GridBounds,
        dims: [usize; 3],
        density_field: &D,
        time: f64,
    ) -> Self
    where
        D: DensityField + ?Sized,
    {
        Self::from_fn(bounds, dims, |point| density_field.density(point, time))
    }

    /// Loads raw little-endian `f32` voxel density samples.
    ///
    /// The file must contain exactly `dims[0] * dims[1] * dims[2]` samples.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or has the wrong byte length.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero or if the dimensions overflow.
    pub fn load_raw(
        bounds: GridBounds,
        dims: [usize; 3],
        path: impl AsRef<Path>,
    ) -> io::Result<Self> {
        validate_dims(dims);
        let expected_bytes = cell_count_for_dims(dims)
            .checked_mul(std::mem::size_of::<f32>())
            .expect("grid byte length should not overflow");
        let bytes = fs::read(path)?;
        if bytes.len() != expected_bytes {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "raw grid has {} bytes, expected {expected_bytes}",
                    bytes.len()
                ),
            ));
        }

        let density = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(Self::new(bounds, dims, density))
    }

    /// Loads a self-describing grid density file.
    ///
    /// The file stores dimensions, bounds, interpolation, optional frame metadata, and a raw
    /// little-endian `f32` density payload.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, has invalid metadata, or has the wrong data
    /// length.
    pub fn load_grid(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::load_grid_with_metadata(path).map(|(grid, _metadata)| grid)
    }

    /// Loads a self-describing grid density file and returns its metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, has invalid metadata, or has the wrong data
    /// length.
    pub fn load_grid_with_metadata(
        path: impl AsRef<Path>,
    ) -> io::Result<(Self, GridDensityMetadata)> {
        let bytes = fs::read(path)?;
        let (metadata, density) = parse_grid_file(&bytes)?;
        let grid = Self::new(metadata.bounds, metadata.dims, density)
            .with_interpolation(metadata.interpolation);
        Ok((grid, metadata))
    }

    /// Loads only the metadata from a self-describing grid density file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or has invalid metadata.
    pub fn load_grid_metadata(path: impl AsRef<Path>) -> io::Result<GridDensityMetadata> {
        let file = fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut header = String::new();
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(invalid_data("grid file missing data marker"));
            }
            if line == GRID_FILE_DATA_MARKER {
                break;
            }
            header.push_str(&line);
        }
        parse_grid_metadata(&header)
    }

    /// Saves raw little-endian `f32` voxel density samples.
    ///
    /// # Errors
    ///
    /// Returns an error when the output file cannot be created or written.
    pub fn save_raw(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        for value in &self.density {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    /// Saves a self-describing grid density file without frame metadata.
    ///
    /// Raw `f32` files remain available through [`Self::save_raw`] when metadata is supplied by the
    /// caller through another channel.
    ///
    /// # Errors
    ///
    /// Returns an error when the output file cannot be created or written.
    pub fn save_grid(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_grid_with_optional_frame(path, None)
    }

    /// Saves a self-describing grid density file with a simulation frame index.
    ///
    /// # Errors
    ///
    /// Returns an error when the output file cannot be created or written.
    pub fn save_grid_with_frame(
        &self,
        path: impl AsRef<Path>,
        frame_index: usize,
    ) -> io::Result<()> {
        self.save_grid_with_optional_frame(path, Some(frame_index))
    }

    /// Returns a copy with a different interpolation mode.
    #[must_use]
    pub fn with_interpolation(mut self, interpolation: GridInterpolation) -> Self {
        self.interpolation = interpolation;
        self
    }

    /// Returns a copy with an explicit density majorant.
    ///
    /// The majorant must be no smaller than the maximum stored density.
    ///
    /// # Panics
    ///
    /// Panics if `max_density` is not positive and finite, or if it is smaller than the stored
    /// density maximum.
    #[must_use]
    pub fn with_max_density(mut self, max_density: f64) -> Self {
        assert!(
            max_density.is_finite() && max_density > 0.0,
            "grid density maximum must be positive and finite"
        );
        let stored_max = self.computed_max_density();
        assert!(
            max_density + f64::EPSILON >= stored_max,
            "grid density maximum must be at least the stored sample maximum"
        );
        self.max_density = max_density;
        self
    }

    /// Returns the grid bounds.
    #[must_use]
    pub const fn bounds(&self) -> GridBounds {
        self.bounds
    }

    /// Returns the grid dimensions.
    #[must_use]
    pub const fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Returns the raw sanitized density samples.
    #[must_use]
    pub fn densities(&self) -> &[f32] {
        &self.density
    }

    /// Returns the interpolation mode.
    #[must_use]
    pub const fn interpolation(&self) -> GridInterpolation {
        self.interpolation
    }

    /// Returns this grid's metadata without a frame index.
    #[must_use]
    pub const fn metadata(&self) -> GridDensityMetadata {
        GridDensityMetadata {
            dims: self.dims,
            bounds: self.bounds,
            interpolation: self.interpolation,
            frame_index: None,
        }
    }

    /// Returns the flattened sample index.
    ///
    /// # Panics
    ///
    /// Panics if any coordinate is outside the grid dimensions.
    #[must_use]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        assert!(
            x < self.dims[0] && y < self.dims[1] && z < self.dims[2],
            "grid index out of bounds"
        );
        index_for_dims(self.dims, x, y, z)
    }

    /// Returns the world-space center of one voxel.
    ///
    /// # Panics
    ///
    /// Panics if any coordinate is outside the grid dimensions.
    #[must_use]
    pub fn cell_center(&self, x: usize, y: usize, z: usize) -> Point {
        assert!(
            x < self.dims[0] && y < self.dims[1] && z < self.dims[2],
            "grid index out of bounds"
        );
        cell_center(self.bounds, self.dims, x, y, z)
    }

    /// Returns the maximum sanitized stored density.
    #[must_use]
    pub fn computed_max_density(&self) -> f64 {
        self.density
            .iter()
            .map(|value| f64::from(*value))
            .fold(0.0_f64, f64::max)
    }

    fn sample_nearest(&self, point: Point) -> f64 {
        let Some([x, y, z]) = self.grid_coordinates(point) else {
            return 0.0;
        };
        let xi = nearest_index(x, self.dims[0]);
        let yi = nearest_index(y, self.dims[1]);
        let zi = nearest_index(z, self.dims[2]);
        f64::from(self.density[self.index(xi, yi, zi)])
    }

    fn sample_trilinear(&self, point: Point) -> f64 {
        let Some([x, y, z]) = self.grid_coordinates(point) else {
            return 0.0;
        };
        let x_axis = axis_lerp(x, self.dims[0]);
        let y_axis = axis_lerp(y, self.dims[1]);
        let z_axis = axis_lerp(z, self.dims[2]);

        let mut value = 0.0;
        for (xi, x_weight) in [(x_axis.lower, 1.0 - x_axis.t), (x_axis.upper, x_axis.t)] {
            for (yi, y_weight) in [(y_axis.lower, 1.0 - y_axis.t), (y_axis.upper, y_axis.t)] {
                for (zi, z_weight) in [(z_axis.lower, 1.0 - z_axis.t), (z_axis.upper, z_axis.t)] {
                    value += x_weight
                        * y_weight
                        * z_weight
                        * f64::from(self.density[self.index(xi, yi, zi)]);
                }
            }
        }
        value
    }

    fn grid_coordinates(&self, point: Point) -> Option<[f64; 3]> {
        if !self.bounds.contains(point) {
            return None;
        }
        let extent = self.bounds.extent();
        Some([
            axis_grid_coordinate(point.x(), self.bounds.min.x(), extent.x(), self.dims[0]),
            axis_grid_coordinate(point.y(), self.bounds.min.y(), extent.y(), self.dims[1]),
            axis_grid_coordinate(point.z(), self.bounds.min.z(), extent.z(), self.dims[2]),
        ])
    }

    fn save_grid_with_optional_frame(
        &self,
        path: impl AsRef<Path>,
        frame_index: Option<usize>,
    ) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        writeln!(file, "{GRID_FILE_MAGIC}")?;
        writeln!(
            file,
            "dims {} {} {}",
            self.dims[0], self.dims[1], self.dims[2]
        )?;
        writeln!(
            file,
            "bounds {:.17} {:.17} {:.17} {:.17} {:.17} {:.17}",
            self.bounds.min.x(),
            self.bounds.min.y(),
            self.bounds.min.z(),
            self.bounds.max.x(),
            self.bounds.max.y(),
            self.bounds.max.z()
        )?;
        writeln!(file, "interpolation {}", self.interpolation.as_file_str())?;
        match frame_index {
            Some(frame_index) => writeln!(file, "frame {frame_index}")?,
            None => writeln!(file, "frame none")?,
        }
        file.write_all(GRID_FILE_DATA_MARKER.as_bytes())?;
        for value in &self.density {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }
}

impl DensityField for GridDensityField {
    fn density(&self, point: Point, _time: f64) -> f64 {
        match self.interpolation {
            GridInterpolation::Nearest => self.sample_nearest(point),
            GridInterpolation::Trilinear => self.sample_trilinear(point),
        }
    }

    fn max_density(&self) -> f64 {
        self.max_density
    }
}

#[derive(Clone, Copy)]
struct AxisLerp {
    lower: usize,
    upper: usize,
    t: f64,
}

fn validate_dims(dims: [usize; 3]) {
    assert!(
        dims.into_iter().all(|dim| dim > 0),
        "grid dimensions must be non-zero"
    );
    let _ = cell_count_for_dims(dims);
}

fn cell_count_for_dims(dims: [usize; 3]) -> usize {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .expect("grid dimensions overflow")
}

fn index_for_dims(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + dims[0] * (y + dims[1] * z)
}

fn cell_center(bounds: GridBounds, dims: [usize; 3], x: usize, y: usize, z: usize) -> Point {
    let extent = bounds.extent();
    Point::new(
        axis_cell_center(bounds.min.x(), extent.x(), dims[0], x),
        axis_cell_center(bounds.min.y(), extent.y(), dims[1], y),
        axis_cell_center(bounds.min.z(), extent.z(), dims[2], z),
    )
}

fn axis_cell_center(minimum: f64, extent: f64, dim: usize, index: usize) -> f64 {
    minimum + (usize_to_f64(index) + 0.5) * extent / usize_to_f64(dim)
}

fn axis_grid_coordinate(value: f64, minimum: f64, extent: f64, dim: usize) -> f64 {
    ((value - minimum) / extent) * usize_to_f64(dim) - 0.5
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn nearest_index(coord: f64, dim: usize) -> usize {
    coord.round().clamp(0.0, usize_to_f64(dim - 1)) as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn axis_lerp(coord: f64, dim: usize) -> AxisLerp {
    let coord = coord.clamp(0.0, usize_to_f64(dim - 1));
    let lower = coord.floor() as usize;
    let upper = (lower + 1).min(dim - 1);
    AxisLerp {
        lower,
        upper,
        t: coord - usize_to_f64(lower),
    }
}

fn positive_majorant(max_density: f64) -> f64 {
    if max_density.is_finite() && max_density > 0.0 {
        max_density
    } else {
        f64::MIN_POSITIVE
    }
}

fn sanitize_density_f32(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

#[allow(clippy::cast_possible_truncation)]
fn sanitize_density_f64(value: f64) -> f32 {
    if value.is_finite() && value > 0.0 {
        value.min(f64::from(f32::MAX)) as f32
    } else {
        0.0
    }
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("grid dimension should fit in u32"))
}

fn parse_grid_file(bytes: &[u8]) -> io::Result<(GridDensityMetadata, Vec<f32>)> {
    let (header, data) = split_grid_file(bytes)?;
    let metadata = parse_grid_metadata(header)?;
    let expected_bytes = checked_cell_count_for_dims(metadata.dims)?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| invalid_data("grid density byte length overflows"))?;
    if data.len() != expected_bytes {
        return Err(invalid_data(format!(
            "grid density payload has {} bytes, expected {expected_bytes}",
            data.len()
        )));
    }

    let density = data
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok((metadata, density))
}

fn split_grid_file(bytes: &[u8]) -> io::Result<(&str, &[u8])> {
    let marker = GRID_FILE_DATA_MARKER.as_bytes();
    let marker_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or_else(|| invalid_data("grid file missing data marker"))?;
    let header = std::str::from_utf8(&bytes[..marker_start])
        .map_err(|_| invalid_data("grid metadata header is not valid UTF-8"))?;
    Ok((header, &bytes[(marker_start + marker.len())..]))
}

fn parse_grid_metadata(header: &str) -> io::Result<GridDensityMetadata> {
    let mut lines = header.lines();
    if lines.next() != Some(GRID_FILE_MAGIC) {
        return Err(invalid_data("grid file magic does not match"));
    }

    let mut dims = None;
    let mut bounds = None;
    let mut interpolation = None;
    let mut frame_index = None;
    for line in lines {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        match key {
            "dims" => {
                let parsed = [
                    parse_usize_part(parts.next(), "grid width")?,
                    parse_usize_part(parts.next(), "grid height")?,
                    parse_usize_part(parts.next(), "grid depth")?,
                ];
                if parts.next().is_some() {
                    return Err(invalid_data("grid dims line has extra fields"));
                }
                checked_cell_count_for_dims(parsed)?;
                dims = Some(parsed);
            }
            "bounds" => {
                let values = [
                    parse_f64_part(parts.next(), "bounds min x")?,
                    parse_f64_part(parts.next(), "bounds min y")?,
                    parse_f64_part(parts.next(), "bounds min z")?,
                    parse_f64_part(parts.next(), "bounds max x")?,
                    parse_f64_part(parts.next(), "bounds max y")?,
                    parse_f64_part(parts.next(), "bounds max z")?,
                ];
                if parts.next().is_some() {
                    return Err(invalid_data("grid bounds line has extra fields"));
                }
                bounds = Some(parse_bounds(values)?);
            }
            "interpolation" => {
                let value = parts
                    .next()
                    .ok_or_else(|| invalid_data("grid interpolation line missing value"))?;
                if parts.next().is_some() {
                    return Err(invalid_data("grid interpolation line has extra fields"));
                }
                interpolation = Some(GridInterpolation::from_file_str(value)?);
            }
            "frame" => {
                let value = parts
                    .next()
                    .ok_or_else(|| invalid_data("grid frame line missing value"))?;
                if parts.next().is_some() {
                    return Err(invalid_data("grid frame line has extra fields"));
                }
                frame_index = match value {
                    "none" => Some(None),
                    value => Some(Some(value.parse::<usize>().map_err(|_| {
                        invalid_data("grid frame index must be a non-negative integer")
                    })?)),
                };
            }
            _ => return Err(invalid_data("unknown grid metadata key")),
        }
    }

    Ok(GridDensityMetadata {
        dims: dims.ok_or_else(|| invalid_data("grid metadata missing dims"))?,
        bounds: bounds.ok_or_else(|| invalid_data("grid metadata missing bounds"))?,
        interpolation: interpolation
            .ok_or_else(|| invalid_data("grid metadata missing interpolation"))?,
        frame_index: frame_index.ok_or_else(|| invalid_data("grid metadata missing frame"))?,
    })
}

fn parse_bounds(values: [f64; 6]) -> io::Result<GridBounds> {
    if !values.into_iter().all(f64::is_finite) {
        return Err(invalid_data("grid bounds must be finite"));
    }
    if !(values[3] > values[0] && values[4] > values[1] && values[5] > values[2]) {
        return Err(invalid_data(
            "grid bounds must have positive extent on every axis",
        ));
    }
    Ok(GridBounds {
        min: Point::new(values[0], values[1], values[2]),
        max: Point::new(values[3], values[4], values[5]),
    })
}

fn checked_cell_count_for_dims(dims: [usize; 3]) -> io::Result<usize> {
    if !dims.into_iter().all(|dim| dim > 0) {
        return Err(invalid_data("grid dimensions must be non-zero"));
    }
    dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .ok_or_else(|| invalid_data("grid dimensions overflow"))
}

fn parse_usize_part(value: Option<&str>, label: &str) -> io::Result<usize> {
    value
        .ok_or_else(|| invalid_data(format!("missing {label}")))?
        .parse::<usize>()
        .map_err(|_| invalid_data(format!("{label} must be a non-negative integer")))
}

fn parse_f64_part(value: Option<&str>, label: &str) -> io::Result<f64> {
    value
        .ok_or_else(|| invalid_data(format!("missing {label}")))?
        .parse::<f64>()
        .map_err(|_| invalid_data(format!("{label} must be a finite number")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    fn unit_bounds() -> GridBounds {
        GridBounds::new(Point::new(0.0, 0.0, 0.0), Point::new(1.0, 1.0, 1.0))
    }

    fn temp_raw_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "gartus_grid_density_{}_{}.raw",
            std::process::id(),
            NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn temp_grid_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "gartus_grid_density_{}_{}.gdf",
            std::process::id(),
            NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn grid_density_samples_exact_cell_centers() {
        let bounds = GridBounds::new(Point::new(0.0, 0.0, 0.0), Point::new(2.0, 1.0, 1.0));
        let grid = GridDensityField::from_fn(bounds, [2, 1, 1], Point::x);

        assert_close(grid.density(grid.cell_center(0, 0, 0), 0.0), 0.5);
        assert_close(grid.density(grid.cell_center(1, 0, 0), 0.0), 1.5);
    }

    #[test]
    fn grid_density_trilinear_interpolates_midpoint() {
        let grid = GridDensityField::new(
            unit_bounds(),
            [2, 2, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        );

        assert_close(grid.density(Point::new(0.5, 0.5, 0.5), 0.0), 3.5);
    }

    #[test]
    fn grid_density_nearest_uses_closest_cell_center() {
        let grid = GridDensityField::new(unit_bounds(), [2, 1, 1], vec![1.0, 3.0])
            .with_interpolation(GridInterpolation::Nearest);

        assert_close(grid.density(Point::new(0.26, 0.5, 0.5), 0.0), 1.0);
        assert_close(grid.density(Point::new(0.74, 0.5, 0.5), 0.0), 3.0);
    }

    #[test]
    fn grid_density_reports_computed_majorant() {
        let grid = GridDensityField::new(unit_bounds(), [4, 1, 1], vec![-1.0, f32::NAN, 2.5, 1.0]);

        assert_close(grid.computed_max_density(), 2.5);
        assert_close(grid.max_density(), 2.5);
        assert_eq!(grid.densities(), &[0.0, 0.0, 2.5, 1.0]);
    }

    #[test]
    fn grid_density_returns_zero_outside_bounds() {
        let grid = GridDensityField::new(unit_bounds(), [1, 1, 1], vec![4.0]);

        assert_close(grid.density(Point::new(-0.01, 0.5, 0.5), 0.0), 0.0);
        assert_close(grid.density(Point::new(0.5, 1.01, 0.5), 0.0), 0.0);
    }

    #[test]
    fn grid_density_raw_round_trip_preserves_samples() {
        let grid = GridDensityField::new(unit_bounds(), [2, 2, 1], vec![0.0, 0.25, 0.5, 0.75]);
        let path = temp_raw_path();

        grid.save_raw(&path).expect("raw save should work");
        let loaded =
            GridDensityField::load_raw(unit_bounds(), [2, 2, 1], &path).expect("raw load works");
        let _ = std::fs::remove_file(path);

        assert_eq!(loaded.dims(), [2, 2, 1]);
        assert_eq!(loaded.densities(), grid.densities());
        assert_close(loaded.max_density(), 0.75);
    }

    #[test]
    fn grid_density_metadata_round_trip_preserves_grid_and_frame() {
        let bounds = GridBounds::new(Point::new(-1.0, 0.0, 2.0), Point::new(1.0, 3.0, 4.0));
        let grid = GridDensityField::new(bounds, [2, 2, 1], vec![0.0, 0.25, 0.5, 0.75])
            .with_interpolation(GridInterpolation::Nearest);
        let path = temp_grid_path();

        grid.save_grid_with_frame(&path, 42)
            .expect("metadata grid save should work");
        let metadata =
            GridDensityField::load_grid_metadata(&path).expect("metadata load should work");
        let (loaded, loaded_metadata) =
            GridDensityField::load_grid_with_metadata(&path).expect("grid load should work");
        let _ = std::fs::remove_file(path);

        assert_eq!(metadata, loaded_metadata);
        assert_eq!(metadata.dims, [2, 2, 1]);
        assert_eq!(metadata.bounds, bounds);
        assert_eq!(metadata.interpolation, GridInterpolation::Nearest);
        assert_eq!(metadata.frame_index, Some(42));
        assert_eq!(loaded.densities(), grid.densities());
        assert_eq!(loaded.interpolation(), GridInterpolation::Nearest);
        assert_close(loaded.max_density(), 0.75);
    }

    #[test]
    fn grid_density_bakes_density_field_at_time() {
        let field = crate::graphics::raytracing::FnDensityField::new(4.0, |point: Point, time| {
            point.x() + time
        });
        let bounds = GridBounds::new(Point::new(0.0, 0.0, 0.0), Point::new(2.0, 1.0, 1.0));
        let grid = GridDensityField::from_density_field(bounds, [2, 1, 1], &field, 0.25);

        assert_close(grid.density(grid.cell_center(0, 0, 0), 0.0), 0.75);
        assert_close(grid.density(grid.cell_center(1, 0, 0), 0.0), 1.75);
    }
}

//! Runtime state for executing parsed MDL commands.

use super::{
    animation::KnobMap,
    ast::{Material, Vec3},
    executor::ExecutionError,
};
use crate::{
    gmath::{
        edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix, stack::MatrixStack,
    },
    graphics::{
        colors::Rgb,
        display::{Canvas, PolygonColorMode, ShadingMode as CanvasShadingMode},
        lighting::{Lighting, PointLight, ReflectionConstants, SurfaceMaterial},
    },
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[cfg(feature = "external")]
use {
    crate::{
        external::MaterialMesh,
        graphics::texture::{Texture, TextureFilter, TextureWrap},
    },
    std::sync::Arc,
};

#[cfg(feature = "external")]
#[derive(Debug, Clone, Default)]
pub(crate) struct AssetCaches {
    mesh_cache: HashMap<PathBuf, Arc<MaterialMesh>>,
    texture_cache: HashMap<PathBuf, Arc<Texture>>,
}

/// Rendering configuration for one MDL execution.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    width: u32,
    height: u32,
    line_color: Rgb,
    background: Rgb,
    wrapped: bool,
    display_enabled: bool,
    source_dir: Option<PathBuf>,
    save_enabled: bool,
    save_override: Option<PathBuf>,
    #[cfg(feature = "external")]
    texture_wrap: (TextureWrap, TextureWrap),
}

impl RenderConfig {
    /// Creates a render config with the `11_anim` MDL defaults:
    /// white draw color on a black background.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            line_color: Rgb::WHITE,
            background: Rgb::BLACK,
            wrapped: true,
            display_enabled: true,
            source_dir: None,
            save_enabled: true,
            save_override: None,
            #[cfg(feature = "external")]
            texture_wrap: (TextureWrap::Clamp, TextureWrap::Clamp),
        }
    }

    /// Creates a render config with a background color and line color.
    #[must_use]
    pub fn new_with_bg(width: u32, height: u32, line_color: Rgb, background: Rgb) -> Self {
        Self {
            width,
            height,
            line_color,
            background,
            wrapped: true,
            display_enabled: true,
            source_dir: None,
            save_enabled: true,
            save_override: None,
            #[cfg(feature = "external")]
            texture_wrap: (TextureWrap::Clamp, TextureWrap::Clamp),
        }
    }

    /// Enables or disables display commands.
    #[must_use]
    pub fn display_enabled(mut self, enabled: bool) -> Self {
        self.display_enabled = enabled;
        self
    }

    /// Sets whether canvas coordinates wrap around image edges.
    #[must_use]
    pub fn wrapped(mut self, wrapped: bool) -> Self {
        self.wrapped = wrapped;
        self
    }

    /// Sets the source directory used to resolve relative mesh paths.
    #[must_use]
    pub fn source_dir(mut self, source_dir: impl Into<PathBuf>) -> Self {
        self.source_dir = Some(source_dir.into());
        self
    }

    /// Enables or disables `save` commands.
    #[must_use]
    pub fn save_enabled(mut self, enabled: bool) -> Self {
        self.save_enabled = enabled;
        self
    }

    /// Redirects `save` commands to a fixed output path.
    #[must_use]
    pub fn save_override(mut self, output: impl Into<PathBuf>) -> Self {
        self.save_override = Some(output.into());
        self
    }

    /// Sets wrap behavior for runtime-loaded textures.
    #[cfg(feature = "external")]
    #[must_use]
    pub fn texture_wrap(mut self, wrap_s: TextureWrap, wrap_t: TextureWrap) -> Self {
        self.texture_wrap = (wrap_s, wrap_t);
        self
    }

    pub(crate) fn create_canvas(&self) -> Canvas {
        let mut canvas = Canvas::new_with_bg(self.width, self.height, self.background);
        canvas.line = self.line_color;
        canvas.wrapped = self.wrapped;
        canvas.set_shading_mode(CanvasShadingMode::Flat);
        canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
        canvas
    }
}

/// A reusable set of material constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialConstants {
    /// Reflection coefficients.
    pub material: Material,
    /// Object color.
    pub color: Vec3,
}

impl From<MaterialConstants> for SurfaceMaterial {
    fn from(constants: MaterialConstants) -> Self {
        // MDL Phong shading takes its hue from the reflection coefficients. The object color is
        // a flat/Gouraud fallback, so folding it into the diffuse albedo here would double-tint
        // Phong-authored material constants.
        constants.material.into()
    }
}

/// A point light declared in MDL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// Light color.
    pub color: Vec3,
    /// Light position.
    pub position: Vec3,
}

/// Camera state declared in MDL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Camera eye point.
    pub eye: Vec3,
    /// Camera aim point.
    pub aim: Vec3,
    /// Focal length.
    pub focal: f64,
}

/// One named runtime symbol.
#[derive(Debug, Clone, PartialEq)]
pub enum Symbol {
    /// Numeric knob value.
    Knob(f64),
    /// Saved knob-list values.
    KnobList(KnobMap),
    /// Material constants.
    Constants(MaterialConstants),
    /// Named point light.
    Light(Light),
    /// Saved coordinate-system matrix.
    CoordSystem(Matrix),
}

/// Mutable state used while executing one MDL program.
#[derive(Debug)]
pub struct Runtime {
    canvas: Canvas,
    canvas_baseline: CanvasBaseline,
    scene: SceneState,
    output: OutputState,
    scratch: ScratchGeometry,
    #[cfg(feature = "external")]
    mesh_cache: HashMap<PathBuf, Arc<MaterialMesh>>,
    #[cfg(feature = "external")]
    texture_cache: HashMap<PathBuf, Arc<Texture>>,
}

#[derive(Debug, Clone)]
struct CanvasBaseline {
    pixels: Vec<Rgb>,
    line: Rgb,
    lighting: Lighting,
    polygon_color_mode: PolygonColorMode,
    shading_mode: CanvasShadingMode,
}

#[derive(Debug)]
struct SceneState {
    stack: MatrixStack,
    symbols: HashMap<String, Symbol>,
    frame_knobs: HashMap<String, f64>,
    lights: Vec<Light>,
    ambient: Vec3,
    camera: Option<Camera>,
}

#[derive(Debug)]
struct OutputState {
    basename: String,
    frames: usize,
    generate_rayfiles: bool,
    display_enabled: bool,
    source_dir: Option<PathBuf>,
    save_enabled: bool,
    save_override: Option<PathBuf>,
    #[cfg(feature = "external")]
    texture_wrap: (TextureWrap, TextureWrap),
}

#[derive(Debug)]
struct ScratchGeometry {
    tmp_edge: EdgeMatrix,
    tmp_polygon: PolygonMatrix,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DrawState {
    line: Rgb,
    ambient_reflection: ReflectionConstants,
    diffuse_reflection: ReflectionConstants,
    specular_reflection: ReflectionConstants,
}

impl Runtime {
    /// Creates a new runtime from render configuration.
    #[must_use]
    pub fn new(config: &RenderConfig) -> Self {
        let canvas = config.create_canvas();
        Self::with_canvas(config, canvas)
    }

    /// Creates a new runtime from render configuration and an existing canvas.
    #[must_use]
    pub fn with_canvas(config: &RenderConfig, canvas: Canvas) -> Self {
        let output = OutputState::from_config(config);
        let canvas_baseline = CanvasBaseline::from_canvas(&canvas);
        Self {
            canvas,
            canvas_baseline,
            scene: SceneState::new(),
            output,
            scratch: ScratchGeometry::new(),
            #[cfg(feature = "external")]
            mesh_cache: HashMap::new(),
            #[cfg(feature = "external")]
            texture_cache: HashMap::new(),
        }
    }

    #[cfg(feature = "external")]
    #[must_use]
    pub(crate) fn with_asset_caches(config: &RenderConfig, caches: AssetCaches) -> Self {
        let mut runtime = Self::new(config);
        runtime.mesh_cache = caches.mesh_cache;
        runtime.texture_cache = caches.texture_cache;
        runtime
    }

    #[cfg(feature = "external")]
    #[must_use]
    pub(crate) fn asset_caches(&self) -> AssetCaches {
        AssetCaches {
            mesh_cache: self.mesh_cache.clone(),
            texture_cache: self.texture_cache.clone(),
        }
    }

    /// Returns the current canvas.
    #[must_use = "the returned canvas is only useful if inspected"]
    pub fn canvas(&self) -> &Canvas {
        &self.canvas
    }

    /// Returns the current canvas mutably.
    pub fn canvas_mut(&mut self) -> &mut Canvas {
        &mut self.canvas
    }

    /// Consumes the runtime and returns its canvas.
    #[must_use = "the canvas is dropped if the return value is ignored"]
    pub fn into_canvas(self) -> Canvas {
        self.canvas
    }

    #[cfg(feature = "filters")]
    pub(crate) fn replace_canvas(&mut self, canvas: Canvas) {
        self.canvas = canvas;
    }

    /// Returns the current coordinate-system stack top.
    #[must_use = "the returned transform is only useful if inspected"]
    pub fn top_transform(&self) -> &Matrix {
        self.scene.stack.top()
    }

    /// Returns the coordinate-system stack depth.
    #[must_use]
    pub fn stack_len(&self) -> usize {
        self.scene.stack.len()
    }

    /// Returns a named symbol.
    #[must_use]
    pub fn symbol(&self, name: &str) -> Option<&Symbol> {
        self.scene.symbols.get(name)
    }

    /// Returns a saved coordinate system.
    #[must_use]
    pub fn coord_system(&self, name: &str) -> Option<&Matrix> {
        match self.scene.symbols.get(name) {
            Some(Symbol::CoordSystem(matrix)) => Some(matrix),
            _ => None,
        }
    }

    /// Returns declared point lights.
    #[must_use]
    pub fn lights(&self) -> &[Light] {
        &self.scene.lights
    }

    /// Returns the ambient light color.
    #[must_use]
    pub fn ambient(&self) -> Vec3 {
        self.scene.ambient
    }

    /// Returns the camera state.
    #[must_use]
    pub fn camera(&self) -> Option<Camera> {
        self.scene.camera
    }

    /// Returns the animation basename.
    #[must_use]
    pub fn basename(&self) -> &str {
        &self.output.basename
    }

    /// Returns the requested frame count.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.output.frames
    }

    /// Returns whether ray-file generation was requested.
    #[must_use]
    pub fn generate_rayfiles(&self) -> bool {
        self.output.generate_rayfiles
    }

    /// Returns the source directory used for relative mesh paths.
    #[must_use]
    pub fn source_dir(&self) -> Option<&Path> {
        self.output.source_dir.as_deref()
    }

    pub(crate) fn push_stack(&mut self) {
        self.scene.stack.push();
    }

    pub(crate) fn pop_stack(&mut self) -> Result<(), ExecutionError> {
        self.scene
            .stack
            .pop()
            .map(|_| ())
            .ok_or(ExecutionError::StackUnderflow)
    }

    pub(crate) fn apply_transform(&mut self, transform: Matrix) {
        self.scene.stack.apply(transform);
    }

    pub(crate) fn set_top_identity(&mut self) {
        self.scene.stack.replace_top(Matrix::identity_matrix(4));
    }

    pub(crate) fn clear_canvas(&mut self) {
        self.canvas.clear_canvas();
    }

    pub(crate) fn reset(&mut self) {
        self.canvas.clear_canvas();
        self.canvas.set_lighting(Lighting::default());
        self.scene.reset();
        self.scratch.clear();
    }

    pub(crate) fn reset_for_frame(&mut self) {
        self.canvas_baseline.restore(&mut self.canvas);
        self.scene.reset();
        self.scene.frame_knobs.clear();
        self.output.reset_frame_state();
        self.scratch.clear();
    }

    pub(crate) fn set_knob(&mut self, name: String, value: f64) {
        self.scene.symbols.insert(name, Symbol::Knob(value));
    }

    pub(crate) fn seed_frame_knobs<'a>(
        &mut self,
        knobs: impl IntoIterator<Item = (&'a String, &'a f64)>,
    ) {
        self.scene.frame_knobs.clear();
        for (name, value) in knobs {
            self.scene.frame_knobs.insert(name.clone(), *value);
        }
    }

    pub(crate) fn knob_value(&self, knob: Option<&str>) -> Result<f64, ExecutionError> {
        let Some(name) = knob else {
            return Ok(1.0);
        };
        if let Some(value) = self.scene.frame_knobs.get(name) {
            return Ok(*value);
        }
        match self.scene.symbols.get(name) {
            Some(Symbol::Knob(value)) => Ok(*value),
            _ => Err(ExecutionError::UnknownKnob(name.to_string())),
        }
    }

    pub(crate) fn set_all_knobs(&mut self, value: f64) {
        for symbol in self.scene.symbols.values_mut() {
            if let Symbol::Knob(knob) = symbol {
                *knob = value;
            }
        }
    }

    pub(crate) fn save_knobs(&mut self, name: String) {
        let mut knobs = self
            .scene
            .symbols
            .iter()
            .filter_map(|(name, symbol)| match symbol {
                Symbol::Knob(value) => Some((name.clone(), *value)),
                Symbol::KnobList(_)
                | Symbol::Constants(_)
                | Symbol::Light(_)
                | Symbol::CoordSystem(_) => None,
            })
            .collect::<KnobMap>();
        knobs.extend(
            self.scene
                .frame_knobs
                .iter()
                .map(|(name, value)| (name.clone(), *value)),
        );
        self.scene.symbols.insert(name, Symbol::KnobList(knobs));
    }

    pub(crate) fn save_coord_system(&mut self, name: String) {
        self.scene
            .symbols
            .insert(name, Symbol::CoordSystem(self.scene.stack.top().clone()));
    }

    pub(crate) fn set_constants(&mut self, name: String, material: Material, color: Vec3) {
        self.scene.symbols.insert(
            name,
            Symbol::Constants(MaterialConstants { material, color }),
        );
    }

    pub(crate) fn set_ambient(&mut self, color: Vec3) {
        self.scene.ambient = color;
        self.canvas.lighting_mut().ambient = rgb_from_vec3(color);
    }

    pub(crate) fn add_light(&mut self, name: Option<String>, light: Light) {
        let first_user_light = self.scene.lights.is_empty();
        let point_light = PointLight::positional(
            crate::gmath::vector::Vector::new(light.position.x, light.position.y, light.position.z),
            rgb_from_vec3(light.color),
        );
        let lighting = self.canvas.lighting_mut();
        lighting.point_light = point_light;
        if first_user_light {
            lighting.point_lights.clear();
        }
        lighting.point_lights.push(point_light);
        if let Some(name) = name {
            self.scene.symbols.insert(name, Symbol::Light(light));
        }
        self.scene.lights.push(light);
    }

    pub(crate) fn set_camera(&mut self, eye: Vec3, aim: Vec3) {
        let focal = self.scene.camera.map_or(1.0, |camera| camera.focal);
        self.scene.camera = Some(Camera { eye, aim, focal });
    }

    pub(crate) fn set_focal(&mut self, focal: f64) {
        self.scene.camera = Some(match self.scene.camera {
            Some(camera) => Camera { focal, ..camera },
            None => Camera {
                eye: Vec3::new(0.0, 0.0, 1.0),
                aim: Vec3::new(0.0, 0.0, 0.0),
                focal,
            },
        });
    }

    pub(crate) fn set_basename(&mut self, basename: String) {
        self.output.basename = basename;
    }

    pub(crate) fn set_frames(&mut self, frames: usize) {
        self.output.frames = frames;
    }

    pub(crate) fn set_generate_rayfiles(&mut self) {
        self.output.generate_rayfiles = true;
    }

    pub(crate) fn transform_for(
        &self,
        coord_system: Option<&str>,
    ) -> Result<Matrix, ExecutionError> {
        let Some(name) = coord_system else {
            return Ok(self.scene.stack.top().clone());
        };
        match self.scene.symbols.get(name) {
            Some(Symbol::CoordSystem(matrix)) => Ok(matrix.clone()),
            _ => Err(ExecutionError::UnknownCoordSystem(name.to_string())),
        }
    }

    pub(crate) fn material_for(
        &self,
        constants: Option<&str>,
    ) -> Result<Option<MaterialConstants>, ExecutionError> {
        let Some(name) = constants else {
            return Ok(None);
        };
        match self.scene.symbols.get(name) {
            Some(Symbol::Constants(constants)) => Ok(Some(*constants)),
            _ => Err(ExecutionError::UnknownConstants(name.to_string())),
        }
    }

    pub(crate) fn apply_draw_state(&mut self, constants: Option<MaterialConstants>) -> DrawState {
        let lighting = self.canvas.lighting_ref();
        let previous = DrawState {
            line: self.canvas.line_color(),
            ambient_reflection: lighting.ambient_reflection,
            diffuse_reflection: lighting.diffuse_reflection,
            specular_reflection: lighting.specular_reflection,
        };

        if let Some(constants) = constants {
            let material = constants.material;
            let lighting = self.canvas.lighting_mut();
            lighting.ambient_reflection =
                ReflectionConstants::new(material.kar, material.kag, material.kab);
            lighting.diffuse_reflection =
                ReflectionConstants::new(material.kdr, material.kdg, material.kdb);
            lighting.specular_reflection =
                ReflectionConstants::new(material.ksr, material.ksg, material.ksb);

            self.canvas.set_line_pixel(rgb_from_vec3(constants.color));
        }

        previous
    }

    pub(crate) fn restore_draw_state(&mut self, previous: DrawState) {
        self.canvas.set_line_pixel(previous.line);
        let lighting = self.canvas.lighting_mut();
        lighting.ambient_reflection = previous.ambient_reflection;
        lighting.diffuse_reflection = previous.diffuse_reflection;
        lighting.specular_reflection = previous.specular_reflection;
    }

    pub(crate) fn with_tmp_edges<R>(&mut self, build: impl FnOnce(&mut EdgeMatrix) -> R) -> R {
        self.scratch.tmp_edge.clear();
        build(&mut self.scratch.tmp_edge)
    }

    pub(crate) fn transform_tmp_edges(&mut self, transform: &Matrix) {
        self.scratch.tmp_edge.apply_in_place(transform);
    }

    pub(crate) fn draw_tmp_edges(&mut self) {
        self.canvas.draw_lines(&self.scratch.tmp_edge);
    }

    pub(crate) fn with_tmp_polygons<R>(
        &mut self,
        build: impl FnOnce(&mut PolygonMatrix) -> R,
    ) -> R {
        self.scratch.tmp_polygon.clear();
        build(&mut self.scratch.tmp_polygon)
    }

    pub(crate) fn transform_tmp_polygons(&mut self, transform: &Matrix) {
        self.scratch.tmp_polygon.apply_in_place(transform);
    }

    pub(crate) fn draw_tmp_polygons(&mut self) {
        self.canvas.draw_polygons(&self.scratch.tmp_polygon);
    }

    #[cfg(feature = "external")]
    pub(crate) fn draw_tmp_polygons_with_vertex_normal_plan(
        &mut self,
        plan: &crate::graphics::draw::VertexNormalPlan,
    ) {
        self.canvas
            .draw_polygons_with_vertex_normal_plan(&self.scratch.tmp_polygon, Some(plan));
    }

    pub(crate) fn display(&self) -> Result<(), ExecutionError> {
        if self.output.display_enabled {
            self.canvas.display().map_err(ExecutionError::Io)?;
        }
        Ok(())
    }

    pub(crate) fn save(&self, filename: &str) -> Result<(), ExecutionError> {
        if !self.output.save_enabled {
            return Ok(());
        }

        let path = self
            .output
            .save_override
            .as_deref()
            .unwrap_or_else(|| Path::new(filename));
        self.save_to_path(path)
    }

    pub(crate) fn save_to_path(&self, path: &Path) -> Result<(), ExecutionError> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(ExecutionError::Io)?;
        }

        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("ppm") => self
                .canvas
                .save_binary(path_to_str(path)?)
                .map_err(ExecutionError::Io),
            _ => self
                .canvas
                .save_extension(path_to_str(path)?)
                .map_err(ExecutionError::Io),
        }
    }

    #[cfg(feature = "external")]
    pub(crate) fn resolve_mesh_path(&self, filename: &str, source_name: Option<&Path>) -> PathBuf {
        let path = Path::new(filename);
        if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(source_dir) = source_name
            .and_then(Path::parent)
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            source_dir.join(path)
        } else if let Some(source_dir) = &self.output.source_dir {
            source_dir.join(path)
        } else {
            path.to_path_buf()
        }
    }

    #[cfg(feature = "external")]
    pub(crate) fn load_mesh_cached(
        &mut self,
        path: &Path,
    ) -> Result<Arc<MaterialMesh>, ExecutionError> {
        if let Some(mesh) = self.mesh_cache.get(path) {
            return Ok(Arc::clone(mesh));
        }

        let mesh =
            crate::external::meshify_with_materials(path_to_str(path)?).map_err(|error| {
                ExecutionError::Mesh {
                    filename: path.display().to_string(),
                    error: error.to_string(),
                }
            })?;
        let mesh = Arc::new(mesh);
        self.mesh_cache
            .insert(path.to_path_buf(), Arc::clone(&mesh));
        Ok(mesh)
    }

    #[cfg(feature = "external")]
    pub(crate) fn load_texture_cached(
        &mut self,
        path: &Path,
    ) -> Result<Arc<Texture>, ExecutionError> {
        if let Some(texture) = self.texture_cache.get(path) {
            return Ok(Arc::clone(texture));
        }

        let image = crate::external::ppmify(path_to_str(path)?, false).map_err(|error| {
            ExecutionError::Texture {
                filename: path.display().to_string(),
                error: error.to_string(),
            }
        })?;
        let texture = Arc::new(
            Texture::from_canvas(image)
                .wrap(self.output.texture_wrap.0, self.output.texture_wrap.1)
                .filter(TextureFilter::Linear)
                .mipmapped(),
        );
        self.texture_cache
            .insert(path.to_path_buf(), Arc::clone(&texture));
        Ok(texture)
    }
}

impl SceneState {
    fn new() -> Self {
        Self {
            stack: MatrixStack::new(),
            symbols: HashMap::new(),
            frame_knobs: HashMap::new(),
            lights: Vec::new(),
            ambient: default_ambient(),
            camera: None,
        }
    }

    fn reset(&mut self) {
        self.stack = MatrixStack::new();
        self.symbols.clear();
        self.lights.clear();
        self.ambient = default_ambient();
        self.camera = None;
    }
}

impl CanvasBaseline {
    fn from_canvas(canvas: &Canvas) -> Self {
        Self {
            pixels: canvas.pixels().to_vec(),
            line: canvas.line_color(),
            lighting: canvas.lighting(),
            polygon_color_mode: canvas.polygon_color_mode(),
            shading_mode: canvas.shading_mode(),
        }
    }

    fn restore(&self, canvas: &mut Canvas) {
        canvas.restore_pixels(&self.pixels);
        canvas.set_line_pixel(self.line);
        canvas.set_lighting(self.lighting.clone());
        canvas.set_polygon_color_mode(self.polygon_color_mode);
        canvas.set_shading_mode(self.shading_mode);
    }
}

impl OutputState {
    fn from_config(config: &RenderConfig) -> Self {
        Self {
            basename: "frame".to_string(),
            frames: 1,
            generate_rayfiles: false,
            display_enabled: config.display_enabled,
            source_dir: config.source_dir.clone(),
            save_enabled: config.save_enabled,
            save_override: config.save_override.clone(),
            #[cfg(feature = "external")]
            texture_wrap: config.texture_wrap,
        }
    }

    fn reset_frame_state(&mut self) {
        self.basename = "frame".to_string();
        self.frames = 1;
        self.generate_rayfiles = false;
    }
}

impl ScratchGeometry {
    fn new() -> Self {
        Self {
            tmp_edge: EdgeMatrix::new(),
            tmp_polygon: PolygonMatrix::new(),
        }
    }

    fn clear(&mut self) {
        self.tmp_edge.clear();
        self.tmp_polygon.clear();
    }
}

fn path_to_str(path: &Path) -> Result<&str, ExecutionError> {
    path.to_str().ok_or_else(|| {
        ExecutionError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path is not valid UTF-8",
        ))
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn rgb_from_vec3(color: Vec3) -> Rgb {
    Rgb::new(
        color.x.round().clamp(0.0, 255.0) as u8,
        color.y.round().clamp(0.0, 255.0) as u8,
        color.z.round().clamp(0.0, 255.0) as u8,
    )
}

fn default_ambient() -> Vec3 {
    let ambient = Lighting::default().ambient;
    Vec3::new(
        f64::from(ambient.red),
        f64::from(ambient.green),
        f64::from(ambient.blue),
    )
}

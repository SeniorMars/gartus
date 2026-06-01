use crate::gmath::edge_matrix::EdgeMatrix;
use crate::gmath::matrix::Matrix;
use crate::gmath::polygon_matrix::PolygonMatrix;
use crate::graphics::{colors::Rgb, display::Canvas};
use std::collections::HashMap;
use std::fmt;
use std::{
    fs, io,
    path::{Path, PathBuf},
    str::FromStr,
};

/**
```text
Goes through the file named filename and performs all of the actions listed in that file.
The file follows the following format:
     Every command is a single character that takes up a line
     Any command that requires arguments must have those arguments
     in the second line. The commands are as follows:
        sphere: generate a sphere, apply the current coordinate system, draw -
            takes 4 arguments (cx, cy, cz, r)
        torus: generate a torus, apply the current coordinate system, draw -
            takes 5 arguments (cx, cy, cz, r1, r2)
        box: generate a rectangular prism, apply the current coordinate system, draw -
            takes 6 arguments (x, y, z, width, height, depth)
        mesh: load triangles from an OBJ or ASCII STL file, apply current CS, draw -
            takes 1 argument (file name)
            OBJ texture coordinates, normals, materials, groups, objects, and smoothing are ignored
        mesh_reverse: load triangles from an OBJ or ASCII STL file with winding reversed -
            takes 1 argument (file name)
        circle: generate a circle, apply the current coordinate system, draw -
            takes 4 arguments (cx, cy, cz, r)
        hermite: generate a hermite curve, apply the current CS, draw -
            takes 8 arguments (x0, y0, x1, y1, rx0, ry0, rx1, ry1)
        bezier: generate a third degree bezier curve, apply the current CS, draw -
            takes 8 arguments (x0, y0, x1, y1, x2, y2, x3, y3)
        beziern: generate a nth degree bezier curve, apply the current CS, draw -
            takes the n-degree and (n + 2) * 2 arguments for x, y points
        line: generate a line segment, apply the current coordinate system, draw -
            takes 6 arguments (x0, y0, z0, x1, y1, z1)
        scale: multiply the current top of the CS stack by a scale matrix -
            takes 3 arguments (sx, sy, sz)
        move: multiply the current top of the CS stack by a translation matrix -
            takes 3 arguments (tx, ty, tz)
        rotate: multiply the current top of the CS stack by a rotation matrix -
            takes 2 arguments (axis, theta) axis should be x y or z
        push: push a copy of the current top of the CS stack onto the stack
        pop: remove the top of the CS stack
        set: set a variable to a value
            takes 2 arguments (variable_name, value)
        reflect: multiply the current top of the CS stack by a reflection matrix -
            takes 1 argument (axis) - should be x y or z
        shear: multiply the current top of the CS stack by a shearing matrix -
            takes 3 arguments (axis, sh_factor, sh_factor) axis should be x, y, or z
        color: changes the line's color -- should be ONLY RGB or a color constant
            takes 3 argument representing the new color parameters
            takes 1 argument representing the new color constant
        filter: apply a filter to the canvas
            takes 1 or 2 argument representing the filter to be applied and the threshold
            options: "grayscale", "sepia", "reflect", "blur", "sobel", "invert", "edge",
                "emboss", "oil", "watercolor", "solarize", "black_and_white",
                "brightness", "posterize", "gaussian", "contrast", "bilateral",
                "unsharp", "histogram", "clahe", "canny", "floyd_steinberg"
        reset: reset CS stack, variables, and canvas
        display: show the current canvas
        save: save the current canvas to a file -
            takes 1 argument (file name)
        quit: end parsing
```
*/
#[derive(Debug)]
pub struct Parser {
    /// The name of the file being parsed
    file_name: String,
    /// The current top of the coordinate system stack
    trans_matrix: Matrix,
    /// The coordinate system stack for hierarchical modeling
    trans_stack: Vec<Matrix>,
    /// Symbol table for variables
    symbols: HashMap<String, f64>,
    /// The [Canvas] where the image will be drawn in
    canvas: Canvas,
    /// Whether parser `display` commands should spawn the external viewer.
    display_enabled: bool,
    /// Stack of directories for resolving relative include and mesh paths.
    source_dirs: Vec<PathBuf>,
    /// Temporary edge matrix to avoid allocations
    tmp_edge: EdgeMatrix,
    /// Temporary polygon matrix to avoid allocations
    tmp_polygon: PolygonMatrix,
}

#[derive(Debug)]
/// Custom Errors for Parser
#[allow(clippy::module_name_repetitions)]
pub enum ParserError {
    /// An I/O error while reading, saving, or displaying.
    Io(io::Error),
    /// An error that specifies errors with Matrices..
    MatrixError(usize, String, String),
    /// An error that specifies errors with given arguments
    ArgumentError(usize, String),
    /// An unknown command for the Parser
    CommandError(usize, String),
    /// An error while loading an external mesh file.
    MeshError(usize, String, String),
    /// Stack underflow error
    StackUnderflow(usize),
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::Io(err) => write!(f, "I/O error: {err}"),
            ParserError::MatrixError(line_num, line, matrix_type) => write!(
                f,
                "There was an error creating the {matrix_type} matrix with line: {line}:{line_num}"
            ),
            ParserError::CommandError(line_num, line) => {
                write!(f, "There was an unknown command: {line}:{line_num}")
            }
            ParserError::MeshError(line_num, file_name, err) => {
                write!(
                    f,
                    "Could not load mesh `{file_name}` at line {line_num}: {err}"
                )
            }
            ParserError::ArgumentError(line_num, line) => {
                write!(
                    f,
                    "Read spec. There was an error parsing the arguments in line: {line}:{line_num}"
                )
            }
            ParserError::StackUnderflow(line_num) => {
                write!(f, "Stack underflow at line {line_num}")
            }
        }
    }
}

impl std::error::Error for ParserError {}

#[allow(dead_code)]
impl Parser {
    /// Returns a parser that can parse through `file_name`
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `color` - A [Rgb] that represents the color of the drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Rgb};
    /// use crate::gartus::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let porygon = Parser::new("tests/porygon_script", 512, 512, &purplish);
    /// ```
    #[must_use]
    pub fn new(file_name: &str, width: u32, height: u32, color: &Rgb) -> Self {
        Self {
            file_name: file_name.to_string(),
            trans_matrix: Matrix::identity_matrix(4),
            trans_stack: Vec::new(),
            symbols: HashMap::new(),
            canvas: Canvas::new(width, height, *color),
            display_enabled: true,
            source_dirs: Vec::new(),
            tmp_edge: EdgeMatrix::new(),
            tmp_polygon: PolygonMatrix::new(),
        }
    }

    /// Returns a parser that can parse through `file_name` that starts with [Canvas] filled by `bg`.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `color` - A [Rgb] that represents the color of the drawing line
    /// * `bg` - A [Rgb] the default background color of self.canvas
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let outline = Rgb::new(235, 219, 178);
    /// let porygon = Parser::new_with_bg("./tests/porygon_script", 512, 512, &purplish, &outline);
    /// ```
    #[must_use]
    pub fn new_with_bg(file_name: &str, width: u32, height: u32, color: &Rgb, bg: &Rgb) -> Self {
        let mut canvas = Canvas::new_with_bg(width, height, *bg);
        canvas.line = *color;
        Self {
            file_name: file_name.to_string(),
            trans_matrix: Matrix::identity_matrix(4),
            trans_stack: Vec::new(),
            symbols: HashMap::new(),
            canvas,
            display_enabled: true,
            source_dirs: Vec::new(),
            tmp_edge: EdgeMatrix::new(),
            tmp_polygon: PolygonMatrix::new(),
        }
    }

    /// Get a mutable reference to the parser's canvas.
    pub fn canvas_mut(&mut self) -> &mut Canvas {
        &mut self.canvas
    }

    /// Get a reference to the parser's canvas.
    pub fn canvas(&self) -> &Canvas {
        &self.canvas
    }

    /// Enables or disables external display for parser `display` commands.
    pub fn set_display_enabled(&mut self, enabled: bool) {
        self.display_enabled = enabled;
    }

    /// Parses and runs through the commands in `self.file_name`
    ///
    /// # Errors
    ///
    /// Returns a `ParserError`
    pub fn parse_file(&mut self) -> Result<(), ParserError> {
        let path = PathBuf::from(&self.file_name);
        let contents = fs::read_to_string(&path).map_err(ParserError::Io)?;
        self.parse_source(&contents, path.parent())
    }

    /// Parses and runs through the commands in a string.
    ///
    /// # Errors
    ///
    /// Returns a `ParserError`
    pub fn parse_string(&mut self, contents: &str) -> Result<(), ParserError> {
        self.parse_source(contents, None)
    }

    fn parse_source(
        &mut self,
        contents: &str,
        source_dir: Option<&Path>,
    ) -> Result<(), ParserError> {
        if let Some(source_dir) = source_dir.filter(|dir| !dir.as_os_str().is_empty()) {
            self.source_dirs.push(source_dir.to_path_buf());
        }

        let result = self.parse_contents(contents);

        if source_dir.is_some_and(|dir| !dir.as_os_str().is_empty()) {
            self.source_dirs.pop();
        }

        result
    }

    fn parse_contents(&mut self, contents: &str) -> Result<(), ParserError> {
        let mut iter = contents.lines().enumerate();
        while let Some((line_num, line)) = iter.next() {
            let command = line.trim();
            match command {
                comment if comment.starts_with('#') => {}
                "" | "apply" => {}
                "quit" => return Ok(()),
                "ident" => self.trans_matrix = Matrix::identity_matrix(4),
                "push" => self.trans_stack.push(self.trans_matrix.clone()),
                "pop" => {
                    self.trans_matrix = self
                        .trans_stack
                        .pop()
                        .ok_or(ParserError::StackUnderflow(line_num))?;
                }
                "set" => self.parse_set(Self::next_arg_line(&mut iter, line_num, command)?)?,
                "display" => self.display()?,
                "clear" => self.canvas.clear_canvas(),
                "reset" => self.parse_reset(),
                _ => {
                    self.handle_command(&mut iter, line_num, command)?;
                }
            }
        }
        Ok(())
    }

    fn resolve_script_path(&self, file_name: &str) -> PathBuf {
        let path = Path::new(file_name.trim());
        if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(source_dir) = self.source_dirs.last() {
            source_dir.join(path)
        } else {
            path.to_path_buf()
        }
    }

    fn next_arg_line<'a>(
        iter: &mut std::iter::Enumerate<std::str::Lines<'a>>,
        command_line: usize,
        command: &str,
    ) -> Result<(usize, &'a str), ParserError> {
        iter.next().ok_or_else(|| {
            ParserError::ArgumentError(command_line, format!("missing arguments for `{command}`"))
        })
    }

    fn resolve_value(&self, s: &str) -> Option<f64> {
        if let Ok(v) = s.parse::<f64>() {
            return Some(v);
        }
        self.symbols.get(s).copied()
    }

    fn finite_value(&self, token: &str, line_num: usize, line: &str) -> Result<f64, ParserError> {
        let value = self
            .resolve_value(token)
            .ok_or_else(|| ParserError::ArgumentError(line_num, line.to_string()))?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(ParserError::ArgumentError(line_num, line.to_string()))
        }
    }

    fn parse_u8_value(&self, token: &str, line_num: usize, line: &str) -> Result<u8, ParserError> {
        let value = self.finite_value(token, line_num, line)?;
        if value.fract() != 0.0 || !(0.0..=255.0).contains(&value) {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Ok(value as u8)
    }

    fn parse_positive_usize_value(
        &self,
        token: &str,
        line_num: usize,
        line: &str,
    ) -> Result<usize, ParserError> {
        let value = self.finite_value(token, line_num, line)?;
        if value.fract() != 0.0 || value < 1.0 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }

        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        {
            if value > usize::MAX as f64 {
                return Err(ParserError::ArgumentError(line_num, line.to_string()));
            }
            Ok(value as usize)
        }
    }

    fn parse_args_resolved(
        &self,
        line: &str,
        expected: usize,
        line_num: usize,
    ) -> Result<Vec<f64>, ParserError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != expected {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let mut res = Vec::with_capacity(expected);
        for p in parts {
            let val = self.finite_value(p, line_num, line)?;
            res.push(val);
        }
        Ok(res)
    }

    fn handle_command(
        &mut self,
        iter: &mut std::iter::Enumerate<std::str::Lines<'_>>,
        cline_num: usize,
        command: &str,
    ) -> Result<(), ParserError> {
        match command {
            "line" => self.parse_line(Self::next_arg_line(iter, cline_num, command)?),
            "scale" => self.parse_scale(Self::next_arg_line(iter, cline_num, command)?),
            "move" => self.parse_move(Self::next_arg_line(iter, cline_num, command)?),
            "rotate" => self.parse_rotate(Self::next_arg_line(iter, cline_num, command)?),
            "reflect" => self.parse_reflect(Self::next_arg_line(iter, cline_num, command)?),
            "shear" => self.parse_shear(Self::next_arg_line(iter, cline_num, command)?),
            "color" => self.parse_color(Self::next_arg_line(iter, cline_num, command)?),
            "circle" => self.parse_circle(Self::next_arg_line(iter, cline_num, command)?),
            "hermite" => self.parse_hermite(Self::next_arg_line(iter, cline_num, command)?),
            "bezier" => self.parse_bezier(Self::next_arg_line(iter, cline_num, command)?),
            "beziern" => {
                let n_degree_line = Self::next_arg_line(iter, cline_num, "beziern degree")?;
                let n_degree_s = n_degree_line.1.trim();
                let n_degree = n_degree_s.parse::<usize>().map_err(|_| {
                    ParserError::ArgumentError(n_degree_line.0, n_degree_line.1.to_string())
                })?;

                let coords_line = Self::next_arg_line(iter, cline_num, "beziern coords")?;
                let mut coords = Vec::new();
                for p in coords_line.1.split_whitespace() {
                    let val = self.finite_value(p, coords_line.0, coords_line.1)?;
                    coords.push(val);
                }

                if coords.len() == (n_degree + 1) * 2 {
                    let mut x_points = Vec::with_capacity(n_degree + 1);
                    let mut y_points = Vec::with_capacity(n_degree + 1);
                    for i in (0..coords.len()).step_by(2) {
                        x_points.push(coords[i]);
                        y_points.push(coords[i + 1]);
                    }
                    let mut tmp = EdgeMatrix::new();
                    tmp.add_beziern(n_degree, &x_points, &y_points);
                    self.canvas.draw_lines(&tmp.apply(&self.trans_matrix));
                    Ok(())
                } else {
                    Err(ParserError::ArgumentError(
                        coords_line.0,
                        coords_line.1.to_string(),
                    ))
                }
            }
            "box" => self.parse_box(Self::next_arg_line(iter, cline_num, command)?),
            "sphere" => self.parse_sphere(Self::next_arg_line(iter, cline_num, command)?),
            "torus" => self.parse_torus(Self::next_arg_line(iter, cline_num, command)?),
            "cylinder" => self.parse_cylinder(Self::next_arg_line(iter, cline_num, command)?),
            "cone" => self.parse_cone(Self::next_arg_line(iter, cline_num, command)?),
            "pyramid" => self.parse_pyramid(Self::next_arg_line(iter, cline_num, command)?),
            "bezier_surface" => self.parse_bezier_surface(iter, cline_num),
            #[cfg(feature = "external")]
            "mesh" => self.parse_mesh(Self::next_arg_line(iter, cline_num, command)?),
            #[cfg(not(feature = "external"))]
            "mesh" => Self::parse_mesh_unavailable(Self::next_arg_line(iter, cline_num, command)?),
            #[cfg(feature = "external")]
            "mesh_reverse" => {
                self.parse_mesh_reverse(Self::next_arg_line(iter, cline_num, command)?)
            }
            #[cfg(not(feature = "external"))]
            "mesh_reverse" => {
                Self::parse_mesh_unavailable(Self::next_arg_line(iter, cline_num, command)?)
            }
            "include" => self.parse_include(Self::next_arg_line(iter, cline_num, command)?),
            "save" => self.save(Self::next_arg_line(iter, cline_num, command)?),
            "display" => self.display(),
            "filter" => self.parse_filter(Self::next_arg_line(iter, cline_num, command)?),
            _ => Err(ParserError::CommandError(cline_num, command.to_string())),
        }
    }

    fn parse_set(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let name = parts[0].to_string();
        let val = self.finite_value(parts[1], line_num, line)?;
        self.symbols.insert(name, val);
        Ok(())
    }

    fn parse_rotate(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let axis = parts[0];
        let theta = self.finite_value(parts[1], line_num, line)?;

        let rotate_matrix = match axis {
            "x" => Matrix::rotate_x(theta),
            "y" => Matrix::rotate_y(theta),
            "z" => Matrix::rotate_z(theta),
            _ => {
                return Err(ParserError::MatrixError(
                    line_num,
                    line.to_string(),
                    "rotate".to_string(),
                ));
            }
        };
        self.trans_matrix = &self.trans_matrix * &rotate_matrix;
        Ok(())
    }

    fn parse_reflect(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        if args.len() != 1 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let axis = args[0];
        let reflect_matrix = match axis {
            "x" => Matrix::reflect_xz(),
            "y" => Matrix::reflect_yz(),
            "z" => Matrix::reflect_xy(),
            _ => {
                return Err(ParserError::MatrixError(
                    line_num,
                    line.to_string(),
                    "reflect".to_string(),
                ));
            }
        };
        self.trans_matrix = &self.trans_matrix * &reflect_matrix;
        Ok(())
    }

    fn parse_shear(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        if args.len() != 3 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let axis = args[0];
        let sh_factor_one = self.finite_value(args[1], line_num, line)?;
        let sh_factor_two = self.finite_value(args[2], line_num, line)?;

        let shear_matrix = match axis {
            "x" => Matrix::shearing_x(sh_factor_one, sh_factor_two),
            "y" => Matrix::shearing_y(sh_factor_one, sh_factor_two),
            "z" => Matrix::shearing_z(sh_factor_one, sh_factor_two),
            _ => {
                return Err(ParserError::MatrixError(
                    line_num,
                    line.to_string(),
                    "shear".to_string(),
                ));
            }
        };
        self.trans_matrix = &self.trans_matrix * &shear_matrix;
        Ok(())
    }

    fn parse_color(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        match args.as_slice() {
            [name] => match Rgb::name_to_const(&name.to_lowercase()) {
                Some(color) => {
                    self.canvas.line = color;
                    Ok(())
                }
                None => Err(ParserError::ArgumentError(line_num, line.to_string())),
            },
            [r_s, g_s, b_s] => {
                let red = self.parse_u8_value(r_s, line_num, line)?;
                let green = self.parse_u8_value(g_s, line_num, line)?;
                let blue = self.parse_u8_value(b_s, line_num, line)?;
                self.canvas.line = Rgb::new(red, green, blue);
                Ok(())
            }
            _ => Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
    }

    fn parse_circle(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 4, line_num)?;
        self.tmp_edge.clear();
        self.tmp_edge
            .add_circle(args[0], args[1], args[2], args[3], 0.001);
        self.canvas
            .draw_lines(&self.tmp_edge.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_hermite(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 8, line_num)?;
        let p0 = (args[0], args[1]);
        let p1 = (args[2], args[3]);
        let r0 = (args[4], args[5]);
        let r1 = (args[6], args[7]);
        let mut tmp = EdgeMatrix::new();
        tmp.add_hermite(p0, p1, r0, r1);
        self.canvas.draw_lines(&tmp.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_bezier(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 8, line_num)?;
        let p0 = (args[0], args[1]);
        let p1 = (args[2], args[3]);
        let p2 = (args[4], args[5]);
        let p3 = (args[6], args[7]);
        let mut tmp = EdgeMatrix::new();
        tmp.add_bezier3(p0, p1, p2, p3);
        self.canvas.draw_lines(&tmp.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_scale(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 3, line_num)?;
        let dilate_matrix = Matrix::scale(args[0], args[1], args[2]);
        self.trans_matrix = &self.trans_matrix * &dilate_matrix;
        Ok(())
    }

    fn parse_move(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 3, line_num)?;
        let translation_matrix = Matrix::translate(args[0], args[1], args[2]);
        self.trans_matrix = &self.trans_matrix * &translation_matrix;
        Ok(())
    }

    fn parse_line(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 6, line_num)?;
        self.tmp_edge.clear();
        self.tmp_edge
            .push_edge(args[0], args[1], args[2], args[3], args[4], args[5]);
        self.canvas
            .draw_lines(&self.tmp_edge.apply(&self.trans_matrix));
        Ok(())
    }

    fn display(&mut self) -> Result<(), ParserError> {
        if self.display_enabled {
            self.canvas.display().map_err(ParserError::Io)?;
        }
        Ok(())
    }

    fn save(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        let extension = Path::new(file_name)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);

        match extension.as_deref() {
            Some("ppm") => self.canvas.save_binary(file_name).map_err(ParserError::Io),
            Some("png" | "jpg" | "jpeg") => self
                .canvas
                .save_extension(file_name)
                .map_err(ParserError::Io),
            _ => Err(ParserError::ArgumentError(line_num, file_name.to_string())),
        }
    }

    #[allow(dead_code)]
    fn parse_as<T: FromStr>(line: &str) -> Result<Vec<T>, T::Err> {
        line.split_whitespace().map(str::parse).collect()
    }

    /// Set the parser's color.
    pub fn set_color(&mut self, color: &Rgb) {
        self.canvas.line = *color;
    }

    /// Get a reference to the parser's trans matrix.
    pub fn trans_matrix(&self) -> &Matrix {
        &self.trans_matrix
    }

    fn parse_filter(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.len() {
            2 => {
                self.canvas = match args[0] {
                    "solarize" => {
                        let threshold = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.solarize(threshold)
                    }
                    "black_and_white" => {
                        let threshold = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.black_and_white(threshold)
                    }
                    "brightness" => {
                        let brightness = args[1]
                            .parse::<i16>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.adjust_brightness(brightness)
                    }
                    "posterize" => {
                        let levels = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.posterize(levels)
                    }
                    "gaussian" => {
                        let radius = args[1]
                            .parse::<f32>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.gaussian_blur(radius)
                    }
                    "contrast" => {
                        let constrast = args[1]
                            .parse::<f32>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.adjust_contrast(constrast)
                    }
                    _ => return Err(ParserError::ArgumentError(line_num, line.to_string())),
                };
                Ok(())
            }
            1 => {
                self.canvas = match args[0] {
                    "grayscale" => self.canvas.grayscale(),
                    "sepia" => self.canvas.sepia(),
                    "reflect" => self.canvas.reflect(),
                    "blur" => self.canvas.blur(),
                    "sobel" => self.canvas.sobel(),
                    "invert" => self.canvas.invert(),
                    "edge" => self.canvas.laplacian_edge_detection(),
                    "emboss" => self.canvas.emboss(),
                    "oil" => self.canvas.oil_painting(),
                    "watercolor" => self.canvas.watercolor(),
                    "bilateral" => self.canvas.bilateral_filter(2, 3.0, 32.0),
                    "unsharp" => self.canvas.unsharp_mask(1.0, 1.0),
                    "histogram" | "histogram_equalization" => self.canvas.histogram_equalization(),
                    "clahe" => self.canvas.clahe(32, 16),
                    "canny" => self.canvas.canny(40, 100),
                    "floyd_steinberg" | "floyd" => self.canvas.floyd_steinberg_dither(),
                    _ => return Err(ParserError::ArgumentError(line_num, line.to_string())),
                };
                Ok(())
            }
            _ => Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
    }

    fn parse_reset(&mut self) {
        self.canvas.clear_canvas();
        self.trans_matrix = Matrix::identity_matrix(4);
        self.trans_stack.clear();
        self.symbols.clear();
    }

    fn parse_box(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 6, line_num)?;
        self.tmp_polygon.clear();
        self.tmp_polygon
            .add_box((args[0], args[1], args[2]), args[3], args[4], args[5]);
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_sphere(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 4, line_num)?;
        self.tmp_polygon.clear();
        self.tmp_polygon
            .add_sphere((args[0], args[1], args[2]), args[3], 24);
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_torus(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.tmp_polygon.clear();
        self.tmp_polygon
            .add_torus((args[0], args[1], args[2]), args[3], args[4], 24);
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_cylinder(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.tmp_polygon.clear();
        self.tmp_polygon
            .add_cylinder((args[0], args[1], args[2]), args[3], args[4], 24);
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_cone(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.tmp_polygon.clear();
        self.tmp_polygon
            .add_cone((args[0], args[1], args[2]), args[3], args[4], 24);
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_pyramid(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.tmp_polygon.clear();
        self.tmp_polygon
            .add_pyramid((args[0], args[1], args[2]), args[3], args[4]);
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    fn parse_bezier_surface(
        &mut self,
        iter: &mut std::iter::Enumerate<std::str::Lines<'_>>,
        cline_num: usize,
    ) -> Result<(), ParserError> {
        let steps_line = Self::next_arg_line(iter, cline_num, "bezier_surface steps")?;
        let steps =
            self.parse_positive_usize_value(steps_line.1.trim(), steps_line.0, steps_line.1)?;

        let mut all_coords = Vec::with_capacity(48);
        while all_coords.len() < 48 {
            let next = Self::next_arg_line(iter, cline_num, "bezier_surface coords")?;
            for p in next.1.split_whitespace() {
                let val = self.finite_value(p, next.0, next.1)?;
                all_coords.push(val);
            }
        }

        let mut controls = [[(0.0, 0.0, 0.0); 4]; 4];
        for (i, row) in controls.iter_mut().enumerate() {
            for (j, control) in row.iter_mut().enumerate() {
                let base = (i * 4 + j) * 3;
                *control = (all_coords[base], all_coords[base + 1], all_coords[base + 2]);
            }
        }

        let mut tmp = PolygonMatrix::new();
        tmp.add_bezier_surface(controls, steps);
        self.canvas.draw_polygons(&tmp.apply(&self.trans_matrix));
        Ok(())
    }

    #[cfg(feature = "external")]
    fn parse_mesh(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        let path = self.resolve_script_path(file_name);
        let mut tmp = PolygonMatrix::new();
        crate::external::add_mesh(path.to_string_lossy().as_ref(), &mut tmp).map_err(|err| {
            ParserError::MeshError(line_num, path.display().to_string(), err.to_string())
        })?;
        self.canvas.draw_polygons(&tmp.apply(&self.trans_matrix));
        Ok(())
    }

    #[cfg(feature = "external")]
    fn parse_mesh_reverse(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        let path = self.resolve_script_path(file_name);
        self.tmp_polygon.clear();
        crate::external::add_mesh(path.to_string_lossy().as_ref(), &mut self.tmp_polygon).map_err(
            |err| ParserError::MeshError(line_num, path.display().to_string(), err.to_string()),
        )?;
        self.tmp_polygon.reverse_winding();
        self.canvas
            .draw_polygons(&self.tmp_polygon.apply(&self.trans_matrix));
        Ok(())
    }

    #[cfg(not(feature = "external"))]
    fn parse_mesh_unavailable(line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        Err(ParserError::MeshError(
            line_num,
            file_name.trim().to_string(),
            "mesh command requires the `external` feature".to_string(),
        ))
    }

    fn parse_include(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (_line_num, file_name) = line;
        let path = self.resolve_script_path(file_name);
        let contents = fs::read_to_string(&path).map_err(ParserError::Io)?;
        self.parse_source(&contents, path.parent())
    }
}

#[ignore = "Display test, not meant for CI"]
#[test]
fn parser_test() {
    let mut test = Parser::new("./tests/script_testing", 500, 500, &Rgb::new(0, 255, 0));
    test.parse_file().expect("Script is valid");
}

#[cfg(test)]
mod tests {
    use super::{Parser, ParserError};
    use crate::graphics::colors::Rgb;
    use std::fs;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gartus-parser-{name}-{}.cg", std::process::id()))
    }

    fn temp_file_with_extension(name: &str, extension: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gartus-parser-{name}-{}.{}",
            std::process::id(),
            extension
        ))
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gartus-parser-{name}-{}", std::process::id()))
    }

    #[test]
    fn missing_file_returns_io_error() {
        let mut parser = Parser::new("/definitely/not/a/real/script.cg", 2, 2, &Rgb::GREEN);
        assert!(matches!(parser.parse_file(), Err(ParserError::Io(_))));
    }

    #[test]
    fn missing_argument_line_returns_parser_error() {
        let path = temp_file("missing-arg");
        fs::write(&path, "line\n").expect("write temp script");

        let mut parser = Parser::new(path.to_str().expect("utf8 path"), 2, 2, &Rgb::GREEN);
        let error = parser.parse_file().expect_err("script should fail");

        assert!(matches!(error, ParserError::ArgumentError(0, _)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn beziern_degree_must_be_integer() {
        let path = temp_file("beziern-degree");
        fs::write(&path, "beziern\n3.7\n0 0 1 1 2 2 3 3\n").expect("write temp script");

        let mut parser = Parser::new(path.to_str().expect("utf8 path"), 2, 2, &Rgb::GREEN);
        let error = parser.parse_file().expect_err("script should fail");

        assert!(matches!(error, ParserError::ArgumentError(1, _)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_push_pop() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("push\nmove\n10 10 10\npop")
            .expect("push pop valid");
        assert_eq!(
            parser.trans_matrix(),
            &crate::gmath::matrix::Matrix::identity_matrix(4)
        );
    }

    #[test]
    fn test_variables() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("set\nx 100\nmove\nx 0 0")
            .expect("set valid");
        assert!((parser.trans_matrix().get(0, 3) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn transform_order_is_top_times_t() {
        // move then scale: CS = I*T*S. Applied to point [1,0,0]: scale first → [2,0,0], then translate → [12,0,0]
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("move\n10 0 0\nscale\n2 2 2")
            .expect("valid");
        let m = parser.trans_matrix();
        // T(10,0,0) * S(2,2,2) = [[2,0,0,10],[0,2,0,0],[0,0,2,0],[0,0,0,1]]
        assert!((m.get(0, 0) - 2.0).abs() < 1e-9);
        assert!((m.get(0, 3) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn shapes_draw_immediately_to_canvas() {
        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser = Parser::new_with_bg("test", 100, 100, &fg, &bg);
        // Move sphere into canvas center and draw it
        parser
            .parse_string("move\n50 50 0\nsphere\n0 0 0 20")
            .expect("valid");
        let has_fg = parser.canvas().pixels().contains(&fg);
        assert!(has_fg, "sphere should draw to canvas immediately");
    }

    #[test]
    fn push_pop_restores_coordinate_system() {
        // Shapes drawn in different CS should produce different results
        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser = Parser::new_with_bg("test", 200, 200, &fg, &bg);

        // Draw a box at (20, 100, 0) then push, move to (150, 100, 0), draw another, pop
        parser
            .parse_string(
                "move\n20 100 0\nbox\n0 0 0 10 10 1\npush\nmove\n130 0 0\nbox\n0 0 0 10 10 1\npop",
            )
            .expect("valid");

        // After pop, trans_matrix should be back to T(20,100,0)
        let m = parser.trans_matrix();
        assert!((m.get(0, 3) - 20.0).abs() < 1e-9);
        assert!((m.get(1, 3) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_include_command() {
        let path = temp_file("include-target");
        fs::write(&path, "move\n5 5 0\nline\n0 0 0 1 1 0\n").expect("write temp script");

        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser = Parser::new_with_bg("test", 20, 20, &fg, &bg);
        parser
            .parse_string(&format!("include\n{}", path.to_str().unwrap()))
            .expect("include valid");

        // Line was drawn: canvas should have fg pixels
        let has_fg = parser.canvas().pixels().contains(&fg);
        assert!(has_fg, "included script should draw to canvas");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn include_paths_resolve_relative_to_script_file() {
        let dir = temp_dir("relative-include");
        fs::create_dir_all(&dir).expect("create temp dir");
        let main_path = dir.join("main.cg");
        let child_path = dir.join("child.cg");
        fs::write(&main_path, "include\nchild.cg\n").expect("write main script");
        fs::write(&child_path, "move\n1 1 0\nline\n0 0 0 1 1 0\n").expect("write child script");

        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser =
            Parser::new_with_bg(main_path.to_str().expect("utf8 path"), 20, 20, &fg, &bg);
        parser.parse_file().expect("include valid");

        let has_fg = parser.canvas().pixels().contains(&fg);
        assert!(has_fg, "relative-included script should draw to canvas");
        let _ = fs::remove_file(main_path);
        let _ = fs::remove_file(child_path);
        let _ = fs::remove_dir(dir);
    }

    #[cfg(feature = "external")]
    #[test]
    fn test_mesh_command() {
        let path = temp_file_with_extension("mesh", "obj");
        // Triangle facing viewer (counterclockwise from front) centered near canvas
        fs::write(&path, "v 40 40 0\nv 60 40 0\nv 50 60 0\nf 1 2 3\n").expect("write temp obj");

        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser = Parser::new_with_bg("test", 100, 100, &fg, &bg);
        parser
            .parse_string(&format!("mesh\n{}", path.to_str().unwrap()))
            .expect("mesh valid");

        let has_fg = parser.canvas().pixels().contains(&fg);
        assert!(has_fg, "mesh should draw to canvas immediately");
        let _ = fs::remove_file(path);
    }

    #[cfg(feature = "external")]
    #[test]
    fn mesh_paths_resolve_relative_to_script_file() {
        let dir = temp_dir("relative-mesh");
        fs::create_dir_all(&dir).expect("create temp dir");
        let script_path = dir.join("main.cg");
        let mesh_path = dir.join("triangle.obj");
        fs::write(&script_path, "mesh\ntriangle.obj\n").expect("write script");
        fs::write(&mesh_path, "v 40 40 0\nv 60 40 0\nv 50 60 0\nf 1 2 3\n").expect("write obj");

        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser =
            Parser::new_with_bg(script_path.to_str().expect("utf8 path"), 100, 100, &fg, &bg);
        parser.parse_file().expect("mesh valid");

        let has_fg = parser.canvas().pixels().contains(&fg);
        assert!(has_fg, "relative-path mesh should draw to canvas");
        let _ = fs::remove_file(script_path);
        let _ = fs::remove_file(mesh_path);
        let _ = fs::remove_dir(dir);
    }

    #[cfg(feature = "external")]
    #[test]
    fn nested_include_mesh_paths_resolve_relative_to_included_script() {
        let dir = temp_dir("nested-include-mesh");
        let scripts_dir = dir.join("scripts");
        let subdir = scripts_dir.join("sub");
        let meshes_dir = scripts_dir.join("meshes");
        fs::create_dir_all(&subdir).expect("create script dir");
        fs::create_dir_all(&meshes_dir).expect("create mesh dir");

        let main_path = scripts_dir.join("main.cg");
        let child_path = subdir.join("child.cg");
        let mesh_path = meshes_dir.join("triangle.obj");
        fs::write(&main_path, "include\nsub/child.cg\n").expect("write main script");
        fs::write(&child_path, "mesh\n../meshes/triangle.obj\n").expect("write child script");
        fs::write(&mesh_path, "v 40 40 0\nv 60 40 0\nv 50 60 0\nf 1 2 3\n").expect("write obj");

        let bg = Rgb::new(0, 0, 0);
        let fg = Rgb::new(255, 255, 255);
        let mut parser =
            Parser::new_with_bg(main_path.to_str().expect("utf8 path"), 100, 100, &fg, &bg);
        parser.parse_file().expect("nested include mesh valid");

        let has_fg = parser.canvas().pixels().contains(&fg);
        assert!(
            has_fg,
            "nested-include relative-path mesh should draw to canvas"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(not(feature = "external"))]
    #[test]
    fn mesh_command_reports_missing_external_feature() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);

        let error = parser
            .parse_string("mesh\ntriangle.obj")
            .expect_err("mesh should require external feature");

        assert!(matches!(error, ParserError::MeshError(1, _, _)));
        assert!(error.to_string().contains("external"));
    }

    #[test]
    fn color_rejects_non_byte_values() {
        for value in ["300", "-1", "1.5", "NaN"] {
            let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
            let error = parser
                .parse_string(&format!("color\n{value} 0 0"))
                .expect_err("invalid color component should fail");

            assert!(matches!(error, ParserError::ArgumentError(1, _)));
        }
    }

    #[test]
    fn parse_ident_resets_transform_matrix() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("scale\n2 2 2\nrotate\nz 45\nident")
            .expect("ident valid");
        assert_eq!(
            parser.trans_matrix(),
            &crate::gmath::matrix::Matrix::identity_matrix(4)
        );
    }

    #[test]
    fn parse_apply_is_noop() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("scale\n2 2 2\napply")
            .expect("apply valid");
        let m = parser.trans_matrix();
        assert!((m.get(0, 0) - 2.0).abs() < 1e-9);
        assert!((m.get(1, 1) - 2.0).abs() < 1e-9);
        assert!((m.get(2, 2) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn set_accepts_previous_symbol_values() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("set\nbase 3\nset\nscale_x base")
            .expect("set should resolve existing symbol");

        parser
            .parse_string("move\nscale_x 0 0")
            .expect("move should use symbol from set");
        assert!((parser.trans_matrix().get(0, 3) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn bezier_surface_steps_must_be_positive_integer() {
        for value in ["0", "-1", "1.5", "NaN"] {
            let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
            let error = parser
                .parse_string(&format!("bezier_surface\n{value}"))
                .expect_err("invalid surface step count should fail");

            assert!(matches!(error, ParserError::ArgumentError(1, _)));
        }
    }
}

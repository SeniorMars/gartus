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
        sphere: add a sphere to the polygon matrix -
            takes 4 arguments (cx, cy, cz, r)
        torus: add a torus to the polygon matrix -
            takes 5 arguments (cx, cy, cz, r1, r2)
        box: add a rectangular prism to the polygon matrix -
            takes 6 arguments (x, y, z, width, height, depth)
        mesh: add triangles from an OBJ or ASCII STL file to the polygon matrix -
            takes 1 argument (file name)
            OBJ texture coordinates, normals, materials, groups, objects, and smoothing are ignored
        mesh_reverse: add triangles from an OBJ or ASCII STL file with winding reversed -
            takes 1 argument (file name)
        reverse_winding: reverse all triangle winding in the polygon matrix
        circle: add a circle to the edge matrix -
            takes 4 arguments (cx, cy, cz, r)
        hermite: add a hermite curve to the edge matrix -
            takes 8 arguments (x0, y0, x1, y1, rx0, ry0, rx1, ry1)
        bezier: add a third degree bezier curve to the edge matrix -
            takes 8 arguments (x0, y0, x1, y1, x2, y2, x3, y3)
        beziern: add a nth degree bezier curve to the edge matrix -
            takes the n-degree and (n + 2) * 2 arguments for points for x, y i.e., "n x0 y0 x1 y1 x2 y2 ... xn yn"
        line: add a line to the edge matrix -
            takes 6 arguemnts (x0, y0, z0, x1, y1, z1)
        ident: set the transform matrix to the identity matrix -
        scale: create a scale matrix,
            then multiply the transform matrix by the scale matrix -
            takes 3 arguments (sx, sy, sz)
        move: create a translation matrix,
            then multiply the transform matrix by the translation matrix -
            takes 3 arguments (tx, ty, tz)
        rotate: create a rotation matrix,
            then multiply the transform matrix by the rotation matrix -
            takes 2 arguments (axis, theta) axis should be x y or z
        push: push a copy of the current transform matrix onto the stack
        pop: pop the top matrix from the stack and set it as the current transform matrix
        set: set a variable to a value
            takes 2 arguments (variable_name, value)
        reflect: create a reflection matrix,
            then multiply the transform matrix by the rotation matrix -
            takes a argument (axis) - should be x y or z
        shear: create a shearing matrix,
            then multiply the transform matrix by the shearing matrix -
            takes 3 arguments (axis, sh_factor, sh_factor)  axis should be x, y, or z
        color: changes the line's color -- should be ONLY RGB or a color constant
            takes 3 argument representing the new color parameters
            takes 1 argument representing the new color constant
        filter: apply a filter to the canvas
            takes 1 or 2 argument representing the filter to be applied and the threshold
            options: "grayscale", "sepia", "reflect", "blur", "sobel", "invert", "edge",
                "emboss", "oil", "watercolor", "solarize", "black_and_white",
                "brightness", "posterize", "gaussian", "contrast", "bilateral",
                "unsharp", "histogram", "clahe", "canny", "floyd_steinberg"
        apply: apply the current transformation matrix to the edge and polygon matrices
        clear: clear the edge and polygon matrices and the canvas
        reset: reset transformation matrix, clear matrices, stack, variables and canvas
        display: clear the screen, then
            draw the lines and polygons to the screen
            display the screen
        save: clear the screen, then
            draw the lines and polygons to the screen
            save the screen to a file -
            takes 1 argument (file name)
        quit: end parsing
```
*/
#[derive(Debug)]
pub struct Parser {
    /// The name of the file being parsed
    file_name: String,
    /// The [`EdgeMatrix`] where points will be appended to draw onto the [Canvas]
    edge_matrix: EdgeMatrix,
    /// The [`PolygonMatrix`] where triangles will be appended to draw onto the [Canvas]
    polygon_matrix: PolygonMatrix,
    /// First edge column that still needs the next parser `apply`.
    edge_apply_start: usize,
    /// First polygon column that still needs the next parser `apply`.
    polygon_apply_start: usize,
    /// The [Matrix] that transformations will be applied to
    trans_matrix: Matrix,
    /// The transformation stack for hierarchical modeling
    trans_stack: Vec<Matrix>,
    /// Symbol table for variables
    symbols: HashMap<String, f64>,
    /// The [Canvas] where the image will be drawn in
    canvas: Canvas,
    /// Whether parser `display` commands should spawn the external viewer.
    display_enabled: bool,
    /// Stack of directories for resolving relative include and mesh paths.
    source_dirs: Vec<PathBuf>,
    /// Whether the canvas needs to be rerendered from the edge matrix.
    canvas_dirty: bool,
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
            edge_matrix: EdgeMatrix::new(),
            polygon_matrix: PolygonMatrix::new(),
            edge_apply_start: 0,
            polygon_apply_start: 0,
            trans_matrix: Matrix::identity_matrix(4),
            trans_stack: Vec::new(),
            symbols: HashMap::new(),
            canvas: Canvas::new(width, height, *color),
            display_enabled: true,
            source_dirs: Vec::new(),
            canvas_dirty: true,
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
            edge_matrix: EdgeMatrix::new(),
            polygon_matrix: PolygonMatrix::new(),
            edge_apply_start: 0,
            polygon_apply_start: 0,
            trans_matrix: Matrix::identity_matrix(4),
            trans_stack: Vec::new(),
            symbols: HashMap::new(),
            canvas,
            display_enabled: true,
            source_dirs: Vec::new(),
            canvas_dirty: true,
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

    /// Returns true if the canvas needs to be rerendered.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.canvas_dirty
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
                "" => {}
                "quit" => return Ok(()),
                "push" => self.trans_stack.push(self.trans_matrix.clone()),
                "pop" => {
                    self.trans_matrix = self
                        .trans_stack
                        .pop()
                        .ok_or(ParserError::StackUnderflow(line_num))?;
                }
                "set" => self.parse_set(Self::next_arg_line(&mut iter, line_num, command)?)?,
                "ident" => self.trans_matrix = Matrix::identity_matrix(4),
                "apply" => self.parse_apply(),
                "display" => self.display()?,
                "clear" => {
                    self.edge_matrix = EdgeMatrix::new();
                    self.polygon_matrix = PolygonMatrix::new();
                    self.edge_apply_start = 0;
                    self.polygon_apply_start = 0;
                    self.canvas.clear_canvas();
                    self.canvas_dirty = false;
                }
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
                    self.edge_matrix.add_beziern(n_degree, &x_points, &y_points);
                    self.canvas_dirty = true;
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
            "reverse_winding" => {
                self.polygon_matrix.reverse_winding();
                self.canvas_dirty = true;
                Ok(())
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
        let val = parts[1]
            .parse::<f64>()
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
        if !val.is_finite() {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
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
        self.trans_matrix = &rotate_matrix * &self.trans_matrix;
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
        self.trans_matrix = &reflect_matrix * &self.trans_matrix;
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
        self.trans_matrix = &shear_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_color(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        match args.as_slice() {
            [name] => match Rgb::name_to_const(&name.to_lowercase()) {
                Some(color) => {
                    self.canvas.line = color;
                    self.canvas_dirty = true;
                    Ok(())
                }
                None => Err(ParserError::ArgumentError(line_num, line.to_string())),
            },
            [r_s, g_s, b_s] => {
                let red = self.parse_u8_value(r_s, line_num, line)?;
                let green = self.parse_u8_value(g_s, line_num, line)?;
                let blue = self.parse_u8_value(b_s, line_num, line)?;
                self.canvas.line = Rgb::new(red, green, blue);
                self.canvas_dirty = true;
                Ok(())
            }
            _ => Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
    }

    fn parse_circle(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 4, line_num)?;
        self.edge_matrix
            .add_circle(args[0], args[1], args[2], args[3], 0.001);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_hermite(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 8, line_num)?;
        let p0 = (args[0], args[1]);
        let p1 = (args[2], args[3]);
        let r0 = (args[4], args[5]);
        let r1 = (args[6], args[7]);
        self.edge_matrix.add_hermite(p0, p1, r0, r1);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_bezier(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 8, line_num)?;
        let p0 = (args[0], args[1]);
        let p1 = (args[2], args[3]);
        let p2 = (args[4], args[5]);
        let p3 = (args[6], args[7]);
        self.edge_matrix.add_bezier3(p0, p1, p2, p3);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_scale(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 3, line_num)?;
        let dilate_matrix = Matrix::scale(args[0], args[1], args[2]);
        self.trans_matrix = &dilate_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_move(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 3, line_num)?;
        let translation_matrix = Matrix::translate(args[0], args[1], args[2]);
        self.trans_matrix = &translation_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_line(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 6, line_num)?;
        self.edge_matrix
            .push_edge(args[0], args[1], args[2], args[3], args[4], args[5]);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_apply(&mut self) {
        self.edge_matrix
            .apply_from_col_mut(self.edge_apply_start, &self.trans_matrix);
        self.polygon_matrix
            .apply_from_col_mut(self.polygon_apply_start, &self.trans_matrix);
        self.edge_apply_start = self.edge_matrix.cols();
        self.polygon_apply_start = self.polygon_matrix.cols();
        self.canvas_dirty = true;
    }

    fn render_scene(&mut self) {
        self.canvas.clear_canvas();
        self.canvas.try_draw_lines(&self.edge_matrix);
        self.canvas.draw_polygons(&self.polygon_matrix);
        self.canvas_dirty = false;
    }

    fn ensure_rendered(&mut self) {
        if self.canvas_dirty {
            self.render_scene();
        }
    }

    fn display(&mut self) -> Result<(), ParserError> {
        self.ensure_rendered();
        if self.display_enabled {
            self.canvas.display().map_err(ParserError::Io)?;
        }
        Ok(())
    }

    fn save(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        self.ensure_rendered();
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

    fn parse_as<T: FromStr>(line: &str) -> Result<Vec<T>, T::Err> {
        line.split_whitespace().map(str::parse).collect()
    }

    /// Set the parser's color.
    pub fn set_color(&mut self, color: &Rgb) {
        self.canvas.line = *color;
        self.canvas_dirty = true;
    }

    /// Get a reference to the parser's trans matrix.
    pub fn trans_matrix(&self) -> &Matrix {
        &self.trans_matrix
    }

    /// Get a reference to the parser's edge matrix.
    #[must_use]
    pub fn edge_matrix(&self) -> &EdgeMatrix {
        &self.edge_matrix
    }

    /// Get a reference to the parser's polygon matrix.
    #[must_use]
    pub fn polygon_matrix(&self) -> &PolygonMatrix {
        &self.polygon_matrix
    }

    /// Allows you to modify the parser's edge matrix.
    ///
    /// * `func`: A function that takes a mutable reference to the parser's edge matrix.
    ///
    /// have fun
    pub fn with_edge_matrix<F>(&mut self, func: F)
    where
        F: FnOnce(&mut EdgeMatrix),
    {
        func(&mut self.edge_matrix);
        self.edge_apply_start = self.edge_apply_start.min(self.edge_matrix.cols());
        self.canvas_dirty = true;
    }

    /// Applies a function to the internal edge matrix and marks canvas dirty.
    pub fn edge_matrix_fun(&mut self, func: &dyn Fn(&mut EdgeMatrix)) {
        self.with_edge_matrix(|edges| func(edges));
    }

    fn parse_filter(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        self.ensure_rendered();

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
                self.canvas_dirty = false;
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
                self.canvas_dirty = false;
                Ok(())
            }
            _ => Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
    }

    fn parse_reset(&mut self) {
        self.canvas.clear_canvas();
        self.edge_matrix = EdgeMatrix::new();
        self.polygon_matrix = PolygonMatrix::new();
        self.edge_apply_start = 0;
        self.polygon_apply_start = 0;
        self.trans_matrix = Matrix::identity_matrix(4);
        self.trans_stack.clear();
        self.symbols.clear();
        self.canvas_dirty = false;
    }

    fn parse_box(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 6, line_num)?;
        self.polygon_matrix
            .add_box((args[0], args[1], args[2]), args[3], args[4], args[5]);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_sphere(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 4, line_num)?;
        self.polygon_matrix
            .add_sphere((args[0], args[1], args[2]), args[3], 24);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_torus(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.polygon_matrix
            .add_torus((args[0], args[1], args[2]), args[3], args[4], 24);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_cylinder(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.polygon_matrix
            .add_cylinder((args[0], args[1], args[2]), args[3], args[4], 24);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_cone(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.polygon_matrix
            .add_cone((args[0], args[1], args[2]), args[3], args[4], 24);
        self.canvas_dirty = true;
        Ok(())
    }

    fn parse_pyramid(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line_str) = line;
        let args = self.parse_args_resolved(line_str, 5, line_num)?;
        self.polygon_matrix
            .add_pyramid((args[0], args[1], args[2]), args[3], args[4]);
        self.canvas_dirty = true;
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

        self.polygon_matrix.add_bezier_surface(controls, steps);
        self.canvas_dirty = true;
        Ok(())
    }

    #[cfg(feature = "external")]
    fn parse_mesh(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        let path = self.resolve_script_path(file_name);
        crate::external::add_mesh(path.to_string_lossy().as_ref(), &mut self.polygon_matrix)
            .map_err(|err| {
                ParserError::MeshError(line_num, path.display().to_string(), err.to_string())
            })?;
        self.canvas_dirty = true;
        Ok(())
    }

    #[cfg(feature = "external")]
    fn parse_mesh_reverse(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let start_col = self.polygon_matrix.cols();
        self.parse_mesh(line)?;
        self.polygon_matrix.reverse_winding_from_col(start_col);
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
    fn extra_whitespace_in_arguments_is_accepted() {
        let path = temp_file("whitespace");
        fs::write(&path, "line\n0   0  0    1 1 0\n").expect("write temp script");

        let mut parser = Parser::new(path.to_str().expect("utf8 path"), 2, 2, &Rgb::GREEN);
        parser.parse_file().expect("script should parse");

        assert_eq!(parser.edge_matrix().cols(), 2);
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
    fn apply_only_transforms_pending_geometry() {
        let mut parser = Parser::new("test", 50, 50, &Rgb::GREEN);

        parser
            .parse_string(
                "line\n0 0 0 1 0 0\nmove\n10 0 0\napply\nident\nline\n0 0 0 1 0 0\nmove\n0 20 0\napply",
            )
            .expect("script valid");

        assert_eq!(
            parser.edge_matrix().as_matrix().data(),
            &[
                10.0, 0.0, 0.0, 1.0, 11.0, 0.0, 0.0, 1.0, 0.0, 20.0, 0.0, 1.0, 1.0, 20.0, 0.0, 1.0
            ]
        );
    }

    #[test]
    fn push_pop_scopes_applied_geometry() {
        let mut parser = Parser::new("test", 50, 50, &Rgb::GREEN);

        parser
            .parse_string(
                "push\nmove\n10 0 0\nline\n0 0 0 1 0 0\napply\npop\nmove\n0 20 0\nline\n0 0 0 1 0 0\napply",
            )
            .expect("script valid");

        assert_eq!(
            parser.edge_matrix().as_matrix().data(),
            &[
                10.0, 0.0, 0.0, 1.0, 11.0, 0.0, 0.0, 1.0, 0.0, 20.0, 0.0, 1.0, 1.0, 20.0, 0.0, 1.0
            ]
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_variables() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string("set\nx 100\nmove\nx 0 0")
            .expect("set valid");
        assert_eq!(parser.trans_matrix().get(0, 3), 100.0);
    }

    #[test]
    fn test_new_primitives() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);

        parser
            .parse_string("cylinder\n0 0 0 10 20")
            .expect("cylinder valid");
        assert!(!parser.polygon_matrix().is_empty());
        let mut manual = crate::gmath::polygon_matrix::PolygonMatrix::new();
        manual.add_cylinder((0.0, 0.0, 0.0), 10.0, 20.0, 24);
        assert_eq!(parser.polygon_matrix().cols(), manual.cols());

        parser
            .parse_string("clear\ncone\n0 0 0 10 20")
            .expect("cone valid");
        manual = crate::gmath::polygon_matrix::PolygonMatrix::new();
        manual.add_cone((0.0, 0.0, 0.0), 10.0, 20.0, 24);
        assert_eq!(parser.polygon_matrix().cols(), manual.cols());

        parser
            .parse_string("clear\npyramid\n0 0 0 10 20")
            .expect("pyramid valid");
        manual = crate::gmath::polygon_matrix::PolygonMatrix::new();
        manual.add_pyramid((0.0, 0.0, 0.0), 10.0, 20.0);
        assert_eq!(parser.polygon_matrix().cols(), manual.cols());
    }

    #[test]
    fn reverse_winding_command_flips_polygon_faces() {
        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser.parse_string("box\n0 0 0 1 1 1").expect("box valid");
        let before = parser.polygon_matrix().as_matrix().data().to_vec();

        parser
            .parse_string("reverse_winding")
            .expect("reverse winding valid");
        let after = parser.polygon_matrix().as_matrix().data();

        assert_eq!(&after[0..4], &before[0..4]);
        assert_eq!(&after[4..8], &before[8..12]);
        assert_eq!(&after[8..12], &before[4..8]);
    }

    #[test]
    fn test_include_command() {
        let path = temp_file("include-target");
        fs::write(&path, "line\n0 0 0 1 1 1\n").expect("write temp script");

        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string(&format!("include\n{}", path.to_str().unwrap()))
            .expect("include valid");

        assert_eq!(parser.edge_matrix().cols(), 2);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn include_paths_resolve_relative_to_script_file() {
        let dir = temp_dir("relative-include");
        fs::create_dir_all(&dir).expect("create temp dir");
        let main_path = dir.join("main.cg");
        let child_path = dir.join("child.cg");
        fs::write(&main_path, "include\nchild.cg\n").expect("write main script");
        fs::write(&child_path, "line\n0 0 0 1 1 1\n").expect("write child script");

        let mut parser = Parser::new(main_path.to_str().expect("utf8 path"), 10, 10, &Rgb::GREEN);
        parser.parse_file().expect("include valid");

        assert_eq!(parser.edge_matrix().cols(), 2);
        let _ = fs::remove_file(main_path);
        let _ = fs::remove_file(child_path);
        let _ = fs::remove_dir(dir);
    }

    #[cfg(feature = "external")]
    #[test]
    fn test_mesh_command() {
        let path = temp_file_with_extension("mesh", "obj");
        fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write temp obj");

        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser
            .parse_string(&format!("mesh\n{}", path.to_str().unwrap()))
            .expect("mesh valid");

        assert_eq!(parser.polygon_matrix().cols(), 3);
        assert!(parser.is_dirty());
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
        fs::write(&mesh_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write obj");

        let mut parser = Parser::new(
            script_path.to_str().expect("utf8 path"),
            10,
            10,
            &Rgb::GREEN,
        );
        parser.parse_file().expect("mesh valid");

        assert_eq!(parser.polygon_matrix().cols(), 3);
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
        fs::write(&mesh_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write obj");

        let mut parser = Parser::new(main_path.to_str().expect("utf8 path"), 10, 10, &Rgb::GREEN);
        parser.parse_file().expect("nested include mesh valid");

        assert_eq!(parser.polygon_matrix().cols(), 3);
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(feature = "external")]
    #[test]
    fn mesh_reverse_reverses_only_imported_winding() {
        let path = temp_file_with_extension("mesh-reverse", "obj");
        fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write temp obj");

        let mut parser = Parser::new("test", 10, 10, &Rgb::GREEN);
        parser.parse_string("box\n0 0 0 1 1 1").expect("box valid");
        let existing = parser.polygon_matrix().as_matrix().data().to_vec();
        parser
            .parse_string(&format!("mesh_reverse\n{}", path.to_str().unwrap()))
            .expect("mesh reverse valid");

        let data = parser.polygon_matrix().as_matrix().data();
        assert_eq!(&data[..existing.len()], existing.as_slice());
        assert_eq!(
            &data[existing.len()..],
            &[
                0.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 1.0,
            ]
        );
        let _ = fs::remove_file(path);
    }

    #[cfg(feature = "external")]
    #[test]
    fn example_teapot_mesh_script_parses() {
        let mut parser = Parser::new(
            "examples/data/scripts/teapot_mesh.cg",
            800,
            800,
            &Rgb::GREEN,
        );
        parser.set_display_enabled(false);

        parser.parse_file().expect("teapot mesh script valid");

        assert!(!parser.polygon_matrix().is_empty());
        assert!(!parser.is_dirty());
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

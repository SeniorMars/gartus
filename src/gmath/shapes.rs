use super::matrix::Matrix;
use std::f64::consts::PI;

// 2D Shapes
impl Matrix {
    /// Adds a new circle centered at (cx, cy, cz) with radius r, and a precision of the circle
    ///
    /// # Arguments
    ///
    /// * `cx` - The x corrdinate of the center of the circle
    /// * `cy` - The y corrdinate of the center of the circle
    /// * `cz` - The z corrdinate of the center of the circle
    /// * `r` - The radius of the circle
    /// * `step` - The precision of the circle
    pub fn add_circle(&mut self, cx: f64, cy: f64, cz: f64, r: f64, step: f64) {
        self.add_parametric_curve(
            |t: f64| r * (t * 2.0 * PI).cos() + cx,
            |t: f64| r * (t * 2.0 * PI).sin() + cy,
            cz,
            step,
        );
    }

    /// Adds a new triangle to the edge matrix connecting points (x1, y1, z1), (x2, y2, z2), and (x3, y3, z3)
    ///
    /// * `x1`: First vertex x coordinate
    /// * `y1`: First vertex y coordinate
    /// * `z1`: First vertex z coordinate
    /// * `x2`: Second vertex x coordinate
    /// * `y2`: Second vertex y coordinate
    /// * `z2`: Second vertex z coordinate
    /// * `x3`: Third vertex x coordinate
    /// * `y3`: Third vertex y coordinate
    /// * `z3`: Third vertex z coordinate
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "fancy_math")]
    pub fn add_triangle(
        &mut self,
        x1: f64,
        y1: f64,
        z1: f64,
        x2: f64,
        y2: f64,
        z2: f64,
        x3: f64,
        y3: f64,
        z3: f64,
    ) {
        // Add line segment AB
        self.add_parametric_curve(
            |t: f64| x1 + t * (x2 - x1),
            |t: f64| y1 + t * (y2 - y1),
            z1,
            1.0,
        );

        // Add line segment BC
        self.add_parametric_curve(
            |t: f64| x2 + t * (x3 - x2),
            |t: f64| y2 + t * (y3 - y2),
            z2,
            1.0,
        );

        // Add line segment CA
        self.add_parametric_curve(
            |t: f64| x3 + t * (x1 - x3),
            |t: f64| y3 + t * (y1 - y3),
            z3,
            1.0,
        );
    }
}

// 3D Shapes
impl Matrix {
    #[allow(clippy::cast_precision_loss)]
    /// Adds a box to the edge matrix
    ///
    /// * `width`: the width of the box
    /// * `height`: the height of the box
    /// * `depth`: the depth of the box
    /// * `vertex`: the top left front of the box:
    ///     - `x`: the x coordinate of the vertex
    ///     - `y`: the y coordinate of the vertex
    ///     - `z`: the z coordinate of the vertex
    pub fn add_box(
        &mut self,
        (x, y, z): (f64, f64, f64),
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let (height, width, depth) = (height as f64, width as f64, depth as f64);

        let p1 = (x, y, z);
        let p2 = (x, y - height, z);
        let p3 = (x + width, y, z);
        let p4 = (x + width, y - height, z);

        let p5 = (x, y, z - depth);
        let p6 = (x, y - height, z - depth);
        let p7 = (x + width, y, z - depth);
        let p8 = (x + width, y - height, z - depth);

        // one face
        self.add_edge_tuple(p1, p2);
        self.add_edge_tuple(p2, p4);
        self.add_edge_tuple(p1, p3);
        self.add_edge_tuple(p3, p4);

        // another face
        self.add_edge_tuple(p5, p6);
        self.add_edge_tuple(p6, p8);
        self.add_edge_tuple(p5, p7);
        self.add_edge_tuple(p7, p8);

        // connecting faces
        self.add_edge_tuple(p1, p5);
        self.add_edge_tuple(p2, p6);
        self.add_edge_tuple(p3, p7);
        self.add_edge_tuple(p4, p8);
    }

    #[allow(clippy::cast_precision_loss)]
    /// Adds a sphere to the edge matrix
    ///
    /// * `radius`: the radius of the sphere
    /// * `steps`: the precision of the sphere
    pub fn add_sphere(&mut self, (cx, cy, cz): (f64, f64, f64), radius: f64, steps: usize) {
        let pps = steps + 1;
        let step_by = 2. * PI / steps as f64;
        let mut points: Vec<(f64, f64, f64)> = Vec::with_capacity(steps * pps);

        for rot in 0..steps {
            let phi = rot as f64 * step_by;
            for cir in 0..=steps {
                let theta = cir as f64 * step_by;

                let x = radius * theta.cos() + cx;
                let y = radius * theta.sin() * phi.sin() + cy;
                let z = radius * theta.sin() * phi.cos() + cz;

                points.push((x, y, z));
            }
        }

        for point in points {
            let point_next = (point.0 + 1.0, point.1 + 1.0, point.2 + 1.0);
            self.add_edge_tuple(point, point_next);
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    /// Adds a torus to the edge matrix
    ///
    /// * `radius_sm`: the radius of the small circle
    /// * `radius_big`: the radius of the big circle
    /// * `steps`: the precision of the torus
    pub fn add_torus(
        &mut self,
        (cx, cy, cz): (f64, f64, f64),
        radius_sm: f64,
        radius_big: f64,
        steps: usize,
    ) {
        let estimated_capacity = steps * (steps + 1);

        let mut points: Vec<(f64, f64, f64)> = Vec::with_capacity(estimated_capacity);

        let step_by = 2. * PI / steps as f64;

        for torus_ang_norm in 0..steps {
            let steps = steps as f64;
            let torus_ang = torus_ang_norm as f64 * step_by;
            for cir_ang_norm in 0..=(steps as usize) {
                let circ_ang = cir_ang_norm as f64 * step_by;

                let x = torus_ang.cos() * (radius_sm * circ_ang.cos() + radius_big) + cx;
                let y = radius_sm * circ_ang.sin() + cy;
                let z = -torus_ang.sin() * (radius_sm * circ_ang.cos() + radius_big) + cz;

                points.push((x, y, z));
            }
        }

        for point in points {
            let point_next = (point.0 + 1.0, point.1 + 1.0, point.2 + 1.0);
            self.add_edge_tuple(point, point_next);
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    /// Adds a cylinder to the edge matrix
    ///
    /// * (x, y, z): The coordinates of where to generate the cylinder
    /// * `radius`: The radius of the cylinder
    /// * `height`: The height of the cylinder
    /// * `steps`: The precision of the cylinder
    pub fn add_cylinder(
        &mut self,
        (x, y, z): (f64, f64, f64),
        radius: f64,
        height: f64,
        steps: usize,
    ) {
        let theta_step = 2.0 * std::f64::consts::PI / steps as f64;
        let height_step = height / steps as f64;

        for i in 0..steps {
            let theta = i as f64 * theta_step;
            for j in 0..steps {
                let h = j as f64 * height_step;

                let x_point = x + radius * theta.cos();
                let y_point = y + radius * theta.sin();
                let z_point = z + h;

                // Assuming you have a method to add points to your edge matrix.
                self.add_point(x_point, y_point, z_point);

                // If you want to connect the points to form the cylinder's surface,
                // You might want to add lines between consecutive points.
                if i < steps - 1 || j < steps - 1 {
                    let next_theta = (i + 1) as f64 * theta_step;
                    let next_h = (j + 1) as f64 * height_step;

                    let next_x = x + radius * next_theta.cos();
                    let next_y = y + radius * next_theta.sin();
                    let next_z = z + next_h;

                    self.add_edge(x_point, y_point, z_point, next_x, next_y, next_z);
                }
            }
        }

        // Add bottom base
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let x_point = x + radius * theta.cos();
            let y_point = y + radius * theta.sin();
            let z_point = z; // bottom base

            let next_theta = ((i + 1) % steps) as f64 * theta_step;
            let next_x = x + radius * next_theta.cos();
            let next_y = y + radius * next_theta.sin();

            // Connect the center to the current point
            self.add_edge(x, y, z, x_point, y_point, z_point);
            // Connect the current point to the next point
            self.add_edge(x_point, y_point, z_point, next_x, next_y, z);
            // Connect the next point back to the center
            self.add_edge(next_x, next_y, z, x, y, z);
        }

        // Add top base
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let x_point = x + radius * theta.cos();
            let y_point = y + radius * theta.sin();
            let z_point = z + height; // top base

            let next_theta = ((i + 1) % steps) as f64 * theta_step;
            let next_x = x + radius * next_theta.cos();
            let next_y = y + radius * next_theta.sin();

            // Connect the center to the current point
            self.add_edge(x, y, z + height, x_point, y_point, z_point);
            // Connect the current point to the next point
            self.add_edge(x_point, y_point, z_point, next_x, next_y, z + height);
            // Connect the next point back to the center
            self.add_edge(next_x, next_y, z + height, x, y, z + height);
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    /// Adds a cone to the edge matrix
    ///
    /// * `radius`: Radius of the base
    /// * `height`: Height of the cone
    /// * `steps`: Precision of the cone
    pub fn add_cone(&mut self, (x, y, z): (f64, f64, f64), radius: f64, height: f64, steps: usize) {
        let theta_step = 2.0 * std::f64::consts::PI / steps as f64;

        // Add base (circle)
        for i in 0..steps {
            let theta = i as f64 * theta_step;
            let x_point = x + radius * theta.cos();
            let y_point = y + radius * theta.sin();
            let z_point = z;

            let next_theta = ((i + 1) % steps) as f64 * theta_step;
            let next_x = x + radius * next_theta.cos();
            let next_y = y + radius * next_theta.sin();

            // Connect the center to the current point for base
            self.add_edge(x, y, z, x_point, y_point, z_point);
            // Connect the current point to the top for the lateral surface
            self.add_edge(x_point, y_point, z_point, x, y, z + height);
            // Connect the current point to the next point for the base
            self.add_edge(x_point, y_point, z_point, next_x, next_y, z);
        }
    }

    /// Adds a pyramid to the edge matrix
    ///
    /// * `base_length`: The length of the base
    /// * `height`: The height of the pyramid
    pub fn add_pyramid(&mut self, (x, y, z): (f64, f64, f64), base_length: f64, height: f64) {
        let half_length = base_length / 2.0;

        // Four corners of the square base
        let corners = [
            (x - half_length, y, z - half_length),
            (x + half_length, y, z - half_length),
            (x + half_length, y, z + half_length),
            (x - half_length, y, z + half_length),
        ];

        // Connect base corners to top
        for &corner in &corners {
            self.add_edge(corner.0, corner.1, corner.2, x, y + height, z);
        }

        // Connect base corners to each other
        for i in 0..corners.len() {
            let next_i = (i + 1) % corners.len();
            self.add_edge(
                corners[i].0,
                corners[i].1,
                corners[i].2,
                corners[next_i].0,
                corners[next_i].1,
                corners[next_i].2,
            );
        }
    }
}

#[cfg(test)]
mod test {

    use crate::prelude::{Canvas, Rgb};

    use super::*;

    #[test]
    fn add_box_test() {
        let mut test = Matrix::new(4, 0, Vec::new());
        test.add_box((10.0, 10.0, 10.0), 10, 10, 10);

        assert_eq!(test.len(), 24 * 4);

        // self * other
        let transformations =
            Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(45.0) * Matrix::rotate_z(45.0);
        test = transformations * test;

        let mut canvas = Canvas::new_with_bg(100, 100, 255, Rgb::new(24, 26, 27));

        canvas.set_line_pixel(&Rgb::new(255, 255, 255));

        canvas.draw_lines(&test);

        canvas.display().expect("Failed to display canvas");
    }

    #[test]
    fn add_sphere_test() {
        let mut test = Matrix::new(4, 0, Vec::new());

        test.add_sphere((50.0, 50.0, 50.0), 25.0, 24);

        // step = 24
        // pps = 25
        // 24 * 25 = 600
        // but we have 2 points per edge
        // and each point is 4 coordinates
        // so 600 * 2 * 4 = 4800
        // assert_eq!(test.len(), 24 * 25 * 8);

        let mut canvas = Canvas::new_with_bg(100, 100, 255, Rgb::new(24, 26, 27));

        canvas.set_line_pixel(&Rgb::GREEN);

        canvas.draw_lines(&test);

        canvas.display().expect("Failed to display canvas");
    }

    // create a test to test torus
    #[test]
    fn add_torus_test() {
        let mut test = Matrix::new(4, 0, Vec::new());

        test.add_torus((50.0, 50.0, 50.0), 12.5, 25.0, 30);

        assert_eq!(test.len(), 30 * 31 * 8);

        let mut canvas = Canvas::new_with_bg(100, 100, 255, Rgb::new(24, 26, 27));

        canvas.set_line_pixel(&Rgb::GREEN);

        canvas.draw_lines(&test);

        canvas.display().expect("Failed to display canvas");
    }

    // create a test to test cylinder
    #[test]
    fn add_cylinder_test() {
        let mut test = Matrix::new(4, 0, Vec::new());

        test.add_cylinder((25.0, 25.0, 25.0), 25.0, 25.0, 36);

        let transformations =
            Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(45.0) * Matrix::rotate_z(45.0);
        // test = Matrix::rotate_y(45.0) * test;

        test = transformations * test;
        // assert_eq!(test.len(), 30 * 31 * 8);

        let mut canvas = Canvas::new_with_bg(100, 100, 255, Rgb::new(24, 26, 27));

        canvas.set_line_pixel(&Rgb::GREEN);

        canvas.draw_lines(&test);

        canvas.display().expect("Failed to display canvas");
    }

    // create a test to test cone
    #[test]
    fn add_cone_test() {
        let mut test = Matrix::new(4, 0, Vec::new());

        test.add_cone((25.0, 25.0, 25.0), 25.0, 25.0, 36);

        // let transformations =
        //     Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(25.0) * Matrix::rotate_z(45.0);
        test = Matrix::rotate_y(90.0) * test;

        // test = transformations * test;
        // assert_eq!(test.len(), 30 * 31 * 8);

        let mut canvas = Canvas::new_with_bg(100, 100, 255, Rgb::new(24, 26, 27));

        canvas.set_line_pixel(&Rgb::GREEN);

        canvas.draw_lines(&test);

        canvas.display().expect("Failed to display canvas");
    }

    // create a test to test pyramid
    #[test]
    fn add_pyramid_test() {
        let mut test = Matrix::new(4, 0, Vec::new());

        test.add_pyramid((25.0, 25.0, 25.0), 25.0, 25.0);

        // let transformations =
        //     Matrix::translate(50.0, 50.0, 0.0) * Matrix::rotate_x(25.0) * Matrix::rotate_z(45.0);
        test = Matrix::rotate_x(22.5) * test;

        // test = transformations * test;
        // assert_eq!(test.len(), 30 * 31 * 8);

        let mut canvas = Canvas::new_with_bg(100, 100, 255, Rgb::new(24, 26, 27));

        canvas.set_line_pixel(&Rgb::GREEN);

        canvas.draw_lines(&test);

        canvas.display().expect("Failed to display canvas");
    }
}

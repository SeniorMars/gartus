use super::colors::Rgb;
use crate::gmath::random::SampleRng;
use crate::gmath::ray::Ray;
use crate::gmath::{
    geometry::CameraPose,
    vector::{Point, Vector},
};
use crate::graphics::raytracing::{
    Hittable, INFINITY, Interval, LinearColor, component_mul, degrees_to_radians,
};
use crate::graphics::raytracing::{SHADOW_ACNE_EPSILON, normal_scene_color};
use crate::{
    gmath::{edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::display::Canvas,
};
use std::io::{self, Write};

/// A simple perspective camera for projecting 3D points onto a 2D canvas.
#[derive(Debug, Clone, Copy)]
pub struct Camera3D {
    width: u32,
    height: u32,
    camera_distance: f64,
    focal_length: f64,
    center_y_factor: f64,
    near_depth: f64,
    lookfrom: Option<Point>,
    lookat: Point,
    vup: Vector,
}

/// A projected 2D point plus its camera-space depth.
#[derive(Debug, Clone, Copy)]
pub struct ScreenPoint {
    /// Horizontal screen coordinate.
    pub x: f64,
    /// Vertical screen coordinate.
    pub y: f64,
    /// Camera-space depth used for sorting and shading.
    pub depth: f64,
}

/// A projected colored line segment.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedSegment {
    /// First projected endpoint.
    pub a: ScreenPoint,
    /// Second projected endpoint.
    pub b: ScreenPoint,
    /// Segment draw color.
    pub color: Rgb,
}

/// A simple pinhole camera that emits one ray through each image pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCamera {
    aspect_ratio: f64,
    image_width: u32,
    image_height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    rng_seed: u64,
    vertical_fov: f64,
    lookfrom: Point,
    lookat: Point,
    view_up: Vector,
    defocus_angle: f64,
    focus_distance: f64,
    shutter_start: f64,
    shutter_end: f64,
    background: LinearColor,
    camera_center: Point,
    pixel00_loc: Point,
    pixel_delta_u: Vector,
    pixel_delta_v: Vector,
    defocus_disk_u: Vector,
    defocus_disk_v: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RayCameraParams {
    image_width: u32,
    aspect_ratio: f64,
    samples_per_pixel: u32,
    max_depth: u32,
    rng_seed: u64,
    vertical_fov: f64,
    lookfrom: Point,
    lookat: Point,
    view_up: Vector,
    defocus_angle: f64,
    focus_distance: f64,
    shutter_start: f64,
    shutter_end: f64,
    background: LinearColor,
}

impl Camera3D {
    /// Creates a camera centered in a canvas.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            camera_distance: 900.0,
            focal_length: 700.0,
            center_y_factor: 0.5,
            near_depth: 80.0,
            lookfrom: None,
            lookat: Point::new(0.0, 0.0, 0.0),
            vup: Vector::new(0.0, 1.0, 0.0),
        }
    }

    /// Sets the distance added to incoming z values before projection.
    #[must_use]
    pub fn with_camera_distance(mut self, camera_distance: f64) -> Self {
        self.camera_distance = camera_distance;
        self
    }

    /// Sets the focal length used for perspective scaling.
    #[must_use]
    pub fn with_focal_length(mut self, focal_length: f64) -> Self {
        self.focal_length = focal_length;
        self
    }

    /// Sets the focal length from a vertical field-of-view angle in degrees.
    ///
    /// # Panics
    ///
    /// Panics if `vertical_fov` is not finite or is outside `0..180` degrees.
    #[must_use]
    pub fn with_vertical_fov(mut self, vertical_fov: f64) -> Self {
        assert!(
            vertical_fov.is_finite() && 0.0 < vertical_fov && vertical_fov < 180.0,
            "vertical field of view must be finite and in 0..180 degrees"
        );
        let theta = vertical_fov.to_radians();
        self.focal_length = f64::from(self.height) * 0.5 / (theta * 0.5).tan();
        self
    }

    /// Sets the vertical screen center as a fraction of canvas height.
    #[must_use]
    pub fn with_center_y_factor(mut self, center_y_factor: f64) -> Self {
        self.center_y_factor = center_y_factor;
        self
    }

    /// Sets the minimum projected depth.
    #[must_use]
    pub fn with_near_depth(mut self, near_depth: f64) -> Self {
        self.near_depth = near_depth;
        self
    }

    /// Positions the projection camera at `lookfrom`, aimed at `lookat`.
    ///
    /// The default camera is equivalent to looking from `(0, 0, -camera_distance)`
    /// toward the origin, preserving the historical projection behavior.
    ///
    /// # Panics
    ///
    /// Panics if `lookfrom` and `lookat` are the same point.
    #[must_use]
    pub fn with_look_at(mut self, lookfrom: Point, lookat: Point) -> Self {
        assert!(
            (lookat - lookfrom).length_squared() > f64::EPSILON,
            "lookfrom and lookat must be distinct"
        );
        self.lookfrom = Some(lookfrom);
        self.lookat = lookat;
        self
    }

    /// Sets the camera-relative up direction.
    ///
    /// # Panics
    ///
    /// Panics if `vup` is zero.
    #[must_use]
    pub fn with_view_up(mut self, vup: Vector) -> Self {
        assert!(
            vup.length_squared() > f64::EPSILON,
            "view-up vector must be nonzero"
        );
        self.vup = vup;
        self
    }

    fn effective_lookfrom(&self) -> Point {
        self.lookfrom
            .unwrap_or_else(|| Point::new(0.0, 0.0, -self.camera_distance))
    }

    fn camera_frame(&self) -> Option<(Point, Vector, Vector, Vector)> {
        let lookfrom = self.effective_lookfrom();
        let frame = CameraPose::new(lookfrom, self.lookat, self.vup).frame()?;
        Some((frame.origin, -frame.right, frame.up, frame.forward))
    }

    /// Projects a homogeneous point into 2D screen coordinates.
    #[must_use]
    pub fn project(&self, point: &[f64]) -> Option<ScreenPoint> {
        if point.len() < 3 {
            return None;
        }
        let (lookfrom, right, up, forward) = self.camera_frame()?;
        let point = Point::new(point[0], point[1], point[2]);
        let camera_relative = point - lookfrom;
        let depth = camera_relative.dot(forward);
        if depth < self.near_depth {
            return None;
        }
        let scale = self.focal_length / depth;
        Some(ScreenPoint {
            x: f64::from(self.width) * 0.5 + camera_relative.dot(right) * scale,
            y: f64::from(self.height) * self.center_y_factor - camera_relative.dot(up) * scale,
            depth,
        })
    }

    /// Projects transformed mesh triangle edges into colored wireframe segments.
    ///
    /// `color_for_triangle` receives the triangle index and average projected triangle depth.
    pub fn project_mesh_wireframe_segments<F>(
        &self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        stride: usize,
        mut color_for_triangle: F,
    ) -> Vec<ProjectedSegment>
    where
        F: FnMut(usize, f64) -> Rgb,
    {
        let stride = stride.max(1);
        let mut segments = Vec::new();
        for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
            if idx % stride != 0 {
                continue;
            }
            let Some(a) = self.project(&p0) else {
                continue;
            };
            let Some(b) = self.project(&p1) else {
                continue;
            };
            let Some(c) = self.project(&p2) else {
                continue;
            };
            let depth = (a.depth + b.depth + c.depth) / 3.0;
            let color = color_for_triangle(idx, depth);
            segments.push(ProjectedSegment { a, b, color });
            segments.push(ProjectedSegment { a: b, b: c, color });
            segments.push(ProjectedSegment { a: c, b: a, color });
        }
        segments
    }
}

impl Default for RayCamera {
    fn default() -> Self {
        Self::new(100, 1.0)
    }
}

impl RayCamera {
    /// Creates a camera with the requested image width and ideal aspect ratio.
    ///
    /// The image height is rounded down from `image_width / aspect_ratio`, with a
    /// minimum height of one pixel. The viewport is sized from the actual integer
    /// image dimensions so pixel spacing remains square.
    ///
    /// # Panics
    ///
    /// Panics if `image_width` is zero or `aspect_ratio` is not positive and finite.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(image_width: u32, aspect_ratio: f64) -> Self {
        assert!(image_width > 0, "image width must be positive");
        assert!(
            aspect_ratio.is_finite() && aspect_ratio > 0.0,
            "aspect ratio must be positive and finite"
        );

        Self::initialized(RayCameraParams {
            image_width,
            aspect_ratio,
            samples_per_pixel: 1,
            max_depth: 10,
            rng_seed: 1,
            vertical_fov: 90.0,
            lookfrom: Point::new(0.0, 0.0, 0.0),
            lookat: Point::new(0.0, 0.0, -1.0),
            view_up: Vector::new(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_distance: 1.0,
            shutter_start: 0.0,
            shutter_end: 1.0,
            background: LinearColor::new(0.70, 0.80, 1.00),
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn initialized(params: RayCameraParams) -> Self {
        Self::validate_view(&params);

        let image_height = ((f64::from(params.image_width) / params.aspect_ratio) as u32).max(1);
        let theta = degrees_to_radians(params.vertical_fov);
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h * params.focus_distance;
        let viewport_width =
            viewport_height * (f64::from(params.image_width) / f64::from(image_height));
        let camera_center = params.lookfrom;

        let frame = CameraPose::new(params.lookfrom, params.lookat, params.view_up)
            .frame()
            .expect("validated camera basis");
        let w = frame.backward();
        let u = frame.right;
        let v = frame.up;

        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;
        let pixel_delta_u = viewport_u / f64::from(params.image_width);
        let pixel_delta_v = viewport_v / f64::from(image_height);

        let viewport_upper_left =
            camera_center - params.focus_distance * w - viewport_u / 2.0 - viewport_v / 2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        let defocus_radius =
            params.focus_distance * degrees_to_radians(params.defocus_angle * 0.5).tan();
        let defocus_disk_u = defocus_radius * u;
        let defocus_disk_v = defocus_radius * v;

        Self {
            aspect_ratio: params.aspect_ratio,
            image_width: params.image_width,
            image_height,
            samples_per_pixel: params.samples_per_pixel.max(1),
            max_depth: params.max_depth,
            rng_seed: params.rng_seed,
            vertical_fov: params.vertical_fov,
            lookfrom: params.lookfrom,
            lookat: params.lookat,
            view_up: params.view_up,
            defocus_angle: params.defocus_angle,
            focus_distance: params.focus_distance,
            shutter_start: params.shutter_start,
            shutter_end: params.shutter_end,
            background: params.background,
            camera_center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
            defocus_disk_u,
            defocus_disk_v,
        }
    }

    /// Returns a copy of the camera initialized from its current public render parameters.
    #[must_use]
    fn initialize(self) -> Self {
        Self::initialized(RayCameraParams {
            image_width: self.image_width,
            aspect_ratio: self.aspect_ratio,
            samples_per_pixel: self.samples_per_pixel,
            max_depth: self.max_depth,
            rng_seed: self.rng_seed,
            vertical_fov: self.vertical_fov,
            lookfrom: self.lookfrom,
            lookat: self.lookat,
            view_up: self.view_up,
            defocus_angle: self.defocus_angle,
            focus_distance: self.focus_distance,
            shutter_start: self.shutter_start,
            shutter_end: self.shutter_end,
            background: self.background,
        })
    }

    fn validate_view(params: &RayCameraParams) {
        assert!(
            params.vertical_fov.is_finite()
                && 0.0 < params.vertical_fov
                && params.vertical_fov < 180.0,
            "vertical field of view must be finite and in 0..180 degrees"
        );
        let w = params.lookfrom - params.lookat;
        assert!(
            w.length_squared() > f64::EPSILON,
            "lookfrom and lookat must be distinct"
        );
        assert!(
            params.view_up.length_squared() > f64::EPSILON,
            "view-up vector must be nonzero"
        );
        assert!(
            CameraPose::new(params.lookfrom, params.lookat, params.view_up)
                .frame()
                .is_some(),
            "view-up vector must not be parallel to the viewing direction"
        );
        assert!(
            params.defocus_angle.is_finite() && (0.0..180.0).contains(&params.defocus_angle),
            "defocus angle must be finite and in 0..180 degrees"
        );
        assert!(
            params.focus_distance.is_finite() && params.focus_distance > 0.0,
            "focus distance must be positive and finite"
        );
        assert!(
            params.shutter_start.is_finite()
                && params.shutter_end.is_finite()
                && params.shutter_start <= params.shutter_end,
            "shutter interval must be finite and ordered"
        );
        assert!(
            params.background.x().is_finite()
                && params.background.y().is_finite()
                && params.background.z().is_finite(),
            "background color components must be finite"
        );
    }

    /// Sets the target image width and recomputes derived camera values.
    ///
    /// # Panics
    ///
    /// Panics if `image_width` is zero.
    #[must_use]
    pub fn with_image_width(mut self, image_width: u32) -> Self {
        assert!(image_width > 0, "image width must be positive");
        self.image_width = image_width;
        self.initialize()
    }

    /// Sets the target aspect ratio and recomputes derived camera values.
    ///
    /// # Panics
    ///
    /// Panics if `aspect_ratio` is not positive and finite.
    #[must_use]
    pub fn with_aspect_ratio(mut self, aspect_ratio: f64) -> Self {
        assert!(
            aspect_ratio.is_finite() && aspect_ratio > 0.0,
            "aspect ratio must be positive and finite"
        );
        self.aspect_ratio = aspect_ratio;
        self.initialize()
    }

    /// Sets the random samples taken per pixel for world rendering.
    #[must_use]
    pub fn with_samples_per_pixel(mut self, samples_per_pixel: u32) -> Self {
        self.samples_per_pixel = samples_per_pixel.max(1);
        self.initialize()
    }

    /// Sets the maximum ray-bounce recursion depth for diffuse world rendering.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the deterministic random seed used for antialiasing samples.
    #[must_use]
    pub fn with_rng_seed(mut self, rng_seed: u64) -> Self {
        self.rng_seed = rng_seed;
        self
    }

    /// Sets the vertical field of view in degrees and recomputes derived camera values.
    ///
    /// # Panics
    ///
    /// Panics if `vertical_fov` is not finite or is outside `0..180` degrees.
    #[must_use]
    pub fn with_vertical_fov(mut self, vertical_fov: f64) -> Self {
        self.vertical_fov = vertical_fov;
        self.initialize()
    }

    /// Positions the camera at `lookfrom`, aimed at `lookat`.
    ///
    /// # Panics
    ///
    /// Panics if `lookfrom` and `lookat` are the same point, or if the current view-up vector is
    /// parallel to the new viewing direction.
    #[must_use]
    pub fn with_look_at(mut self, lookfrom: Point, lookat: Point) -> Self {
        self.lookfrom = lookfrom;
        self.lookat = lookat;
        self.initialize()
    }

    /// Sets the camera-relative up direction.
    ///
    /// # Panics
    ///
    /// Panics if `view_up` is zero or parallel to the current viewing direction.
    #[must_use]
    pub fn with_view_up(mut self, view_up: Vector) -> Self {
        self.view_up = view_up;
        self.initialize()
    }

    /// Sets the variation angle of rays through each pixel for defocus blur.
    ///
    /// A zero angle keeps the camera as a pinhole camera.
    ///
    /// # Panics
    ///
    /// Panics if `defocus_angle` is not finite or is outside `0..180` degrees.
    #[must_use]
    pub fn with_defocus_angle(mut self, defocus_angle: f64) -> Self {
        self.defocus_angle = defocus_angle;
        self.initialize()
    }

    /// Sets the distance from the camera origin to the plane of perfect focus.
    ///
    /// # Panics
    ///
    /// Panics if `focus_distance` is not positive and finite.
    #[must_use]
    pub fn with_focus_distance(mut self, focus_distance: f64) -> Self {
        self.focus_distance = focus_distance;
        self.initialize()
    }

    /// Sets the camera shutter interval used for sampled rays.
    ///
    /// # Panics
    ///
    /// Panics if either endpoint is non-finite, or if `start > end`.
    #[must_use]
    pub fn with_shutter_interval(mut self, start: f64, end: f64) -> Self {
        assert!(
            start.is_finite() && end.is_finite() && start <= end,
            "shutter interval must be finite and ordered"
        );
        self.shutter_start = start;
        self.shutter_end = end;
        self.initialize()
    }

    /// Sets the color returned by world rendering when a ray misses all scene objects.
    ///
    /// # Panics
    ///
    /// Panics if any color component is non-finite.
    #[must_use]
    pub fn with_background(mut self, background: LinearColor) -> Self {
        assert!(
            background.x().is_finite() && background.y().is_finite() && background.z().is_finite(),
            "background color components must be finite"
        );
        self.background = background;
        self
    }

    /// Returns the camera's ideal aspect ratio.
    #[must_use]
    pub fn aspect_ratio(self) -> f64 {
        self.aspect_ratio
    }

    /// Returns the rendered image width in pixels.
    #[must_use]
    pub fn image_width(self) -> u32 {
        self.image_width
    }

    /// Returns the rendered image height in pixels.
    #[must_use]
    pub fn image_height(self) -> u32 {
        self.image_height
    }

    /// Returns the number of random samples per pixel used by [`Self::render_world`].
    #[must_use]
    pub fn samples_per_pixel(self) -> u32 {
        self.samples_per_pixel
    }

    /// Returns the maximum ray-bounce recursion depth used by [`Self::render_world`].
    #[must_use]
    pub fn max_depth(self) -> u32 {
        self.max_depth
    }

    /// Returns the vertical field of view in degrees.
    #[must_use]
    pub fn vertical_fov(self) -> f64 {
        self.vertical_fov
    }

    /// Returns the defocus cone angle in degrees.
    #[must_use]
    pub fn defocus_angle(self) -> f64 {
        self.defocus_angle
    }

    /// Returns the distance from the camera origin to the plane of perfect focus.
    #[must_use]
    pub fn focus_distance(self) -> f64 {
        self.focus_distance
    }

    /// Returns the camera shutter interval used for sampled rays.
    #[must_use]
    pub fn shutter_interval(self) -> (f64, f64) {
        (self.shutter_start, self.shutter_end)
    }

    /// Returns the color used when world-rendered rays miss the scene.
    #[must_use]
    pub fn background(self) -> LinearColor {
        self.background
    }

    /// Returns the camera origin point.
    #[must_use]
    pub fn camera_center(self) -> Point {
        self.camera_center
    }

    /// Returns the point this camera is aimed at.
    #[must_use]
    pub fn lookat(self) -> Point {
        self.lookat
    }

    /// Returns the camera-relative up direction.
    #[must_use]
    pub fn view_up(self) -> Vector {
        self.view_up
    }

    /// Returns a ray from the camera center through the center of pixel `(x, y)`.
    ///
    /// Pixel coordinates are in storage order: `(0, 0)` is the upper-left pixel,
    /// rows scan left to right, and rows advance downward.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is outside the camera image dimensions.
    #[must_use]
    pub fn ray_for_pixel(self, x: u32, y: u32) -> Ray {
        assert!(x < self.image_width, "pixel x must be inside the image");
        assert!(y < self.image_height, "pixel y must be inside the image");

        let pixel_center = self.pixel00_loc
            + f64::from(x) * self.pixel_delta_u
            + f64::from(y) * self.pixel_delta_v;
        Ray::with_time(
            self.camera_center,
            pixel_center - self.camera_center,
            self.shutter_start,
        )
    }

    fn ray_for_pixel_sample(self, x: u32, y: u32, rng: &mut SampleRng) -> Ray {
        let offset = Self::sample_square(rng);
        let pixel_sample = self.pixel00_loc
            + (f64::from(x) + offset.x()) * self.pixel_delta_u
            + (f64::from(y) + offset.y()) * self.pixel_delta_v;
        let ray_origin = if self.defocus_angle <= 0.0 {
            self.camera_center
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_time = rng.random_range(self.shutter_start, self.shutter_end);
        Ray::with_time(ray_origin, pixel_sample - ray_origin, ray_time)
    }

    fn sample_square(rng: &mut SampleRng) -> Vector {
        Vector::new(rng.random_double() - 0.5, rng.random_double() - 0.5, 0.0)
    }

    fn defocus_disk_sample(self, rng: &mut SampleRng) -> Point {
        let point = rng.random_in_unit_disk();
        self.camera_center + point.x() * self.defocus_disk_u + point.y() * self.defocus_disk_v
    }

    fn pixel_seed(seed: u64, x: u32, y: u32) -> u64 {
        let mut z = seed
            ^ u64::from(x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(y).wrapping_mul(0xD1B5_4A32_D192_ED03);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn ray_color(
        ray: &Ray,
        depth: u32,
        world: &dyn Hittable,
        background: LinearColor,
        rng: &mut SampleRng,
    ) -> LinearColor {
        let mut current_ray = *ray;
        let mut attenuation = LinearColor::new(1.0, 1.0, 1.0);
        let mut color = LinearColor::default();

        for _ in 0..depth {
            let Some(record) = world.hit_with_rng(
                &current_ray,
                Interval::new(SHADOW_ACNE_EPSILON, INFINITY),
                rng,
            ) else {
                return color + component_mul(attenuation, background);
            };

            let emitted = record.material.emitted(record.u, record.v, record.point);
            color += component_mul(attenuation, emitted);

            let Some(scatter) = record.material.scatter(&current_ray, &record, rng) else {
                return color;
            };

            attenuation = component_mul(attenuation, scatter.attenuation);
            current_ray = scatter.ray;
        }

        color
    }

    fn render_world_pixel(self, x: u32, y: u32, world: &dyn Hittable) -> Rgb {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        for _ in 0..self.samples_per_pixel {
            let ray = self.ray_for_pixel_sample(x, y, &mut rng);
            pixel_color += Self::ray_color(&ray, self.max_depth, world, self.background, &mut rng);
        }
        Rgb::from_linear_color(pixel_color / f64::from(self.samples_per_pixel))
    }

    fn render_normal_pixel(self, x: u32, y: u32, world: &dyn Hittable) -> Rgb {
        let mut rng = SampleRng::new(Self::pixel_seed(self.rng_seed, x, y));
        let mut pixel_color = LinearColor::default();
        for _ in 0..self.samples_per_pixel {
            let ray = self.ray_for_pixel_sample(x, y, &mut rng);
            pixel_color += normal_scene_color(&ray, world);
        }
        Rgb::from_linear_color(pixel_color / f64::from(self.samples_per_pixel))
    }

    fn image_canvas(width: u32, height: u32, pixels: Vec<Rgb>) -> Canvas {
        Canvas::from_pixels_with_options(width, height, pixels, true, false)
    }

    /// Renders a canvas by evaluating `ray_color` for each emitted camera ray.
    pub fn render<F>(self, mut ray_color: F) -> Canvas
    where
        F: FnMut(&Ray) -> LinearColor,
    {
        let mut pixels = Vec::with_capacity(self.image_width as usize * self.image_height as usize);
        for y in 0..self.image_height {
            for x in 0..self.image_width {
                pixels.push(Rgb::from(ray_color(&self.ray_for_pixel(x, y))));
            }
        }
        Canvas::from_pixels_with_options(self.image_width, self.image_height, pixels, true, false)
    }

    /// Renders a hittable world using this camera's antialiasing sample count.
    pub fn render_world(self, world: &dyn Hittable) -> Canvas {
        let camera = self.initialize();
        Canvas::from_fn_independent_with_options(
            camera.image_width,
            camera.image_height,
            |x, y| camera.render_world_pixel(x, y, world),
            true,
            false,
        )
    }

    /// Renders a hittable world as surface-normal colors for debugging.
    pub fn render_world_normals(self, world: &dyn Hittable) -> Canvas {
        let camera = self.initialize();
        Canvas::from_fn_independent_with_options(
            camera.image_width,
            camera.image_height,
            |x, y| camera.render_normal_pixel(x, y, world),
            true,
            false,
        )
    }

    /// Renders a canvas while writing scanline progress messages to `log`.
    ///
    /// Use `std::io::stderr()` for book-style progress reporting that stays separate
    /// from generated PPM image output.
    ///
    /// # Errors
    ///
    /// Returns any write error produced by `log`.
    pub fn render_with_progress<F, W>(self, mut log: W, mut ray_color: F) -> io::Result<Canvas>
    where
        F: FnMut(&Ray) -> LinearColor,
        W: Write,
    {
        let mut pixels = Vec::with_capacity(self.image_width as usize * self.image_height as usize);
        for y in 0..self.image_height {
            write!(log, "\rScanlines remaining: {} ", self.image_height - y)?;
            log.flush()?;
            for x in 0..self.image_width {
                pixels.push(Rgb::from(ray_color(&self.ray_for_pixel(x, y))));
            }
        }
        writeln!(log, "\rDone.                 ")?;

        Ok(Self::image_canvas(
            self.image_width,
            self.image_height,
            pixels,
        ))
    }

    /// Renders a hittable world with antialiasing while writing scanline progress messages.
    ///
    /// # Errors
    ///
    /// Returns any write error produced by `log`.
    pub fn render_world_with_progress<W>(
        self,
        world: &dyn Hittable,
        mut log: W,
    ) -> io::Result<Canvas>
    where
        W: Write,
    {
        let camera = self.initialize();
        let mut pixels =
            Vec::with_capacity(camera.image_width as usize * camera.image_height as usize);
        for y in 0..camera.image_height {
            write!(log, "\rScanlines remaining: {} ", camera.image_height - y)?;
            log.flush()?;
            for x in 0..camera.image_width {
                pixels.push(camera.render_world_pixel(x, y, world));
            }
        }
        writeln!(log, "\rDone.                 ")?;

        Ok(Self::image_canvas(
            camera.image_width,
            camera.image_height,
            pixels,
        ))
    }
}

impl ProjectedSegment {
    /// Creates a projected segment if both source points project in front of the camera.
    #[must_use]
    pub fn from_points(camera: &Camera3D, p0: &[f64], p1: &[f64], color: Rgb) -> Option<Self> {
        Some(Self {
            a: camera.project(p0)?,
            b: camera.project(p1)?,
            color,
        })
    }

    /// Returns the average projected depth of the segment.
    #[must_use]
    pub fn average_depth(&self) -> f64 {
        (self.a.depth + self.b.depth) * 0.5
    }
}

/// Sorts projected segments back-to-front for painter-style wireframe rendering.
pub fn sort_segments_back_to_front(segments: &mut [ProjectedSegment]) {
    segments.sort_by(|a, b| {
        b.average_depth()
            .partial_cmp(&a.average_depth())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

impl Canvas {
    /// Draws already-projected colored segments.
    pub fn draw_projected_segments<I>(&mut self, segments: I)
    where
        I: IntoIterator<Item = ProjectedSegment>,
    {
        for segment in segments {
            self.draw_line(
                segment.color,
                segment.a.x,
                segment.a.y,
                segment.b.x,
                segment.b.y,
            );
        }
    }

    /// Projects and draws transformed edge lines without allocating a transformed edge matrix.
    pub fn draw_projected_edges(
        &mut self,
        edges: &EdgeMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        color: Rgb,
    ) {
        for (p0, p1) in edges.transformed_edges(transform) {
            if let Some(segment) = ProjectedSegment::from_points(camera, &p0, &p1, color) {
                self.draw_projected_segments([segment]);
            }
        }
    }

    /// Projects and draws transformed mesh triangle wireframes without allocating a transformed mesh.
    pub fn draw_projected_mesh_wireframe(
        &mut self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        color: Rgb,
        stride: usize,
    ) {
        let stride = stride.max(1);
        for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
            if idx % stride != 0 {
                continue;
            }
            let Some(ab) = ProjectedSegment::from_points(camera, &p0, &p1, color) else {
                continue;
            };
            let Some(bc) = ProjectedSegment::from_points(camera, &p1, &p2, color) else {
                continue;
            };
            let Some(ca) = ProjectedSegment::from_points(camera, &p2, &p0, color) else {
                continue;
            };
            self.draw_projected_segments([ab, bc, ca]);
        }
    }

    /// Projects, depth-sorts, and draws a transformed mesh as triangle wireframe segments.
    ///
    /// `color_for_triangle` receives the triangle index and average projected triangle depth.
    pub fn draw_projected_mesh_wireframe_depth_sorted<F>(
        &mut self,
        mesh: &PolygonMatrix,
        transform: &Matrix,
        camera: &Camera3D,
        stride: usize,
        color_for_triangle: F,
    ) where
        F: FnMut(usize, f64) -> Rgb,
    {
        let mut segments =
            camera.project_mesh_wireframe_segments(mesh, transform, stride, color_for_triangle);
        sort_segments_back_to_front(&mut segments);
        self.draw_projected_segments(segments);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn projected_mesh_wireframe_returns_three_segments_per_visible_triangle() {
        let mut mesh = PolygonMatrix::new();
        mesh.add_polygon((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        let camera = Camera3D::new(100, 100);
        let segments = camera.project_mesh_wireframe_segments(
            &mesh,
            &Matrix::identity_matrix(4),
            1,
            |_, _| Rgb::WHITE,
        );

        assert_eq!(segments.len(), 3);
    }

    #[test]
    fn camera3d_default_projection_matches_legacy_camera_distance() {
        let camera = Camera3D::new(100, 100);
        let point = camera.project(&[0.0, 0.0, 0.0]).expect("visible");

        assert_close(point.x, 50.0);
        assert_close(point.y, 50.0);
        assert_close(point.depth, 900.0);
    }

    #[test]
    fn camera3d_can_be_positioned_with_look_at() {
        let camera = Camera3D::new(100, 100)
            .with_look_at(Point::new(0.0, 0.0, -10.0), Point::new(0.0, 0.0, 0.0))
            .with_focal_length(10.0)
            .with_near_depth(0.1);

        let center = camera.project(&[0.0, 0.0, 0.0]).expect("center visible");
        let right = camera.project(&[1.0, 0.0, 0.0]).expect("right visible");

        assert_close(center.x, 50.0);
        assert_close(center.y, 50.0);
        assert_close(center.depth, 10.0);
        assert!(right.x > center.x);
    }

    #[test]
    fn camera3d_vertical_fov_sets_projection_scale() {
        let camera = Camera3D::new(100, 100)
            .with_look_at(Point::new(0.0, 0.0, -10.0), Point::new(0.0, 0.0, 0.0))
            .with_vertical_fov(90.0)
            .with_near_depth(0.1);

        let top = camera.project(&[0.0, 10.0, 0.0]).expect("top visible");

        assert_close(top.y, 0.0);
    }
    #[test]
    fn ray_camera_uses_actual_integer_image_ratio() {
        let camera = RayCamera::new(400, 16.0 / 9.0);
        assert_eq!(camera.image_width(), 400);
        assert_eq!(camera.image_height(), 225);
    }
    #[test]
    fn ray_camera_sends_center_pixel_forward() {
        let camera = RayCamera::new(400, 16.0 / 9.0);
        let ray = camera.ray_for_pixel(200, 112);

        assert_close(ray.origin().x(), 0.0);
        assert_close(ray.origin().y(), 0.0);
        assert_close(ray.origin().z(), 0.0);
        assert!(ray.direction().z() < 0.0);
        assert!(ray.direction().x().abs() < 0.01);
        assert!(ray.direction().y().abs() < 0.01);
    }
    #[test]
    fn ray_camera_tracks_antialiasing_sample_count() {
        let camera = RayCamera::default()
            .with_image_width(40)
            .with_aspect_ratio(16.0 / 9.0)
            .with_samples_per_pixel(25)
            .with_max_depth(50);

        assert_eq!(camera.image_width(), 40);
        assert_eq!(camera.image_height(), 22);
        assert_eq!(camera.samples_per_pixel(), 25);
        assert_eq!(camera.max_depth(), 50);
        assert_close(camera.defocus_angle(), 0.0);
        assert_close(camera.focus_distance(), 1.0);
    }

    #[test]
    fn ray_camera_tracks_background_color() {
        let background = LinearColor::new(0.1, 0.2, 0.3);
        let camera = RayCamera::new(20, 1.0).with_background(background);
        let empty_world = crate::graphics::raytracing::HittableList::new();

        let canvas = camera.render_world(&empty_world);

        assert_eq!(camera.background(), background);
        assert_eq!(canvas.pixels()[0], Rgb::from_linear_color(background));
    }

    #[test]
    fn ray_camera_vertical_fov_controls_ray_spread() {
        let wide = RayCamera::new(101, 1.0).with_vertical_fov(90.0);
        let narrow = RayCamera::new(101, 1.0).with_vertical_fov(20.0);

        let wide_top = wide.ray_for_pixel(50, 0).direction().normalized();
        let narrow_top = narrow.ray_for_pixel(50, 0).direction().normalized();

        assert!(wide_top.y().abs() > narrow_top.y().abs());
    }
    #[test]
    fn ray_camera_can_be_positioned_with_look_at() {
        let lookfrom = Point::new(-2.0, 2.0, 1.0);
        let lookat = Point::new(0.0, 0.0, -1.0);
        let camera = RayCamera::new(101, 1.0)
            .with_look_at(lookfrom, lookat)
            .with_view_up(Vector::new(0.0, 1.0, 0.0));

        let ray = camera.ray_for_pixel(50, 50);
        let expected_direction = (lookat - lookfrom).normalized();
        let actual_direction = ray.direction().normalized();

        assert_eq!(*ray.origin(), lookfrom);
        assert_close(actual_direction.dot(expected_direction), 1.0);
    }
    #[test]
    fn ray_camera_defocus_blur_offsets_sample_origin() {
        let mut rng = SampleRng::new(17);
        let pinhole = RayCamera::new(101, 1.0);
        let defocused = pinhole.with_defocus_angle(10.0).with_focus_distance(3.4);

        let pinhole_ray = pinhole.ray_for_pixel_sample(50, 50, &mut rng);
        let defocused_ray = defocused.ray_for_pixel_sample(50, 50, &mut rng);

        assert_eq!(*pinhole_ray.origin(), pinhole.camera_center());
        assert_ne!(*defocused_ray.origin(), defocused.camera_center());
    }

    #[test]
    fn ray_camera_samples_shutter_interval() {
        let mut rng = SampleRng::new(19);
        let camera = RayCamera::new(101, 1.0).with_shutter_interval(0.25, 0.75);
        let center_ray = camera.ray_for_pixel(50, 50);
        let sampled_ray = camera.ray_for_pixel_sample(50, 50, &mut rng);

        assert_close(center_ray.time(), 0.25);
        assert!((0.25..0.75).contains(&sampled_ray.time()));
        assert_eq!(camera.shutter_interval(), (0.25, 0.75));
    }

    #[test]
    fn ray_camera_world_render_is_seeded_and_deterministic() {
        let world = crate::graphics::raytracing::normal_sphere_world();
        let camera = RayCamera::new(20, 16.0 / 9.0)
            .with_samples_per_pixel(4)
            .with_max_depth(3)
            .with_rng_seed(123);

        let first = camera.render_world(&world);
        let second = camera.render_world(&world);

        assert_eq!(first.pixels(), second.pixels());
    }
    #[test]
    fn ray_camera_world_render_uses_image_coordinate_canvas() {
        let world = crate::graphics::raytracing::normal_sphere_world();
        let canvas = RayCamera::new(4, 1.0).render_world(&world);

        assert!(canvas.upper_left_origin);
        assert!(!canvas.wrapped);
    }
}

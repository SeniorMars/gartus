use super::{colors::Rgb, display::Canvas, draw::triangle_normal, texture::Texture};

const PERSPECTIVE_EPS: f64 = 1e-12;

/// A screen-space vertex with normalized texture coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturedVertex {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate for z-buffering.
    pub z: f64,
    /// Reciprocal perspective depth used for perspective-correct texture interpolation.
    pub inv_w: f64,
    /// Horizontal texture coordinate, usually in `0..=1`.
    pub s: f64,
    /// Vertical texture coordinate, usually in `0..=1`.
    pub t: f64,
}

impl TexturedVertex {
    /// Creates a textured vertex.
    #[must_use]
    pub fn new(x_coord: f64, y_coord: f64, z_coord: f64, s_coord: f64, t_coord: f64) -> Self {
        Self::from_projected(
            x_coord,
            y_coord,
            z_coord,
            texture_inv_depth_from_z(z_coord),
            s_coord,
            t_coord,
        )
    }

    /// Creates a textured vertex with an explicit reciprocal perspective depth.
    #[must_use]
    pub const fn from_projected(
        x_coord: f64,
        y_coord: f64,
        z_coord: f64,
        inv_w: f64,
        s_coord: f64,
        t_coord: f64,
    ) -> Self {
        Self {
            x: x_coord,
            y: y_coord,
            z: z_coord,
            inv_w,
            s: s_coord,
            t: t_coord,
        }
    }

    fn is_finite(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
            && self.inv_w.is_finite()
            && self.s.is_finite()
            && self.t.is_finite()
    }

    pub(crate) fn position_tuple(self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

impl Canvas {
    /// Draws a textured triangle with linear interpolation of texture coordinates.
    pub fn draw_textured_triangle(&mut self, texture: &Texture, vertices: [TexturedVertex; 3]) {
        self.draw_textured_triangle_with_color(texture, vertices, |sample| sample);
    }

    /// Draws a textured triangle and multiplies each sampled texel by `modulation`.
    pub fn draw_textured_triangle_modulated(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        modulation: Rgb,
    ) {
        self.draw_textured_triangle_with_color(texture, vertices, |sample| {
            modulate_rgb(sample, modulation)
        });
    }

    fn draw_textured_triangle_with_color(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        mut color: impl FnMut(Rgb) -> Rgb,
    ) {
        self.draw_textured_triangle_with_optional_color(texture, vertices, |sample| {
            Some(color(sample))
        });
    }

    fn draw_textured_triangle_with_optional_color(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        mut color: impl FnMut(Rgb) -> Option<Rgb>,
    ) {
        if !vertices.iter().all(TexturedVertex::is_finite) {
            return;
        }

        let normal = triangle_normal(
            vertices[0].position_tuple(),
            vertices[1].position_tuple(),
            vertices[2].position_tuple(),
        );
        if normal[2] <= 0.0 {
            return;
        }

        self.raster_textured_triangle_unculled(texture, vertices, &mut color);
    }

    /// Draws a textured triangle without backface culling.
    pub fn draw_textured_triangle_unculled(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
    ) {
        self.draw_textured_triangle_unculled_with_color(texture, vertices, |sample| sample);
    }

    /// Draws a modulated textured triangle without backface culling.
    pub fn draw_textured_triangle_modulated_unculled(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        modulation: Rgb,
    ) {
        self.draw_textured_triangle_unculled_with_color(texture, vertices, |sample| {
            modulate_rgb(sample, modulation)
        });
    }

    fn draw_textured_triangle_unculled_with_color(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        mut color: impl FnMut(Rgb) -> Rgb,
    ) {
        self.draw_textured_triangle_unculled_with_optional_color(texture, vertices, |sample| {
            Some(color(sample))
        });
    }

    /// Draws a modulated unculled textured triangle and skips texels accepted by `transparent`.
    pub fn draw_textured_triangle_modulated_unculled_keyed(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        modulation: Rgb,
        transparent: impl Fn(Rgb) -> bool,
    ) {
        self.draw_textured_triangle_unculled_with_optional_color(texture, vertices, |sample| {
            if transparent(sample) {
                None
            } else {
                Some(modulate_rgb(sample, modulation))
            }
        });
    }

    fn draw_textured_triangle_unculled_with_optional_color(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        mut color: impl FnMut(Rgb) -> Option<Rgb>,
    ) {
        if !vertices.iter().all(TexturedVertex::is_finite) {
            return;
        }

        self.raster_textured_triangle_unculled(texture, vertices, &mut color);
    }

    /// Draws a textured quad as two textured triangles.
    ///
    /// Texture coordinates are assigned as bottom-left, bottom-right, top-right, top-left.
    pub fn draw_textured_quad(&mut self, texture: &Texture, points: [(f64, f64, f64); 4]) {
        let vertices = [
            TexturedVertex::new(points[0].0, points[0].1, points[0].2, 0.0, 0.0),
            TexturedVertex::new(points[1].0, points[1].1, points[1].2, 1.0, 0.0),
            TexturedVertex::new(points[2].0, points[2].1, points[2].2, 1.0, 1.0),
            TexturedVertex::new(points[3].0, points[3].1, points[3].2, 0.0, 1.0),
        ];

        self.draw_textured_triangle(texture, [vertices[0], vertices[1], vertices[2]]);
        self.draw_textured_triangle(texture, [vertices[0], vertices[2], vertices[3]]);
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_lines
    )]
    fn raster_textured_triangle_unculled(
        &mut self,
        texture: &Texture,
        vertices: [TexturedVertex; 3],
        color: &mut impl FnMut(Rgb) -> Option<Rgb>,
    ) {
        let mut min_x = vertices
            .iter()
            .map(|vertex| vertex.x)
            .fold(f64::INFINITY, f64::min)
            .floor() as i64;
        let mut max_x = vertices
            .iter()
            .map(|vertex| vertex.x)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as i64;
        let mut min_y = vertices
            .iter()
            .map(|vertex| vertex.y)
            .fold(f64::INFINITY, f64::min)
            .floor() as i64;
        let mut max_y = vertices
            .iter()
            .map(|vertex| vertex.y)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as i64;

        if !self.wrapped {
            let width = i64::from(self.width());
            let height = i64::from(self.height());
            min_x = min_x.max(0);
            max_x = max_x.min(width - 1);
            min_y = min_y.max(0);
            max_y = max_y.min(height - 1);
        }

        let Some(bounds) = texture_bounds(min_x, max_x, min_y, max_y) else {
            return;
        };
        let denom = barycentric_denominator(vertices);
        if denom.abs() < f64::EPSILON {
            return;
        }

        let [a, b, c] = vertices;
        let dw0_dx = (b.y - c.y) / denom;
        let dw0_dy = (c.x - b.x) / denom;
        let dw1_dx = (c.y - a.y) / denom;
        let dw1_dy = (a.x - c.x) / denom;

        let q0 = a.inv_w;
        let q1 = b.inv_w;
        let q2 = c.inv_w;

        let ctx = TexturedTriangleContext {
            dw0_dx,
            dw1_dx,
            dw0_dy,
            dw1_dy,
            q0,
            q1,
            q2,
            s0_q0: a.s * q0,
            s1_q1: b.s * q1,
            s2_q2: c.s * q2,
            t0_q0: a.t * q0,
            t1_q1: b.t * q1,
            t2_q2: c.t * q2,
        };

        let sampler = texture.active_sampler();
        let use_mips = sampler.uses_mips();
        let clipped_unwrapped = !self.wrapped;
        let block_count = if use_mips {
            usize::try_from((bounds.max_x - bounds.min_x) / 2 + 1)
                .expect("texture bounds width is non-negative")
        } else {
            0
        };
        let mut previous_lod_row = vec![None; block_count];
        for y in bounds.min_y..=bounds.max_y {
            let Some((mut x0, mut x1)) = covered_x_span(vertices, y as f64) else {
                continue;
            };
            x0 = x0.max(bounds.min_x);
            x1 = x1.min(bounds.max_x);
            if x0 > x1 {
                continue;
            }

            let mut current_lod_row = if use_mips && (y - bounds.min_y) % 2 == 0 {
                vec![None; block_count]
            } else {
                Vec::new()
            };
            let (mut w0, mut w1, _) = barycentric_weights(vertices, x0 as f64, y as f64, denom);
            for x in x0..=x1 {
                let w2 = 1.0 - w0 - w1;
                if inside_barycentric(w0, w1, w2) {
                    let z =
                        w0.mul_add(vertices[0].z, w1.mul_add(vertices[1].z, w2 * vertices[2].z));
                    let visible_index = if clipped_unwrapped {
                        self.visible_pixel_index_clipped_unchecked(x, y, z)
                    } else {
                        self.visible_pixel_index(x, y, z)
                    };
                    let Some(index) = visible_index else {
                        w0 += ctx.dw0_dx;
                        w1 += ctx.dw1_dx;
                        continue;
                    };
                    let (s, t) = perspective_texture_coordinates_fast(vertices, &ctx, w0, w1, w2);
                    let lod = if use_mips {
                        let mut lod_cache = LodCache {
                            x,
                            min_x: bounds.min_x,
                            row_offset: y - bounds.min_y,
                            previous: &previous_lod_row,
                            current: &mut current_lod_row,
                        };
                        lod_for_pixel(texture, &ctx, vertices, (s, t), (w0, w1), &mut lod_cache)
                    } else {
                        0.0
                    };
                    let sample = sampler.sample(s, t, lod);
                    if let Some(color) = color(sample) {
                        self.plot_z_index_unchecked(index, color, z);
                    }
                }
                w0 += ctx.dw0_dx;
                w1 += ctx.dw1_dx;
            }
            if !current_lod_row.is_empty() {
                previous_lod_row = current_lod_row;
            }
        }
    }
}

struct TextureBounds {
    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
}

fn texture_bounds(min_x: i64, max_x: i64, min_y: i64, max_y: i64) -> Option<TextureBounds> {
    if min_x > max_x || min_y > max_y {
        return None;
    }
    Some(TextureBounds {
        min_x,
        max_x,
        min_y,
        max_y,
    })
}

fn barycentric_denominator(vertices: [TexturedVertex; 3]) -> f64 {
    let [a, b, c] = vertices;
    (b.y - c.y).mul_add(a.x - c.x, (c.x - b.x) * (a.y - c.y))
}

fn barycentric_weights(
    vertices: [TexturedVertex; 3],
    sample_x: f64,
    sample_y: f64,
    denominator: f64,
) -> (f64, f64, f64) {
    let [a, b, c] = vertices;
    let w0 = (b.y - c.y).mul_add(sample_x - c.x, (c.x - b.x) * (sample_y - c.y)) / denominator;
    let w1 = (c.y - a.y).mul_add(sample_x - c.x, (a.x - c.x) * (sample_y - c.y)) / denominator;
    let w2 = 1.0 - w0 - w1;
    (w0, w1, w2)
}

fn inside_barycentric(w0: f64, w1: f64, w2: f64) -> bool {
    const EDGE_EPS: f64 = -1e-9;
    w0 >= EDGE_EPS && w1 >= EDGE_EPS && w2 >= EDGE_EPS
}

#[allow(clippy::cast_possible_truncation)]
fn covered_x_span(vertices: [TexturedVertex; 3], y: f64) -> Option<(i64, i64)> {
    let mut intersections = [0.0; 3];
    let mut count = 0;
    for [a, b] in [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[0]],
    ] {
        if let Some(x) = edge_x_at_y(a, b, y) {
            intersections[count] = x;
            count += 1;
        }
    }

    match count {
        0 => None,
        1 => {
            let x = intersections[0].round() as i64;
            Some((x, x))
        }
        _ => {
            let mut min_x = intersections[0];
            let mut max_x = intersections[0];
            for x in intersections.iter().take(count).skip(1) {
                min_x = min_x.min(*x);
                max_x = max_x.max(*x);
            }
            Some((min_x.floor() as i64, max_x.ceil() as i64))
        }
    }
}

fn edge_x_at_y(a: TexturedVertex, b: TexturedVertex, y: f64) -> Option<f64> {
    if (a.y - b.y).abs() < f64::EPSILON {
        let min_y = a.y.min(b.y);
        let max_y = a.y.max(b.y);
        if y >= min_y && y <= max_y {
            return Some(a.x.min(b.x));
        }
        return None;
    }

    let min_y = a.y.min(b.y);
    let max_y = a.y.max(b.y);
    if y < min_y || y > max_y {
        return None;
    }

    let t = (y - a.y) / (b.y - a.y);
    Some(a.x + (b.x - a.x) * t)
}

struct LodCache<'a> {
    x: i64,
    min_x: i64,
    row_offset: i64,
    previous: &'a [Option<f64>],
    current: &'a mut [Option<f64>],
}

fn lod_for_pixel(
    texture: &Texture,
    ctx: &TexturedTriangleContext,
    vertices: [TexturedVertex; 3],
    texcoord: (f64, f64),
    weights: (f64, f64),
    cache: &mut LodCache<'_>,
) -> f64 {
    let block_index =
        usize::try_from((cache.x - cache.min_x) / 2).expect("texture block index is non-negative");
    if cache.row_offset % 2 == 1 {
        if let Some(Some(lod)) = cache.previous.get(block_index) {
            return *lod;
        }
    } else if let Some(Some(lod)) = cache.current.get(block_index) {
        return *lod;
    }

    let (s, t) = texcoord;
    let (w0, w1) = weights;
    let lod = texture_lod(texture, ctx, vertices, s, t, w0, w1);
    if cache.row_offset % 2 == 0
        && let Some(slot) = cache.current.get_mut(block_index)
    {
        *slot = Some(lod);
    }
    lod
}

struct TexturedTriangleContext {
    dw0_dx: f64,
    dw1_dx: f64,
    dw0_dy: f64,
    dw1_dy: f64,
    q0: f64,
    q1: f64,
    q2: f64,
    s0_q0: f64,
    s1_q1: f64,
    s2_q2: f64,
    t0_q0: f64,
    t1_q1: f64,
    t2_q2: f64,
}

fn perspective_texture_coordinates_fast(
    vertices: [TexturedVertex; 3],
    ctx: &TexturedTriangleContext,
    w0: f64,
    w1: f64,
    w2: f64,
) -> (f64, f64) {
    let denom = w0.mul_add(ctx.q0, w1.mul_add(ctx.q1, w2 * ctx.q2));
    if denom.abs() < PERSPECTIVE_EPS {
        return (
            w0.mul_add(vertices[0].s, w1.mul_add(vertices[1].s, w2 * vertices[2].s)),
            w0.mul_add(vertices[0].t, w1.mul_add(vertices[1].t, w2 * vertices[2].t)),
        );
    }

    let s = w0.mul_add(ctx.s0_q0, w1.mul_add(ctx.s1_q1, w2 * ctx.s2_q2)) / denom;
    let t = w0.mul_add(ctx.t0_q0, w1.mul_add(ctx.t1_q1, w2 * ctx.t2_q2)) / denom;
    (s, t)
}

fn texture_lod(
    texture: &Texture,
    ctx: &TexturedTriangleContext,
    vertices: [TexturedVertex; 3],
    s: f64,
    t: f64,
    w0: f64,
    w1: f64,
) -> f64 {
    let right_w0 = w0 + ctx.dw0_dx;
    let right_w1 = w1 + ctx.dw1_dx;
    let right_w2 = 1.0 - right_w0 - right_w1;

    let down_w0 = w0 + ctx.dw0_dy;
    let down_w1 = w1 + ctx.dw1_dy;
    let down_w2 = 1.0 - down_w0 - down_w1;

    let (right_s, right_t) =
        perspective_texture_coordinates_fast(vertices, ctx, right_w0, right_w1, right_w2);
    let (down_s, down_t) =
        perspective_texture_coordinates_fast(vertices, ctx, down_w0, down_w1, down_w2);
    texture.lod_from_derivatives(right_s - s, right_t - t, down_s - s, down_t - t)
}

fn texture_inv_depth_from_z(z: f64) -> f64 {
    let depth = z.abs();
    if depth < PERSPECTIVE_EPS {
        1.0
    } else {
        1.0 / depth
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn modulate_rgb(texture: Rgb, modulation: Rgb) -> Rgb {
    let channel = |texture: u8, modulation: u8| {
        ((u16::from(texture) * u16::from(modulation) + 127) / 255) as u8
    };
    Rgb::new(
        channel(texture.red, modulation.red),
        channel(texture.green, modulation.green),
        channel(texture.blue, modulation.blue),
    )
}

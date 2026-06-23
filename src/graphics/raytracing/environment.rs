//! Environment lighting and importance sampling.

use super::LinearColor;
use crate::{
    gmath::{
        random::SampleRng,
        vector::{Point, Vector},
    },
    graphics::{
        colors::LinearRgb,
        display::Canvas,
        texture::{
            SurfaceTexture, Texture as BitmapTexture, TextureFilter, TextureSample, TextureWrap,
        },
    },
};

const TAU: f64 = std::f64::consts::TAU;

/// Lat-long environment light with luminance-weighted importance sampling.
#[derive(Clone, Debug)]
pub struct EnvironmentLight {
    texture: BitmapTexture,
    constant_radiance: Option<LinearColor>,
    weights: Vec<f64>,
    cdf: Vec<f64>,
    total_weight: f64,
    width: usize,
    height: usize,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
impl EnvironmentLight {
    /// Creates an environment from a bitmap texture.
    #[must_use]
    pub fn new(texture: BitmapTexture) -> Self {
        let mut environment = Self {
            texture: texture
                .wrap(TextureWrap::Repeat, TextureWrap::Clamp)
                .filter(TextureFilter::Linear),
            constant_radiance: None,
            weights: Vec::new(),
            cdf: Vec::new(),
            total_weight: 0.0,
            width: 0,
            height: 0,
        };
        environment.rebuild_distribution();
        environment
    }

    /// Creates an environment from an existing canvas.
    #[must_use]
    pub fn from_canvas(canvas: Canvas) -> Self {
        Self::new(BitmapTexture::from_canvas(canvas))
    }

    /// Creates a constant-color environment.
    #[must_use]
    pub fn constant(color: LinearColor) -> Self {
        let mut environment = Self::from_canvas(Canvas::from_pixels_rgb_only(
            1,
            1,
            vec![color.raw_encode()],
            true,
            false,
        ));
        environment.constant_radiance = Some(color);
        environment.weights.clear();
        environment.cdf.clear();
        environment.total_weight = 0.0;
        environment
    }

    /// Loads an environment image file.
    ///
    /// # Errors
    ///
    /// Returns an error if the image cannot be loaded or converted into a canvas.
    #[cfg(feature = "external")]
    pub fn from_file(path: impl AsRef<str>) -> Result<Self, Box<dyn std::error::Error>> {
        let canvas = crate::external::ppmify(path.as_ref(), false)?;
        Ok(Self::from_canvas(canvas))
    }

    /// Returns the underlying texture.
    #[must_use]
    pub const fn texture(&self) -> &BitmapTexture {
        &self.texture
    }

    /// Returns radiance for a world-space direction.
    #[must_use]
    pub fn radiance(&self, direction: Vector) -> LinearColor {
        if let Some(color) = self.constant_radiance {
            return color;
        }
        let (u, v) = direction_to_latlong_uv(direction);
        self.texture
            .sample_linear(TextureSample::new(u, v, Point::default()))
    }

    /// Samples an incident direction and its solid-angle PDF.
    #[must_use]
    pub fn sample_direction(&self, rng: &mut SampleRng) -> (Vector, f64) {
        if self.total_weight <= f64::EPSILON || self.weights.is_empty() {
            return (
                rng.random_unit_vector_spherical(),
                1.0 / (4.0 * std::f64::consts::PI),
            );
        }

        let target = rng.random_double() * self.total_weight;
        let index = self.cdf.partition_point(|value| *value < target);
        let index = index.min(self.weights.len() - 1);
        let x = index % self.width;
        let y = index / self.width;
        let u = (x as f64 + rng.random_double()) / self.width as f64;
        let theta0 = std::f64::consts::PI * (y as f64 / self.height as f64);
        let theta1 = std::f64::consts::PI * ((y + 1) as f64 / self.height as f64);
        let cos_theta0 = theta0.cos();
        let cos_theta1 = theta1.cos();
        let cos_theta = cos_theta0 + rng.random_double() * (cos_theta1 - cos_theta0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = TAU * u;
        let direction = Vector::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
        (direction, self.pdf_value(direction))
    }

    /// Returns the solid-angle PDF for a world-space direction.
    #[must_use]
    pub fn pdf_value(&self, direction: Vector) -> f64 {
        if self.total_weight <= f64::EPSILON || self.weights.is_empty() {
            return 1.0 / (4.0 * std::f64::consts::PI);
        }

        let direction = direction.normalized();
        let theta = direction.y().clamp(-1.0, 1.0).acos();
        let phi = direction.z().atan2(direction.x()).rem_euclid(TAU);
        let x = ((phi / TAU) * self.width as f64).floor() as usize;
        let y = ((theta / std::f64::consts::PI) * self.height as f64).floor() as usize;
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let index = y * self.width + x;
        let probability = self.weights[index] / self.total_weight;
        let solid_angle = self.pixel_solid_angle(y);
        if solid_angle <= f64::EPSILON {
            0.0
        } else {
            probability / solid_angle
        }
    }

    fn rebuild_distribution(&mut self) {
        let image = self.texture.image();
        self.width = usize::try_from(image.width()).expect("environment width should fit usize");
        self.height = usize::try_from(image.height()).expect("environment height should fit usize");
        self.weights.clear();
        self.cdf.clear();
        self.total_weight = 0.0;

        if self.width == 0 || self.height == 0 {
            return;
        }

        self.weights.reserve_exact(self.width * self.height);
        self.cdf.reserve_exact(self.width * self.height);
        for y in 0..self.height {
            let theta = std::f64::consts::PI * ((y as f64 + 0.5) / self.height as f64);
            let sin_theta = theta.sin().max(0.0);
            for x in 0..self.width {
                let index = y * self.width + x;
                let color = LinearRgb::from_rgb_srgb(image.pixels()[index]);
                let weight = luminance(color).max(0.0) * sin_theta;
                self.total_weight += weight;
                self.weights.push(weight);
                self.cdf.push(self.total_weight);
            }
        }
    }

    fn pixel_solid_angle(&self, y: usize) -> f64 {
        let theta0 = std::f64::consts::PI * (y as f64 / self.height as f64);
        let theta1 = std::f64::consts::PI * ((y + 1) as f64 / self.height as f64);
        (TAU / self.width as f64) * (theta0.cos() - theta1.cos()).max(0.0)
    }
}

fn direction_to_latlong_uv(direction: Vector) -> (f64, f64) {
    let direction = direction.normalized();
    let phi = direction.z().atan2(direction.x()).rem_euclid(TAU);
    let theta = direction.y().clamp(-1.0, 1.0).acos();
    (phi / TAU, 1.0 - theta / std::f64::consts::PI)
}

fn luminance(color: LinearRgb) -> f64 {
    0.2126 * color.red + 0.7152 * color.green + 0.0722 * color.blue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::colors::Rgb;
    #[test]
    fn environment_light_constant_has_uniform_pdf() {
        let environment = EnvironmentLight::constant(LinearColor::new(0.25, 0.5, 0.75));
        let direction = Vector::new(1.0, 0.0, 0.0);

        assert_eq!(
            environment.radiance(direction),
            LinearColor::new(0.25, 0.5, 0.75)
        );
        assert!(
            (environment.pdf_value(direction) - 1.0 / (4.0 * std::f64::consts::PI)).abs() < 1e-12
        );
    }

    #[test]
    fn environment_light_constant_preserves_hdr_radiance() {
        let environment = EnvironmentLight::constant(LinearColor::new(4.0, 2.0, 1.5));
        let radiance = environment.radiance(Vector::new(0.0, 1.0, 0.0));

        assert_eq!(radiance, LinearColor::new(4.0, 2.0, 1.5));
        assert!(
            (environment.pdf_value(Vector::new(0.0, 1.0, 0.0))
                - 1.0 / (4.0 * std::f64::consts::PI))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn environment_light_samples_brighter_texels_more_often() {
        let canvas = Canvas::from_pixels_rgb_only(2, 1, vec![Rgb::BLACK, Rgb::WHITE], true, false);
        let environment = EnvironmentLight::from_canvas(canvas);
        let bright_direction = Vector::new(-1.0, 0.0, 0.0);
        let dark_direction = Vector::new(1.0, 0.0, 0.0);

        assert!(environment.pdf_value(bright_direction) > environment.pdf_value(dark_direction));
        let mut rng = SampleRng::new(5);
        let (_direction, pdf) = environment.sample_direction(&mut rng);
        assert!(pdf.is_finite() && pdf > 0.0);
    }

    #[test]
    fn environment_light_samples_texels_uniformly_over_solid_angle() {
        let canvas = Canvas::from_pixels_rgb_only(1, 2, vec![Rgb::WHITE, Rgb::BLACK], true, false);
        let environment = EnvironmentLight::from_canvas(canvas);
        let mut rng = SampleRng::new(42);
        let mut mean_cos_theta = 0.0;
        let sample_count = 4096;

        for _ in 0..sample_count {
            let (direction, pdf) = environment.sample_direction(&mut rng);
            assert!(
                (pdf - 1.0 / (2.0 * std::f64::consts::PI)).abs() < 1.0e-12,
                "{pdf}"
            );
            assert!(direction.y() >= -1.0e-12, "{direction:?}");
            mean_cos_theta += direction.normalized().y();
        }
        mean_cos_theta /= f64::from(sample_count);

        assert!((mean_cos_theta - 0.5).abs() < 0.03, "{mean_cos_theta}");
    }
}

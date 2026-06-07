//! This example implements "The Double-Decker Cyber-Carousel" in a 3D orthographic pipeline:
//! 1. A tiered mechanical carousel stage with concentric platforms, gear teeth, support column, and base cones.
//! 2. Two concentric rings of characters rotating in opposite directions on the lower stage:
//!    - Outer ring: Kirby (dtorresr), WALL-E (ysato), Duck (aylu), and Penguin (bamado) rotating clockwise.
//!    - Inner ring: Amy Lee (alee3), Yuyingq (yuyingq), Gautamj (gautamj), and Eleehuaj (eleehuaj) rotating counter-clockwise.
//! 3. Solid Phong/Toon meshes drawn first, with overlay wireframe outlines with a Z bias for a neon cyberpunk style.
//! 4. Accurate bounding-box normalization so no characters or spire pieces clip or sink below their platforms.
//! 5. Floating neon 3D shapes on the upper platform rotating in the opposite direction.
//! 6. Stacked, clockwork centerpiece spire sitting on top of the column with precise, collision-free layout.
//! 7. Orbiting point lights with wireframe holographic energy cages.
//! 8. Outer tilted Saturn-like planetary ring and low-intensity concentric floor grid circles.
//! 9. Starry sky backdrop with twinkling stars and a swirling 3D vortex of multi-colored space dust drifting upwards.
//!
//! Saves the final animation to `final/scotty_carousel.gif` and a preview to `final/scotty_carousel.png`.

use gartus::{external, gmath::vector::Vector, prelude::*};
use std::{error::Error, f64::consts::PI};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 960;
const FRAMES: usize = 96;

const KIRBY_PATH: &str = "examples/data/meshes/scotty/dtorresr.obj";
const WALLE_PATH: &str = "examples/data/meshes/scotty/ysato.obj";
const DUCK_PATH: &str = "examples/data/meshes/scotty/aylu.obj";
const PENGUIN_PATH: &str = "examples/data/meshes/scotty/bamado.obj";

const ALEE3_PATH: &str = "examples/data/meshes/scotty/alee3.obj";
const YUYINGQ_PATH: &str = "examples/data/meshes/scotty/yuyingq.obj";
const GAUTAMJ_PATH: &str = "examples/data/meshes/scotty/gautamj.obj";
const ELEEHUAJ_PATH: &str = "examples/data/meshes/scotty/eleehuaj.obj";

const OUTPUT_GIF: &str = "final/scotty_carousel.gif";
const OUTPUT_PREVIEW: &str = "final/scotty_carousel.png";

struct Model {
    mesh: PolygonMatrix,
    normalize: Matrix,
    material: PhongMaterial,
    outline_color: Rgb,
}

struct Scene {
    // Outer Ring
    kirby: Model,
    walle: Model,
    duck: Model,
    penguin: Model,
    // Inner Ring
    alee3: Model,
    yuyingq: Model,
    gautamj: Model,
    eleehuaj: Model,
    stage: PolygonMatrix,
    halo_ring: PolygonMatrix,
    planetary_ring: PolygonMatrix,
    floor_grid: PolygonMatrix,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("Could not render Scotty Carousel example:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    println!("Loading student meshes from scotty/...");
    let scene = load_scene()?;

    println!("Starting render pipeline for {FRAMES} frames...");
    let options = AnimationRenderOptions::new(
        "final/scotty_carousel_frames",
        "scotty-carousel-",
        FRAMES,
        OUTPUT_GIF,
    )
    .delay_cs(4)
    .preview(FRAMES / 3, OUTPUT_PREVIEW)
    .unique_frame_dir(true);

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(frame, &scene)))?;

    println!("Saved preview to {OUTPUT_PREVIEW}");
    println!("Saved animation to {OUTPUT_GIF}");
    Ok(())
}

fn load_model(
    path: &str,
    size: f64,
    material: PhongMaterial,
    outline_color: Rgb,
) -> Result<Model, Box<dyn Error>> {
    let mesh = external::meshify(path)?;
    let norm_init = external::normalize_mesh_transform(&mesh, size, external::MeshUpAxis::Y);
    let normalized = mesh.apply(&norm_init);
    let bounds = normalized
        .bounds()
        .ok_or_else(|| format!("Mesh {path} has no bounds"))?;
    // Shift model up so its lowest Y coordinate is exactly 0.0
    let adjust = Matrix::translate(0.0, -bounds.min.1, 0.0);
    let normalize = adjust * norm_init;
    Ok(Model {
        mesh,
        normalize,
        material,
        outline_color,
    })
}

fn load_scene() -> Result<Scene, Box<dyn Error>> {
    // Outer Ring
    let kirby = load_model(
        KIRBY_PATH,
        110.0,
        PhongMaterial::new(
            ReflectionConstants::new(0.24, 0.16, 0.16),
            ReflectionConstants::new(0.72, 0.44, 0.44),
            ReflectionConstants::new(0.8, 0.6, 0.6),
            30.0,
        ),
        Rgb::new(255, 120, 180),
    )?;

    let walle = load_model(
        WALLE_PATH,
        100.0,
        PhongMaterial::GOLD,
        Rgb::new(255, 160, 0),
    )?;

    let duck = load_model(
        DUCK_PATH,
        110.0,
        PhongMaterial::EMERALD,
        Rgb::new(0, 255, 128),
    )?;

    let penguin = load_model(
        PENGUIN_PATH,
        95.0,
        PhongMaterial::TURQUOISE,
        Rgb::new(0, 180, 255),
    )?;

    // Inner Ring
    let alee3 = load_model(ALEE3_PATH, 65.0, PhongMaterial::GOLD, Rgb::new(255, 215, 0))?;

    let yuyingq = load_model(
        YUYINGQ_PATH,
        65.0,
        PhongMaterial::TURQUOISE,
        Rgb::new(0, 255, 230),
    )?;

    let gautamj = load_model(
        GAUTAMJ_PATH,
        65.0,
        PhongMaterial::EMERALD,
        Rgb::new(50, 255, 100),
    )?;

    let eleehuaj = load_model(
        ELEEHUAJ_PATH,
        65.0,
        PhongMaterial::RUBY,
        Rgb::new(255, 0, 255),
    )?;

    // Build the mechanical carousel stage
    let stage = build_carousel_stage();

    // Build a floating neon halo ring (torus)
    let mut halo_ring = PolygonMatrix::new();
    halo_ring.add_torus((0.0, 0.0, 0.0), 6.0, 310.0, 32);

    // Build a planetary ring
    let mut planetary_ring = PolygonMatrix::new();
    planetary_ring.add_torus((0.0, 0.0, 0.0), 5.0, 440.0, 48);

    // Build floor grid rings
    let mut floor_grid = PolygonMatrix::new();
    for r in [180.0, 300.0, 420.0, 540.0] {
        let mut tr = PolygonMatrix::new();
        tr.add_torus((0.0, 0.0, 0.0), 2.5, r, 36);
        floor_grid.extend(&tr);
    }

    Ok(Scene {
        kirby,
        walle,
        duck,
        penguin,
        alee3,
        yuyingq,
        gautamj,
        eleehuaj,
        stage,
        halo_ring,
        planetary_ring,
        floor_grid,
    })
}

/// Programmatically construct a tiered mechanical stage with concentric platforms and gear teeth
fn build_carousel_stage() -> PolygonMatrix {
    let mut stage = PolygonMatrix::new();
    let rot = Matrix::rotate_x(-90.0); // Rotate cylinders by -90 to go from Z to Y

    // 1. Lower wide base (radius 380, thickness 22, Y = -11 to 11)
    let mut lower_base = PolygonMatrix::new();
    lower_base.add_cylinder((0.0, 0.0, -11.0), 380.0, 22.0, 36);
    stage.extend(&lower_base.apply(&rot));

    // Gear teeth on the lower base (32 teeth)
    let teeth_count = 32;
    for i in 0..teeth_count {
        let angle = i as f64 / teeth_count as f64 * PI * 2.0;
        let mut tooth = PolygonMatrix::new();
        tooth.add_centered_box((380.0, 0.0, 0.0), 16.0, 24.0, 16.0);
        let placed_tooth = tooth.apply(&Matrix::rotate_z(angle * 180.0 / PI));
        stage.extend(&placed_tooth.apply(&rot));
    }

    // 2. Middle lower platform (radius 340, thickness 16, Y = 8 to 24)
    let mut mid_base = PolygonMatrix::new();
    mid_base.add_cylinder((0.0, 0.0, 8.0), 340.0, 16.0, 36);
    stage.extend(&mid_base.apply(&rot));

    // Gear teeth on the middle platform (28 teeth)
    let teeth_count_mid = 28;
    for i in 0..teeth_count_mid {
        let angle = i as f64 / teeth_count_mid as f64 * PI * 2.0;
        let mut tooth = PolygonMatrix::new();
        tooth.add_centered_box((340.0, 0.0, 16.0), 12.0, 18.0, 12.0);
        let placed_tooth = tooth.apply(&Matrix::rotate_z(angle * 180.0 / PI));
        stage.extend(&placed_tooth.apply(&rot));
    }

    // 3. Lower column (radius 50, height 74, starting at Y = 16)
    let mut lower_column = PolygonMatrix::new();
    lower_column.add_cylinder((0.0, 0.0, 16.0), 50.0, 74.0, 24);
    stage.extend(&lower_column.apply(&rot));

    // 4. Middle upper platform (radius 220, thickness 14, Y = 90 to 104)
    let mut upper_base = PolygonMatrix::new();
    upper_base.add_cylinder((0.0, 0.0, 90.0), 220.0, 14.0, 36);
    stage.extend(&upper_base.apply(&rot));

    // Gear teeth on the upper platform (20 teeth)
    let teeth_count_up = 20;
    for i in 0..teeth_count_up {
        let angle = i as f64 / teeth_count_up as f64 * PI * 2.0;
        let mut tooth = PolygonMatrix::new();
        tooth.add_centered_box((220.0, 0.0, 97.0), 10.0, 14.0, 10.0);
        let placed_tooth = tooth.apply(&Matrix::rotate_z(angle * 180.0 / PI));
        stage.extend(&placed_tooth.apply(&rot));
    }

    // 5. Upper column (radius 30, height 136, starting at Y = 104)
    let mut upper_column = PolygonMatrix::new();
    upper_column.add_cylinder((0.0, 0.0, 104.0), 30.0, 136.0, 24);
    stage.extend(&upper_column.apply(&rot));

    // 6. Stylized cones at the lower base pointing upwards for mechanical detail (12 cones)
    for i in 0..12 {
        let angle = i as f64 / 12.0 * PI * 2.0;
        let cx = angle.cos() * 310.0;
        let cz = angle.sin() * 310.0;
        let mut cone = PolygonMatrix::new();
        cone.add_cone((0.0, 0.0, 0.0), 12.0, 24.0, 8);
        let placed_cone = cone.apply(&rot).apply(&Matrix::translate(cx, 16.0, cz));
        stage.extend(&placed_cone);
    }

    // 7. Stylized cones at the upper base pointing upwards (6 cones)
    for i in 0..6 {
        let angle = i as f64 / 6.0 * PI * 2.0;
        let cx = angle.cos() * 190.0;
        let cz = angle.sin() * 190.0;
        let mut cone = PolygonMatrix::new();
        cone.add_cone((0.0, 0.0, 0.0), 8.0, 18.0, 8);
        let placed_cone = cone.apply(&rot).apply(&Matrix::translate(cx, 104.0, cz));
        stage.extend(&placed_cone);
    }

    stage
}

fn render_frame(frame: usize, scene: &Scene) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(0, 0, 0));
    canvas.wrapped = false;
    canvas.upper_left_origin = false; // Y goes up

    let t = frame as f64 / FRAMES as f64;
    let phase = t * PI * 2.0;

    // 1. Draw Starry Sky Background Gradient (Deep Purple/Navy to Near Black)
    for y in 0..HEIGHT {
        let u = y as f64 / HEIGHT as f64;
        let r = (12.0 * (1.0 - u)) as u8;
        let g = (8.0 * (1.0 - u)) as u8;
        let b = (26.0 * (1.0 - u) + 10.0 * u) as u8;
        let color = Rgb::new(r, g, b);
        for x in 0..WIDTH {
            canvas[y as usize * WIDTH as usize + x as usize] = color;
        }
    }

    // Twinkling background stars
    let mut rng = SampleRng::new(555);
    for i in 0..150 {
        let x = rng.random_range(0.0, WIDTH as f64) as i64;
        let y = rng.random_range(0.0, HEIGHT as f64) as i64;
        let twinkle = 0.45 + 0.55 * ((phase * 2.5 + i as f64 * 0.72).sin());
        let star_color = dim(Rgb::new(230, 245, 255), twinkle);
        canvas.plot(&star_color, x, y);
    }

    // 2. Camera Setup & Global View Settings
    let cx = f64::from(WIDTH) * 0.5;
    let cy = f64::from(HEIGHT) * 0.42; // Center stage lowered slightly for tilt view
    let cz = 0.0;
    let view_angle_x = 22.0; // Look down at the carousel stage
    let carousel_spin = t * 360.0;

    // Global camera rotation matrix
    let view_transform = Matrix::translate(cx, cy, cz)
        * Matrix::rotate_x(view_angle_x)
        * Matrix::rotate_y(carousel_spin);

    // 3. Dynamic Positional Lights Setup
    // Central warm lighthouse light at the top of the column
    let center_light_pos = transform_position(&view_transform, Vector::new(0.0, 240.0, 0.0));
    let center_light = PointLight::positional(center_light_pos, Rgb::new(255, 220, 160))
        .with_inverse_square_attenuation(350.0);

    // Orbiting Cyan Light
    let cyan_angle = phase * 1.5;
    let cyan_local = Vector::new(cyan_angle.cos() * 450.0, 180.0, cyan_angle.sin() * 450.0);
    let cyan_light_pos = transform_position(&view_transform, cyan_local);
    let cyan_light = PointLight::positional(cyan_light_pos, Rgb::new(0, 240, 255))
        .with_inverse_square_attenuation(400.0);

    // Orbiting Magenta Light (different speed, height, and counter-rotating)
    let magenta_angle = -phase * 1.2 + PI;
    let magenta_local = Vector::new(
        magenta_angle.cos() * 480.0,
        60.0,
        magenta_angle.sin() * 480.0,
    );
    let magenta_light_pos = transform_position(&view_transform, magenta_local);
    let magenta_light = PointLight::positional(magenta_light_pos, Rgb::new(255, 20, 180))
        .with_inverse_square_attenuation(400.0);

    let lights = vec![center_light, cyan_light, magenta_light];

    // 3.5 Draw Holographic Floor Grid (low intensity cyber-grid)
    let floor_transform = view_transform.clone()
        * Matrix::translate(0.0, -40.0, 0.0)
        * Matrix::rotate_y(t * -360.0 * 0.2); // rotates very slowly
    let floor_mesh = scene.floor_grid.apply(&floor_transform);
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(30, 60, 100)); // dim blue-gray glowing grid
    canvas.draw_polygons(&floor_mesh);

    // 4. Draw Carousel Stage (Phong shading, metallic CHROME material)
    let stage_mesh = scene.stage.apply(&view_transform);
    canvas.set_shading_mode(ShadingMode::Phong);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.set_lighting(lighting_for_material(PhongMaterial::CHROME, &lights));
    canvas.draw_polygons(&stage_mesh);

    // Draw the stage wireframe outlines in dark blue for high tech look
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(20, 40, 90));
    canvas.draw_polygons(&stage_mesh.apply(&Matrix::translate(0.0, 0.0, 0.5))); // tiny z-bias

    // 5. Draw Floating Halo Rings (Wireframe glowing toruses)
    // Ring 1: Cyan ring rotating around the lower base platform
    let ring1_local = Matrix::translate(0.0, 24.0, 0.0)
        * Matrix::rotate_x(8.0 * (phase * 1.5).cos())
        * Matrix::rotate_y(t * -360.0 * 2.0);
    let ring1_mesh = scene
        .halo_ring
        .apply(&(view_transform.clone() * ring1_local));
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(0, 255, 255));
    canvas.draw_polygons(&ring1_mesh);

    // Ring 2: Magenta ring orbiting the upper column, tilted
    let mut top_halo = PolygonMatrix::new();
    top_halo.add_torus((0.0, 0.0, 0.0), 5.0, 190.0, 24);
    let ring2_local = Matrix::translate(0.0, 140.0, 0.0)
        * Matrix::rotate_z(14.0)
        * Matrix::rotate_y(t * 360.0 * 1.3);
    let ring2_mesh = top_halo.apply(&(view_transform.clone() * ring2_local));
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(255, 0, 255));
    canvas.draw_polygons(&ring2_mesh);

    // Ring 3: Tilted Golden planetary ring orbiting the entire base
    let planet_ring_transform =
        view_transform.clone() * Matrix::rotate_x(15.0) * Matrix::rotate_y(t * -360.0 * 0.5);
    let planet_ring_mesh = scene.planetary_ring.apply(&planet_ring_transform);
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(255, 170, 0));
    canvas.draw_polygons(&planet_ring_mesh);

    // 6. Draw Rotating Mechanical Spire Stack (Clockwork centerpiece)
    let mut spire_y = 240.0; // Top of the upper column (104 + 136)

    // Spire 1: Icosahedron (rotates clockwise)
    let mut ico = PolygonMatrix::new();
    ico.add_icosahedron((0.0, 0.0, 0.0), 28.0);
    let ico_bounds = ico.bounds().unwrap();
    let ico_height = ico_bounds.max.1 - ico_bounds.min.1;
    let ico = ico.apply(&Matrix::translate(0.0, -ico_bounds.min.1, 0.0));
    let ico_transform = view_transform.clone()
        * Matrix::translate(0.0, spire_y, 0.0)
        * Matrix::rotate_y(t * 360.0 * 2.5);
    canvas.set_shading_mode(ShadingMode::Phong);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.set_lighting(lighting_for_material(PhongMaterial::GOLD, &lights));
    canvas.draw_polygons(&ico.apply(&ico_transform));
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(255, 215, 0));
    canvas.draw_polygons(&ico.apply(&(ico_transform * Matrix::translate(0.0, 0.0, 0.8))));

    spire_y += ico_height + 4.0;

    // Spire 2: Dodecahedron (rotates counter-clockwise)
    let mut dodeca = PolygonMatrix::new();
    dodeca.add_dodecahedron((0.0, 0.0, 0.0), 20.0);
    let dodeca_bounds = dodeca.bounds().unwrap();
    let dodeca_height = dodeca_bounds.max.1 - dodeca_bounds.min.1;
    let dodeca = dodeca.apply(&Matrix::translate(0.0, -dodeca_bounds.min.1, 0.0));
    let dodeca_transform = view_transform.clone()
        * Matrix::translate(0.0, spire_y, 0.0)
        * Matrix::rotate_y(t * -360.0 * 2.0);
    canvas.set_shading_mode(ShadingMode::Phong);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.set_lighting(lighting_for_material(PhongMaterial::RUBY, &lights));
    canvas.draw_polygons(&dodeca.apply(&dodeca_transform));
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(255, 120, 160));
    canvas.draw_polygons(&dodeca.apply(&(dodeca_transform * Matrix::translate(0.0, 0.0, 0.8))));

    spire_y += dodeca_height + 4.0;

    // Spire 3: Small sphere cap
    let mut spire_sphere = PolygonMatrix::new();
    spire_sphere.add_sphere((0.0, 0.0, 0.0), 14.0, 12);
    let sphere_bounds = spire_sphere.bounds().unwrap();
    let sphere_height = sphere_bounds.max.1 - sphere_bounds.min.1;
    let spire_sphere = spire_sphere.apply(&Matrix::translate(0.0, -sphere_bounds.min.1, 0.0));
    let sphere_transform = view_transform.clone() * Matrix::translate(0.0, spire_y, 0.0);
    canvas.set_shading_mode(ShadingMode::Phong);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.set_lighting(lighting_for_material(PhongMaterial::CHROME, &lights));
    canvas.draw_polygons(&spire_sphere.apply(&sphere_transform));

    spire_y += sphere_height + 4.0;

    // Spire 4: Needle Cone
    let mut spire_cone = PolygonMatrix::new();
    spire_cone.add_cone((0.0, 0.0, 0.0), 8.0, 20.0, 8);
    let spire_cone = spire_cone.apply(&Matrix::rotate_x(-90.0));
    let cone_bounds = spire_cone.bounds().unwrap();
    let spire_cone = spire_cone.apply(&Matrix::translate(0.0, -cone_bounds.min.1, 0.0));
    let cone_transform = view_transform.clone() * Matrix::translate(0.0, spire_y, 0.0);
    canvas.draw_polygons(&spire_cone.apply(&cone_transform));

    // 7. Draw Outer Tier Characters (Orbit radius 290, rotate clockwise with platform)
    let outer_orbit_radius = 290.0;
    let outer_character_data = [
        // 0. Kirby (dtorresr.obj) - angle 0 deg
        (
            &scene.kirby,
            0.0,
            12.0 * (1.0 + (phase * 2.0).sin()), // positive bob only (no clipping!)
            20.0 * (phase * 3.0).cos(),
        ),
        // 1. WALL-E (ysato.obj) - angle 90 deg
        (
            &scene.walle,
            PI * 0.5,
            12.0 * (1.0 + (phase * 2.0 + PI * 0.5).sin()),
            t * 720.0,
        ),
        // 2. Duck (aylu.obj) - angle 180 deg
        (
            &scene.duck,
            PI,
            12.0 * (1.0 + (phase * 2.0 + PI).sin()),
            12.0 * (phase * 3.0).sin(),
        ),
        // 3. Penguin (bamado.obj) - angle 270 deg
        (
            &scene.penguin,
            PI * 1.5,
            12.0 * (1.0 + (phase * 2.0 + PI * 1.5).sin()),
            -carousel_spin, // stays facing camera
        ),
    ];

    for (model, angle_offset, bob_y, self_spin) in outer_character_data {
        let lx = angle_offset.cos() * outer_orbit_radius;
        let lz = angle_offset.sin() * outer_orbit_radius;
        let ly = 24.0 + bob_y; // sits above lower platform top (Y = 24)

        let model_view = view_transform.clone()
            * Matrix::translate(lx, ly, lz)
            * Matrix::rotate_y(self_spin)
            * model.normalize.clone();

        // Filled
        canvas.set_shading_mode(ShadingMode::Toon);
        canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
        canvas.set_lighting(lighting_for_material(model.material, &lights));
        let posed_mesh = model.mesh.apply(&model_view);
        canvas.draw_polygons(&posed_mesh);

        // Neon outline
        let outline_mesh = posed_mesh.apply(&Matrix::translate(0.0, 0.0, 1.2)); // Z-bias
        canvas.set_shading_mode(ShadingMode::Wireframe);
        canvas.set_line_pixel(model.outline_color);
        canvas.draw_polygons(&outline_mesh);
    }

    // 7.5 Draw Inner Tier Characters (Orbit radius 180, rotate counter-clockwise relative to platform)
    let inner_orbit_radius = 180.0;
    let inner_character_data = [
        // 0. Amy Lee (alee3.obj) - angle 45 deg
        (
            &scene.alee3,
            PI * 0.25,
            8.0 * (1.0 + (phase * 2.5 + PI * 0.25).sin()),
            t * -720.0,
        ),
        // 1. Yuyingq (yuyingq.obj) - angle 135 deg
        (
            &scene.yuyingq,
            PI * 0.5 + PI * 0.25,
            8.0 * (1.0 + (phase * 2.5 + PI * 0.75).sin()),
            20.0 * (phase * 2.5).cos(),
        ),
        // 2. Gautamj (gautamj.obj) - angle 225 deg
        (
            &scene.gautamj,
            PI + PI * 0.25,
            8.0 * (1.0 + (phase * 2.5 + PI * 1.25).sin()),
            -carousel_spin,
        ),
        // 3. Eleehuaj (eleehuaj.obj) - angle 315 deg
        (
            &scene.eleehuaj,
            PI * 1.5 + PI * 0.25,
            8.0 * (1.0 + (phase * 2.5 + PI * 1.75).sin()),
            t * 360.0,
        ),
    ];

    for (model, angle_offset, bob_y, self_spin) in inner_character_data {
        let current_angle = angle_offset - phase * 2.0; // orbits counter-clockwise
        let lx = current_angle.cos() * inner_orbit_radius;
        let lz = current_angle.sin() * inner_orbit_radius;
        let ly = 24.0 + bob_y;

        let model_view = view_transform.clone()
            * Matrix::translate(lx, ly, lz)
            * Matrix::rotate_y(self_spin)
            * model.normalize.clone();

        // Filled
        canvas.set_shading_mode(ShadingMode::Toon);
        canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
        canvas.set_lighting(lighting_for_material(model.material, &lights));
        let posed_mesh = model.mesh.apply(&model_view);
        canvas.draw_polygons(&posed_mesh);

        // Neon outline
        let outline_mesh = posed_mesh.apply(&Matrix::translate(0.0, 0.0, 1.2)); // Z-bias
        canvas.set_shading_mode(ShadingMode::Wireframe);
        canvas.set_line_pixel(model.outline_color);
        canvas.draw_polygons(&outline_mesh);
    }

    // 8. Draw Upper Tier Floating Crystals (Orbit radius 130, rotates counter-clockwise)
    let upper_orbit_radius = 130.0;
    let upper_carousel_spin = -t * 360.0 * 1.5;
    let upper_view_transform = Matrix::translate(cx, cy, cz)
        * Matrix::rotate_x(view_angle_x)
        * Matrix::rotate_y(upper_carousel_spin);

    let upper_solids_data = [
        // 0. Neon Pink Dodecahedron
        (
            0.0,
            Rgb::new(255, 50, 180),
            PhongMaterial::RUBY,
            10.0 * (1.0 + (phase * 2.5 + PI * 0.25).sin()),
            {
                let mut pm = PolygonMatrix::new();
                pm.add_dodecahedron((0.0, 0.0, 0.0), 20.0);
                let bounds = pm.bounds().unwrap();
                pm.apply(&Matrix::translate(0.0, -bounds.min.1, 0.0))
            },
        ),
        // 1. Neon Cyan Icosahedron
        (
            PI * 0.5,
            Rgb::new(0, 230, 255),
            PhongMaterial::TURQUOISE,
            10.0 * (1.0 + (phase * 2.5 + PI * 0.75).sin()),
            {
                let mut pm = PolygonMatrix::new();
                pm.add_icosahedron((0.0, 0.0, 0.0), 22.0);
                let bounds = pm.bounds().unwrap();
                pm.apply(&Matrix::translate(0.0, -bounds.min.1, 0.0))
            },
        ),
        // 2. Neon Green Cone
        (
            PI,
            Rgb::new(50, 255, 120),
            PhongMaterial::EMERALD,
            10.0 * (1.0 + (phase * 2.5 + PI * 1.25).sin()),
            {
                let mut pm = PolygonMatrix::new();
                pm.add_cone((0.0, 0.0, 0.0), 12.0, 25.0, 8);
                let pm = pm.apply(&Matrix::rotate_x(-90.0)); // point up
                let bounds = pm.bounds().unwrap();
                pm.apply(&Matrix::translate(0.0, -bounds.min.1, 0.0))
            },
        ),
        // 3. Neon Orange Torus
        (
            PI * 1.5,
            Rgb::new(255, 140, 0),
            PhongMaterial::GOLD,
            10.0 * (1.0 + (phase * 2.5 + PI * 1.75).sin()),
            {
                let mut pm = PolygonMatrix::new();
                pm.add_torus((0.0, 0.0, 0.0), 5.0, 18.0, 16);
                let bounds = pm.bounds().unwrap();
                pm.apply(&Matrix::translate(0.0, -bounds.min.1, 0.0))
            },
        ),
    ];

    for (angle_offset, outline_color, material, bob_y, pm) in upper_solids_data {
        let lx = angle_offset.cos() * upper_orbit_radius;
        let lz = angle_offset.sin() * upper_orbit_radius;
        let ly = 104.0 + 10.0 + bob_y; // sits above upper platform top (Y = 104)

        let model_view = upper_view_transform.clone()
            * Matrix::translate(lx, ly, lz)
            * Matrix::rotate_y(t * 360.0 * 2.0);

        // Filled
        canvas.set_shading_mode(ShadingMode::Toon);
        canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
        canvas.set_lighting(lighting_for_material(material, &lights));
        let posed_mesh = pm.apply(&model_view);
        canvas.draw_polygons(&posed_mesh);

        // Neon outline
        canvas.set_shading_mode(ShadingMode::Wireframe);
        canvas.set_line_pixel(outline_color);
        canvas.draw_polygons(&posed_mesh.apply(&Matrix::translate(0.0, 0.0, 1.2)));
    }

    // 9. Floating space dust vortex orbiting the carousel
    let mut dust_rng = SampleRng::new(999);
    for i in 0..120 {
        let r = dust_rng.random_range(160.0, 480.0);
        let start_theta = dust_rng.random_range(0.0, PI * 2.0);
        let speed = dust_rng.random_range(0.5, 2.0) * (if i % 2 == 0 { 1.0 } else { -1.0 });
        let theta = start_theta + phase * speed * 0.3;

        let lx = theta.cos() * r;
        let lz = theta.sin() * r;
        let start_y = dust_rng.random_range(-50.0, 400.0);
        let ly = (start_y + t * 450.0) % 500.0 - 50.0; // drift upward

        let camera_only_transform = Matrix::translate(cx, cy, cz) * Matrix::rotate_x(view_angle_x);
        let pos = transform_position(&camera_only_transform, Vector::new(lx, ly, lz));
        let size = 2 + (i % 3);
        let pulse = 0.4 + 0.6 * ((phase * 1.5 + i as f64 * 0.72).sin());
        // Mix colors: cyan, magenta, gold
        let color = match i % 3 {
            0 => dim(Rgb::new(0, 240, 255), pulse),  // cyan
            1 => dim(Rgb::new(255, 20, 180), pulse), // magenta
            _ => dim(Rgb::new(255, 215, 0), pulse),  // gold
        };

        canvas.fill_disc(
            pos.x().round() as i64,
            pos.y().round() as i64,
            size as i64,
            color,
        );
    }

    // 10. Draw floating light beacon indicators
    let p_cyan = (cyan_light_pos.x(), cyan_light_pos.y(), cyan_light_pos.z());
    let p_magenta = (
        magenta_light_pos.x(),
        magenta_light_pos.y(),
        magenta_light_pos.z(),
    );
    let p_center = (
        center_light_pos.x(),
        center_light_pos.y(),
        center_light_pos.z(),
    );

    canvas.fill_disc(
        p_cyan.0.round() as i64,
        p_cyan.1.round() as i64,
        8,
        Rgb::new(0, 240, 255),
    );
    canvas.fill_disc(
        p_cyan.0.round() as i64,
        p_cyan.1.round() as i64,
        4,
        Rgb::new(255, 255, 255),
    );

    canvas.fill_disc(
        p_magenta.0.round() as i64,
        p_magenta.1.round() as i64,
        8,
        Rgb::new(255, 20, 180),
    );
    canvas.fill_disc(
        p_magenta.0.round() as i64,
        p_magenta.1.round() as i64,
        4,
        Rgb::new(255, 255, 255),
    );

    canvas.fill_disc(
        p_center.0.round() as i64,
        p_center.1.round() as i64,
        10,
        Rgb::new(255, 200, 100),
    );
    canvas.fill_disc(
        p_center.0.round() as i64,
        p_center.1.round() as i64,
        5,
        Rgb::new(255, 255, 255),
    );

    // Energy cages around orbiting beacons
    let mut beacon_sphere = PolygonMatrix::new();
    beacon_sphere.add_sphere((0.0, 0.0, 0.0), 12.0, 6);

    let cyan_cage = beacon_sphere.apply(&Matrix::translate(p_cyan.0, p_cyan.1, p_cyan.2));
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.set_line_pixel(Rgb::new(0, 240, 255));
    canvas.draw_polygons(&cyan_cage);

    let magenta_cage =
        beacon_sphere.apply(&Matrix::translate(p_magenta.0, p_magenta.1, p_magenta.2));
    canvas.set_line_pixel(Rgb::new(255, 20, 180));
    canvas.draw_polygons(&magenta_cage);

    canvas
}

fn lighting_for_material(material: PhongMaterial, lights: &[PointLight]) -> Lighting {
    Lighting {
        view: Vector::new(0.0, 0.0, 1.0),
        ambient: Rgb::new(20, 15, 30),
        point_lights: lights.to_vec(),
        ambient_reflection: material.ambient,
        diffuse_reflection: material.diffuse,
        specular_reflection: material.specular,
        specular_exponent: material.specular_exponent(),
        ..Lighting::default()
    }
}

fn dim(color: Rgb, factor: f64) -> Rgb {
    Rgb::new(
        scale_channel(color.red, factor),
        scale_channel(color.green, factor),
        scale_channel(color.blue, factor),
    )
}

fn scale_channel(channel: u8, factor: f64) -> u8 {
    (f64::from(channel) * factor).round().clamp(0.0, 255.0) as u8
}

fn transform_position(matrix: &Matrix, pos: Vector) -> Vector {
    let x = pos.x();
    let y = pos.y();
    let z = pos.z();
    let w = 1.0;
    Vector::new(
        matrix[(0, 0)] * x + matrix[(0, 1)] * y + matrix[(0, 2)] * z + matrix[(0, 3)] * w,
        matrix[(1, 0)] * x + matrix[(1, 1)] * y + matrix[(1, 2)] * z + matrix[(1, 3)] * w,
        matrix[(2, 0)] * x + matrix[(2, 1)] * y + matrix[(2, 2)] * z + matrix[(2, 3)] * w,
    )
}

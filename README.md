# Gartus

`gartus` is a Rust graphics library for experimenting with the full stack of
small-renderer building blocks: pixels, colors, matrices, mesh generation,
camera projection, rasterization, procedural textures, animation, MDL scripts,
and path tracing.

## Gallery



## Overview

The project started as a classroom-style computer graphics engine built around a
`Canvas`, transformation matrices, polygon meshes, lighting, and a Motion
Description Language. It has since grown into a reusable library with two
rendering paths:

- a fast canvas/raster path for drawing, meshes, turtle graphics, filters, and
  MDL scenes
- a physically based path-tracing path for analytic geometry, materials,
  importance sampling, volumes, SDF ray marching, and renderer-neutral scene data

## What the Library Provides

### Math and Geometry

The `gmath` module contains the shared math layer used by both renderers:

- `Point`, `Vector`, and `Ray`
- matrices, transformation stacks, and quaternions
- edge and polygon matrices
- analytic sphere, moving sphere, quad, and triangle geometry
- camera bases and camera poses
- Perlin noise, procedural helpers, and deterministic sample RNGs
- directional sampling utilities and PDFs

Mesh helpers live close to the math layer. `PolygonMatrix` can build common
triangle meshes such as boxes, prisms, crystals, spheres, tori, and height
maps, then send those meshes through either the raster pipeline or the ray
pipeline.

### Canvas Raster Graphics

The raster side is centered on `Canvas`.

It supports direct image work, line and polygon drawing, z-buffered filled
triangles, color interpolation, Phong-style lighting, textured rasterization,
filters, bitmap output, and simple 3D projection through `Camera3D`.

```rust
use gartus::prelude::*;

let mut mesh = PolygonMatrix::new();
mesh.add_centered_box(Point::new(0.0, 0.0, 0.0), 2.0, 2.0, 2.0);

let mut scene = SurfaceScene::new();
scene.add_mesh(
    mesh,
    SurfaceMaterial::new(
        LinearRgb::new(0.08, 0.06, 0.04),
        LinearRgb::new(0.85, 0.55, 0.30),
        LinearRgb::new(0.5, 0.45, 0.38),
        32.0,
    ),
);

let camera = Camera3D::new(500, 500)
    .with_look_at(Point::new(3.0, 2.0, 5.0), Point::new(0.0, 0.0, 0.0));

let image = scene.rasterize(&camera);
```

### Renderer-Neutral Scene Data

`SurfaceScene`, `SurfaceMesh`, `SurfaceMaterial`, and the shared texture types
are designed to sit above any one renderer. The same scene data can be rasterized
with `SurfaceScene::rasterize` or path traced with
`PathTracer::render_scene`.

This is the preferred layer for ordinary mesh/material content. For one-off
path-traced previews, use the convenience renderer:

```rust
use gartus::prelude::*;

let surface_scene = SurfaceScene::new();
let camera = RayCamera::new(400, 1.0);
let image = PathTracer::new(camera).render_scene(&surface_scene);
```

For animation, camera iteration, or sample-count iteration, compile once and
render the compiled scene repeatedly so the primitive table and BVH are reused:

```rust
use gartus::prelude::*;

let surface_scene = SurfaceScene::new();
let ray_scene = surface_scene.to_ray_scene();

let preview = PathTracer::new(RayCamera::new(400, 1.0)).render(&ray_scene);
let final_image = PathTracer::new(
    RayCamera::new(800, 1.0).with_stratified_grid_width(16),
)
.render(&ray_scene);
```

Use lower-level ray-tracing types directly when you need ray-specific behavior
such as glass, metal, emissive geometry, participating media, SDFs, or custom
hittables.

### Path Tracing

The `graphics::raytracing` module provides a compact path tracer inspired by the
*Ray Tracing* book series, with additional library-oriented APIs.

Core pieces include:

- `RayCamera` for path-tracing camera setup
- `PathTracer` for rendering worlds and shared surface scenes
- `Hittable`, `HittableList`, `HittableLayers`, and `RayScene`
- analytic `Sphere`, `MovingSphere`, `Quad`, triangle meshes, boxes, transforms,
  and matrix instances
- `Lambertian`, `Metal`, `Dielectric`, `DiffuseLight`, and `Isotropic`
  materials
- checker, image, solid, noise, turbulence, and marble textures
- BVH acceleration for built-in geometry and arbitrary bounded hittables
- explicit light sampling with `SamplingTargetList` and
  `WeightedSamplingTargetList`
- stratified sampling, adaptive sampling, defocus blur, motion blur, and
  configurable recursion depth

For mesh-heavy renderer-neutral scenes, call `SurfaceScene::to_ray_scene()` once
and render the compiled `RayScene` repeatedly. `PathTracer::render_scene` is
intentionally ergonomic, but it recompiles this scene every call. For built-in
ray geometry, build a `RayScene` directly so its compact primitive/material
tables and cached BVH are reused. For book-style scenes or custom hittables, use
`HittableList`.

```rust
use gartus::prelude::*;

let camera = RayCamera::new(400, 1.0)
    .with_stratified_grid_width(16)
    .with_max_depth(12)
    .with_background(LinearColor::new(0.02, 0.03, 0.05))
    .with_look_at(Point::new(3.0, 2.0, 4.0), Point::new(0.0, 0.0, 0.0));

let mut world = HittableList::new();
world.add(Sphere::with_material(
    Point::new(0.0, 0.0, 0.0),
    1.0,
    Lambertian::new(LinearColor::new(0.7, 0.4, 0.9)),
));

let image = PathTracer::new(camera).render(&world);
```

### Volumes

`gartus` supports both constant and non-uniform participating media.

- `ConstantMedium` handles book-style constant-density fog and smoke.
- `DensityField` and `NonUniformMedium` support spatially varying fog, dust,
  clouds, nebulae, and animated density fields.
- `FnDensityField` makes small procedural fields easy to write inline.

Non-uniform media use Woodcock/delta tracking against the field's maximum
density, so empty or low-density regions do not need to be explicitly meshed.

```rust
use gartus::prelude::*;

let dust = FnDensityField::new(0.05, |p: Point, time| {
    let wave = 0.5 + 0.5 * (p.x() * 2.0 + p.z() * 0.7 + time).sin();
    0.05 * wave
});

let medium = NonUniformMedium::new(
    Sphere::new(Point::new(0.0, 1.0, 0.0), 3.0),
    dust,
    LinearColor::new(0.8, 0.65, 0.45),
);
```

### Signed Distance Fields

The SDF API lets applications bring their own ray-marched geometry without
adding every fractal or implicit surface to the library.

- implement `DistanceField`
- provide world-space bounds
- wrap the field with `SdfObject`
- use any existing ray-tracing material

The example `raytracing_mandelbulb.rs` keeps its Mandelbulb estimator local to
the example and uses the library only for the generic SDF machinery.

```rust
use gartus::prelude::*;

let field = FnDistanceField::new(
    Bounds3::new((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    |p: Point| (p - Point::new(0.0, 0.0, 0.0)).length() - 1.0,
);

let object = SdfObject::new(
    field,
    Lambertian::new(LinearColor::new(0.4, 0.7, 1.0)),
);
```

### Animation

`FrameRecorder` and `AnimationRenderOptions` provide frame recording and GIF
encoding. With the `rayon` feature enabled, `FrameRecorder::render_gif_auto`
uses the parallel renderer; without it, the same call falls back to sequential
rendering.

Animation options support unique frame directories, preview output, GIF delay,
and progress reporting.

```rust
use gartus::prelude::*;

let options = AnimationRenderOptions::new(
    "anim",
    "frame-",
    24,
    "final/demo.gif",
)
.preview(12, "final/demo.png")
.show_progress(true);

FrameRecorder::render_gif_auto(options, |_frame| {
    let canvas = Canvas::new(320, 240, Rgb::BLACK);
    Ok(canvas)
})
.expect("render gif");
```

### MDL Scripts

The `mdl` module contains the Motion Description Language front end and runtime.
MDL scripts can build transformation stacks, meshes, animations, materials,
camera state, lights, and output commands.

The MDL runtime can rasterize accumulated polygon scenes, and scripts can select
the ray-tracing backend with:

```text
shading raytrace
```

The legacy two-line parser remains available behind the `old_parser` feature,
but new script work should use `mdl`.

## Examples

Examples are intentionally broad because the library covers several rendering
styles.

Raster and mesh examples:

- `advanced_shapes`
- `gallery_3d`
- `mesh_teapot`
- `mesh_joltik`
- `mesh_kirby_lighting`
- `terrain`
- `transformation_matrices`
- `cosmic_loom`
- `quantum_portal`
- `celestial_dragon`

Path-tracing examples:

- `raytracing_weekend`
- `next_weekend`
- `life`
- `raytracing_fireflies`
- `raytracing_kaleidoscope`
- `raytracing_prism_rain`
- `raytracing_mandelbulb`

Turtle and script-oriented examples:

- `frac_tree`
- `maze`
- `student_mdl`
- `walle_mdl`
- `curve_cathedral`

Most examples write output into `final/` or `final/raytracing/`.

## Running

Use the normal dev profile for quick checks:

```bash
cargo check
cargo run --example gallery_3d
```

Use the render profile for full path-traced images:

```bash
cargo run --profile render --example life
cargo run --profile render --example raytracing_mandelbulb
```

For path-tracing iteration, lower image width, depth, and samples first:

```rust
const IMAGE_WIDTH: u32 = 300;
const MAX_DEPTH: u32 = 8;
```

For final stills, prefer stratified grids:

```rust
RayCamera::new(600, 1.0)
    .with_stratified_grid_width(32)
    .with_max_depth(20);
```

For previews, adaptive sampling can stop pixels that have already converged:

```rust
RayCamera::new(400, 1.0)
    .with_adaptive_sampling(16, 256, 0.01)
    .with_max_depth(12);
```

Indoor path-traced scenes with small emitters usually converge faster when you
pass a dedicated `SamplingTargetList` or `WeightedSamplingTargetList` to
`PathTracer::render_with_lights`.

## Feature Flags

Default features:

- `external`: external image/mesh loading helpers
- `filters`: canvas filters
- `turtle`: turtle graphics
- `rayon`: parallel animation rendering

Optional compatibility feature:

- `old_parser`: legacy parser API

## Project Layout

```text
src/gmath/                 math, geometry, rays, sampling, noise
src/graphics/              canvas, cameras, colors, scenes, animation
src/graphics/raytracing/   path tracer, materials, BVH, volumes, SDFs
src/mdl/                   MDL lexer/parser/semantic/runtime pipeline
examples/                  raster, mesh, MDL, animation, and ray examples
```

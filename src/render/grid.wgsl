// Renders an m × n field of phase samples as a grid of coloured cells.
//
// There is no per-cell geometry: the whole image is one oversized triangle, and
// the fragment shader works out which cell each fragment lands in. That keeps
// the cost independent of m × n and lets the cell walls be specified exactly in
// device pixels.

// Laid out as three `vec4<f32>` so that the uniform address space alignment
// rules are satisfied without any padding games. Mirrored by `GridUniforms` in
// `mod.rs`; the two must be changed together.
struct Uniforms {
    // Data-space coordinates of the viewport corners:
    // xy = top-left, zw = bottom-right.
    bounds: vec4<f32>,

    // x = columns, y = rows, z = physical pixels per cell, w = wall width in pixels.
    grid: vec4<f32>,

    // x = value at t = 0, y = 1 / (value range), z = wrap mode, w = wall opacity.
    shading: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var data_texture: texture_2d<f32>;
@group(0) @binding(2) var lut_texture: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;

const PI: f32 = 3.1415927;
const TAU: f32 = 6.2831855;

// Cell walls are drawn dark. `Colormap::Grayscale` keeps a non-zero floor so
// that they stay visible even where the data bottoms out.
const WALL_COLOR = vec3<f32>(0.04, 0.04, 0.05);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) data_pos: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // One oversized triangle: uv = (0,0), (2,0), (0,2), so the clip-space
    // corners are (-1,-1), (3,-1), (-1,3) and the viewport is fully covered.
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = uv * 2.0 - 1.0;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);

    // The data-space position is interpolated from corners computed on the CPU,
    // rather than reconstructed from `@builtin(position)`. That keeps this
    // shader independent of where the viewport happens to sit in the frame
    // buffer, and of how egui rounded it to whole pixels.
    //
    // Clip-space y = +1 is the *top* of the viewport, which is where row 0
    // lives, so the data y axis runs opposite to uv.y.
    out.data_pos = vec2<f32>(
        mix(u.bounds.x, u.bounds.z, uv.x),
        mix(u.bounds.w, u.bounds.y, uv.y),
    );
    return out;
}

// Wraps `x` to (-π, π]. Mirrors `phase::wrap` on the Rust side.
fn wrap_to_pi(x: f32) -> f32 {
    let y = PI - x;
    return PI - (y - TAU * floor(y / TAU));
}

// 0-1 linear from 0-1 sRGB gamma. Same conversion egui's own shader uses, so
// that our colours composite identically with the rest of the UI.
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// Colours one fragment, in gamma (sRGB) space with premultiplied alpha.
fn shade(data_pos: vec2<f32>) -> vec4<f32> {
    let extent = u.grid.xy;

    // Outside the field: fully transparent, so the panel background shows.
    if data_pos.x < 0.0 || data_pos.y < 0.0 || data_pos.x >= extent.x || data_pos.y >= extent.y {
        return vec4<f32>(0.0);
    }

    let cell = vec2<i32>(floor(data_pos));
    let value = textureLoad(data_texture, cell, 0).r;

    // NaN marks a masked sample, which is common in real interferograms.
    if value != value {
        return vec4<f32>(0.0);
    }

    var t: f32;
    if u.shading.z > 0.5 {
        t = (wrap_to_pi(value) + PI) / TAU;
    } else {
        t = (value - u.shading.x) * u.shading.y;
    }

    // `textureSampleLevel` rather than `textureSample`: no derivatives are
    // needed here, and this stays valid under non-uniform control flow.
    var rgb = textureSampleLevel(
        lut_texture,
        lut_sampler,
        vec2<f32>(clamp(t, 0.0, 1.0), 0.5),
        0.0,
    ).rgb;

    let wall_opacity = u.shading.w;
    if wall_opacity > 0.0 {
        let pixels_per_cell = max(u.grid.z, 1.0e-6);
        let half_width_px = 0.5 * u.grid.w;

        // Half the wall width, expressed in cells, and never more than a
        // quarter of a cell so the clamp below cannot invert its bounds.
        let half_width_data = min(
            half_width_px / pixels_per_cell,
            0.25 * min(extent.x, extent.y),
        );

        // Walls sit on the integer grid lines 0..=cols and 0..=rows. The two
        // outermost lines are inset by half a wall, so the border is drawn at
        // full width instead of having half of it fall outside the image.
        let line = clamp(
            round(data_pos),
            vec2<f32>(half_width_data),
            extent - half_width_data,
        );
        let distance_px = abs(data_pos - line) * pixels_per_cell;
        let distance_to_wall = min(distance_px.x, distance_px.y);

        // One pixel of feathering on each side keeps the wall smooth without
        // multisampling.
        let coverage = 1.0 - smoothstep(
            half_width_px - 0.5,
            half_width_px + 0.5,
            distance_to_wall,
        );
        rgb = mix(rgb, WALL_COLOR, coverage * wall_opacity);
    }

    return vec4<f32>(rgb, 1.0);
}

// For a render target that does *not* convert to sRGB itself.
@fragment
fn fs_main_gamma_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade(in.data_pos);
}

// For an sRGB render target, which applies the transfer function on write.
@fragment
fn fs_main_linear_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = shade(in.data_pos);
    // Alpha is 0 or 1 here, so this stays premultiplied either way.
    return vec4<f32>(linear_from_gamma_rgb(color.rgb) * color.a, color.a);
}

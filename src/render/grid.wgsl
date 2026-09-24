// Renders an m × n field of phase samples as a grid of coloured cells, with an
// optional overlay describing an unwrapping of it.
//
// There is no per-cell geometry: the whole image is one oversized triangle, and
// the fragment shader works out which cell each fragment lands in. That keeps
// the cost independent of m × n and lets the cell walls be specified exactly in
// device pixels.

// Laid out as four `vec4<f32>` so that the uniform address space alignment
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

    // x = overlay enabled, y = residue radius in pixels,
    // z = highlight disagreeing edges, w = draw residues.
    overlay: vec4<f32>,

    // x = green cut edges, y = draw the integration path's walls, zw = unused.
    overlay_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var data_texture: texture_2d<f32>;
@group(0) @binding(2) var lut_texture: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
// One texel per pixel: [right traversal, right |jump|, down traversal, down |jump|].
@group(0) @binding(4) var edge_texture: texture_2d<u32>;
// One texel per inner corner, holding the residue charge plus one.
@group(0) @binding(5) var residue_texture: texture_2d<u32>;

const PI: f32 = 3.1415927;
const TAU: f32 = 6.2831855;

// Traversal codes, matching `Traversal::code` on the Rust side.
const TRAVERSAL_CUT: u32 = 0u;
// Also stands for "no integration path was supplied", so such walls are drawn
// plainly rather than as part of a walk that is not known.
const TRAVERSAL_NO_ROLE: u32 = 3u;

// Cell walls are drawn dark. `Colormap::Grayscale` keeps a non-zero floor so
// that they stay visible even where the data bottoms out.
const WALL_COLOR = vec3<f32>(0.04, 0.04, 0.05);

// Optional colour for the cut edges, which together form the spanning tree of
// the dual graph.
const CUT_WALL_GREEN = vec3<f32>(0.13, 0.69, 0.30);

// Edges where the integration delta is not the wrapped delta.
const HIGHLIGHT_COLOR = vec3<f32>(0.91, 0.09, 0.53);

// Residue charges, matching the figures: orange +1, aqua -1.
const RESIDUE_POSITIVE = vec3<f32>(0.96, 0.51, 0.12);
const RESIDUE_NEGATIVE = vec3<f32>(0.36, 0.78, 0.91);

// Walls the integration passes through are drawn faint and dashed; the solid
// ones are the cut edges, which together form a spanning tree of the dual.
const TREE_WALL_ALPHA: f32 = 0.30;
const DASH_PERIOD: f32 = 0.5;

// Highlighted walls are drawn wider so they survive next to ordinary ones.
const HIGHLIGHT_WIDTH_SCALE: f32 = 1.8;

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

// Wraps `x` to (-pi, pi]. Mirrors `phase::wrap` on the Rust side.
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

// How much of this fragment a wall covers, plus the colour to paint it.
struct Wall {
    color: vec3<f32>,
    coverage: f32,
};

// Shades one family of walls: those perpendicular to a single axis.
//
// `across` is the coordinate perpendicular to the wall and `along` the one
// parallel to it, with `extent_across` cells in the perpendicular direction.
// `texel` locates the edge in `edge_texture` and `channel` picks the pair of
// components describing it, so the caller resolves the axis and this does not
// have to know which one it is looking at.
fn wall_along_axis(
    across: f32,
    along: f32,
    extent_across: f32,
    max_half_data: f32,
    texel: vec2<i32>,
    channel: u32,
    pixels_per_cell: f32,
    base_half_px: f32,
) -> Wall {
    let line = round(across);

    // Walls sit on the integer grid lines. The two outermost are inset by half
    // a wall, so the border is drawn at full width instead of having half of it
    // fall outside the image.
    let half_data = min(base_half_px / pixels_per_cell, max_half_data);
    let center = clamp(line, half_data, extent_across - half_data);
    let distance_px = abs(across - center) * pixels_per_cell;

    var out: Wall;
    out.color = WALL_COLOR;
    var half_px = base_half_px;
    var alpha = 1.0;

    let index = i32(line);
    let is_interior = index > 0 && index < i32(extent_across);
    if u.overlay.x > 0.5 && is_interior {
        let state = textureLoad(edge_texture, texel, 0);
        let traversal = state[channel];
        let jump = state[channel + 1u];

        if jump != 0u && u.overlay.z > 0.5 {
            // The integration moved the phase by something other than the
            // wrapped difference across this edge.
            out.color = HIGHLIGHT_COLOR;
            half_px = base_half_px * HIGHLIGHT_WIDTH_SCALE;
        } else if traversal == TRAVERSAL_CUT {
            // A wall the integration never crosses. These are the spanning
            // tree of the dual graph, and can be picked out in green.
            if u.overlay_flags.x > 0.5 {
                out.color = CUT_WALL_GREEN;
            }
        } else if traversal != TRAVERSAL_NO_ROLE {
            // Part of the integration path: drawn faint and dashed, since the
            // walk passes straight through it, or left out altogether.
            if u.overlay_flags.y > 0.5 {
                alpha = TREE_WALL_ALPHA * step(0.5, fract(along / DASH_PERIOD));
            } else {
                alpha = 0.0;
            }
        }
    }

    out.coverage = (1.0 - smoothstep(half_px - 0.5, half_px + 0.5, distance_px)) * alpha;
    return out;
}

// Colours the residue marker, if any, at the inner corner nearest `p`.
fn residue_overlay(p: vec2<f32>, extent: vec2<f32>, pixels_per_cell: f32) -> Wall {
    var out: Wall;
    out.color = vec3<f32>(0.0);
    out.coverage = 0.0;

    let radius_px = u.overlay.y;
    if u.overlay.x <= 0.5 || u.overlay.w <= 0.5 || radius_px <= 0.0 {
        return out;
    }

    // Inner corners are the integer lattice points strictly inside the field;
    // corner (row, col) of the dual sits at data-space point (col + 1, row + 1).
    let corner = round(p);
    let x = i32(corner.x);
    let y = i32(corner.y);
    if x < 1 || y < 1 || x >= i32(extent.x) || y >= i32(extent.y) {
        return out;
    }

    let charge = textureLoad(residue_texture, vec2<i32>(x - 1, y - 1), 0).r;
    if charge == 1u {
        // Neutral: nothing to draw.
        return out;
    }

    let distance_px = length(p - corner) * pixels_per_cell;
    out.color = select(RESIDUE_NEGATIVE, RESIDUE_POSITIVE, charge == 2u);
    out.coverage = 1.0 - smoothstep(radius_px - 1.0, radius_px + 1.0, distance_px);
    return out;
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

    let pixels_per_cell = max(u.grid.z, 1.0e-6);
    let wall_fade = u.shading.w;

    if wall_fade > 0.0 {
        let base_half_px = 0.5 * u.grid.w;
        // Never wider than a quarter of a cell, so the inset clamp above cannot
        // invert its bounds on a tiny field.
        let max_half_data = 0.25 * min(extent.x, extent.y);

        // Walls at constant x separate two pixels side by side: the one at
        // line `i` in row `r` is the right edge of pixel (r, i - 1).
        let vertical = wall_along_axis(
            data_pos.x,
            data_pos.y,
            extent.x,
            max_half_data,
            vec2<i32>(i32(round(data_pos.x)) - 1, i32(floor(data_pos.y))),
            0u,
            pixels_per_cell,
            base_half_px,
        );

        // Walls at constant y separate two pixels stacked vertically: the one
        // at line `i` in column `c` is the bottom edge of pixel (i - 1, c).
        let horizontal = wall_along_axis(
            data_pos.y,
            data_pos.x,
            extent.y,
            max_half_data,
            vec2<i32>(i32(floor(data_pos.x)), i32(round(data_pos.y)) - 1),
            2u,
            pixels_per_cell,
            base_half_px,
        );

        rgb = mix(rgb, vertical.color, vertical.coverage * wall_fade);
        rgb = mix(rgb, horizontal.color, horizontal.coverage * wall_fade);
    }

    // Residues sit on top, and unlike the walls they do not fade with zoom:
    // they are sparse, and they are the obstruction that forces the
    // highlighted edges to exist at all.
    let residue = residue_overlay(data_pos, extent, pixels_per_cell);
    rgb = mix(rgb, residue.color, residue.coverage);

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

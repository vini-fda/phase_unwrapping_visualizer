//! Colormaps, and the lookup tables that let the CPU and the GPU agree on them.
//!
//! Both the sidebar legend and the `wgpu` fragment shader colour their pixels by
//! sampling the *same* table produced by [`Colormap::lut`], so the two cannot
//! drift apart.
//!
//! Components are in **gamma (sRGB) space**, matching how the palette was
//! authored; converting to linear light, when the render target needs it, is the
//! renderer's job.

/// How the viewer is interpreting the field's values.
///
/// This single choice drives the image and the legend together: a cyclic
/// quantity gets a cyclic colormap, and an unbounded one does not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum DisplayMode {
    /// Raw values, mapped over the field's own `[min, max]`.
    Unbounded,
    /// Values wrapped to `(-π, π]`.
    #[default]
    Wrapped,
}

impl DisplayMode {
    /// The colormap this mode is displayed with.
    pub fn colormap(self) -> Colormap {
        match self {
            Self::Unbounded => Colormap::Grayscale,
            Self::Wrapped => Colormap::CubehelixCycle,
        }
    }

    /// `true` when values are wrapped before being coloured.
    pub fn is_wrapped(self) -> bool {
        self == Self::Wrapped
    }

    /// Short label for a toggle button.
    pub fn label(self) -> &'static str {
        match self {
            Self::Unbounded => "Unbounded",
            Self::Wrapped => "Wrapped",
        }
    }
}

/// A mapping from a normalized value in `[0, 1]` to an sRGB colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum Colormap {
    /// Perceptually plain dark-to-light ramp, for unbounded values.
    Grayscale,
    /// Cyclic cubehelix palette, for wrapped phase: `sample(0) == sample(1)`,
    /// so there is no seam at `±π`.
    CubehelixCycle,
}

/// Control points of the cubehelix cycle, from the BEAM colour palette.
///
/// These are the eight knots of a closed loop: the curve runs
/// `0 → 1 → … → 7 → 0`, which is what makes the map cyclic.
const CUBEHELIX_KNOTS: [[f32; 3]; 8] = [
    [110.0 / 255.0, 60.0 / 255.0, 170.0 / 255.0],
    [210.0 / 255.0, 60.0 / 255.0, 160.0 / 255.0],
    [1.0, 110.0 / 255.0, 70.0 / 255.0],
    [200.0 / 255.0, 200.0 / 255.0, 50.0 / 255.0],
    [80.0 / 255.0, 245.0 / 255.0, 100.0 / 255.0],
    [25.0 / 255.0, 200.0 / 255.0, 180.0 / 255.0],
    [60.0 / 255.0, 130.0 / 255.0, 220.0 / 255.0],
    [100.0 / 255.0, 70.0 / 255.0, 190.0 / 255.0],
];

/// Darkest grey the [`Colormap::Grayscale`] ramp reaches.
///
/// Deliberately above zero: the cell walls the viewer draws on top of the image
/// are dark, and a pure-black floor would swallow them wherever the data bottoms
/// out, hiding the grid exactly where the user zoomed in to see it.
const GRAY_FLOOR: f32 = 0.10;

impl Colormap {
    /// Samples the colormap at `t`.
    ///
    /// `t` is clamped to `[0, 1]` for non-cyclic maps and taken modulo `1` for
    /// cyclic ones, so every input produces a colour.
    pub fn sample(self, t: f32) -> [f32; 3] {
        match self {
            Self::Grayscale => {
                let v = GRAY_FLOOR + (1.0 - GRAY_FLOOR) * t.clamp(0.0, 1.0);
                [v, v, v]
            }
            Self::CubehelixCycle => cubehelix_cycle(t),
        }
    }

    /// `true` when `sample(0)` and `sample(1)` are the same colour.
    pub fn is_cyclic(self) -> bool {
        match self {
            Self::Grayscale => false,
            Self::CubehelixCycle => true,
        }
    }

    /// Builds an RGBA8 lookup table with `len` opaque texels.
    ///
    /// Texel `i` holds the colour at `t = (i + 0.5) / len`. That half-texel
    /// offset is the convention a linearly-filtered texture sampler expects, so
    /// sampling the table at `t` reproduces `sample(t)` rather than a ramp
    /// shifted by half a texel.
    pub fn lut(self, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len * 4);
        for i in 0..len {
            let t = (i as f32 + 0.5) / len as f32;
            for channel in self.sample(t) {
                bytes.push(to_byte(channel));
            }
            bytes.push(u8::MAX);
        }
        bytes
    }
}

/// Quantizes a colour component to 8 bits, rounding rather than truncating.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the input is clamped to [0, 1], so the result is in [0, 255]"
)]
fn to_byte(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Evaluates the closed cubehelix loop at `t`, taken modulo `1`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "`x` is in [0, knots) after `rem_euclid`, so its floor is a valid index"
)]
fn cubehelix_cycle(t: f32) -> [f32; 3] {
    let knots = CUBEHELIX_KNOTS.len();

    // The loop has as many segments as knots, because the last segment closes it
    // by running from knot 7 back to knot 0.
    let x = t.rem_euclid(1.0) * knots as f32;
    let segment = (x.floor() as usize) % knots;
    let local = x - x.floor();

    let knot = |offset: usize| CUBEHELIX_KNOTS[(segment + offset) % knots];
    let (p0, p1, p2, p3) = (knot(knots - 1), knot(0), knot(1), knot(2));

    let mut rgb = [0.0; 3];
    for (channel, out) in rgb.iter_mut().enumerate() {
        *out =
            catmull_rom(p0[channel], p1[channel], p2[channel], p3[channel], local).clamp(0.0, 1.0);
    }
    rgb
}

/// Uniform Catmull–Rom interpolation between `p1` and `p2`.
///
/// `p0` and `p3` are the neighbouring control points; they set the tangents.
/// The spline passes exactly through its control points (`f(0) == p1`,
/// `f(1) == p2`) and is C¹ continuous where segments meet, which is what keeps
/// the cubehelix ramp free of seams.
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (3.0f32.mul_add(p1, -p0) - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Colormap; 2] = [Colormap::Grayscale, Colormap::CubehelixCycle];

    fn max_diff(a: [f32; 3], b: [f32; 3]) -> f32 {
        (0..3).fold(0.0f32, |acc, i| acc.max((a[i] - b[i]).abs()))
    }

    /// The bug this port fixes: the original interpolation never reached the next
    /// control point, so the ramp jumped at every one of the eight segment
    /// boundaries. A proper spline passes *through* its knots.
    #[test]
    fn cubehelix_passes_through_its_control_points() {
        for (i, knot) in CUBEHELIX_KNOTS.iter().enumerate() {
            let t = i as f32 / CUBEHELIX_KNOTS.len() as f32;
            let sampled = Colormap::CubehelixCycle.sample(t);
            assert!(
                max_diff(sampled, *knot) < 1e-5,
                "knot {i} at t = {t}: expected {knot:?}, got {sampled:?}"
            );
        }
    }

    /// Continuity across the whole domain, segment boundaries included.
    #[test]
    fn colormaps_are_continuous() {
        for map in ALL {
            let steps = 4096;
            let mut previous = map.sample(0.0);
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                let current = map.sample(t);
                assert!(
                    max_diff(current, previous) < 0.02,
                    "{map:?} jumped at t = {t}: {previous:?} -> {current:?}"
                );
                previous = current;
            }
        }
    }

    /// A cyclic map must close the loop, or wrapped phase shows a seam at ±π.
    #[test]
    fn cyclic_colormap_closes_the_loop() {
        let map = Colormap::CubehelixCycle;
        assert!(map.is_cyclic(), "cubehelix cycle is cyclic by construction");
        assert!(
            max_diff(map.sample(0.0), map.sample(1.0)) < 1e-5,
            "sample(0) and sample(1) must be the same colour"
        );
        assert!(
            max_diff(map.sample(0.001), map.sample(1.001)) < 1e-5,
            "the map must continue smoothly past t = 1"
        );
    }

    #[test]
    fn components_stay_in_range() {
        for map in ALL {
            for i in -100..=1100 {
                let t = i as f32 / 1000.0;
                for c in map.sample(t) {
                    assert!(
                        (0.0..=1.0).contains(&c),
                        "{map:?} produced {c} at t = {t}, outside [0, 1]"
                    );
                }
            }
        }
    }

    #[test]
    fn grayscale_is_monotonic_and_never_pure_black() {
        let map = Colormap::Grayscale;
        let mut previous = -1.0;
        for i in 0..=1000 {
            let t = i as f32 / 1000.0;
            let v = map.sample(t)[0];
            assert!(v >= previous, "grayscale must not decrease at t = {t}");
            assert!(
                v >= GRAY_FLOOR - 1e-6,
                "grayscale dipped below the floor at t = {t}: {v}"
            );
            previous = v;
        }
    }

    #[test]
    fn lut_uses_the_half_texel_convention() {
        let len = 256;
        let lut = Colormap::CubehelixCycle.lut(len);
        assert_eq!(lut.len(), len * 4, "the table must be RGBA8");

        for i in [0, 1, 97, len - 1] {
            let t = (i as f32 + 0.5) / len as f32;
            let expected = Colormap::CubehelixCycle.sample(t);
            for channel in 0..3 {
                assert_eq!(
                    lut[i * 4 + channel],
                    to_byte(expected[channel]),
                    "texel {i} channel {channel} should hold the colour at t = {t}"
                );
            }
            assert_eq!(lut[i * 4 + 3], u8::MAX, "texel {i} must be opaque");
        }
    }

    #[test]
    fn display_mode_picks_a_cyclic_map_for_cyclic_data() {
        assert!(
            DisplayMode::Wrapped.colormap().is_cyclic(),
            "wrapped phase is cyclic, so its colormap must be too"
        );
        assert!(
            !DisplayMode::Unbounded.colormap().is_cyclic(),
            "unbounded values must not be shown with a cyclic ramp"
        );
    }
}

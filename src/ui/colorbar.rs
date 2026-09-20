//! The sidebar legend: the active colormap, annotated with the values it spans.

use egui::{
    Align2, Color32, ColorImage, CornerRadius, FontId, Rect, Sense, StrokeKind, TextureHandle,
    TextureOptions, Ui, pos2, vec2,
};

use crate::colormap::{Colormap, DisplayMode};
use crate::render::LUT_LEN;

/// Height of the ramp, in points.
const BAR_HEIGHT: f32 = 220.0;

/// Width of the ramp, in points.
const BAR_WIDTH: f32 = 22.0;

/// Length of a tick mark, in points.
const TICK_LENGTH: f32 = 5.0;

/// Gap between a tick mark and its label, in points.
const LABEL_GAP: f32 = 4.0;

/// Draws the colormap legend, caching the ramp texture between frames.
///
/// The ramp is built from the very same [`Colormap::lut`] the shader samples,
/// so the legend cannot disagree with the image it describes.
#[derive(Default)]
pub struct Colorbar {
    cached: Option<(Colormap, TextureHandle)>,
}

impl Colorbar {
    /// Draws the ramp for `mode`, labelled with the values at its ends.
    ///
    /// `value_range` is `(min, max)` in the field's own units; it is ignored in
    /// wrapped mode, where the range is always `(-π, π]`.
    pub fn show(&mut self, ui: &mut Ui, mode: DisplayMode, value_range: (f32, f32)) {
        let colormap = mode.colormap();
        let texture = self.texture(ui, colormap);

        let (rect, _response) =
            ui.allocate_exact_size(vec2(ui.available_width(), BAR_HEIGHT), Sense::hover());
        let bar = Rect::from_min_size(rect.min, vec2(BAR_WIDTH, rect.height()));

        let painter = ui.painter();
        painter.image(
            texture.id(),
            bar,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        let visuals = ui.visuals();
        let stroke = visuals.widgets.noninteractive.fg_stroke;
        painter.rect_stroke(bar, CornerRadius::ZERO, stroke, StrokeKind::Inside);

        let font = FontId::proportional(11.0);
        for (fraction, label) in ticks(mode, value_range) {
            // Fraction 0 is the bottom of the ramp, 1 the top.
            let y = bar.bottom() - fraction * bar.height();
            painter.line_segment(
                [pos2(bar.right(), y), pos2(bar.right() + TICK_LENGTH, y)],
                stroke,
            );
            painter.text(
                pos2(bar.right() + TICK_LENGTH + LABEL_GAP, y),
                Align2::LEFT_CENTER,
                label,
                font.clone(),
                visuals.text_color(),
            );
        }
    }

    /// The ramp texture for `colormap`, rebuilt only when the colormap changes.
    fn texture(&mut self, ui: &Ui, colormap: Colormap) -> TextureHandle {
        if let Some((cached, texture)) = &self.cached
            && *cached == colormap
        {
            return texture.clone();
        }

        let lut = colormap.lut(LUT_LEN);
        // A one-pixel-wide column, top to bottom, so the largest value is drawn
        // at the top the way an axis is normally read.
        let mut pixels = Vec::with_capacity(LUT_LEN * 4);
        for texel in lut.chunks_exact(4).rev() {
            pixels.extend_from_slice(texel);
        }

        let image = ColorImage::from_rgba_unmultiplied([1, LUT_LEN], &pixels);
        let texture = ui.ctx().load_texture(
            format!("colorbar_{colormap:?}"),
            image,
            TextureOptions::LINEAR,
        );
        self.cached = Some((colormap, texture.clone()));
        texture
    }
}

/// The tick marks for a mode, as `(fraction along the ramp, label)`.
fn ticks(mode: DisplayMode, value_range: (f32, f32)) -> Vec<(f32, String)> {
    match mode {
        DisplayMode::Wrapped => ["−π", "−π/2", "0", "π/2", "π"]
            .into_iter()
            .enumerate()
            .map(|(i, label)| (i as f32 / 4.0, label.to_owned()))
            .collect(),
        DisplayMode::Unbounded => {
            let (min, max) = value_range;
            (0..5)
                .map(|i| {
                    let fraction = i as f32 / 4.0;
                    let value = (max - min).mul_add(fraction, min);
                    (fraction, format_value(value))
                })
                .collect()
        }
    }
}

/// Formats a value compactly: phase in radians spans a few tens at most, but a
/// field of unwrapped values can be arbitrarily large.
fn format_value(value: f32) -> String {
    if value != 0.0 && (value.abs() < 0.01 || value.abs() >= 10_000.0) {
        format!("{value:.2e}")
    } else {
        format!("{value:.2}")
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    /// The values, in radians, that the wrapped legend's fixed labels stand for.
    const WRAPPED_TICK_VALUES: [f32; 5] = [-PI, -PI / 2.0, 0.0, PI / 2.0, PI];

    #[test]
    fn wrapped_ticks_span_the_whole_cycle() {
        let ticks = ticks(DisplayMode::Wrapped, (0.0, 1.0));
        assert_eq!(ticks.len(), 5, "-π, -π/2, 0, π/2, π");
        assert_eq!(ticks[0].0, 0.0, "-π sits at the bottom of the ramp");
        assert_eq!(ticks[4].0, 1.0, "π sits at the top");

        // The labels are fixed strings; this pins them to the values the shader
        // actually maps to those positions.
        for (i, (fraction, _)) in ticks.iter().enumerate() {
            let value = (2.0 * PI).mul_add(*fraction, -PI);
            assert!(
                (value - WRAPPED_TICK_VALUES[i]).abs() < 1e-5,
                "tick {i} at fraction {fraction} stands for {value}, not {}",
                WRAPPED_TICK_VALUES[i]
            );
        }
    }

    #[test]
    fn unbounded_ticks_interpolate_the_data_range() {
        let ticks = ticks(DisplayMode::Unbounded, (-3.0, 7.0));
        assert_eq!(ticks.len(), 5, "five evenly spaced ticks");
        assert_eq!(ticks[0].1, "-3.00", "the bottom tick is the minimum");
        assert_eq!(ticks[2].1, "2.00", "the middle tick is the midpoint");
        assert_eq!(ticks[4].1, "7.00", "the top tick is the maximum");
    }

    #[test]
    fn large_and_tiny_values_fall_back_to_scientific_notation() {
        assert_eq!(format_value(0.0), "0.00", "zero stays plain");
        assert_eq!(format_value(12.5), "12.50", "ordinary values stay plain");
        assert_eq!(
            format_value(1.0e6),
            "1.00e6",
            "large values are abbreviated"
        );
        assert_eq!(format_value(1.0e-5), "1.00e-5", "tiny values too");
    }
}

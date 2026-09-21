//! The node representation: pixels as graph vertices, phase differences as
//! edges.
//!
//! This is the same graph as the cell view, drawn the other way round — and the
//! same data space, so it shares the viewer's [`ViewTransform`] unchanged.
//! Pixel `(row, col)` sits at `(col + 0.5, row + 0.5)` and the residue between
//! four pixels at `(col + 1, row + 1)`, which is exactly where the cell view
//! puts the corresponding inner corner.
//!
//! Unlike the cell view this is drawn on the CPU with egui shapes. Arrowheads
//! and dash patterns are fiddly in a fragment shader and pointless when zoomed
//! out, so instead the view is gated on zoom and culled to what is on screen:
//! at a readable scale that is at most a few thousand primitives.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, Ui, pos2, vec2};

use crate::colormap::DisplayMode;
use crate::graph::{EdgeState, Traversal, Unwrapping};
use crate::view::ViewTransform;

/// Below this many points per cell the diagram is unreadable and is replaced by
/// a hint.
pub const MIN_POINTS_PER_CELL: f32 = 16.0;

/// Above this, each node is labelled with its value.
const LABEL_POINTS_PER_CELL: f32 = 56.0;

/// Edges whose integration delta is not the wrapped delta.
const HIGHLIGHT: Color32 = Color32::from_rgb(232, 23, 135);

/// Residue charges, matching the figures.
const RESIDUE_POSITIVE: Color32 = Color32::from_rgb(245, 130, 31);
const RESIDUE_NEGATIVE: Color32 = Color32::from_rgb(92, 199, 232);

/// Edges walked against their canonical orientation.
const BACKWARD: Color32 = Color32::from_rgb(230, 85, 75);

/// Draws the node diagram of `unwrapping` into `rect`.
///
/// Returns `false` without drawing anything if the zoom is too low for the
/// diagram to mean anything; the caller shows a hint instead.
pub fn show(
    ui: &Ui,
    rect: Rect,
    view: &ViewTransform,
    unwrapping: &Unwrapping,
    mode: DisplayMode,
    value_range: (f32, f32),
) -> bool {
    if view.points_per_cell() < MIN_POINTS_PER_CELL {
        return false;
    }

    let painter = ui.painter_at(rect);
    let visible = view.visible_data_rect(rect);
    let (rows, cols) = (unwrapping.rows(), unwrapping.cols());

    // Cull to the cells on screen, with one cell of margin so edges and markers
    // that straddle the border are still drawn.
    let first_row = clamp_index(visible.min.y - 1.0, rows);
    let last_row = clamp_index(visible.max.y + 1.0, rows);
    let first_col = clamp_index(visible.min.x - 1.0, cols);
    let last_col = clamp_index(visible.max.x + 1.0, cols);

    let scale = view.points_per_cell();
    let at = |row: usize, col: usize| {
        view.data_to_screen(pos2(col as f32 + 0.5, row as f32 + 0.5), rect)
    };

    let visuals = ui.visuals();
    let forward = visuals.text_color();
    let cut = visuals.weak_text_color();

    // Edges first, so the nodes sit on top of their ends.
    for row in first_row..last_row {
        for col in first_col..last_col {
            if col + 1 < cols
                && let Some(state) = unwrapping.horizontal_edge(row, col)
            {
                draw_edge(
                    &painter,
                    at(row, col),
                    at(row, col + 1),
                    state,
                    scale,
                    forward,
                    cut,
                );
            }
            if row + 1 < rows
                && let Some(state) = unwrapping.vertical_edge(row, col)
            {
                draw_edge(
                    &painter,
                    at(row, col),
                    at(row + 1, col),
                    state,
                    scale,
                    forward,
                    cut,
                );
            }
        }
    }

    // Residues sit at the centre of each group of four pixels.
    let radius = (scale * 0.13).clamp(3.0, 9.0);
    for row in first_row..last_row {
        for col in first_col..last_col {
            let Some(charge) = unwrapping.residue(row, col) else {
                continue;
            };
            let center = view.data_to_screen(pos2(col as f32 + 1.0, row as f32 + 1.0), rect);
            if charge == 0 {
                painter.circle_filled(center, (radius * 0.28).max(1.0), cut);
                continue;
            }
            let color = if charge > 0 {
                RESIDUE_POSITIVE
            } else {
                RESIDUE_NEGATIVE
            };
            painter.circle_filled(center, radius, color);
            painter.text(
                center,
                Align2::CENTER_CENTER,
                if charge > 0 { "+" } else { "−" },
                FontId::proportional(radius * 1.5),
                Color32::BLACK,
            );
        }
    }

    // Nodes, coloured by the value they carry so the diagram still shows the
    // field rather than just its topology.
    let node_radius = (scale * 0.1).clamp(2.0, 7.0);
    let label = scale >= LABEL_POINTS_PER_CELL;
    let field = unwrapping.unwrapped();
    for row in first_row..last_row {
        for col in first_col..last_col {
            let Some(value) = field.get(row, col) else {
                continue;
            };
            let center = at(row, col);
            painter.circle(
                center,
                node_radius,
                node_color(mode, value, value_range),
                Stroke::new(1.0, visuals.window_stroke.color),
            );
            if label {
                painter.text(
                    center + vec2(0.0, -node_radius - 2.0),
                    Align2::CENTER_BOTTOM,
                    format!("{value:.2}"),
                    FontId::monospace(scale * 0.13),
                    forward,
                );
            }
        }
    }

    true
}

/// What to say when the diagram cannot show a walk.
pub fn path_hint(unwrapping: &Unwrapping) -> Option<&'static str> {
    (!unwrapping.has_path()).then_some(
        "No integration path provided — edges are drawn plain, with no arrows.          Open one with File → Open integration path…",
    )
}

/// Where the diagram stops being legible, for the caller's hint.
pub fn zoom_hint(points_per_cell: f32) -> Option<String> {
    (points_per_cell < MIN_POINTS_PER_CELL).then(|| {
        format!(
            "Zoom in to read the node graph — {points_per_cell:.0} px/cell, needs {MIN_POINTS_PER_CELL:.0}"
        )
    })
}

/// The colour a node carries.
///
/// Both the colormap and the position along it come from the display mode, so
/// a node is coloured exactly as the cell view would colour the same sample —
/// wrapped first when the mode says so, rather than stretched across the
/// field's whole unbounded range.
fn node_color(mode: DisplayMode, value: f32, value_range: (f32, f32)) -> Color32 {
    let t = crate::render::colormap_position(value, mode.is_wrapped(), value_range);
    let [r, g, b] = mode.colormap().sample(t);
    Color32::from_rgb(to_byte(r), to_byte(g), to_byte(b))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the input is clamped to [0, 1], so the result is in [0, 255]"
)]
fn to_byte(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Clamps a data-space coordinate to a usable index bound.
///
/// The two infinities are handled apart from `NaN`: an infinite *upper* bound
/// should cull nothing, whereas `NaN` cannot index anything at all. Collapsing
/// both to zero would leave the diagram silently empty.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is bounded by `limit` before narrowing"
)]
fn clamp_index(coordinate: f32, limit: usize) -> usize {
    if coordinate.is_nan() || coordinate <= 0.0 {
        return 0;
    }
    if coordinate >= limit as f32 {
        return limit;
    }
    coordinate as usize
}

/// Draws one edge of `G` between two node centres.
///
/// Tree edges get an arrowhead pointing the way the integration walked them;
/// cut edges are drawn as a faint dotted line, since the walk never crosses
/// them. An edge whose integration delta disagrees with the wrapped delta is
/// drawn in the highlight colour whatever its role.
fn draw_edge(
    painter: &Painter,
    source: Pos2,
    target: Pos2,
    state: EdgeState,
    scale: f32,
    forward_color: Color32,
    cut_color: Color32,
) {
    let inset = scale * 0.16;
    let Some((from, to)) = shorten(source, target, inset) else {
        return;
    };

    let (color, width) = match (state.is_consistent(), state.traversal) {
        (false, _) => (HIGHLIGHT, 2.4),
        // No integration path was supplied, so nothing is known about this
        // edge's role: it is drawn plainly rather than guessed at.
        (true, None) => (cut_color, 1.2),
        (true, Some(Traversal::Cut)) => (cut_color, 1.0),
        (true, Some(Traversal::Backward)) => (BACKWARD, 1.6),
        (true, Some(Traversal::Forward)) => (forward_color, 1.6),
    };
    let stroke = Stroke::new(width, color);

    match state.traversal {
        None => {
            painter.line_segment([from, to], stroke);
        }
        Some(Traversal::Cut) => dotted(painter, from, to, stroke),
        // The walk enters the target, so the head goes there for a forward edge
        // and at the source for a backward one.
        Some(Traversal::Forward) => arrow(painter, from, to, stroke, scale),
        Some(Traversal::Backward) => arrow(painter, to, from, stroke, scale),
    }
}

/// Pulls both ends of a segment in by `inset`, so edges stop short of the node
/// markers instead of running underneath them. `None` if nothing is left.
fn shorten(source: Pos2, target: Pos2, inset: f32) -> Option<(Pos2, Pos2)> {
    let delta = target - source;
    let length = delta.length();
    if length <= 2.0 * inset {
        return None;
    }
    let unit = delta / length;
    Some((source + unit * inset, target - unit * inset))
}

/// A line with an arrowhead at `to`.
fn arrow(painter: &Painter, from: Pos2, to: Pos2, stroke: Stroke, scale: f32) {
    painter.line_segment([from, to], stroke);

    let delta = to - from;
    let length = delta.length();
    if length <= f32::EPSILON {
        return;
    }
    let unit = delta / length;
    let normal = vec2(-unit.y, unit.x);
    let head = (scale * 0.13).clamp(3.0, 12.0).min(length);
    let base = to - unit * head;

    painter.add(egui::Shape::convex_polygon(
        vec![to, base + normal * head * 0.45, base - normal * head * 0.45],
        stroke.color,
        Stroke::NONE,
    ));
}

/// A dotted line, for the edges the integration never crosses.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the step count is clamped to a small positive range before narrowing"
)]
fn dotted(painter: &Painter, from: Pos2, to: Pos2, stroke: Stroke) {
    let delta = to - from;
    let length = delta.length();
    if length <= f32::EPSILON {
        return;
    }
    let spacing = 4.0;
    // Bounded so an absurdly long segment cannot spin here.
    let steps = (length / spacing).round().clamp(1.0, 4096.0) as usize;
    let unit = delta / length;
    for step in 0..steps {
        let start = from + unit * (step as f32 * spacing);
        let end = from + unit * ((step as f32 + 0.45) * spacing);
        painter.line_segment([start, end], stroke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Rect;

    fn viewport() -> Rect {
        Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0))
    }

    /// The two representations must agree about where things are, or panning
    /// between them would jump.
    #[test]
    fn a_residue_sits_at_the_corner_shared_by_its_four_pixels() {
        let vp = viewport();
        let view = ViewTransform::fit(8, 8, vp);

        // Residue (row, col) is the corner below-right of pixel (row, col),
        // which is the midpoint of the four pixel centres around it.
        for (row, col) in [(0, 0), (3, 5), (6, 6)] {
            let residue = view.data_to_screen(pos2(col as f32 + 1.0, row as f32 + 1.0), vp);
            let corners = [
                (row, col),
                (row, col + 1),
                (row + 1, col),
                (row + 1, col + 1),
            ]
            .map(|(r, c)| view.data_to_screen(pos2(c as f32 + 0.5, r as f32 + 0.5), vp));

            let mean = corners
                .iter()
                .fold(vec2(0.0, 0.0), |acc, p| acc + p.to_vec2())
                / 4.0;
            assert!(
                (residue.to_vec2() - mean).length() < 1e-3,
                "residue ({row}, {col}) must sit at the centre of its four pixels"
            );
        }
    }

    /// The bug this fixes: the node view coloured by the raw value, so in
    /// wrapped mode a field spanning many turns was smeared across the
    /// colormap instead of being wrapped first. Two samples a whole turn apart
    /// are the same phase and must look it.
    #[test]
    fn wrapped_mode_colours_a_node_by_its_phase_not_its_value() {
        use std::f32::consts::TAU;
        // A range like a real unwrapped field's: many turns wide.
        let range = (0.0, 30.0);

        for value in [0.3, 2.0, 5.5] {
            assert_eq!(
                node_color(DisplayMode::Wrapped, value, range),
                node_color(DisplayMode::Wrapped, value + TAU, range),
                "{value} and {value} + 2π are the same wrapped phase"
            );
            assert_eq!(
                node_color(DisplayMode::Wrapped, value, range),
                node_color(DisplayMode::Wrapped, value + 3.0 * TAU, range),
                "however many turns apart"
            );
        }

        // The unbounded mode must keep telling them apart, or it would be
        // showing the wrapped field under a different name.
        assert_ne!(
            node_color(DisplayMode::Unbounded, 2.0, range),
            node_color(DisplayMode::Unbounded, 2.0 + TAU, range),
            "unbounded mode reads the value itself"
        );
    }

    /// A node and the cell under it are the same sample, so they must be the
    /// same colour — the two views take entirely different paths to it.
    #[test]
    fn a_node_is_coloured_exactly_as_its_cell_would_be() {
        let range = (-4.0, 19.0);
        for mode in [DisplayMode::Unbounded, DisplayMode::Wrapped] {
            for value in [-4.0, 0.0, 1.5, 7.25, 19.0, 100.0] {
                let shader_t = crate::render::colormap_position(value, mode.is_wrapped(), range);
                let [r, g, b] = mode.colormap().sample(shader_t);
                assert_eq!(
                    node_color(mode, value, range),
                    Color32::from_rgb(to_byte(r), to_byte(g), to_byte(b)),
                    "{mode:?} at {value} must match what the shader computes"
                );
            }
        }
    }

    #[test]
    fn culling_keeps_indices_inside_the_field() {
        assert_eq!(clamp_index(-5.0, 10), 0, "off the top-left clamps to 0");
        assert_eq!(clamp_index(3.7, 10), 3, "inside the field truncates");
        assert_eq!(
            clamp_index(99.0, 10),
            10,
            "past the end clamps to the limit"
        );
        assert_eq!(clamp_index(f32::NAN, 10), 0, "NaN cannot index anything");
        assert_eq!(
            clamp_index(f32::INFINITY, 10),
            10,
            "an infinite bound clamps to the limit"
        );
    }

    #[test]
    fn shortening_drops_segments_with_nothing_left() {
        let (from, to) = shorten(pos2(0.0, 0.0), pos2(10.0, 0.0), 2.0).expect("room to draw");
        assert!(
            (from.x - 2.0).abs() < 1e-5 && (to.x - 8.0).abs() < 1e-5,
            "both ends must be pulled in by the inset, got {from:?} -> {to:?}"
        );
        assert!(
            shorten(pos2(0.0, 0.0), pos2(3.0, 0.0), 2.0).is_none(),
            "an edge shorter than twice the inset has nothing left to draw"
        );
    }

    #[test]
    fn the_zoom_hint_appears_exactly_below_the_threshold() {
        assert!(
            zoom_hint(MIN_POINTS_PER_CELL).is_none(),
            "at the threshold the diagram is drawn"
        );
        assert!(
            zoom_hint(MIN_POINTS_PER_CELL - 0.1).is_some(),
            "below it the hint takes over"
        );
    }
}

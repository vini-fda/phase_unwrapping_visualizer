//! The pannable, zoomable image of the phase field.

use std::sync::Arc;

use eframe::egui_wgpu;
use egui::{CursorIcon, Key, Pos2, Rect, Response, Sense, Ui};

use crate::colormap::DisplayMode;
use crate::phase::{self, PhaseField};
use crate::render::{GridCallback, GridUniforms, OverlaySource};
use crate::view::ViewTransform;

/// Zoom applied per notch of the scroll wheel, as an exponent: zooming is
/// multiplicative, so that the same gesture feels the same at every scale.
const SCROLL_ZOOM_RATE: f32 = 0.0025;

/// Zoom factor for one press of a zoom button or key.
pub const ZOOM_STEP: f32 = 1.25;

/// Below this, a zoom request is treated as "no change".
const ZOOM_EPSILON: f32 = 1.0e-6;

/// The cell under the pointer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoverInfo {
    /// Row index of the cell.
    pub row: usize,
    /// Column index of the cell.
    pub col: usize,
    /// The stored sample.
    pub value: f32,
    /// The sample wrapped to `(-π, π]`.
    pub wrapped: f32,
}

/// What [`PhaseImage::show`] reports back to the app.
pub struct PhaseImageOutput {
    /// The widget's response, for further interaction.
    pub response: Response,
    /// The cell under the pointer, if any.
    pub hover: Option<HoverInfo>,
}

/// The image widget.
pub struct PhaseImage<'a> {
    /// Samples to draw. Shared with the render callback rather than copied.
    pub field: &'a Arc<PhaseField>,

    /// How to colour the samples.
    pub mode: DisplayMode,

    /// The values mapped to the two ends of the colormap.
    pub value_range: (f32, f32),

    /// Pan/zoom state. `None` means "not positioned yet": the widget fits the
    /// field to the viewport the first time it knows how big the viewport is.
    pub view: &'a mut Option<ViewTransform>,

    /// A zoom factor requested from outside, e.g. by a toolbar button. `1.0`
    /// for none; applied about the centre of the view rather than the pointer.
    pub external_zoom: f32,

    /// An unwrapping to draw over the field: per-edge wall colours and residue
    /// markers. `None` draws the field alone.
    pub overlay: Option<OverlaySource>,
}

/// Allocates the viewer's area and applies this frame's pan and zoom, without
/// drawing anything into it.
///
/// Both representations share this, so a gesture means the same thing in
/// either and switching between them never moves the view.
///
/// `min_points_per_cell` is the zoom below which the calling representation has
/// nothing to show, and only applies to the first fit: it is a floor on where
/// the view *starts*, never on where the user may take it.
pub fn interact(
    ui: &mut Ui,
    rows: usize,
    cols: usize,
    view: &mut Option<ViewTransform>,
    external_zoom: f32,
    min_points_per_cell: f32,
) -> (Rect, Response) {
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

    let transform = view
        .get_or_insert_with(|| ViewTransform::fit_at_least(rows, cols, rect, min_points_per_cell));

    if response.dragged() {
        transform.pan_by_screen_delta(response.drag_delta());
    }

    if (external_zoom - 1.0).abs() > ZOOM_EPSILON {
        transform.zoom_about(rect.center(), external_zoom, rect);
    }

    if response.hovered() {
        let factor = zoom_from_input(ui);
        if (factor - 1.0).abs() > ZOOM_EPSILON {
            let anchor = response.hover_pos().unwrap_or_else(|| rect.center());
            transform.zoom_about(anchor, factor, rect);
        }
    }

    let response = if response.dragged() {
        response.on_hover_cursor(CursorIcon::Grabbing)
    } else {
        response.on_hover_cursor(CursorIcon::Grab)
    };

    (rect, response)
}

/// The cell under the pointer, for the sidebar readout.
pub fn hover_at(
    view: &ViewTransform,
    field: &PhaseField,
    response: &Response,
    rect: Rect,
) -> Option<HoverInfo> {
    let pointer = response.hover_pos()?;
    cell_at(view, field, pointer, rect)
}

impl PhaseImage<'_> {
    /// Draws the image, taking up all the remaining space in `ui`.
    pub fn show(self, ui: &mut Ui) -> PhaseImageOutput {
        let (rows, cols) = (self.field.rows(), self.field.cols());
        // The cell view draws something at any scale, so it fits to contain and
        // asks for no floor.
        let (rect, response) = interact(ui, rows, cols, self.view, self.external_zoom, 0.0);

        let view = self
            .view
            .as_ref()
            .copied()
            .unwrap_or_else(|| ViewTransform::fit(rows, cols, rect));

        let hover = hover_at(&view, self.field, &response, rect);

        // `points_per_cell` is in egui points; the shader sizes cell walls in
        // physical pixels, so the display's scale factor belongs here.
        let pixels_per_cell = view.points_per_cell() * ui.ctx().pixels_per_point();

        let uniforms = GridUniforms::new(
            view.visible_data_rect(rect),
            rows,
            cols,
            pixels_per_cell,
            self.value_range,
            self.mode.is_wrapped(),
            self.overlay.as_ref().map(|overlay| overlay.options),
        );

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            GridCallback::new(Arc::clone(self.field), self.mode.colormap(), uniforms)
                .with_overlay(self.overlay),
        ));

        PhaseImageOutput { response, hover }
    }
}

/// Collects this frame's zoom gestures into a single multiplicative factor.
fn zoom_from_input(ui: &Ui) -> f32 {
    ui.input(|input| {
        // A pinch, or ctrl+scroll, arrives as a zoom delta; egui keeps it out of
        // the scroll delta, so the two never double up.
        let mut factor = input.zoom_delta();
        if (factor - 1.0).abs() <= ZOOM_EPSILON {
            factor = (input.smooth_scroll_delta.y * SCROLL_ZOOM_RATE).exp();
        }
        if input.key_pressed(Key::Plus) || input.key_pressed(Key::Equals) {
            factor *= ZOOM_STEP;
        }
        if input.key_pressed(Key::Minus) {
            factor /= ZOOM_STEP;
        }
        factor
    })
}

/// Resolves a screen position to the cell under it, if it is over the field.
#[expect(
    clippy::cast_possible_truncation,
    reason = "negatives are rejected above, and a coordinate too large to index \
              saturates to a row or column `PhaseField::get` then rejects"
)]
fn cell_at(
    view: &ViewTransform,
    field: &PhaseField,
    pointer: Pos2,
    viewport: Rect,
) -> Option<HoverInfo> {
    let data = view.screen_to_data(pointer, viewport);
    if data.x < 0.0 || data.y < 0.0 {
        // Casting a negative float to `usize` saturates to 0, which would
        // wrongly report the first row or column as hovered.
        return None;
    }
    let col = data.x as usize;
    let row = data.y as usize;
    let value = field.get(row, col)?;
    Some(HoverInfo {
        row,
        col,
        value,
        wrapped: phase::wrap(value),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};

    fn field() -> PhaseField {
        PhaseField::new((0..12).map(|i| i as f32).collect(), 3, 4).expect("12 == 3 × 4")
    }

    fn viewport() -> Rect {
        Rect::from_min_size(pos2(100.0, 50.0), vec2(400.0, 300.0))
    }

    #[test]
    fn hover_reports_the_cell_under_the_pointer() {
        let vp = viewport();
        let field = field();
        let view = ViewTransform::fit(field.rows(), field.cols(), vp);

        for (row, col) in [(0, 0), (0, 3), (2, 0), (2, 3), (1, 2)] {
            let centre = view.data_to_screen(pos2(col as f32 + 0.5, row as f32 + 0.5), vp);
            let hovered = cell_at(&view, &field, centre, vp).expect("centre is over the field");
            assert_eq!(
                (hovered.row, hovered.col),
                (row, col),
                "the centre of cell ({row}, {col}) must resolve back to it"
            );
            assert_eq!(
                hovered.value,
                field.get(row, col).expect("in bounds"),
                "the reported value must be the cell's own sample"
            );
        }
    }

    #[test]
    fn hover_is_none_outside_the_field() {
        let vp = viewport();
        let field = field();
        let view = ViewTransform::fit(field.rows(), field.cols(), vp);

        // Just outside each edge of the image, in data space.
        for data in [
            pos2(-0.01, 1.0),
            pos2(1.0, -0.01),
            pos2(4.01, 1.0),
            pos2(1.0, 3.01),
        ] {
            let screen = view.data_to_screen(data, vp);
            assert!(
                cell_at(&view, &field, screen, vp).is_none(),
                "{data:?} is outside the field and must not report a cell"
            );
        }
    }

    #[test]
    fn hover_reports_the_wrapped_value_too() {
        let vp = viewport();
        let field = PhaseField::new(vec![7.0], 1, 1).expect("1 == 1 × 1");
        let view = ViewTransform::fit(1, 1, vp);
        let hovered = cell_at(&view, &field, vp.center(), vp).expect("the cell fills the view");
        assert_eq!(hovered.value, 7.0, "the raw sample is passed through");
        assert_eq!(
            hovered.wrapped,
            phase::wrap(7.0),
            "the wrapped value must agree with `phase::wrap`"
        );
    }
}

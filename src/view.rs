//! The data ↔ screen mapping behind panning and zooming.
//!
//! # Coordinate systems
//!
//! *Data space* has its origin at the top-left corner of the field, one unit per
//! cell, with `y` growing **downwards** so that rows run in the same direction
//! on screen as they do in memory. Cell `(row, col)` covers
//! `[col, col + 1] × [row, row + 1]`, so its centre is at `(col + 0.5, row + 0.5)`
//! and the whole field occupies `[0, cols] × [0, rows]`.
//!
//! *Screen space* is egui's: points (not physical pixels), `y` down.
//!
//! The mapping is a uniform scale plus a translation — never an anisotropic one,
//! so cells always stay square.

use egui::{Pos2, Rect, Vec2, pos2};

/// Smallest allowed zoom, in screen points per cell.
pub const MIN_POINTS_PER_CELL: f32 = 1.0e-3;

/// Largest allowed zoom, in screen points per cell.
pub const MAX_POINTS_PER_CELL: f32 = 1.0e4;

/// A pan/zoom state: which data point sits at the centre of the viewport, and
/// how large a cell is drawn.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ViewTransform {
    /// Data-space point shown at the viewport's centre.
    center: Pos2,

    /// Screen points per cell. Always in `[MIN_POINTS_PER_CELL, MAX_POINTS_PER_CELL]`.
    points_per_cell: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            center: Pos2::ZERO,
            points_per_cell: 1.0,
        }
    }
}

impl ViewTransform {
    /// Builds a transform, clamping the zoom into the supported range.
    ///
    /// A non-finite `points_per_cell` falls back to `1.0` rather than poisoning
    /// every later mapping with `NaN`.
    pub fn new(center: Pos2, points_per_cell: f32) -> Self {
        let points_per_cell = if points_per_cell.is_finite() {
            points_per_cell.clamp(MIN_POINTS_PER_CELL, MAX_POINTS_PER_CELL)
        } else {
            1.0
        };
        Self {
            center,
            points_per_cell,
        }
    }

    /// The zoom that makes the whole `rows × cols` field visible inside
    /// `viewport`, centred, with the image touching the viewport on whichever
    /// axis is the binding constraint ("contain", not "cover": nothing is
    /// cropped).
    pub fn fit(rows: usize, cols: usize, viewport: Rect) -> Self {
        let center = pos2(cols as f32 / 2.0, rows as f32 / 2.0);
        if rows == 0 || cols == 0 {
            return Self::new(center, 1.0);
        }
        let scale = (viewport.width() / cols as f32).min(viewport.height() / rows as f32);
        Self::new(center, scale)
    }

    /// Screen points per cell.
    pub fn points_per_cell(&self) -> f32 {
        self.points_per_cell
    }

    /// Data-space point shown at the viewport's centre.
    pub fn center(&self) -> Pos2 {
        self.center
    }

    /// Maps a data-space point to screen space.
    pub fn data_to_screen(&self, data: Pos2, viewport: Rect) -> Pos2 {
        viewport.center() + (data - self.center) * self.points_per_cell
    }

    /// Maps a screen-space point to data space. Exact inverse of
    /// [`Self::data_to_screen`].
    pub fn screen_to_data(&self, screen: Pos2, viewport: Rect) -> Pos2 {
        self.center + (screen - viewport.center()) / self.points_per_cell
    }

    /// Drags the image by a screen-space delta, so the content follows the
    /// pointer one-to-one.
    pub fn pan_by_screen_delta(&mut self, delta: Vec2) {
        self.center -= delta / self.points_per_cell;
    }

    /// Multiplies the zoom by `factor`, keeping whatever data point is under
    /// `anchor` pinned to `anchor`.
    ///
    /// This is what makes scroll-wheel zoom feel like it is zooming "into" the
    /// cursor instead of the middle of the screen.
    pub fn zoom_about(&mut self, anchor: Pos2, factor: f32, viewport: Rect) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let anchor_data = self.screen_to_data(anchor, viewport);
        *self = Self::new(self.center, self.points_per_cell * factor);
        // Re-centre so that `anchor_data` lands back on `anchor` at the new zoom.
        self.center = anchor_data - (anchor - viewport.center()) / self.points_per_cell;
    }

    /// The data-space rectangle currently covered by `viewport`.
    ///
    /// This is what the renderer interpolates across the viewport, so it is the
    /// single place the CPU and the shader have to agree on.
    pub fn visible_data_rect(&self, viewport: Rect) -> Rect {
        Rect::from_min_max(
            self.screen_to_data(viewport.min, viewport),
            self.screen_to_data(viewport.max, viewport),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::vec2;

    fn viewport() -> Rect {
        Rect::from_min_size(pos2(37.0, 11.0), vec2(640.0, 480.0))
    }

    fn assert_close(a: Pos2, b: Pos2, tolerance: f32, what: &str) {
        assert!(
            (a.x - b.x).abs() < tolerance && (a.y - b.y).abs() < tolerance,
            "{what}: {a:?} != {b:?}"
        );
    }

    #[test]
    fn screen_and_data_are_inverses() {
        let view = ViewTransform::new(pos2(12.5, -3.25), 7.5);
        let vp = viewport();
        for (x, y) in [(0.0, 0.0), (100.0, 250.0), (-40.0, 900.0), (677.0, 491.0)] {
            let screen = pos2(x, y);
            let round_tripped = view.data_to_screen(view.screen_to_data(screen, vp), vp);
            assert_close(round_tripped, screen, 1e-3, "screen -> data -> screen");
        }
    }

    #[test]
    fn the_viewport_centre_shows_the_transform_centre() {
        let view = ViewTransform::new(pos2(4.0, 9.0), 3.0);
        let vp = viewport();
        assert_close(
            view.data_to_screen(view.center(), vp),
            vp.center(),
            1e-4,
            "the centre must map to the viewport centre at any zoom",
        );
    }

    #[test]
    fn zoom_keeps_the_anchor_pinned() {
        let vp = viewport();
        let mut view = ViewTransform::new(pos2(8.0, 5.0), 4.0);
        let anchor = pos2(120.0, 400.0);
        let before = view.screen_to_data(anchor, vp);

        for factor in [1.2, 0.5, 3.0, 0.9] {
            view.zoom_about(anchor, factor, vp);
            let after = view.screen_to_data(anchor, vp);
            assert_close(
                after,
                before,
                1e-3,
                "the data point under the anchor must not move while zooming",
            );
        }
    }

    #[test]
    fn zoom_is_clamped_and_ignores_nonsense_factors() {
        let vp = viewport();
        let mut view = ViewTransform::new(pos2(0.0, 0.0), 1.0);

        for _ in 0..100 {
            view.zoom_about(vp.center(), 10.0, vp);
        }
        assert_eq!(
            view.points_per_cell(),
            MAX_POINTS_PER_CELL,
            "zooming in forever must saturate, not overflow"
        );

        for _ in 0..200 {
            view.zoom_about(vp.center(), 0.1, vp);
        }
        assert_eq!(
            view.points_per_cell(),
            MIN_POINTS_PER_CELL,
            "zooming out forever must saturate, not reach zero"
        );

        let unchanged = view;
        view.zoom_about(vp.center(), 0.0, vp);
        view.zoom_about(vp.center(), -1.0, vp);
        view.zoom_about(vp.center(), f32::NAN, vp);
        assert_eq!(
            view, unchanged,
            "a zero, negative or NaN factor must be ignored"
        );
    }

    #[test]
    fn pan_follows_the_pointer_one_to_one() {
        let vp = viewport();
        let view = ViewTransform::new(pos2(10.0, 10.0), 8.0);
        let grabbed = pos2(200.0, 300.0);
        let data_under_pointer = view.screen_to_data(grabbed, vp);

        let delta = vec2(-53.0, 17.0);
        let mut panned = view;
        panned.pan_by_screen_delta(delta);

        assert_close(
            panned.data_to_screen(data_under_pointer, vp),
            grabbed + delta,
            1e-3,
            "the grabbed cell must stay under the pointer while dragging",
        );
    }

    #[test]
    fn fit_contains_the_whole_field() {
        let vp = viewport();
        for (rows, cols) in [(10, 10), (3, 200), (200, 3), (1, 1), (0, 0)] {
            let view = ViewTransform::fit(rows, cols, vp);
            let visible = view.visible_data_rect(vp);
            let image = Rect::from_min_max(pos2(0.0, 0.0), pos2(cols as f32, rows as f32));

            assert!(
                visible.contains_rect(image.expand(-1e-3)),
                "{rows}×{cols}: fit must show the whole field, visible = {visible:?}"
            );

            if rows > 0 && cols > 0 {
                // "Contain" means the image touches the viewport on exactly the
                // binding axis; the other axis is letterboxed.
                let fills_width = (visible.width() - cols as f32).abs() < 1e-3;
                let fills_height = (visible.height() - rows as f32).abs() < 1e-3;
                assert!(
                    fills_width || fills_height,
                    "{rows}×{cols}: fit must be tight on at least one axis, visible = {visible:?}"
                );
            }
        }
    }

    #[test]
    fn fit_centres_the_field() {
        let vp = viewport();
        let view = ViewTransform::fit(7, 13, vp);
        assert_close(
            view.data_to_screen(pos2(6.5, 3.5), vp),
            vp.center(),
            1e-3,
            "the middle of the field belongs at the middle of the viewport",
        );
    }

    #[test]
    fn visible_rect_matches_the_corners() {
        let vp = viewport();
        let view = ViewTransform::new(pos2(3.0, 4.0), 12.0);
        let visible = view.visible_data_rect(vp);
        assert_close(
            visible.min,
            view.screen_to_data(vp.min, vp),
            1e-4,
            "top-left corner",
        );
        assert_close(
            visible.max,
            view.screen_to_data(vp.max, vp),
            1e-4,
            "bottom-right corner",
        );
        assert!(
            (visible.width() * view.points_per_cell() - vp.width()).abs() < 1e-3,
            "the visible width must be the viewport width divided by the zoom"
        );
    }
}

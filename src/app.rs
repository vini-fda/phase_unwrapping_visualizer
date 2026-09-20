//! The viewer application: state, layout, and the wiring between them.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::colormap::DisplayMode;
use crate::phase::PhaseField;
use crate::render::GridRenderer;
use crate::ui::{Colorbar, HoverInfo, PhaseImage, ZOOM_STEP};
use crate::view::ViewTransform;

/// Size of the demo field.
const DEMO_ROWS: usize = 192;
/// Size of the demo field.
const DEMO_COLS: usize = 256;
/// Fringes across the demo field, horizontally and vertically.
const DEMO_CYCLES: (f32, f32) = (6.0, 2.5);

/// The viewer.
///
/// Only the display choices are persisted; the field itself and everything
/// derived from it are rebuilt on startup, so a stale saved state can never
/// disagree with the data actually loaded.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct PhaseVisualizerApp {
    /// How the values are coloured.
    mode: DisplayMode,

    /// Pan/zoom. `None` until the image view has been laid out once and can
    /// fit the field to the viewport.
    #[serde(skip)]
    view: Option<ViewTransform>,

    /// The samples. Shared with the render callback, never copied.
    #[serde(skip)]
    field: Arc<PhaseField>,

    /// Bumped whenever `field` is replaced, to invalidate the GPU's copy.
    #[serde(skip)]
    generation: u64,

    /// `(min, max)` of the finite samples, cached: it only changes with the
    /// field, and scanning every sample per frame would be wasteful.
    #[serde(skip)]
    value_range: (f32, f32),

    #[serde(skip)]
    hover: Option<HoverInfo>,

    #[serde(skip)]
    colorbar: Colorbar,
}

impl Default for PhaseVisualizerApp {
    fn default() -> Self {
        let mut app = Self {
            mode: DisplayMode::default(),
            view: None,
            field: Arc::new(PhaseField::default()),
            generation: 0,
            value_range: (0.0, 1.0),
            hover: None,
            colorbar: Colorbar::default(),
        };
        app.set_field(PhaseField::linear_gradient(
            DEMO_ROWS,
            DEMO_COLS,
            DEMO_CYCLES.0,
            DEMO_CYCLES.1,
        ));
        app
    }
}

impl PhaseVisualizerApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // The custom renderer lives in the wgpu render state's callback
        // resources, so it outlives individual frames and individual callbacks.
        if let Some(render_state) = cc.wgpu_render_state.as_ref() {
            GridRenderer::install(render_state);
        } else {
            log::error!("no wgpu render state; the phase image cannot be drawn");
        }

        let mut app: Self = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();

        // `field` is `#[serde(skip)]`, so a restored app comes back with an
        // empty one. Load the data again, which also refreshes everything
        // derived from it.
        app.set_field(PhaseField::linear_gradient(
            DEMO_ROWS,
            DEMO_COLS,
            DEMO_CYCLES.0,
            DEMO_CYCLES.1,
        ));
        app
    }

    /// Replaces the displayed field and refreshes the state derived from it.
    fn set_field(&mut self, field: PhaseField) {
        self.value_range = field.finite_range().unwrap_or((0.0, 1.0));
        self.field = Arc::new(field);
        self.generation = self.generation.wrapping_add(1);
        self.hover = None;
        // The old view may be pointing somewhere that no longer exists.
        self.view = None;
    }

    /// The values at the two ends of the colormap, for the current mode.
    fn displayed_range(&self) -> (f32, f32) {
        if self.mode.is_wrapped() {
            (-PI, PI)
        } else {
            self.value_range
        }
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) -> f32 {
        let mut zoom = 1.0;

        ui.heading("Phase");
        ui.add_space(4.0);

        ui.label(format!(
            "{} × {} samples",
            self.field.rows(),
            self.field.cols()
        ));
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            for mode in [DisplayMode::Unbounded, DisplayMode::Wrapped] {
                ui.selectable_value(&mut self.mode, mode, mode.label());
            }
        });
        ui.add_space(8.0);

        self.colorbar.show(ui, self.mode, self.displayed_range());
        ui.add_space(12.0);

        ui.horizontal(|ui| {
            if ui.button("Fit to content").clicked() {
                // Cleared here, re-fitted by the image view, which is the only
                // place that knows how large the viewport is.
                self.view = None;
            }
            if ui.button("−").on_hover_text("Zoom out").clicked() {
                zoom /= ZOOM_STEP;
            }
            if ui.button("+").on_hover_text("Zoom in").clicked() {
                zoom *= ZOOM_STEP;
            }
        });

        if let Some(view) = self.view.as_ref() {
            ui.label(format!("{:.2} px/cell", view.points_per_cell()));
        }
        ui.add_space(12.0);

        ui.separator();
        ui.add_space(4.0);
        ui.label("Under cursor");
        match self.hover {
            Some(hover) => {
                ui.monospace(format!("row {}, col {}", hover.row, hover.col));
                ui.monospace(format!("value    {:+.4}", hover.value));
                ui.monospace(format!("wrapped  {:+.4}", hover.wrapped));
            }
            None => {
                ui.weak("—");
            }
        }

        zoom
    }
}

impl eframe::App for PhaseVisualizerApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top_panel").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ui.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        let zoom = egui::Panel::right("sidebar")
            .default_size(220.0)
            .show(ui, |ui| self.sidebar(ui))
            .inner;

        egui::CentralPanel::default().show(ui, |ui| {
            let output = PhaseImage {
                field: &self.field,
                generation: self.generation,
                mode: self.mode,
                value_range: self.displayed_range(),
                view: &mut self.view,
                external_zoom: zoom,
            }
            .show(ui);

            self.hover = output.hover;
            if output.response.double_clicked() {
                // Double-click is the usual "reset the view" gesture.
                self.view = None;
            }
        });
    }
}

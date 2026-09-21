//! The viewer application: state, layout, and the wiring between them.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::colormap::DisplayMode;
use crate::demo::SceneSettings;
use crate::file_dialog::{FileDialog, Payload};
use crate::inputs::{Inputs, Origin, Resolved, Slot, Supplied};
use crate::phase::PhaseField;
use crate::phase_file;
use crate::render::{GridRenderer, OverlayOptions, OverlaySource};
use crate::ui::{Colorbar, HoverInfo, PhaseImage, ZOOM_STEP, hover_at, interact, node_view};
use crate::view::ViewTransform;

/// Which field the viewer is showing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
enum Tab {
    /// The phase before wrapping: what the unwrapping is trying to recover.
    ///
    /// Demo-only. A real interferogram arrives already wrapped and there is
    /// nothing to compare against — which is the whole difficulty.
    #[default]
    Truth,
    /// The observable phase, wrapped into `(-π, π]`.
    Wrapped,
    /// A candidate unwrapping of it, with the integration path that produced it.
    Unwrapped,
}

impl Tab {
    /// The tabs in pipeline order: the phase, what is observed of it, and what
    /// is recovered from that.
    const ALL: [Self; 3] = [Self::Truth, Self::Wrapped, Self::Unwrapped];

    fn label(self) -> &'static str {
        match self {
            Self::Truth => "Truth",
            Self::Wrapped => "Wrapped",
            Self::Unwrapped => "Unwrapped",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Self::Truth => {
                "φ before wrapping — the ramp the demo generated. Not observable in practice."
            }
            Self::Wrapped => "ψ = wrap(truth), the observable phase in (-π, π]",
            Self::Unwrapped => "A candidate unwrapping of ψ, and the path that produced it",
        }
    }
}

/// How the pixel graph is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
enum Representation {
    /// Pixels as square faces, edges as the walls between them.
    #[default]
    Cell,
    /// Pixels as graph vertices, edges as arrows between them.
    Node,
}

impl Representation {
    fn label(self) -> &'static str {
        match self {
            Self::Cell => "Cell",
            Self::Node => "Node",
        }
    }
}

/// The resolved scene, plus the ranges the colormaps span.
struct Loaded {
    scene: Resolved,
    original_range: (f32, f32),
    wrapped_range: (f32, f32),
    unwrapped_range: (f32, f32),
}

impl Loaded {
    fn new(scene: Resolved) -> Self {
        Self {
            original_range: scene
                .original
                .as_ref()
                .and_then(|field| field.finite_range())
                .unwrap_or((0.0, 1.0)),
            wrapped_range: scene.wrapped.finite_range().unwrap_or((0.0, 1.0)),
            unwrapped_range: scene.unwrapped.finite_range().unwrap_or((0.0, 1.0)),
            scene,
        }
    }
}

/// The viewer.
///
/// Only the display choices and the demo settings are persisted; the scene
/// itself is rebuilt on startup, so a stale saved state can never disagree with
/// the data actually loaded.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct PhaseVisualizerApp {
    tab: Tab,
    representation: Representation,
    mode: DisplayMode,
    settings: SceneSettings,
    overlay_options: OverlayOptions,

    /// Pan/zoom, shared by both tabs and both representations: the fields have
    /// the same shape, so switching never moves the view.
    #[serde(skip)]
    view: Option<ViewTransform>,

    #[serde(skip)]
    scene: Option<Loaded>,

    #[serde(skip)]
    inputs: Inputs,

    #[serde(skip)]
    dialog: FileDialog,

    #[serde(skip)]
    hover: Option<HoverInfo>,

    #[serde(skip)]
    colorbar: Colorbar,

    /// Set when the scene could not be built, so the UI can say why instead of
    /// silently showing nothing.
    #[serde(skip)]
    error: Option<String>,
}

impl Default for PhaseVisualizerApp {
    fn default() -> Self {
        let mut app = Self {
            tab: Tab::default(),
            representation: Representation::default(),
            mode: DisplayMode::default(),
            settings: SceneSettings::default(),
            overlay_options: OverlayOptions::default(),
            view: None,
            scene: None,
            inputs: Inputs::default(),
            dialog: FileDialog::default(),
            hover: None,
            colorbar: Colorbar::default(),
            error: None,
        };
        app.use_generated_data();
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

        // The scene is `#[serde(skip)]`, so a restored app comes back without
        // one; build it from the settings that *were* restored.
        app.use_generated_data();
        app
    }

    /// Rebuilds the displayed scene from whatever is currently supplied.
    ///
    /// Called after every change to the inputs, so what is on screen always
    /// matches what the sidebar says it came from.
    fn resolve(&mut self) {
        self.hover = None;
        self.view = None;
        match self.inputs.resolve() {
            Ok(scene) => {
                self.scene = Some(Loaded::new(scene));
                self.error = None;
            }
            Err(error) => {
                self.scene = None;
                self.error = Some(error.to_string());
            }
        }
    }

    /// Discards every supplied file and goes back to the generator.
    fn use_generated_data(&mut self) {
        self.inputs = Inputs::generated(self.settings);
        self.resolve();
    }

    /// Regenerates the synthetic original, keeping any other supplied inputs.
    fn regenerate(&mut self) {
        self.inputs.original = Inputs::generated(self.settings).original;
        self.resolve();
    }

    /// Files the contents of an opened file into the slot it was opened for.
    fn accept(&mut self, opened: crate::file_dialog::Opened) {
        let origin = Origin::File(opened.name.clone());
        let payload = match opened.result {
            Ok(payload) => payload,
            Err(error) => {
                self.error = Some(format!("{}: {error}", opened.name));
                return;
            }
        };

        // Keep the previous inputs in hand: if the new file does not fit the
        // others, the viewer should say so and carry on showing what it had.
        let previous = self.inputs.clone();

        match (opened.slot, payload) {
            (Slot::Path, Payload::Path(path)) => {
                self.inputs.path = Some(Supplied {
                    value: std::sync::Arc::new(path),
                    origin,
                });
            }
            (slot, Payload::Phase(field)) => {
                let supplied = Some(Supplied {
                    value: std::sync::Arc::new(field),
                    origin,
                });
                match slot {
                    Slot::Original => self.inputs.original = supplied,
                    Slot::Wrapped => self.inputs.wrapped = supplied,
                    Slot::Unwrapped => self.inputs.unwrapped = supplied,
                    Slot::Path => unreachable!("the path slot is handled above"),
                }
            }
            (slot, _) => {
                self.error = Some(format!(
                    "{}: that file does not hold {}",
                    opened.name,
                    slot.label().to_lowercase()
                ));
                return;
            }
        }

        self.resolve();
        if self.error.is_some() {
            // The combination was rejected; put back what worked.
            self.inputs = previous;
            let message = self.error.clone();
            self.resolve();
            self.error = message.map(|message| format!("{}: {message}", opened.name));
        }
    }

    /// Collects a file the picker or a drag-and-drop has finished reading.
    fn poll_incoming_file(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // A dropped file fills whichever slot its extension names.
            let dropped = ctx.input(|input| input.raw.dropped_files.clone());
            for file in dropped {
                let path = file.path().to_path_buf();
                let name = path
                    .file_name()
                    .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
                let slot = if path.extension().is_some_and(|extension| {
                    extension.eq_ignore_ascii_case(crate::phase_file::PATH_EXTENSION)
                }) {
                    Slot::Path
                } else {
                    Slot::Original
                };
                match file.bytes() {
                    Ok(bytes) => self.dialog.accept(slot, name, &bytes),
                    Err(error) => self.error = Some(format!("{name}: {error}")),
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;

        if let Some(opened) = self.dialog.take() {
            self.accept(opened);
        }
    }

    /// Loads the example that ships with the crate, as the original phase.
    fn load_example(&self) {
        self.dialog.accept(
            Slot::Original,
            phase_file::EXAMPLE_NAME.to_owned(),
            phase_file::EXAMPLE,
        );
    }

    /// The field on show, and the values at the two ends of its colormap.    /// The field on show, and the values at the two ends of its colormap.
    fn displayed(&self) -> Option<(&Arc<PhaseField>, (f32, f32))> {
        let scene = self.scene.as_ref()?;
        let (field, range) = match self.tab {
            Tab::Truth => (scene.scene.original.as_ref()?, scene.original_range),
            Tab::Wrapped => (&scene.scene.wrapped, scene.wrapped_range),
            Tab::Unwrapped => (&scene.scene.unwrapped, scene.unwrapped_range),
        };
        // Wrapped mode always spans a full turn, whatever the data's own range.
        let range = if self.mode.is_wrapped() {
            (-PI, PI)
        } else {
            range
        };
        Some((field, range))
    }

    /// The overlay to draw, if the unwrapped tab is showing the cell view.
    fn overlay(&self) -> Option<OverlaySource> {
        let scene = self.scene.as_ref()?;
        (self.tab == Tab::Unwrapped && self.representation == Representation::Cell).then(|| {
            OverlaySource {
                unwrapping: Arc::clone(&scene.scene.unwrapping),
                options: self.overlay_options,
            }
        })
    }

    fn tab_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for tab in Tab::ALL {
                ui.selectable_value(&mut self.tab, tab, tab.label())
                    .on_hover_text(tab.tooltip());
            }
            if self.tab == Tab::Unwrapped {
                ui.separator();
                ui.label("Representation:");
                for representation in [Representation::Cell, Representation::Node] {
                    ui.selectable_value(
                        &mut self.representation,
                        representation,
                        representation.label(),
                    );
                }
            }
        });
    }

    /// Draws the sidebar and returns any zoom its buttons requested.
    fn sidebar(&mut self, ui: &mut egui::Ui) -> f32 {
        let mut zoom = 1.0;

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading(match self.tab {
                Tab::Truth => "Original phase",
                Tab::Wrapped => "Wrapped phase",
                Tab::Unwrapped => "Candidate unwrapping",
            });
            ui.add_space(2.0);
            ui.label(format!(
                "{} × {} samples",
                self.settings.rows, self.settings.cols
            ));
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                for mode in [DisplayMode::Unbounded, DisplayMode::Wrapped] {
                    ui.selectable_value(&mut self.mode, mode, mode.label());
                }
            });
            ui.add_space(8.0);

            let range = self.displayed().map_or((0.0, 1.0), |(_, range)| range);
            self.colorbar.show(ui, self.mode, range);
            ui.add_space(12.0);

            ui.horizontal(|ui| {
                if ui.button("Fit to content").clicked() {
                    // Cleared here, re-fitted by the view, which is the only
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

            if self.tab == Tab::Unwrapped {
                ui.add_space(12.0);
                ui.separator();
                self.integration_section(ui);
            }

            ui.add_space(12.0);
            ui.separator();
            self.sources_section(ui);

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
        });

        zoom
    }

    /// Counts and the legend for the integration overlay.
    fn integration_section(&mut self, ui: &mut egui::Ui) {
        let Some(scene) = self.scene.as_ref() else {
            return;
        };
        let stats = scene.scene.unwrapping.stats();
        let (rows, cols) = (self.settings.rows, self.settings.cols);

        ui.add_space(4.0);
        ui.label("Integration path");
        if scene.scene.unwrapping.has_path() {
            ui.monospace(format!("tree edges   {:>7}", stats.tree_edges));
            ui.monospace(format!("cut edges    {:>7}", stats.cut_edges));
        } else {
            ui.colored_label(ui.visuals().warn_fg_color, "no path provided")
                .on_hover_text(
                    "The residues and the disagreeing edges below are still exact — they \
                 need only ψ and φ. What is missing is which edges the walk used, so \
                 the walls cannot separate cut edges from the path and the node view \
                 cannot draw arrows.",
                );
        }
        ui.monospace(format!(
            "residues   +{:>3} −{:<3}",
            stats.positive_residues, stats.negative_residues
        ));

        ui.add_space(6.0);
        ui.monospace(format!("disagreeing  {:>7}", stats.inconsistent_edges))
            .on_hover_text("Edges where the integration delta is not the wrapped delta");

        // The bounds quoted in the theory, shown against the measured value.
        let lower = stats.positive_residues.max(stats.negative_residues);
        let upper = rows.saturating_sub(1) * cols.saturating_sub(1);
        ui.weak(format!("bounds: {lower} ≤ L ≤ {upper}"))
            .on_hover_text("At least max(N+, N−), at most (m−1)(n−1)");

        if self.representation == Representation::Cell {
            ui.add_space(10.0);
            ui.label("Overlay");

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.overlay_options.highlight_edges, "");
                swatch(ui, HIGHLIGHT_SWATCH, "delta ≠ wrapped delta");
            })
            .response
            .on_hover_text(
                "Colour the edges the integration moved by something other than the \
                 wrapped delta. Turned off, they revert to the wall their role in the \
                 integration path calls for.",
            );

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.overlay_options.show_residues, "");
                swatch(ui, RESIDUE_POSITIVE_SWATCH, "residue +1");
                swatch(ui, RESIDUE_NEGATIVE_SWATCH, "residue −1");
            })
            .response
            .on_hover_text("Draw the charges at the inner corners");

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.overlay_options.show_tree_edges, "");
                swatch(ui, TREE_WALL_SWATCH, "integration path (dashed)");
            })
            .response
            .on_hover_text(
                "Draw the walls the integration walks through. Off, only the cut \
                 edges remain — the spanning tree of the dual graph on its own.",
            );

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.overlay_options.green_cut_edges, "");
                let color = if self.overlay_options.green_cut_edges {
                    CUT_WALL_GREEN_SWATCH
                } else {
                    CUT_WALL_SWATCH
                };
                swatch(ui, color, "green cut edges (solid)");
            })
            .response
            .on_hover_text(
                "Pick the cut edges out in green. The image border stays neutral: \
                 it bounds the outer face O and is not an edge of G.",
            );
        }
    }

    /// What the viewer is currently working from, slot by slot.
    ///
    /// Every row says where its data came from, or — when nothing was supplied
    /// — what is being done instead and what that costs, so the display is
    /// never quietly standing on something the user did not choose.
    fn sources_section(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.label("Data sources")
            .on_hover_text("Where each input came from. Only the wrapped phase is required.");
        ui.add_space(2.0);

        let mut pick = None;
        let mut clear = None;

        for slot in Slot::ALL {
            let state = self.inputs.state(slot);
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.strong(slot.label()).on_hover_text(slot.tooltip());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if state.is_supplied()
                        && ui
                            .small_button("✕")
                            .on_hover_text("Forget this file")
                            .clicked()
                    {
                        clear = Some(slot);
                    }
                    if ui
                        .small_button("Open…")
                        .on_hover_text(format!("Open a .{} file", slot.extension()))
                        .clicked()
                    {
                        pick = Some(slot);
                    }
                });
            });

            let label = state.label();
            match &state {
                crate::inputs::SlotState::Supplied(_) => {
                    ui.monospace(&label).on_hover_text(&label);
                }
                crate::inputs::SlotState::Derived(_) => {
                    ui.weak(&label);
                }
                crate::inputs::SlotState::Missing(_) => {
                    ui.colored_label(ui.visuals().warn_fg_color, &label);
                }
            }
        }

        if let Some(slot) = clear {
            self.inputs.clear(slot);
            self.resolve();
        }
        if let Some(slot) = pick {
            self.dialog.pick(slot);
        }

        if let Some(error) = self.error.as_ref() {
            ui.add_space(6.0);
            ui.colored_label(ui.visuals().error_fg_color, error);
        }

        // The generator only has anything to say while it is the one supplying
        // the original phase.
        if self.inputs.state(Slot::Original)
            == crate::inputs::SlotState::Supplied(Origin::Generated)
        {
            ui.add_space(10.0);
            self.generator_section(ui);
        }
    }

    /// The knobs behind the synthetic original phase.
    fn generator_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Generator")
            .on_hover_text("Settings for the synthetic original phase");

        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.noise, 0.0..=1.5)
                    .text("noise")
                    .fixed_decimals(2),
            )
            .on_hover_text(
                "Standard deviation of the additive noise, in radians. Zero unwraps perfectly.",
            )
            .changed();

        ui.horizontal(|ui| {
            if ui.button("New seed").clicked() {
                self.settings.seed = self
                    .settings
                    .seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                changed = true;
            }
            ui.weak(format!("seed {}", self.settings.seed));
        });

        if changed {
            self.regenerate();
        }
    }

    /// Draws whichever representation is selected into the central panel.    /// Draws whichever representation is selected into the central panel.
    fn central(&mut self, ui: &mut egui::Ui, zoom: f32) {
        let Some(scene) = self.scene.as_ref() else {
            let message = self.error.clone().unwrap_or_else(|| {
                "No data. Open an original or wrapped phase from the File menu.".to_owned()
            });
            ui.centered_and_justified(|ui| {
                ui.colored_label(ui.visuals().warn_fg_color, message);
            });
            return;
        };

        if self.tab == Tab::Unwrapped && self.representation == Representation::Node {
            let unwrapping = Arc::clone(&scene.scene.unwrapping);
            let range = scene.unwrapped_range;
            let (rect, response) = interact(
                ui,
                unwrapping.rows(),
                unwrapping.cols(),
                &mut self.view,
                zoom,
            );
            let Some(view) = self.view else {
                return;
            };

            self.hover = hover_at(&view, unwrapping.unwrapped(), &response, rect);

            let drawn = node_view::show(ui, rect, &view, &unwrapping, self.mode.colormap(), range);
            if !drawn && let Some(hint) = node_view::zoom_hint(view.points_per_cell()) {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    hint,
                    egui::FontId::proportional(14.0),
                    ui.visuals().weak_text_color(),
                );
            } else if drawn && let Some(hint) = node_view::path_hint(&unwrapping) {
                ui.painter().text(
                    rect.center_bottom() - egui::vec2(0.0, 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    hint,
                    egui::FontId::proportional(12.0),
                    ui.visuals().warn_fg_color,
                );
            }
            return;
        }

        let Some((field, value_range)) = self.displayed() else {
            // The only way to get here is the truth tab with no original phase.
            ui.centered_and_justified(|ui| {
                ui.weak(
                    "No original phase provided.\n\n\
                     Open one with File → Open original phase…, or switch to the \
                     Wrapped or Unwrapped tab.",
                );
            });
            return;
        };
        let field = Arc::clone(field);
        let overlay = self.overlay();

        let output = PhaseImage {
            field: &field,
            mode: self.mode,
            value_range,
            view: &mut self.view,
            external_zoom: zoom,
            overlay,
        }
        .show(ui);

        self.hover = output.hover;
        if output.response.double_clicked() {
            // Double-click is the usual "reset the view" gesture.
            self.view = None;
        }
    }
}

/// Legend colours, matching the constants in `grid.wgsl`.
const HIGHLIGHT_SWATCH: egui::Color32 = egui::Color32::from_rgb(232, 23, 135);
const RESIDUE_POSITIVE_SWATCH: egui::Color32 = egui::Color32::from_rgb(245, 130, 31);
const RESIDUE_NEGATIVE_SWATCH: egui::Color32 = egui::Color32::from_rgb(92, 199, 232);
const CUT_WALL_SWATCH: egui::Color32 = egui::Color32::from_rgb(10, 10, 13);
const CUT_WALL_GREEN_SWATCH: egui::Color32 = egui::Color32::from_rgb(33, 176, 77);
const TREE_WALL_SWATCH: egui::Color32 = egui::Color32::from_rgb(120, 120, 128);

/// A colour chip with a caption, for the overlay legend.
fn swatch(ui: &mut egui::Ui, color: egui::Color32, label: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, color);
        ui.weak(label);
    });
}

impl eframe::App for PhaseVisualizerApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_incoming_file(ui.ctx());

        egui::Panel::top("top_panel").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    for slot in Slot::ALL {
                        if ui
                            .button(format!("Open {}…", slot.label().to_lowercase()))
                            .on_hover_text(slot.tooltip())
                            .clicked()
                        {
                            self.dialog.pick(slot);
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui
                        .button("Open example as original phase")
                        .on_hover_text(phase_file::EXAMPLE_NAME)
                        .clicked()
                    {
                        self.load_example();
                        ui.close();
                    }
                    if ui
                        .button("Reset to generated data")
                        .on_hover_text(
                            "Discard every opened file and go back to the synthetic scene",
                        )
                        .clicked()
                    {
                        self.use_generated_data();
                        ui.close();
                    }

                    // NOTE: no File->Quit on web pages!
                    if !cfg!(target_arch = "wasm32") {
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            ui.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                });
                ui.add_space(16.0);

                egui::widgets::global_theme_preference_buttons(ui);
            });
            ui.add_space(2.0);
            self.tab_bar(ui);
            ui.add_space(2.0);
        });

        let zoom = egui::Panel::right("sidebar")
            .default_size(250.0)
            .show(ui, |ui| self.sidebar(ui))
            .inner;

        egui::CentralPanel::default().show(ui, |ui| self.central(ui, zoom));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Building the scene needs no GPU, so the display wiring is testable even
    /// though the rendering is not.
    fn app() -> PhaseVisualizerApp {
        let app = PhaseVisualizerApp::default();
        assert!(
            app.scene.is_some(),
            "the demo scene must build: {:?}",
            app.error
        );
        app
    }

    fn field_of(app: &PhaseVisualizerApp) -> Arc<PhaseField> {
        Arc::clone(app.displayed().expect("a scene is loaded").0)
    }

    /// The renderer decides whether to re-upload by comparing the `Arc` it was
    /// handed against the one already on the GPU. If both tabs handed it the
    /// same allocation, a tab switch would be invisible to it and the previous
    /// tab's samples would stay on screen.
    #[test]
    fn every_tab_hands_over_a_distinct_field() {
        let mut app = app();

        let fields: Vec<(Tab, Arc<PhaseField>)> = Tab::ALL
            .into_iter()
            .map(|tab| {
                app.tab = tab;
                (tab, field_of(&app))
            })
            .collect();

        for (i, (tab, field)) in fields.iter().enumerate() {
            for (other_tab, other) in &fields[i + 1..] {
                assert!(
                    !Arc::ptr_eq(field, other),
                    "{tab:?} and {other_tab:?} must own separate fields, \
                     or the GPU cache cannot tell them apart"
                );
                assert_ne!(
                    field.as_slice(),
                    other.as_slice(),
                    "{tab:?} and {other_tab:?} must not show the same samples"
                );
            }
        }
    }

    /// The three tabs are the pipeline in order, and the middle one is defined
    /// as the wrapping of the first. Pinning that keeps the demo honest: if
    /// they ever drifted apart, the wrapped tab would stop being an observation
    /// of anything.
    #[test]
    fn the_wrapped_tab_is_the_truth_tab_wrapped() {
        let mut app = app();

        app.tab = Tab::Truth;
        let truth = field_of(&app);
        app.tab = Tab::Wrapped;
        let wrapped = field_of(&app);

        for (index, (&original, &observed)) in
            truth.as_slice().iter().zip(wrapped.as_slice()).enumerate()
        {
            assert_eq!(
                observed,
                crate::phase::wrap(original),
                "sample {index}: ψ must be wrap(truth)"
            );
        }
    }

    /// The truth tab shows the phase before wrapping, so it spans many turns.
    #[test]
    fn the_truth_tab_shows_the_unwrapped_ramp() {
        let mut app = app();
        app.tab = Tab::Truth;
        app.mode = DisplayMode::Unbounded;

        let (field, range) = app.displayed().expect("a scene is loaded");
        assert!(
            field.as_slice().iter().any(|value| value.abs() > PI),
            "the original phase should leave (-π, π]"
        );
        assert!(
            range.1 - range.0 > 2.0 * PI,
            "and its grayscale range should be wider than one turn, got {range:?}"
        );
    }

    #[test]
    fn the_truth_tab_comes_first() {
        assert_eq!(Tab::ALL[0], Tab::Truth, "the pipeline starts at the phase");
        assert_eq!(
            Tab::default(),
            Tab::Truth,
            "and that is where a fresh viewer lands"
        );
    }

    /// The wrapped tab shows the observable, so every sample it displays is
    /// already inside the interval — in grayscale as much as in cubehelix.
    /// Showing the un-wrapped ground truth here made it look like the
    /// integrated field, which is the whole reason this is pinned.
    #[test]
    fn the_wrapped_tab_shows_samples_inside_the_interval() {
        let mut app = app();
        app.tab = Tab::Wrapped;

        for mode in [DisplayMode::Unbounded, DisplayMode::Wrapped] {
            app.mode = mode;
            let (field, range) = app.displayed().expect("a scene is loaded");
            assert!(
                field
                    .as_slice()
                    .iter()
                    .all(|value| -PI < *value && *value <= PI),
                "{mode:?}: the wrapped tab must show ψ, not the phase behind it"
            );
            assert!(
                range.0 >= -PI - 1e-3 && range.1 <= PI + 1e-3,
                "{mode:?}: the colormap must span the wrapped range, got {range:?}"
            );
        }
    }

    /// The candidate unwrapping is only interesting because it leaves the
    /// interval; if it did not, nothing would have been unwrapped.
    #[test]
    fn the_unwrapped_tab_shows_a_field_that_leaves_the_interval() {
        let mut app = app();
        app.tab = Tab::Unwrapped;
        app.mode = DisplayMode::Unbounded;

        let (field, range) = app.displayed().expect("a scene is loaded");
        assert!(
            field.as_slice().iter().any(|value| value.abs() > PI),
            "the integrated field should span many turns"
        );
        assert!(
            range.1 - range.0 > 2.0 * PI,
            "and its grayscale range should be wider than one turn, got {range:?}"
        );
    }

    /// Changing the demo settings must replace the fields, not edit them in
    /// place, for the same reason the tabs must differ.
    #[test]
    fn rebuilding_the_scene_replaces_the_fields() {
        let mut app = app();
        let before = field_of(&app);

        app.settings.noise += 0.3;
        app.use_generated_data();
        let after = field_of(&app);

        assert!(
            !Arc::ptr_eq(&before, &after),
            "a rebuilt scene must hand the renderer a fresh allocation"
        );
    }

    /// Opening an original phase must rebuild every tab, not just the one on
    /// show: the wrapped and unwrapped fields are both derived from it.
    #[test]
    fn opening_an_original_phase_rebuilds_every_tab() {
        let mut app = app();
        app.tab = Tab::Truth;
        let before = field_of(&app);

        app.load_example();
        app.poll_incoming_file(&egui::Context::default());

        assert!(app.error.is_none(), "the shipped example must load cleanly");
        assert_eq!(
            app.inputs.state(Slot::Original),
            crate::inputs::SlotState::Supplied(Origin::File(
                crate::phase_file::EXAMPLE_NAME.to_owned()
            )),
            "the sidebar must be able to name the file"
        );

        let after = field_of(&app);
        assert!(
            !Arc::ptr_eq(&before, &after),
            "the renderer must be handed a fresh allocation"
        );
        for tab in Tab::ALL {
            app.tab = tab;
            let field = field_of(&app);
            assert_eq!(
                (field.rows(), field.cols()),
                (8, 8),
                "{tab:?} must follow the loaded field's shape"
            );
        }
        assert!(
            app.view.is_none(),
            "a differently sized field must be re-fitted"
        );
    }

    /// The whole point of the wrapped slot: a supplied ψ is what the unwrapping
    /// was measured against, so it must never be silently replaced by one
    /// derived from the original.
    #[test]
    fn a_supplied_wrapped_phase_wins_over_a_derived_one() {
        let mut app = app();

        // A wrapped field that is deliberately *not* wrap(original).
        let psi = PhaseField::new(
            vec![0.25; app.settings.rows * app.settings.cols],
            app.settings.rows,
            app.settings.cols,
        )
        .expect("the right number of samples");
        app.inputs.wrapped = Some(Supplied {
            value: Arc::new(psi),
            origin: Origin::File("psi.phase".to_owned()),
        });
        app.resolve();

        assert!(app.error.is_none(), "the shapes agree, so this resolves");
        app.tab = Tab::Wrapped;
        let shown = field_of(&app);
        assert!(
            shown.as_slice().iter().all(|value| *value == 0.25),
            "the wrapped tab must show the ψ that was supplied, not wrap(original)"
        );
    }

    #[test]
    fn without_an_original_the_truth_tab_has_nothing_to_show() {
        let mut app = app();
        app.inputs.clear(Slot::Original);
        // Something still has to provide ψ, or there is no scene at all.
        app.inputs.wrapped = Some(Supplied {
            value: Arc::new(crate::demo::wrap_field(&crate::demo::noisy_ramp(
                8, 8, 2.0, 1.0, 0.5, 3,
            ))),
            origin: Origin::File("psi.phase".to_owned()),
        });
        app.resolve();

        assert!(app.error.is_none(), "a wrapped phase alone is enough");
        app.tab = Tab::Truth;
        assert!(
            app.displayed().is_none(),
            "the truth tab must report emptiness rather than invent a field"
        );
        app.tab = Tab::Wrapped;
        assert!(
            app.displayed().is_some(),
            "while the wrapped tab still works"
        );
        assert_eq!(
            app.inputs.state(Slot::Original),
            crate::inputs::SlotState::Missing("not provided — the Truth tab is empty".to_owned()),
            "and the sidebar must say so"
        );
    }

    #[test]
    fn with_nothing_supplied_there_is_no_scene_and_the_reason_is_given() {
        let mut app = app();
        app.inputs = crate::inputs::Inputs::default();
        app.resolve();

        assert!(app.scene.is_none(), "there is no wrapped phase to show");
        let error = app.error.as_ref().expect("the reason must be reported");
        assert!(
            error.contains("original") && error.contains("wrapped"),
            "the message must say what would fix it, got {error:?}"
        );
    }

    /// Supplying a candidate without a path leaves the residues and the
    /// disagreeing edges intact, but nothing can be said about the walk.
    #[test]
    fn an_unwrapped_phase_without_a_path_loses_only_the_path() {
        let mut app = app();
        let candidate = app
            .scene
            .as_ref()
            .expect("a scene is loaded")
            .scene
            .unwrapped
            .as_ref()
            .clone();

        app.inputs.unwrapped = Some(Supplied {
            value: Arc::new(candidate),
            origin: Origin::File("phi.phase".to_owned()),
        });
        app.resolve();

        let scene = app.scene.as_ref().expect("this resolves");
        assert!(
            !scene.scene.unwrapping.has_path(),
            "no path was supplied with the candidate"
        );
        assert_eq!(
            scene.scene.unwrapping.stats().tree_edges,
            0,
            "so there is no walk to count"
        );
        assert_eq!(
            app.inputs.state(Slot::Path),
            crate::inputs::SlotState::Missing(
                "not provided — no walls or arrows for the path".to_owned()
            ),
            "and the sidebar must say what that costs"
        );
    }

    /// Clearing the candidate must clear the walk with it: a path describes how
    /// one particular candidate was built, and means nothing without it.
    #[test]
    fn clearing_the_unwrapped_phase_clears_its_path() {
        let mut app = app();
        let (rows, cols) = (app.settings.rows, app.settings.cols);
        app.inputs.unwrapped = Some(Supplied {
            value: Arc::new(PhaseField::linear_gradient(rows, cols, 1.0, 1.0)),
            origin: Origin::File("phi.phase".to_owned()),
        });
        app.inputs.path = Some(Supplied {
            value: Arc::new(crate::graph::IntegrationPath::comb(rows, cols)),
            origin: Origin::File("phi.path".to_owned()),
        });

        app.inputs.clear(Slot::Unwrapped);
        assert!(
            app.inputs.path.is_none(),
            "a path without the candidate it produced describes nothing"
        );
    }

    #[test]
    fn a_file_of_the_wrong_shape_is_refused_and_the_view_survives() {
        let mut app = app();
        let before = field_of(&app);

        // The example is 8 × 8, the generated scene is not.
        let example =
            crate::phase_file::decode(crate::phase_file::EXAMPLE).expect("the example decodes");
        app.inputs.unwrapped = Some(Supplied {
            value: Arc::new(example),
            origin: Origin::File("mismatched.phase".to_owned()),
        });
        app.resolve();

        assert!(app.scene.is_none(), "the combination cannot be resolved");
        let error = app.error.as_ref().expect("and the reason is reported");
        assert!(
            error.contains('×'),
            "the message should name both shapes, got {error:?}"
        );

        // Put it back the way it was, as `accept` does on a rejected file.
        app.inputs.clear(Slot::Unwrapped);
        app.resolve();
        assert!(
            Arc::ptr_eq(&before, &field_of(&app)),
            "clearing the offending file restores what was on screen"
        );
    }

    #[test]
    fn resetting_goes_back_to_the_generator() {
        let mut app = app();
        app.load_example();
        app.poll_incoming_file(&egui::Context::default());

        app.use_generated_data();

        assert_eq!(
            app.inputs.state(Slot::Original),
            crate::inputs::SlotState::Supplied(Origin::Generated),
            "the original comes from the generator again"
        );
        let field = field_of(&app);
        assert_eq!(
            (field.rows(), field.cols()),
            (app.settings.rows, app.settings.cols),
            "and the generator's own dimensions come back"
        );
    }

    /// Every slot must describe itself, whatever state it is in — a blank line
    /// in the sidebar would be worse than no sidebar.
    #[test]
    fn every_slot_always_has_something_to_say() {
        let mut app = app();
        for inputs in [
            crate::inputs::Inputs::default(),
            crate::inputs::Inputs::generated(app.settings),
        ] {
            app.inputs = inputs;
            for slot in Slot::ALL {
                let state = app.inputs.state(slot);
                assert!(
                    !state.label().is_empty(),
                    "{slot:?} must say where its data comes from"
                );
                assert!(
                    !slot.tooltip().is_empty(),
                    "{slot:?} must explain what it is for"
                );
            }
        }
    }

    /// The overlay belongs to the unwrapped cell view alone: the node view
    /// draws its own, and the wrapped tab has no integration path to show.
    #[test]
    fn the_overlay_is_offered_only_where_it_is_drawn() {
        let mut app = app();

        for (tab, representation, expected) in [
            (Tab::Truth, Representation::Cell, false),
            (Tab::Truth, Representation::Node, false),
            (Tab::Wrapped, Representation::Cell, false),
            (Tab::Wrapped, Representation::Node, false),
            (Tab::Unwrapped, Representation::Cell, true),
            (Tab::Unwrapped, Representation::Node, false),
        ] {
            app.tab = tab;
            app.representation = representation;
            assert_eq!(
                app.overlay().is_some(),
                expected,
                "{tab:?} + {representation:?} should {} an overlay",
                if expected { "offer" } else { "not offer" }
            );
        }
    }
}

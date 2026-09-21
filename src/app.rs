//! The viewer application: state, layout, and the wiring between them.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::colormap::DisplayMode;
use crate::demo::{Scene, SceneSettings};
use crate::file_dialog::FileDialog;
use crate::graph::Unwrapping;
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

/// Where the field on show came from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Source {
    /// Generated from [`SceneSettings`].
    #[default]
    Demo,
    /// Read out of a `.phase` file.
    File {
        /// What to call it in the UI.
        name: String,
    },
}

/// The scene, plus everything derived from it that the viewer needs per frame.
///
/// The two tabs hold two *separate* `Arc`s, which is what the renderer keys its
/// uploads on: handing it a different one is what makes it refresh the GPU.
struct Loaded {
    /// The phase before wrapping, shown by the truth tab.
    truth: Arc<PhaseField>,
    /// The observable phase ψ, already inside `(-π, π]`.
    wrapped: Arc<PhaseField>,
    /// The candidate unwrapping φ, shared with the render callback.
    unwrapped: Arc<PhaseField>,
    /// The analysis behind the overlay.
    unwrapping: Arc<Unwrapping>,
    truth_range: (f32, f32),
    wrapped_range: (f32, f32),
    unwrapped_range: (f32, f32),
}

impl Loaded {
    fn new(scene: Scene) -> Self {
        let unwrapped = Arc::new(scene.unwrapping.unwrapped().clone());
        Self {
            truth_range: scene.truth.finite_range().unwrap_or((0.0, 1.0)),
            wrapped_range: scene.wrapped.finite_range().unwrap_or((0.0, 1.0)),
            unwrapped_range: unwrapped.finite_range().unwrap_or((0.0, 1.0)),
            truth: Arc::new(scene.truth),
            wrapped: Arc::new(scene.wrapped),
            unwrapped,
            unwrapping: Arc::new(scene.unwrapping),
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
    source: Source,

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
            source: Source::default(),
            dialog: FileDialog::default(),
            hover: None,
            colorbar: Colorbar::default(),
            error: None,
        };
        app.rebuild_scene();
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
        app.rebuild_scene();
        app
    }

    /// Regenerates the demo scene from the current settings.
    fn rebuild_scene(&mut self) {
        self.hover = None;
        match Scene::new(self.settings) {
            Ok(scene) => {
                self.scene = Some(Loaded::new(scene));
                self.error = None;
            }
            Err(error) => {
                self.scene = None;
                self.error = Some(error.to_string());
            }
        }
        // The old view may be pointing somewhere that no longer exists.
        self.view = None;
    }

    /// Replaces the scene with one built around a field read from a file.
    fn load_field(&mut self, name: String, field: PhaseField) {
        self.hover = None;
        self.view = None;
        match Scene::from_truth(field) {
            Ok(scene) => {
                self.scene = Some(Loaded::new(scene));
                self.source = Source::File { name };
                self.error = None;
            }
            Err(error) => {
                // Keep whatever was on screen; only report why the new file
                // could not replace it.
                self.error = Some(format!("{name}: {error}"));
            }
        }
    }

    /// Collects a file the picker or a drag-and-drop has finished reading.
    fn poll_incoming_file(&mut self, ctx: &egui::Context) {
        // A dropped file takes the same route as the picker, so both end up
        // reported the same way.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dropped = ctx.input(|input| input.raw.dropped_files.clone());
            for file in dropped {
                let name = file
                    .path()
                    .file_name()
                    .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
                match file.bytes() {
                    Ok(bytes) => self.dialog.accept(name, &bytes),
                    Err(error) => self.error = Some(format!("{name}: {error}")),
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;

        if let Some(opened) = self.dialog.take() {
            match opened.result {
                Ok(field) => self.load_field(opened.name, field),
                Err(error) => self.error = Some(format!("{}: {error}", opened.name)),
            }
        }
    }

    /// Loads the example that ships with the crate.
    fn load_example(&self) {
        self.dialog
            .accept(phase_file::EXAMPLE_NAME.to_owned(), phase_file::EXAMPLE);
    }

    /// Goes back to the generated scene.
    fn use_demo_data(&mut self) {
        self.source = Source::Demo;
        self.rebuild_scene();
    }

    /// The field on show, and the values at the two ends of its colormap.
    fn displayed(&self) -> Option<(&Arc<PhaseField>, (f32, f32))> {
        let scene = self.scene.as_ref()?;
        let (field, range) = match self.tab {
            Tab::Truth => (&scene.truth, scene.truth_range),
            Tab::Wrapped => (&scene.wrapped, scene.wrapped_range),
            Tab::Unwrapped => (&scene.unwrapped, scene.unwrapped_range),
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
                unwrapping: Arc::clone(&scene.unwrapping),
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
            self.demo_section(ui);

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
        let stats = scene.unwrapping.stats();
        let (rows, cols) = (self.settings.rows, self.settings.cols);

        ui.add_space(4.0);
        ui.label("Integration path");
        ui.monospace(format!("tree edges   {:>7}", stats.tree_edges));
        ui.monospace(format!("cut edges    {:>7}", stats.cut_edges));
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

    /// The knobs behind the synthetic scene.
    fn demo_section(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);

        if let Source::File { name } = self.source.clone() {
            ui.label("Source");
            ui.monospace(&name).on_hover_text(&name);
            if ui.button("Use generated data").clicked() {
                self.use_demo_data();
            }
            if let Some(error) = self.error.as_ref() {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
            return;
        }

        ui.label("Generated data");

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
            self.rebuild_scene();
        }

        if let Some(error) = self.error.as_ref() {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
    }

    /// Draws whichever representation is selected into the central panel.
    fn central(&mut self, ui: &mut egui::Ui, zoom: f32) {
        let Some(scene) = self.scene.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.weak("No scene to show");
            });
            return;
        };

        if self.tab == Tab::Unwrapped && self.representation == Representation::Node {
            let unwrapping = Arc::clone(&scene.unwrapping);
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

            if !node_view::show(ui, rect, &view, &unwrapping, self.mode.colormap(), range)
                && let Some(hint) = node_view::zoom_hint(view.points_per_cell())
            {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    hint,
                    egui::FontId::proportional(14.0),
                    ui.visuals().weak_text_color(),
                );
            }
            return;
        }

        let Some((field, value_range)) = self.displayed() else {
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
                    if ui
                        .button("Open phase data…")
                        .on_hover_text("Open a .phase file as the original phase")
                        .clicked()
                    {
                        self.dialog.pick();
                        ui.close();
                    }
                    if ui
                        .button("Open example")
                        .on_hover_text(phase_file::EXAMPLE_NAME)
                        .clicked()
                    {
                        self.load_example();
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.source != Source::Demo,
                            egui::Button::new("Use generated data"),
                        )
                        .clicked()
                    {
                        self.use_demo_data();
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
        app.rebuild_scene();
        let after = field_of(&app);

        assert!(
            !Arc::ptr_eq(&before, &after),
            "a rebuilt scene must hand the renderer a fresh allocation"
        );
    }

    /// Opening a file must replace every field, not just the one on show: the
    /// wrapped and unwrapped tabs are both derived from the original phase.
    #[test]
    fn opening_a_file_replaces_the_whole_scene() {
        let mut app = app();
        app.tab = Tab::Truth;
        let before = field_of(&app);

        let example =
            crate::phase_file::decode(crate::phase_file::EXAMPLE).expect("the example decodes");
        app.load_field("example.phase".to_owned(), example);

        assert_eq!(
            app.source,
            Source::File {
                name: "example.phase".to_owned()
            },
            "the sidebar must be able to say where the data came from"
        );
        assert!(app.error.is_none(), "a good file reports no error");

        let after = field_of(&app);
        assert!(
            !Arc::ptr_eq(&before, &after),
            "the renderer must be handed a fresh allocation"
        );
        assert_eq!(
            (after.rows(), after.cols()),
            (8, 8),
            "the example is 8 × 8, not the demo's shape"
        );

        // Every tab has to follow, not just the one that happened to be open.
        for tab in Tab::ALL {
            app.tab = tab;
            let field = field_of(&app);
            assert_eq!(
                (field.rows(), field.cols()),
                (8, 8),
                "{tab:?} must show the loaded field's shape"
            );
        }

        assert!(
            app.view.is_none(),
            "a differently sized field must be re-fitted rather than keeping the old view"
        );
    }

    #[test]
    fn going_back_to_generated_data_restores_the_demo() {
        let mut app = app();
        let example =
            crate::phase_file::decode(crate::phase_file::EXAMPLE).expect("the example decodes");
        app.load_field("example.phase".to_owned(), example);

        app.use_demo_data();

        assert_eq!(
            app.source,
            Source::Demo,
            "the source goes back to generated"
        );
        let field = field_of(&app);
        assert_eq!(
            (field.rows(), field.cols()),
            (app.settings.rows, app.settings.cols),
            "and the generator's own dimensions come back"
        );
    }

    /// A file that cannot be read must not blank the viewer.
    #[test]
    fn a_failed_load_keeps_what_was_already_on_screen() {
        let mut app = app();
        let before = field_of(&app);

        app.dialog
            .accept("broken.phase".to_owned(), &crate::phase_file::EXAMPLE[..20]);
        app.poll_incoming_file(&egui::Context::default());

        assert!(
            app.error.is_some(),
            "the failure has to be reported somewhere"
        );
        assert_eq!(app.source, Source::Demo, "and the source must not change");
        assert!(
            Arc::ptr_eq(&before, &field_of(&app)),
            "the field on screen must survive a failed open"
        );
    }

    #[test]
    fn the_example_can_be_opened_without_a_filesystem() {
        let mut app = app();
        app.load_example();
        app.poll_incoming_file(&egui::Context::default());

        assert!(app.error.is_none(), "the shipped example must load cleanly");
        assert_eq!(
            app.source,
            Source::File {
                name: crate::phase_file::EXAMPLE_NAME.to_owned()
            },
            "and be named after the file it came from"
        );
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

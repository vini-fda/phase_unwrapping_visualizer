//! An interactive viewer for `InSAR` phase fields.

#![warn(clippy::all, rust_2018_idioms)]

mod app;
mod render;
mod ui;

// The domain core: no rendering, fully unit tested, and what later features
// (residues, branch cuts, unwrapping itself) will be written against.
pub mod colormap;
pub mod demo;
pub mod file_dialog;
pub mod graph;
pub mod phase;
pub mod phase_file;
pub mod view;

pub use app::PhaseVisualizerApp;

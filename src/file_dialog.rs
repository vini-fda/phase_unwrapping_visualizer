//! Opening `.phase` files through the platform's own file picker.
//!
//! Native and web cannot share one code path: a native dialog blocks and hands
//! back a path, while a browser can only ever hand over bytes asynchronously,
//! and never a path at all. The shape here — a synchronous branch for native, a
//! spawned future for the web, both ending in the same slot the UI polls — is
//! the one [rerun's `file_dialog.rs`][rerun] uses for the same split.
//!
//! [rerun]: https://github.com/rerun-io/rerun/blob/main/crates/viewer_support/re_viewer_context/src/file_dialog.rs

use std::sync::{Arc, Mutex};

use crate::graph::IntegrationPath;
use crate::inputs::Slot;
use crate::phase::PhaseField;
use crate::phase_file;

/// What a chosen file turned out to hold.
#[derive(Debug)]
pub enum Payload {
    /// A field of phase samples.
    Phase(PhaseField),
    /// A walk over the pixels.
    Path(IntegrationPath),
}

/// A file the user chose, decoded or not.
pub struct Opened {
    /// Which input it was being opened for.
    pub slot: Slot,
    /// What to call it in the UI. On the web this is all that is knowable.
    pub name: String,
    /// The contents, or why they could not be read.
    pub result: Result<Payload, String>,
}

/// A picker whose result arrives whenever it arrives.
///
/// The UI calls [`Self::take`] every frame; on native the answer is already
/// there by the time [`Self::pick`] returns, on the web it appears some frames
/// later.
#[derive(Clone, Default)]
pub struct FileDialog {
    /// Where a finished pick waits until the UI collects it. Named apart from
    /// the input [`Slot`] it is filling, which is a different thing entirely.
    pending: Arc<Mutex<Option<Opened>>>,
}

impl FileDialog {
    /// Opens the picker.
    ///
    /// Returns immediately on both platforms; the result shows up in
    /// [`Self::take`].
    #[cfg(target_arch = "wasm32")]
    pub fn pick(&self, slot: Slot) {
        let dialog = rfd::AsyncFileDialog::new()
            .add_filter(slot.label(), &[slot.extension()])
            .set_title(format!("Open {}", slot.label().to_lowercase()));

        // A browser never blocks on a file picker, and never reveals a path:
        // all that comes back is a handle to read bytes out of, later.
        let pending = Arc::clone(&self.pending);
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(handle) = dialog.pick_file().await {
                let name = handle.file_name();
                let bytes = handle.read().await;
                store(&pending, decode_named(slot, name, &bytes));
            }
        });
    }

    /// Opens the picker.
    ///
    /// On macOS a file dialog [may only be opened from the main thread][rfd],
    /// which is where egui runs its update loop, so this is safe to call from
    /// anywhere in the UI.
    ///
    /// [rfd]: https://docs.rs/rfd/latest/rfd/#macos-non-windowed-applications-async-and-threading
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pick(&self, slot: Slot) {
        // The blocking API, so there is no executor to find: the call returns
        // once the user has chosen or cancelled.
        let path = rfd::FileDialog::new()
            .add_filter(slot.label(), &[slot.extension()])
            .set_title(format!("Open {}", slot.label().to_lowercase()))
            .pick_file();

        let Some(path) = path else {
            return; // cancelled
        };

        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );

        match std::fs::read(&path) {
            Ok(bytes) => store(&self.pending, decode_named(slot, name, &bytes)),
            Err(error) => store(
                &self.pending,
                Opened {
                    slot,
                    name,
                    result: Err(error.to_string()),
                },
            ),
        }
    }

    /// Accepts bytes that arrived some other way, such as a dropped file.
    pub fn accept(&self, slot: Slot, name: String, bytes: &[u8]) {
        store(&self.pending, decode_named(slot, name, bytes));
    }

    /// Takes the pending result, if one has arrived.
    pub fn take(&self) -> Option<Opened> {
        self.pending.lock().ok()?.take()
    }
}

/// Parks a result for the UI to collect.
///
/// A poisoned lock is dropped on the floor rather than propagated: the only
/// thing behind it is one pending file, and losing it costs the user a second
/// click.
fn store(pending: &Mutex<Option<Opened>>, opened: Opened) {
    if let Ok(mut pending) = pending.lock() {
        *pending = Some(opened);
    } else {
        log::error!(
            "the file dialog's slot was poisoned; dropping {}",
            opened.name
        );
    }
}

fn decode_named(slot: Slot, name: String, bytes: &[u8]) -> Opened {
    let result = match slot {
        Slot::Path => phase_file::decode_path(bytes).map(Payload::Path),
        _ => phase_file::decode(bytes).map(Payload::Phase),
    };
    Opened {
        slot,
        name,
        result: result.map_err(|error| error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decoded_file_reaches_the_slot_once() {
        let dialog = FileDialog::default();
        assert!(dialog.take().is_none(), "nothing has been picked yet");

        dialog.accept(
            Slot::Original,
            "example.phase".to_owned(),
            phase_file::EXAMPLE,
        );

        let opened = dialog.take().expect("the accepted file must be waiting");
        assert_eq!(opened.name, "example.phase", "the name is carried through");
        assert_eq!(
            opened.slot,
            Slot::Original,
            "the slot is carried through too"
        );
        let Payload::Phase(field) = opened.result.expect("the example decodes") else {
            panic!("a .phase file must decode to phase samples");
        };
        assert_eq!((field.rows(), field.cols()), (8, 8), "8 × 8");

        assert!(
            dialog.take().is_none(),
            "taking must consume it, or the UI would reload every frame"
        );
    }

    #[test]
    fn a_bad_file_reports_why_instead_of_being_dropped() {
        let dialog = FileDialog::default();
        dialog.accept(
            Slot::Wrapped,
            "truncated.phase".to_owned(),
            &phase_file::EXAMPLE[..40],
        );

        let opened = dialog.take().expect("a failure still reaches the UI");
        assert_eq!(
            opened.name, "truncated.phase",
            "named so the user knows which"
        );
        let message = opened.result.expect_err("40 bytes is not a valid file");
        assert!(
            message.contains("header"),
            "the message should say what was wrong, got {message:?}"
        );
    }
}

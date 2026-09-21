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

use crate::phase::PhaseField;
use crate::phase_file::{self, EXTENSION};

/// A file the user chose, decoded or not.
pub struct Opened {
    /// What to call it in the UI. On the web this is all that is knowable.
    pub name: String,
    /// The field, or why it could not be read.
    pub result: Result<PhaseField, String>,
}

/// A picker whose result arrives whenever it arrives.
///
/// The UI calls [`Self::take`] every frame; on native the answer is already
/// there by the time [`Self::pick`] returns, on the web it appears some frames
/// later.
#[derive(Clone, Default)]
pub struct FileDialog {
    slot: Arc<Mutex<Option<Opened>>>,
}

impl FileDialog {
    /// Opens the picker.
    ///
    /// Returns immediately on both platforms; the result shows up in
    /// [`Self::take`].
    #[cfg(target_arch = "wasm32")]
    pub fn pick(&self) {
        let dialog = rfd::AsyncFileDialog::new()
            .add_filter("Phase field", &[EXTENSION])
            .set_title("Open phase data");

        // A browser never blocks on a file picker, and never reveals a path:
        // all that comes back is a handle to read bytes out of, later.
        let slot = Arc::clone(&self.slot);
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(handle) = dialog.pick_file().await {
                let name = handle.file_name();
                let bytes = handle.read().await;
                store(&slot, decode_named(name, &bytes));
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
    pub fn pick(&self) {
        // The blocking API, so there is no executor to find: the call returns
        // once the user has chosen or cancelled.
        let path = rfd::FileDialog::new()
            .add_filter("Phase field", &[EXTENSION])
            .set_title("Open phase data")
            .pick_file();

        let Some(path) = path else {
            return; // cancelled
        };

        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );

        match std::fs::read(&path) {
            Ok(bytes) => store(&self.slot, decode_named(name, &bytes)),
            Err(error) => store(
                &self.slot,
                Opened {
                    name,
                    result: Err(error.to_string()),
                },
            ),
        }
    }

    /// Accepts bytes that arrived some other way, such as a dropped file.
    pub fn accept(&self, name: String, bytes: &[u8]) {
        store(&self.slot, decode_named(name, bytes));
    }

    /// Takes the pending result, if one has arrived.
    pub fn take(&self) -> Option<Opened> {
        self.slot.lock().ok()?.take()
    }
}

/// Parks a result for the UI to collect.
///
/// A poisoned lock is dropped on the floor rather than propagated: the only
/// thing behind it is one pending file, and losing it costs the user a second
/// click.
fn store(slot: &Mutex<Option<Opened>>, opened: Opened) {
    if let Ok(mut slot) = slot.lock() {
        *slot = Some(opened);
    } else {
        log::error!(
            "the file dialog's slot was poisoned; dropping {}",
            opened.name
        );
    }
}

fn decode_named(name: String, bytes: &[u8]) -> Opened {
    Opened {
        name,
        result: phase_file::decode(bytes).map_err(|error| error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decoded_file_reaches_the_slot_once() {
        let dialog = FileDialog::default();
        assert!(dialog.take().is_none(), "nothing has been picked yet");

        dialog.accept("example.phase".to_owned(), phase_file::EXAMPLE);

        let opened = dialog.take().expect("the accepted file must be waiting");
        assert_eq!(opened.name, "example.phase", "the name is carried through");
        let field = opened.result.expect("the example decodes");
        assert_eq!((field.rows(), field.cols()), (8, 8), "8 × 8");

        assert!(
            dialog.take().is_none(),
            "taking must consume it, or the UI would reload every frame"
        );
    }

    #[test]
    fn a_bad_file_reports_why_instead_of_being_dropped() {
        let dialog = FileDialog::default();
        dialog.accept("truncated.phase".to_owned(), &phase_file::EXAMPLE[..40]);

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

//! What the viewer has been given, and what it derives from it.
//!
//! Four things go in, and only one of them is strictly required:
//!
//! | Input | If absent |
//! |---|---|
//! | **Original phase** — the phase before wrapping | the Truth tab has nothing to show |
//! | **Wrapped phase** `ψ` — the observable | derived as `wrap(original)` |
//! | **Unwrapped phase** `φ` — a candidate | integrated here along a comb path |
//! | **Integration path** — the walk that produced `φ` | walls and arrows cannot say what the walk did |
//!
//! What cannot be missing is `ψ`, because everything else is measured against
//! it — so at least one of the original or the wrapped phase has to be there.
//!
//! A supplied `ψ` is never overwritten by one derived from the original. An
//! unwrapper consumed some particular `ψ`, and if the viewer measured against a
//! different one — a different mask, a different wrapping convention, a
//! filtering step in between — every reported disagreement would be an artefact
//! of the mismatch rather than a property of the unwrapping.

use std::sync::Arc;

use crate::demo::{self, SceneSettings};
use crate::graph::{IntegrationPath, Unwrapping, UnwrappingError};
use crate::phase::PhaseField;

/// Where one input came from, for the UI to report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Made up by the demo generator.
    Generated,
    /// Read from a file of this name.
    File(String),
}

impl Origin {
    /// How to name it in the sidebar.
    pub fn label(&self) -> String {
        match self {
            Self::Generated => "synthetic".to_owned(),
            Self::File(name) => name.clone(),
        }
    }
}

/// One supplied input and where it came from.
#[derive(Clone, Debug)]
pub struct Supplied<T> {
    /// The data itself.
    pub value: T,
    /// Where it was obtained.
    pub origin: Origin,
}

/// Which of the four inputs a file is being opened for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// The phase before wrapping.
    Original,
    /// The observable wrapped phase.
    Wrapped,
    /// A candidate unwrapping.
    Unwrapped,
    /// The walk that produced the candidate.
    Path,
}

impl Slot {
    /// Every slot, in the order the pipeline uses them.
    pub const ALL: [Self; 4] = [Self::Original, Self::Wrapped, Self::Unwrapped, Self::Path];

    /// Short name for the sidebar.
    pub fn label(self) -> &'static str {
        match self {
            Self::Original => "Original phase",
            Self::Wrapped => "Wrapped phase",
            Self::Unwrapped => "Unwrapped phase",
            Self::Path => "Integration path",
        }
    }

    /// What this input is for, and what happens without it.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Original => {
                "The phase before wrapping — the thing an unwrapping is trying to recover.\n\n\
                 Optional. Only the Truth tab shows it, and a real interferogram does not come \
                 with one. If given and no wrapped phase is, the wrapped phase is derived from \
                 it as ψ = wrap(original)."
            }
            Self::Wrapped => {
                "The observable phase ψ, inside (-π, π]. Everything else is measured against it.\n\n\
                 Required, but it can come from the original instead: supply one or the other. \
                 If your unwrapping was produced elsewhere, supply the very same ψ it consumed, \
                 or the disagreeing edges describe the mismatch rather than the unwrapping."
            }
            Self::Unwrapped => {
                "A candidate unwrapping φ, congruent to ψ modulo 2π.\n\n\
                 Optional. Without one the viewer makes its own by integrating ψ along a comb \
                 path — down the first column, then across each row — which is the naive raster \
                 method and a poor unwrapper, deliberately."
            }
            Self::Path => {
                "The walk that produced the unwrapped phase, as one byte per pixel naming the \
                 neighbour each was reached from.\n\n\
                 Optional, and only meaningful alongside a supplied unwrapped phase. Without it \
                 the residues and the disagreeing edges are still exact — they need only ψ and φ \
                 — but the walls cannot separate cut edges from the path and the node view cannot \
                 draw arrows."
            }
        }
    }

    /// What the button that gives up this slot's file should say.
    ///
    /// Each names the thing it falls back *to*, because "synthetic" means
    /// something different in each row: a generated field for the original, a
    /// derivation for the wrapped phase, an algorithm for the candidate.
    pub fn fallback_label(self) -> &'static str {
        match self {
            Self::Original | Self::Wrapped | Self::Path => "Use synthetic",
            Self::Unwrapped => "Use naive unwrapping algorithm",
        }
    }

    /// What happens to this slot when its file is given up.
    pub fn fallback_tooltip(self) -> &'static str {
        match self {
            Self::Original => {
                "Stop using this file and generate a synthetic original phase again.\n\n\
                 Not simply dropped: with no original at all, a derived wrapped phase would \
                 go with it and leave nothing to show."
            }
            Self::Wrapped => {
                "Stop using this file. The wrapped phase goes back to being derived from the \
                 original as ψ = wrap(original)."
            }
            Self::Unwrapped => {
                "Stop using this file. The viewer goes back to integrating ψ itself, along a \
                 comb path — and its own path is then known, so the walls and arrows come back."
            }
            Self::Path => {
                "Stop using this file. With a supplied unwrapped phase and no path, the walls \
                 and arrows cannot say what the walk did."
            }
        }
    }

    /// The file extension this slot reads.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Original | Self::Wrapped | Self::Unwrapped => crate::phase_file::EXTENSION,
            Self::Path => crate::phase_file::PATH_EXTENSION,
        }
    }
}

/// Everything the viewer has been given.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    /// The phase before wrapping.
    pub original: Option<Supplied<Arc<PhaseField>>>,
    /// The observable phase, when supplied rather than derived.
    pub wrapped: Option<Supplied<Arc<PhaseField>>>,
    /// A candidate unwrapping, when supplied rather than integrated here.
    pub unwrapped: Option<Supplied<Arc<PhaseField>>>,
    /// The walk behind that candidate.
    pub path: Option<Supplied<Arc<IntegrationPath>>>,
}

/// How a slot is currently being filled — supplied, derived, or not at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Supplied directly, from here.
    Supplied(Origin),
    /// Not supplied, but worked out from what was. The string says how.
    Derived(String),
    /// Not supplied and not derivable. The string says what that costs.
    Missing(String),
}

impl SlotState {
    /// `true` when a file was supplied for this slot, so it can be cleared.
    pub fn is_supplied(&self) -> bool {
        matches!(self, Self::Supplied(_))
    }

    /// One line for the sidebar.
    pub fn label(&self) -> String {
        match self {
            Self::Supplied(origin) => origin.label(),
            Self::Derived(how) | Self::Missing(how) => how.clone(),
        }
    }
}

/// A scene resolved from the inputs, ready to display.
pub struct Resolved {
    /// The phase before wrapping, if there is one.
    pub original: Option<Arc<PhaseField>>,
    /// The observable phase.
    pub wrapped: Arc<PhaseField>,
    /// The candidate and its analysis.
    pub unwrapping: Arc<Unwrapping>,
    /// The candidate's samples, shared with the render callback.
    pub unwrapped: Arc<PhaseField>,
}

/// Why the inputs could not be turned into a scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    /// Neither an original nor a wrapped phase was supplied.
    NoWrappedPhase,
    /// Two inputs describe different shapes.
    ShapeMismatch {
        /// Which input disagreed.
        slot: Slot,
        /// The shape everything else has, as `(rows, cols)`.
        expected: (usize, usize),
        /// The shape this one has, as `(rows, cols)`.
        found: (usize, usize),
    },
    /// The analysis rejected the combination.
    Unwrapping(UnwrappingError),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoWrappedPhase => write!(
                f,
                "no wrapped phase: open either an original phase to wrap, or a wrapped phase directly"
            ),
            Self::ShapeMismatch {
                slot,
                expected,
                found,
            } => write!(
                f,
                "{} is {} × {}, but the rest of the data is {} × {}",
                slot.label(),
                found.0,
                found.1,
                expected.0,
                expected.1
            ),
            Self::Unwrapping(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ResolveError {}

impl Inputs {
    /// The inputs the demo starts from: a generated original, everything else
    /// derived.
    pub fn generated(settings: SceneSettings) -> Self {
        Self {
            original: Some(Supplied {
                value: Arc::new(demo::noisy_ramp(
                    settings.rows,
                    settings.cols,
                    settings.cycles_x,
                    settings.cycles_y,
                    settings.noise,
                    settings.seed,
                )),
                origin: Origin::Generated,
            }),
            ..Self::default()
        }
    }

    /// How each slot is currently being filled, for the sidebar to report.
    pub fn state(&self, slot: Slot) -> SlotState {
        let supplied = match slot {
            Slot::Original => self.original.as_ref().map(|input| &input.origin),
            Slot::Wrapped => self.wrapped.as_ref().map(|input| &input.origin),
            Slot::Unwrapped => self.unwrapped.as_ref().map(|input| &input.origin),
            Slot::Path => self.path.as_ref().map(|input| &input.origin),
        };
        if let Some(origin) = supplied {
            return SlotState::Supplied(origin.clone());
        }

        match slot {
            Slot::Original => {
                SlotState::Missing("not provided — the Truth tab is empty".to_owned())
            }
            Slot::Wrapped => {
                if self.original.is_some() {
                    SlotState::Derived("derived: ψ = wrap(original)".to_owned())
                } else {
                    SlotState::Missing("not provided — nothing to show".to_owned())
                }
            }
            Slot::Unwrapped => SlotState::Derived("integrated here along a comb path".to_owned()),
            Slot::Path => {
                if self.unwrapped.is_some() {
                    SlotState::Missing("not provided — no walls or arrows for the path".to_owned())
                } else {
                    SlotState::Derived("comb path from (0, 0)".to_owned())
                }
            }
        }
    }

    /// Forgets whatever was supplied for `slot`.
    pub fn clear(&mut self, slot: Slot) {
        match slot {
            Slot::Original => self.original = None,
            Slot::Wrapped => self.wrapped = None,
            Slot::Unwrapped => {
                self.unwrapped = None;
                // A path describes a walk that produced a particular candidate.
                // Without the candidate it describes nothing.
                self.path = None;
            }
            Slot::Path => self.path = None,
        }
    }

    /// The shape everything must agree on, taken from the first input there is.
    fn shape(&self) -> Option<(usize, usize)> {
        let field = self
            .wrapped
            .as_ref()
            .or(self.original.as_ref())
            .or(self.unwrapped.as_ref())?;
        Some((field.value.rows(), field.value.cols()))
    }

    /// Turns the inputs into something displayable.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] if there is no wrapped phase to be had, if the
    /// inputs disagree about the shape of the field, or if the analysis rejects
    /// the combination.
    pub fn resolve(&self) -> Result<Resolved, ResolveError> {
        let Some(expected) = self.shape() else {
            return Err(ResolveError::NoWrappedPhase);
        };

        for (slot, shape) in [
            (
                Slot::Original,
                self.original
                    .as_ref()
                    .map(|i| (i.value.rows(), i.value.cols())),
            ),
            (
                Slot::Wrapped,
                self.wrapped
                    .as_ref()
                    .map(|i| (i.value.rows(), i.value.cols())),
            ),
            (
                Slot::Unwrapped,
                self.unwrapped
                    .as_ref()
                    .map(|i| (i.value.rows(), i.value.cols())),
            ),
            (
                Slot::Path,
                self.path.as_ref().map(|i| (i.value.rows(), i.value.cols())),
            ),
        ] {
            if let Some(found) = shape
                && found != expected
            {
                return Err(ResolveError::ShapeMismatch {
                    slot,
                    expected,
                    found,
                });
            }
        }

        // A supplied ψ wins over one derived from the original: it is what the
        // unwrapping was actually measured against.
        let wrapped = match (self.wrapped.as_ref(), self.original.as_ref()) {
            (Some(supplied), _) => Arc::clone(&supplied.value),
            (None, Some(original)) => Arc::new(demo::wrap_field(&original.value)),
            (None, None) => return Err(ResolveError::NoWrappedPhase),
        };

        let (candidate, path) = if let Some(supplied) = self.unwrapped.as_ref() {
            (
                supplied.value.as_ref().clone(),
                self.path.as_ref().map(|path| path.value.as_ref().clone()),
            )
        } else {
            // No candidate given, so make the naive one — and then the path it
            // followed is known exactly, because we chose it.
            let comb = IntegrationPath::comb(expected.0, expected.1);
            let integrated = demo::integrate(&wrapped, &comb).map_err(ResolveError::Unwrapping)?;
            (integrated, Some(comb))
        };

        let unwrapping =
            Unwrapping::new(&wrapped, candidate, path).map_err(ResolveError::Unwrapping)?;

        Ok(Resolved {
            original: self.original.as_ref().map(|input| Arc::clone(&input.value)),
            unwrapped: Arc::new(unwrapping.unwrapped().clone()),
            unwrapping: Arc::new(unwrapping),
            wrapped,
        })
    }
}

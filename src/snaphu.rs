//! The snaphu-rs unwrapper, wired to the viewer's phase fields.
//!
//! [`snaphu_rs`] is a port of SNAPHU, the statistical-cost network-flow
//! phase unwrapping software from Stanford: <https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/>.
//!
//! The viewer drives it at one fixed setting of the parameters, chosen here
//! rather than exposed: the point of the option is to have a real unwrapper to
//! compare against, not to become a front end for SNAPHU's configuration.
//!
//! Only the phase comes back. SNAPHU's answer is a flow field, not a walk, so
//! it has no integration path to report. A candidate from here has none, just
//! like a candidate opened from a file without its `.path`.

use snaphu_rs::data::raster::Raster;
use snaphu_rs::{CostMode, RunConfig, SnaphuError, UnwrapInputs, run_snaphu_inplace};

use crate::phase::PhaseField;

/// The parameters the viewer runs SNAPHU with.
///
/// Stock defaults but for the cost mode. SNAPHU's own default is topography
/// mode, which reads the phase as terrain and prices every arc through a
/// baseline, a wavelength and an orbit geometry. A bare phase field has none
/// of these. Smooth mode asks for none of that: it prices an arc by how
/// far the flow bends the solution, which is all that can be said about phase
/// that arrived without its radar.
fn config() -> RunConfig {
    RunConfig {
        cost_mode: CostMode::Smooth,
        ..RunConfig::default()
    }
}

/// Unwraps `wrapped` with snaphu-rs.
///
/// # Errors
///
/// Returns [`SnaphuError`] if the field is smaller than 2 × 2, if the solver
/// fails, or if the field holds a non-finite sample. Those are the viewer's
/// masked pixels, which SNAPHU cannot handle.
pub fn unwrap(wrapped: &PhaseField) -> Result<PhaseField, SnaphuError> {
    let (rows, cols) = (wrapped.rows(), wrapped.cols());

    // A `Raster` owns its samples, so psi is copied on the way in. The way out is
    // not: `run_snaphu_inplace` writes straight into the buffer that becomes
    // the returned field, rather than allocating a raster we would then move
    // out of. The other three outputs (flows, connected components and the
    // magnitude used) are not requested, so they are not produced.
    let psi = Raster::new(cols, rows, wrapped.as_slice().to_vec());
    let mut unwrapped = vec![0.0f32; rows * cols];
    run_snaphu_inplace(
        &UnwrapInputs::new(&psi),
        &config(),
        &mut unwrapped,
        None,
        None,
        None,
    )?;

    Ok(PhaseField::new(unwrapped, rows, cols)
        .expect("the buffer was allocated as rows × cols samples"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo;
    use std::f32::consts::TAU;

    /// A ramp with no noise has no residues, so every unwrapper must recover it
    /// exactly, up to a constant offset. Unwrapping can never pin that offset
    /// down, and it is a whole number of turns because phi stays congruent to psi.
    #[test]
    fn a_noiseless_ramp_comes_back_whole() {
        let truth = demo::noisy_ramp(24, 32, 2.0, 1.0, 0.0, 7);
        let wrapped = demo::wrap_field(&truth);

        let unwrapped = unwrap(&wrapped).expect("a finite 24 × 32 field unwraps");
        assert_eq!(
            (unwrapped.rows(), unwrapped.cols()),
            (24, 32),
            "the candidate must have the shape it was asked for"
        );

        let offset = unwrapped.as_slice()[0] - truth.as_slice()[0];
        assert!(
            (offset / TAU - (offset / TAU).round()).abs() < 1e-3,
            "the offset must be a whole number of turns, got {offset}"
        );
        for (index, (recovered, expected)) in unwrapped
            .as_slice()
            .iter()
            .zip(truth.as_slice())
            .enumerate()
        {
            assert!(
                (recovered - expected - offset).abs() < 1e-3,
                "sample {index}: {recovered} is not {expected} + {offset}"
            );
        }
    }

    /// Whatever the solution, it has to be a candidate at all: phi ≡ psi (mod 2pi).
    #[test]
    fn the_candidate_stays_congruent_to_the_wrapped_phase() {
        let wrapped = demo::wrap_field(&demo::noisy_ramp(16, 16, 3.0, 1.0, 0.8, 11));

        let unwrapped = unwrap(&wrapped).expect("a finite 16 × 16 field unwraps");

        for (index, (phi, psi)) in unwrapped
            .as_slice()
            .iter()
            .zip(wrapped.as_slice())
            .enumerate()
        {
            let turns = (phi - psi) / TAU;
            assert!(
                (turns - turns.round()).abs() < 1e-3,
                "sample {index}: phi − psi = {} is not a whole number of turns",
                phi - psi
            );
        }
    }

    /// The grid SNAPHU refuses is the one the viewer must report rather than
    /// swallow, so the error has to arrive as an error.
    #[test]
    fn a_field_too_small_to_unwrap_is_refused() {
        let single = PhaseField::new(vec![0.0], 1, 1).expect("one sample is one by one");
        assert_eq!(
            unwrap(&single),
            Err(SnaphuError::TooSmall { rows: 1, cols: 1 }),
            "a 1 × 1 grid has no arcs to solve for"
        );
    }
}

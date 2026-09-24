//! The phase field: an `m × n` grid of `f32` samples, plus the wrapping operator.
//!
//! This module is deliberately free of any `egui`/`wgpu` types so that the parts
//! of the viewer where correctness is actually decidable can be unit tested.

use std::f32::consts::{PI, TAU};

/// Wraps `x` to the half-open interval `(-pi, pi]`.
///
/// This is the `wrapping` operator from the interferometry literature: the
/// observable phase of an interferogram is only ever known modulo `2pi`, and
/// unwrapping is the problem of recovering the original `x` from `wrap(x)`.
///
/// Non-finite inputs produce `NaN`, which callers treat as "masked".
pub fn wrap(x: f32) -> f32 {
    // `rem_euclid` lands in `[0, 2pi)`, so `pi - that` lands in `(-pi, pi]`.
    // Wrapping `pi - x` rather than `x` is what makes the interval closed on
    // the right: `wrap(pi) == pi` and `wrap(-pi) == pi`.
    PI - (PI - x).rem_euclid(TAU)
}

/// Why a [`PhaseField`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseFieldError {
    /// `data.len()` did not match `rows * cols`.
    LengthMismatch {
        /// Number of rows that was asked for.
        rows: usize,
        /// Number of columns that was asked for.
        cols: usize,
        /// Number of samples that was actually supplied.
        len: usize,
    },
    /// `rows * cols` overflowed `usize`.
    SizeOverflow {
        /// Number of rows that was asked for.
        rows: usize,
        /// Number of columns that was asked for.
        cols: usize,
    },
}

impl std::fmt::Display for PhaseFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::LengthMismatch { rows, cols, len } => {
                write!(
                    f,
                    "expected {rows} × {cols} = {} samples, got {len}",
                    rows * cols
                )
            }
            Self::SizeOverflow { rows, cols } => {
                write!(f, "{rows} × {cols} overflows usize")
            }
        }
    }
}

impl std::error::Error for PhaseFieldError {}

/// A rectangular field of phase samples, stored row-major.
///
/// The invariant `data.len() == rows * cols` holds for every value of this
/// type; it is the only reason the accessors can be total.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhaseField {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl PhaseField {
    /// Builds a field from row-major samples.
    ///
    /// A `0 × 0` field is valid and renders as nothing.
    ///
    /// # Errors
    ///
    /// Returns [`PhaseFieldError`] if `data.len() != rows * cols`, or if
    /// `rows * cols` overflows.
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self, PhaseFieldError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(PhaseFieldError::SizeOverflow { rows, cols })?;
        if data.len() != expected {
            return Err(PhaseFieldError::LengthMismatch {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    /// Number of rows (`m`).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (`n`).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `true` if the field holds no samples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The samples, row-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The sample at `(row, col)`, or `None` if it is out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.rows && col < self.cols {
            self.data.get(row * self.cols + col).copied()
        } else {
            None
        }
    }

    /// The `(min, max)` of the finite samples, ignoring `NaN` and infinities.
    ///
    /// `None` when there is no finite sample at all.
    pub fn finite_range(&self) -> Option<(f32, f32)> {
        self.data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(None, |acc, v| match acc {
                None => Some((v, v)),
                Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
            })
    }

    /// A linear phase ramp, for demos: `phi(row, col) = 2pi (cycles_x·u + cycles_y·v)`
    /// where `u` and `v` are the normalized column and row coordinates of the
    /// sample's centre.
    ///
    /// Wrapping this produces the straight, evenly spaced fringes that a flat
    /// tilted surface would give in an interferogram.
    pub fn linear_gradient(rows: usize, cols: usize, cycles_x: f32, cycles_y: f32) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let v = (row as f32 + 0.5) / rows as f32;
            for col in 0..cols {
                let u = (col as f32 + 0.5) / cols as f32;
                data.push(TAU * cycles_x.mul_add(u, cycles_y * v));
            }
        }
        Self { data, rows, cols }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `wrap` must land in `(-pi, pi]`: open on the left, closed on the right.
    #[test]
    fn wrap_lands_in_the_half_open_interval() {
        assert_eq!(wrap(PI), PI, "pi is the included endpoint");
        assert_eq!(wrap(-PI), PI, "-pi ≡ pi (mod 2pi), and -pi is excluded");
        assert_eq!(wrap(0.0), 0.0, "zero is a fixed point");

        for i in -1000..1000 {
            let x = i as f32 * 0.037;
            let w = wrap(x);
            assert!(
                -PI < w && w <= PI,
                "wrap({x}) = {w} escaped the interval (-pi, pi]"
            );
        }
    }

    /// Wrapping must be invariant under whole turns, which is the whole point.
    #[test]
    fn wrap_is_invariant_under_whole_turns() {
        for i in -50..50 {
            let x = i as f32 * 0.1;
            for k in -3..=3 {
                let shifted = wrap(TAU.mul_add(k as f32, x));
                assert!(
                    (shifted - wrap(x)).abs() < 1e-4,
                    "wrap({x} + {k}·2pi) = {shifted} != wrap({x}) = {}",
                    wrap(x)
                );
            }
        }
    }

    #[test]
    fn new_rejects_a_length_mismatch() {
        let err = PhaseField::new(vec![0.0; 5], 2, 3).expect_err("5 != 2 × 3");
        assert_eq!(
            err,
            PhaseFieldError::LengthMismatch {
                rows: 2,
                cols: 3,
                len: 5
            },
            "the error should report the sizes it compared"
        );
    }

    #[test]
    fn get_is_row_major_and_bounds_checked() {
        let field = PhaseField::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2, 3).expect("6 == 2 × 3");
        assert_eq!(field.get(0, 0), Some(0.0), "first sample");
        assert_eq!(field.get(0, 2), Some(2.0), "end of the first row");
        assert_eq!(field.get(1, 0), Some(3.0), "start of the second row");
        assert_eq!(field.get(2, 0), None, "row past the end");
        assert_eq!(field.get(0, 3), None, "column past the end");
    }

    #[test]
    fn finite_range_skips_non_finite_samples() {
        let field =
            PhaseField::new(vec![f32::NAN, -2.0, f32::INFINITY, 7.0], 1, 4).expect("4 == 1 × 4");
        assert_eq!(
            field.finite_range(),
            Some((-2.0, 7.0)),
            "NaN and ∞ must not leak into the range"
        );

        let all_nan = PhaseField::new(vec![f32::NAN; 4], 1, 4).expect("4 == 1 × 4");
        assert_eq!(
            all_nan.finite_range(),
            None,
            "a fully masked field has no range"
        );
    }

    #[test]
    fn linear_gradient_is_linear_along_both_axes() {
        let field = PhaseField::linear_gradient(4, 5, 1.0, 0.0);
        let row: Vec<f32> = (0..5)
            .map(|c| field.get(0, c).expect("in bounds"))
            .collect();
        let step = row[1] - row[0];
        for pair in row.windows(2) {
            assert!(
                (pair[1] - pair[0] - step).abs() < 1e-5,
                "column steps should be uniform, got {pair:?}"
            );
        }
        for r in 0..4 {
            assert!(
                (field.get(r, 0).expect("in bounds") - row[0]).abs() < 1e-5,
                "with cycles_y = 0 the ramp must not vary down a column"
            );
        }
    }
}

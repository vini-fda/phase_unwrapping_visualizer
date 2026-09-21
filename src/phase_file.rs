//! The `.phase` container: a field of `f32` samples behind a fixed-size header.
//!
//! # Layout
//!
//! | offset | size | meaning                                  |
//! |--------|------|------------------------------------------|
//! | 0      | 4    | `m`, the number of rows, as `u32`        |
//! | 4      | 4    | `n`, the number of columns, as `u32`     |
//! | 8      | 56   | reserved, written as zero, ignored on read |
//! | 64     | 4mn  | the samples, row-major                   |
//!
//! Everything is **native-endian**, which keeps reading free on the machine
//! that wrote the file but means a file does not survive a move between a
//! little-endian and a big-endian host. Every mainstream target today is
//! little-endian, so the example file shipped with this crate is readable
//! anywhere it is likely to be opened.
//!
//! The header is padded to 64 bytes so the samples start aligned to a cache
//! line, and so that a later version can claim some of the reserved space —
//! a magic number, a version, a units tag — without moving the data or
//! invalidating files written today, which already have zeros there.
//!
//! # Trusting the header
//!
//! `m` and `n` are `u32`, so `m × n` can reach 2^64 elements and `4mn` can
//! overflow even a `usize`. Nothing here is sized from the header alone: the
//! product is computed with checked arithmetic and then has to *match* the
//! bytes actually present, so a corrupt or hostile header is rejected rather
//! than turned into an allocation.

use std::path::Path;

use crate::phase::{PhaseField, PhaseFieldError};

/// Bytes of header before the samples begin.
pub const HEADER_LEN: usize = 64;

/// Conventional extension for the format.
pub const EXTENSION: &str = "phase";

/// The example file shipped with the crate.
///
/// Embedded rather than read from `assets/` so that opening it needs no
/// filesystem and no guess about the working directory — which is what makes
/// it work on the web too.
pub const EXAMPLE: &[u8] = include_bytes!("../assets/original_phase_example.phase");

/// File name of [`EXAMPLE`].
pub const EXAMPLE_NAME: &str = "original_phase_example.phase";

/// Why a `.phase` file could not be read.
#[derive(Debug)]
pub enum PhaseFileError {
    /// The file could not be read from disk.
    Io(std::io::Error),
    /// Fewer bytes than the header needs.
    TooShort {
        /// How many bytes the file holds.
        len: usize,
    },
    /// A dimension was zero, so there is no field to show.
    Empty {
        /// Rows named by the header.
        rows: u32,
        /// Columns named by the header.
        cols: u32,
    },
    /// `rows × cols × 4` does not fit in a `usize` on this machine.
    TooLarge {
        /// Rows named by the header.
        rows: u32,
        /// Columns named by the header.
        cols: u32,
    },
    /// The header and the file's actual length disagree.
    LengthMismatch {
        /// Rows named by the header.
        rows: u32,
        /// Columns named by the header.
        cols: u32,
        /// Sample bytes the header implies.
        expected: usize,
        /// Sample bytes the file actually holds.
        found: usize,
    },
    /// The samples did not form a valid field.
    Field(PhaseFieldError),
}

impl std::fmt::Display for PhaseFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::TooShort { len } => {
                write!(
                    f,
                    "only {len} bytes, too short for a {HEADER_LEN}-byte header"
                )
            }
            Self::Empty { rows, cols } => {
                write!(f, "header says {rows} × {cols}, which holds no samples")
            }
            Self::TooLarge { rows, cols } => write!(
                f,
                "header says {rows} × {cols}, which is too large to address on this machine"
            ),
            Self::LengthMismatch {
                rows,
                cols,
                expected,
                found,
            } => write!(
                f,
                "header says {rows} × {cols}, which needs {expected} bytes of samples, but the file has {found}"
            ),
            Self::Field(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PhaseFileError {}

impl From<std::io::Error> for PhaseFileError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Reads a field out of the bytes of a `.phase` file.
///
/// # Errors
///
/// Returns [`PhaseFileError`] if the header is missing, names an empty or
/// unaddressable field, or disagrees with how many samples are actually there.
pub fn decode(bytes: &[u8]) -> Result<PhaseField, PhaseFileError> {
    let Some((header, payload)) = bytes.split_at_checked(HEADER_LEN) else {
        return Err(PhaseFileError::TooShort { len: bytes.len() });
    };

    let rows = read_u32(header, 0);
    let cols = read_u32(header, 4);
    // Bytes 8..64 are reserved. They are deliberately not checked, so that a
    // later version can use them without old readers rejecting new files.

    if rows == 0 || cols == 0 {
        return Err(PhaseFileError::Empty { rows, cols });
    }

    let expected = sample_bytes(rows, cols).ok_or(PhaseFileError::TooLarge { rows, cols })?;
    if payload.len() != expected {
        return Err(PhaseFileError::LengthMismatch {
            rows,
            cols,
            expected,
            found: payload.len(),
        });
    }

    let data = payload
        .chunks_exact(4)
        .map(|word| {
            let mut sample = [0u8; 4];
            sample.copy_from_slice(word);
            f32::from_ne_bytes(sample)
        })
        .collect();

    // `rows` and `cols` fit in `usize` here: `sample_bytes` already proved
    // their product does.
    PhaseField::new(data, rows as usize, cols as usize).map_err(PhaseFileError::Field)
}

/// Writes a field out as the bytes of a `.phase` file.
///
/// # Errors
///
/// Returns [`PhaseFileError::TooLarge`] if the field does not fit the `u32`
/// dimensions the format allows.
pub fn encode(field: &PhaseField) -> Result<Vec<u8>, PhaseFileError> {
    let (Ok(rows), Ok(cols)) = (u32::try_from(field.rows()), u32::try_from(field.cols())) else {
        return Err(PhaseFileError::TooLarge {
            rows: u32::MAX,
            cols: u32::MAX,
        });
    };

    let mut bytes = Vec::with_capacity(HEADER_LEN + field.as_slice().len() * 4);
    bytes.extend_from_slice(&rows.to_ne_bytes());
    bytes.extend_from_slice(&cols.to_ne_bytes());
    bytes.resize(HEADER_LEN, 0);
    for sample in field.as_slice() {
        bytes.extend_from_slice(&sample.to_ne_bytes());
    }
    Ok(bytes)
}

/// Reads a `.phase` file from disk.
///
/// # Errors
///
/// Returns [`PhaseFileError`] if the file cannot be read or does not decode.
pub fn read(path: &Path) -> Result<PhaseField, PhaseFileError> {
    decode(&std::fs::read(path)?)
}

/// Writes a field to a `.phase` file.
///
/// # Errors
///
/// Returns [`PhaseFileError`] if the field cannot be encoded or written.
pub fn write(path: &Path, field: &PhaseField) -> Result<(), PhaseFileError> {
    std::fs::write(path, encode(field)?)?;
    Ok(())
}

/// How many bytes of samples a `rows × cols` field needs, or `None` if that
/// cannot be addressed here.
///
/// This is the one piece of arithmetic standing between a header and an
/// allocation, so every step of it is checked.
fn sample_bytes(rows: u32, cols: u32) -> Option<usize> {
    let rows = usize::try_from(rows).ok()?;
    let cols = usize::try_from(cols).ok()?;
    rows.checked_mul(cols)?.checked_mul(4)
}

/// Reads a native-endian `u32` at `offset`, which must be in bounds.
fn read_u32(header: &[u8], offset: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&header[offset..offset + 4]);
    u32::from_ne_bytes(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field() -> PhaseField {
        PhaseField::linear_gradient(3, 5, 1.0, 0.5)
    }

    #[test]
    fn a_field_survives_a_round_trip() {
        let original = field();
        let bytes = encode(&original).expect("3 × 5 fits in u32");
        let decoded = decode(&bytes).expect("what we just encoded must decode");
        assert_eq!(decoded, original, "encoding must not change the samples");
    }

    #[test]
    fn the_header_is_padded_and_the_samples_follow_it() {
        let bytes = encode(&field()).expect("3 × 5 fits in u32");

        assert_eq!(bytes.len(), HEADER_LEN + 3 * 5 * 4, "header plus 4mn bytes");
        assert_eq!(read_u32(&bytes, 0), 3, "rows come first");
        assert_eq!(read_u32(&bytes, 4), 5, "then columns");
        assert!(
            bytes[8..HEADER_LEN].iter().all(|byte| *byte == 0),
            "the reserved span must be written as zero"
        );
        assert_eq!(
            HEADER_LEN % 4,
            0,
            "the samples must start on an f32 boundary"
        );
    }

    /// The reserved bytes are for later use, so a reader must not insist they
    /// are zero — otherwise the first file to use them breaks every old reader.
    #[test]
    fn reserved_header_bytes_are_ignored_rather_than_rejected() {
        let mut bytes = encode(&field()).expect("3 × 5 fits in u32");
        for byte in &mut bytes[8..HEADER_LEN] {
            *byte = 0xAB;
        }
        let decoded = decode(&bytes).expect("a future header field must not break today's reader");
        assert_eq!(decoded, field(), "and must not disturb the samples");
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let bytes = encode(&field()).expect("3 × 5 fits in u32");

        let no_header = decode(&bytes[..HEADER_LEN - 1]).expect_err("header is incomplete");
        assert!(
            matches!(no_header, PhaseFileError::TooShort { len } if len == HEADER_LEN - 1),
            "a stub shorter than the header must say so, got {no_header:?}"
        );

        let short = decode(&bytes[..bytes.len() - 4]).expect_err("one sample is missing");
        assert!(
            matches!(
                short,
                PhaseFileError::LengthMismatch {
                    expected: 60,
                    found: 56,
                    ..
                }
            ),
            "a truncated payload must report both lengths, got {short:?}"
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = encode(&field()).expect("3 × 5 fits in u32");
        bytes.push(0);
        assert!(
            matches!(decode(&bytes), Err(PhaseFileError::LengthMismatch { .. })),
            "a file longer than its header claims is corrupt, not partially valid"
        );
    }

    #[test]
    fn an_empty_field_is_rejected() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(&0u32.to_ne_bytes());
        bytes[4..8].copy_from_slice(&8u32.to_ne_bytes());
        assert!(
            matches!(
                decode(&bytes),
                Err(PhaseFileError::Empty { rows: 0, cols: 8 })
            ),
            "zero rows means there is nothing to display"
        );
    }

    /// The header is 8 bytes and can describe 2^64 samples. Believing it would
    /// mean trying to allocate 64 exabytes from a 64-byte file.
    #[test]
    fn a_header_claiming_an_impossible_size_allocates_nothing() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(&u32::MAX.to_ne_bytes());
        bytes[4..8].copy_from_slice(&u32::MAX.to_ne_bytes());

        let error = decode(&bytes).expect_err("2^64 samples cannot be in a 64-byte file");
        assert!(
            matches!(
                error,
                PhaseFileError::TooLarge { .. } | PhaseFileError::LengthMismatch { .. }
            ),
            "an impossible header must be refused before anything is sized from it, got {error:?}"
        );
    }

    /// The shipped example is generated by a separate implementation of the
    /// spec, so this checks the two agree — and that the file is worth opening:
    /// a single vortex, which leaves exactly one charge behind when wrapped.
    #[test]
    fn the_example_file_decodes_to_a_field_with_one_residue() {
        let truth = decode(EXAMPLE).expect("the shipped example must decode");
        assert_eq!((truth.rows(), truth.cols()), (8, 8), "the example is 8 × 8");
        assert_eq!(
            EXAMPLE.len(),
            HEADER_LEN + 8 * 8 * 4,
            "64-byte header plus 256 bytes of samples"
        );

        let (min, max) = truth.finite_range().expect("the samples are finite");
        assert!(
            max - min > std::f32::consts::TAU,
            "the example must span more than one turn, or wrapping it is trivial: {min}..{max}"
        );

        let scene = crate::demo::Scene::from_truth(truth).expect("8 × 8 is a valid field");
        let stats = scene.unwrapping.stats();
        assert_eq!(
            (stats.positive_residues, stats.negative_residues),
            (1, 0),
            "the example holds a single +1 vortex"
        );
        assert_eq!(
            scene.unwrapping.residue(3, 3),
            Some(1),
            "and it sits at the centre of the grid"
        );
        assert!(
            stats.inconsistent_edges > 0,
            "one residue must force at least one edge to disagree"
        );
    }

    #[test]
    fn sample_bytes_checks_every_multiplication() {
        assert_eq!(sample_bytes(3, 5), Some(60), "3 × 5 × 4");
        assert_eq!(sample_bytes(1, 1), Some(4), "one sample is four bytes");
        assert_eq!(
            sample_bytes(u32::MAX, u32::MAX),
            None,
            "the product of two u32s times four overflows a 64-bit usize"
        );
        assert_eq!(
            sample_bytes(1 << 31, 1 << 31),
            None,
            "so does a product that is merely enormous"
        );
    }
}

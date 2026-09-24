//! The `.phase` container: a field of `f32` samples behind a fixed-size header.
//!
//! # Layout
//!
//! | offset | size | meaning                                     |
//! |--------|------|---------------------------------------------|
//! | 0      | 4    | `m`, the number of rows, as `u32`           |
//! | 4      | 4    | `n`, the number of columns, as `u32`        |
//! | 8      | 1    | payload kind: `0` phase samples, `1` parent directions |
//! | 9      | 55   | reserved, written as zero, ignored on read  |
//! | 64     | …    | the payload, row-major                      |
//!
//! The payload is `4mn` bytes of `f32` for [`Kind::Phase`], and `mn` bytes of
//! [`ParentDirection`] codes for [`Kind::Parents`]. The kind byte lives in what
//! used to be reserved space and is zero in every file written before it
//! existed, which is exactly why zero means "phase samples".
//!
//! Everything is **native-endian**, which keeps reading free on the machine
//! that wrote the file but means a file does not survive a move between a
//! little-endian and a big-endian host. Every mainstream target today is
//! little-endian, so the example file shipped with this crate is readable
//! anywhere it is likely to be opened.
//!
//! The header is padded to 64 bytes so the samples start aligned to a cache
//! line, and so that a later version can use the reserved space for, say, a
//! magic number, a version or a units tag. Files written today have zeros
//! there, so neither the data nor those files would need to change.
//!
//! # Trusting the header
//!
//! `m` and `n` are `u32`, so `m × n` can reach 2^64 elements and `4mn` can
//! overflow even a `usize`. Nothing here is sized from the header alone: the
//! product is computed with checked arithmetic and then has to *match* the
//! bytes actually present, so a corrupt or hostile header is rejected rather
//! than turned into an allocation.

use std::path::Path;

use crate::graph::IntegrationPath;
use crate::phase::{PhaseField, PhaseFieldError};

/// Bytes of header before the samples begin.
pub const HEADER_LEN: usize = 64;

/// Conventional extension for a file of phase samples.
pub const EXTENSION: &str = "phase";

/// Conventional extension for a file of parent directions.
pub const PATH_EXTENSION: &str = "path";

/// What a file's payload holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `4mn` bytes: one `f32` phase sample per pixel.
    Phase,
    /// `mn` bytes: one [`ParentDirection`] code per pixel.
    Parents,
}

impl Kind {
    /// The byte this kind is stored as, at offset 8.
    pub fn code(self) -> u8 {
        match self {
            Self::Phase => 0,
            Self::Parents => 1,
        }
    }

    /// Reads a kind from its byte.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Phase),
            1 => Some(Self::Parents),
            _ => None,
        }
    }

    /// Bytes each pixel occupies in the payload.
    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Phase => 4,
            Self::Parents => 1,
        }
    }

    /// What to call it when a file turns out to hold the wrong one.
    fn describe(self) -> &'static str {
        match self {
            Self::Phase => "phase samples",
            Self::Parents => "an integration path",
        }
    }
}

/// The example file shipped with the crate.
///
/// Embedded rather than read from `assets/` so that opening it needs no
/// filesystem and no guess about the working directory. That is also why it
/// works on the web.
pub const EXAMPLE: &[u8] = include_bytes!("../assets/original_phase_example.phase");

/// File name of [`EXAMPLE`].
pub const EXAMPLE_NAME: &str = "original_phase_example.phase";

/// An integration path matching [`EXAMPLE`], for trying the path slot out.
///
/// It is the comb walk written out explicitly, so opening it alongside a
/// candidate reproduces exactly what the viewer does on its own. It is for
/// checking the file plumbing, not an interesting path in itself.
pub const EXAMPLE_PATH: &[u8] = include_bytes!("../assets/original_phase_example.path");

/// File name of [`EXAMPLE_PATH`].
pub const EXAMPLE_PATH_NAME: &str = "original_phase_example.path";

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
    /// The file holds a payload of a kind the caller did not ask for.
    WrongKind {
        /// What was being opened.
        expected: Kind,
        /// What the file actually holds, if it is a kind this build knows.
        found: Option<Kind>,
    },
    /// The parent directions did not form a valid path.
    Path(crate::graph::PathError),
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
            Self::WrongKind { expected, found } => match found {
                Some(found) => write!(
                    f,
                    "this file holds {}, but {} were expected",
                    found.describe(),
                    expected.describe()
                ),
                None => write!(
                    f,
                    "this file's payload kind is not one this version understands; {} were expected",
                    expected.describe()
                ),
            },
            Self::Path(error) => write!(f, "{error}"),
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
    let (rows, cols, payload) = split(bytes, Kind::Phase)?;

    let data = payload
        .chunks_exact(4)
        .map(|word| {
            let mut sample = [0u8; 4];
            sample.copy_from_slice(word);
            f32::from_ne_bytes(sample)
        })
        .collect();

    // `rows` and `cols` fit in `usize` here: `split` already proved their
    // product does.
    PhaseField::new(data, rows as usize, cols as usize).map_err(PhaseFileError::Field)
}

/// Reads an integration path out of the bytes of a `.path` file.
///
/// # Errors
///
/// Returns [`PhaseFileError`] if the header is missing or inconsistent, if the
/// file holds something other than parent directions, or if the directions do
/// not describe a single walk covering every pixel.
pub fn decode_path(bytes: &[u8]) -> Result<IntegrationPath, PhaseFileError> {
    let (rows, cols, payload) = split(bytes, Kind::Parents)?;
    IntegrationPath::from_codes(payload, rows as usize, cols as usize).map_err(PhaseFileError::Path)
}

/// Validates a header and hands back the dimensions and the payload.
///
/// This is where a header stops being trusted: `m` and `n` are `u32`, so their
/// product can reach 2^64 and the payload size can overflow a `usize`. Both
/// multiplications are checked, and the result then has to *match* the bytes
/// actually present, so an impossible header is refused before anything is
/// sized from it.
fn split(bytes: &[u8], expected: Kind) -> Result<(u32, u32, &[u8]), PhaseFileError> {
    let Some((header, payload)) = bytes.split_at_checked(HEADER_LEN) else {
        return Err(PhaseFileError::TooShort { len: bytes.len() });
    };

    let rows = read_u32(header, 0);
    let cols = read_u32(header, 4);

    let found = Kind::from_code(header[8]);
    if found != Some(expected) {
        return Err(PhaseFileError::WrongKind { expected, found });
    }
    // Bytes 9..64 are reserved. They are deliberately not checked, so that a
    // later version can use them without old readers rejecting new files.

    if rows == 0 || cols == 0 {
        return Err(PhaseFileError::Empty { rows, cols });
    }

    let wanted =
        payload_bytes(rows, cols, expected).ok_or(PhaseFileError::TooLarge { rows, cols })?;
    if payload.len() != wanted {
        return Err(PhaseFileError::LengthMismatch {
            rows,
            cols,
            expected: wanted,
            found: payload.len(),
        });
    }

    Ok((rows, cols, payload))
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

    let mut bytes = header(rows, cols, Kind::Phase);
    bytes.reserve(field.as_slice().len() * 4);
    for sample in field.as_slice() {
        bytes.extend_from_slice(&sample.to_ne_bytes());
    }
    Ok(bytes)
}

/// Writes an integration path out as the bytes of a `.path` file.
///
/// # Errors
///
/// Returns [`PhaseFileError::TooLarge`] if the path does not fit the `u32`
/// dimensions the format allows.
pub fn encode_path(path: &IntegrationPath) -> Result<Vec<u8>, PhaseFileError> {
    let (Ok(rows), Ok(cols)) = (u32::try_from(path.rows()), u32::try_from(path.cols())) else {
        return Err(PhaseFileError::TooLarge {
            rows: u32::MAX,
            cols: u32::MAX,
        });
    };

    let mut bytes = header(rows, cols, Kind::Parents);
    bytes.extend_from_slice(&path.to_codes());
    Ok(bytes)
}

/// Builds the fixed-size header, zero-padded.
fn header(rows: u32, cols: u32, kind: Kind) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_LEN);
    bytes.extend_from_slice(&rows.to_ne_bytes());
    bytes.extend_from_slice(&cols.to_ne_bytes());
    bytes.push(kind.code());
    bytes.resize(HEADER_LEN, 0);
    bytes
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

/// How many bytes of payload a `rows × cols` field of `kind` needs, or `None`
/// if that cannot be addressed here.
///
/// This is the one piece of arithmetic standing between a header and an
/// allocation, so every step of it is checked.
fn payload_bytes(rows: u32, cols: u32, kind: Kind) -> Option<usize> {
    let rows = usize::try_from(rows).ok()?;
    let cols = usize::try_from(cols).ok()?;
    rows.checked_mul(cols)?.checked_mul(kind.bytes_per_pixel())
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
            bytes[9..HEADER_LEN].iter().all(|byte| *byte == 0),
            "the reserved span must be written as zero"
        );
        assert_eq!(
            HEADER_LEN % 4,
            0,
            "the samples must start on an f32 boundary"
        );
    }

    /// The reserved bytes are for later use, so a reader must not insist they
    /// are zero. Otherwise the first file to use them breaks every old reader.
    #[test]
    fn reserved_header_bytes_are_ignored_rather_than_rejected() {
        let mut bytes = encode(&field()).expect("3 × 5 fits in u32");
        for byte in &mut bytes[9..HEADER_LEN] {
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
    /// spec, so this checks the two agree. It also checks the file is worth
    /// opening: a single vortex, which leaves exactly one charge when wrapped.
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

    /// The two example files are generated by a separate implementation of the
    /// spec, so this checks that implementation and this one agree, including
    /// on the kind byte, which is the only thing telling the files apart.
    #[test]
    fn the_example_path_matches_the_example_field() {
        let path = decode_path(EXAMPLE_PATH).expect("the shipped path must decode");
        assert_eq!((path.rows(), path.cols()), (8, 8), "it matches the field");
        assert_eq!(path.root(), (0, 0), "the comb starts at the first pixel");
        assert_eq!(
            path,
            IntegrationPath::comb(8, 8),
            "the file spells out exactly the comb the viewer builds itself"
        );

        // Each file must be refused for the other's slot, or a mix-up would be
        // read as garbage rather than reported.
        assert!(
            matches!(
                decode(EXAMPLE_PATH),
                Err(PhaseFileError::WrongKind {
                    expected: Kind::Phase,
                    found: Some(Kind::Parents)
                })
            ),
            "a path opened as phase samples must be refused by name"
        );
        assert!(
            matches!(
                decode_path(EXAMPLE),
                Err(PhaseFileError::WrongKind {
                    expected: Kind::Parents,
                    found: Some(Kind::Phase)
                })
            ),
            "and the other way round"
        );
    }

    #[test]
    fn a_path_survives_a_round_trip() {
        let original = IntegrationPath::comb(4, 6);
        let bytes = encode_path(&original).expect("4 × 6 fits in u32");
        assert_eq!(bytes.len(), HEADER_LEN + 4 * 6, "one byte per pixel");
        let decoded = decode_path(&bytes).expect("what we just encoded must decode");
        assert_eq!(decoded, original, "encoding must not change the walk");
    }

    #[test]
    fn payload_bytes_checks_every_multiplication() {
        assert_eq!(payload_bytes(3, 5, Kind::Phase), Some(60), "3 × 5 × 4");
        assert_eq!(
            payload_bytes(1, 1, Kind::Phase),
            Some(4),
            "one sample is four bytes"
        );
        assert_eq!(
            payload_bytes(u32::MAX, u32::MAX, Kind::Phase),
            None,
            "the product of two u32s times four overflows a 64-bit usize"
        );
        assert_eq!(
            payload_bytes(1 << 31, 1 << 31, Kind::Phase),
            None,
            "so does a product that is merely enormous"
        );
    }
}

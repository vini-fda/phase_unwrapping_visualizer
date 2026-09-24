//! The pixel graph `G`, its dual `G*`, and the per-edge analysis that decides
//! which walls the viewer highlights.
//!
//! # The two graphs
//!
//! `G = (V, E)` has one vertex per pixel and one edge per adjacent pair, so an
//! `m × n` field gives `m(n-1) + n(m-1) = 2mn - m - n` edges. Integration walks
//! a spanning tree of `G`, which has `mn - 1` edges; the remaining
//! `(m-1)(n-1)` edges are *cut*, and they are exactly the edges crossed by a
//! spanning tree of the dual `G*`.
//!
//! `G*` has one vertex per inner corner, `(m-1) × (n-1)` in all, plus a
//! single vertex `O` standing for the whole outer boundary. Each inner corner
//! carries the residue of the four pixels around it.
//!
//! # What gets highlighted
//!
//! For every edge, integration *should* move the phase by the wrapped
//! difference `wrap(psi_b - psi_a)`. Along the tree that holds by construction, but
//! a candidate unwrapping is only a candidate: this module measures
//! `(delta phi - delta psi) / 2pi` on *every* edge rather than assuming it, and a non-zero
//! result is what the viewer draws in the highlight colour.

use std::f32::consts::TAU;

use crate::phase::{self, PhaseField};

/// Which of a pixel's two canonical neighbours an edge leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// To the pixel one column to the right.
    Horizontal,
    /// To the pixel one row down.
    Vertical,
}

/// An edge of `G`, named by its upper-left endpoint.
///
/// Canonical orientation always runs right or down, so every edge has exactly
/// one name no matter which way the integration happens to traverse it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId {
    /// Row of the edge's canonical source pixel.
    pub row: usize,
    /// Column of the edge's canonical source pixel.
    pub col: usize,
    /// Which neighbour it leads to.
    pub axis: Axis,
}

impl EdgeId {
    /// The edge from `(row, col)` to `(row, col + 1)`.
    pub fn horizontal(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            axis: Axis::Horizontal,
        }
    }

    /// The edge from `(row, col)` to `(row + 1, col)`.
    pub fn vertical(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            axis: Axis::Vertical,
        }
    }

    /// `(source, target)` as `(row, col)` pairs, in canonical orientation.
    pub fn endpoints(self) -> ((usize, usize), (usize, usize)) {
        let target = match self.axis {
            Axis::Horizontal => (self.row, self.col + 1),
            Axis::Vertical => (self.row + 1, self.col),
        };
        ((self.row, self.col), target)
    }
}

/// How the integration walk, started at the seed pixel, uses an edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Traversal {
    /// Not in the spanning tree: the integration never crosses it, and its dual
    /// is part of the spanning tree of `G*`. Drawn as a solid wall.
    #[default]
    Cut,
    /// Walked along its canonical orientation, adding the phase difference.
    Forward,
    /// Walked against its canonical orientation, subtracting it.
    Backward,
}

/// Texel code meaning "this wall has no role in an integration path".
///
/// Covers the slots where no edge exists (the right edge of the last column
/// and the bottom edge of the last row), and every edge when no path was
/// supplied. The shader draws both the same way: a plain wall, neither
/// dashed as part of a walk nor eligible for the cut-edge colour.
pub const EDGE_NO_ROLE: u8 = 3;

impl Traversal {
    /// `true` when the edge is part of the integration path.
    pub fn is_in_tree(self) -> bool {
        self != Self::Cut
    }

    /// How the shader sees this traversal. Must match `grid.wgsl`.
    pub fn code(self) -> u8 {
        match self {
            Self::Cut => 0,
            Self::Forward => 1,
            Self::Backward => 2,
        }
    }
}

/// What the viewer knows about one edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeState {
    /// The edge's role in the integration path, or `None` when no path was
    /// supplied and there is nothing to say about it.
    pub traversal: Option<Traversal>,

    /// `(delta phi - delta psi) / 2pi`, rounded.
    ///
    /// Zero means the integration moved the phase by exactly the wrapped
    /// difference. Anything else is an edge where the unwrapping disagrees with
    /// the data, and is what gets highlighted.
    pub jump: i32,
}

impl EdgeState {
    /// `true` when the integration delta matches the wrapped delta.
    pub fn is_consistent(self) -> bool {
        self.jump == 0
    }
}

/// Which neighbour a pixel was reached from while the integration walked the
/// field.
///
/// One byte per pixel encodes the whole integration path: which edges it used,
/// which way it crossed each of them, and where it started. That is a great
/// deal less than an explicit edge list (`mn` bytes against roughly `9mn`),
/// and it is the form a breadth-first, depth-first or region-growing unwrapper
/// already has in hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ParentDirection {
    /// Where the integration started. Exactly one pixel carries this.
    Root = 0,
    /// Reached from the pixel above, `(row - 1, col)`.
    Up = 1,
    /// Reached from the pixel below, `(row + 1, col)`.
    Down = 2,
    /// Reached from the pixel to the left, `(row, col - 1)`.
    Left = 3,
    /// Reached from the pixel to the right, `(row, col + 1)`.
    Right = 4,
}

impl ParentDirection {
    /// Reads a direction from its on-disk byte.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Root),
            1 => Some(Self::Up),
            2 => Some(Self::Down),
            3 => Some(Self::Left),
            4 => Some(Self::Right),
            _ => None,
        }
    }

    /// The byte this direction is stored as.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// The pixel this one was reached from, or `None` at the root.
    ///
    /// Returns `None` too when the parent would fall outside the field, which
    /// is how an out-of-bounds direction is caught.
    pub fn parent_of(
        self,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    ) -> Option<(usize, usize)> {
        let parent = match self {
            Self::Root => return None,
            Self::Up => (row.checked_sub(1)?, col),
            Self::Down => (row + 1, col),
            Self::Left => (row, col.checked_sub(1)?),
            Self::Right => (row, col + 1),
        };
        (parent.0 < rows && parent.1 < cols).then_some(parent)
    }

    /// How the edge between a pixel and its parent is traversed.
    ///
    /// Edges are named by their upper-left endpoint, so a walk that arrived
    /// from above or from the left ran along that naming and one that arrived
    /// from below or the right ran against it.
    pub fn traversal(self) -> Option<Traversal> {
        match self {
            Self::Root => None,
            Self::Up | Self::Left => Some(Traversal::Forward),
            Self::Down | Self::Right => Some(Traversal::Backward),
        }
    }
}

/// Why a parent array is not a valid integration path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathError {
    /// The array does not have one entry per pixel.
    LengthMismatch {
        /// Rows the path claims.
        rows: usize,
        /// Columns the path claims.
        cols: usize,
        /// Entries actually supplied.
        len: usize,
    },
    /// A byte outside `0..=4`.
    BadDirection {
        /// Row of the offending pixel.
        row: usize,
        /// Column of the offending pixel.
        col: usize,
        /// The byte that was there.
        code: u8,
    },
    /// A pixel points at a parent outside the field.
    ParentOutOfBounds {
        /// Row of the offending pixel.
        row: usize,
        /// Column of the offending pixel.
        col: usize,
    },
    /// The path has no root, or more than one.
    ///
    /// A walk starts in exactly one place; anything else describes several
    /// disconnected walks, or none.
    RootCount {
        /// How many pixels claimed to be the root.
        found: usize,
    },
    /// Following parents from this pixel goes round in circles instead of
    /// reaching the root.
    Cycle {
        /// Row of a pixel on the cycle.
        row: usize,
        /// Column of a pixel on the cycle.
        col: usize,
    },
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::LengthMismatch { rows, cols, len } => write!(
                f,
                "the path has {len} entries, but {rows} × {cols} needs {}",
                rows * cols
            ),
            Self::BadDirection { row, col, code } => {
                write!(
                    f,
                    "pixel ({row}, {col}) has direction byte {code}, which is not 0..=4"
                )
            }
            Self::ParentOutOfBounds { row, col } => {
                write!(
                    f,
                    "pixel ({row}, {col}) points at a parent outside the field"
                )
            }
            Self::RootCount { found } => {
                write!(
                    f,
                    "a path starts at exactly one pixel, but {found} are marked as the root"
                )
            }
            Self::Cycle { row, col } => write!(
                f,
                "following parents from ({row}, {col}) never reaches the root"
            ),
        }
    }
}

impl std::error::Error for PathError {}

/// The path the integration walked, as one parent direction per pixel.
///
/// Validated on construction, so every pixel is guaranteed to reach the single
/// root by following parents. That makes the used edges a spanning tree of `G`
/// without ever having to count them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegrationPath {
    parents: Vec<ParentDirection>,
    rows: usize,
    cols: usize,
    root: (usize, usize),
}

impl IntegrationPath {
    /// Validates a parent array.
    ///
    /// # Errors
    ///
    /// Returns [`PathError`] if the array is the wrong length, holds a byte
    /// outside `0..=4`, points outside the field, has other than exactly one
    /// root, or contains a cycle.
    pub fn from_codes(codes: &[u8], rows: usize, cols: usize) -> Result<Self, PathError> {
        let expected = rows.saturating_mul(cols);
        if codes.len() != expected {
            return Err(PathError::LengthMismatch {
                rows,
                cols,
                len: codes.len(),
            });
        }

        let mut parents = Vec::with_capacity(codes.len());
        let mut root = None;
        let mut roots = 0;
        for (index, &code) in codes.iter().enumerate() {
            let (row, col) = (index / cols.max(1), index % cols.max(1));
            let direction = ParentDirection::from_code(code).ok_or(PathError::BadDirection {
                row,
                col,
                code,
            })?;
            if direction == ParentDirection::Root {
                roots += 1;
                root = Some((row, col));
            } else if direction.parent_of(row, col, rows, cols).is_none() {
                return Err(PathError::ParentOutOfBounds { row, col });
            }
            parents.push(direction);
        }

        let Some(root) = root.filter(|_| roots == 1) else {
            return Err(PathError::RootCount { found: roots });
        };

        let path = Self {
            parents,
            rows,
            cols,
            root,
        };
        path.check_reaches_root()?;
        Ok(path)
    }

    /// Every pixel must reach the root; otherwise the "path" contains a loop
    /// that no walk could have produced.
    ///
    /// Each chain is followed once and its pixels marked, so the whole check is
    /// linear however tangled the parents are.
    fn check_reaches_root(&self) -> Result<(), PathError> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Mark {
            Unvisited,
            OnCurrentChain,
            ReachesRoot,
        }

        let mut marks = vec![Mark::Unvisited; self.parents.len()];
        let mut chain = Vec::new();

        for start in 0..self.parents.len() {
            if marks[start] != Mark::Unvisited {
                continue;
            }
            chain.clear();
            let mut at = start;
            loop {
                match marks[at] {
                    Mark::ReachesRoot => break,
                    Mark::OnCurrentChain => {
                        let (row, col) = (at / self.cols, at % self.cols);
                        return Err(PathError::Cycle { row, col });
                    }
                    Mark::Unvisited => {}
                }
                marks[at] = Mark::OnCurrentChain;
                chain.push(at);

                let (row, col) = (at / self.cols, at % self.cols);
                match self.parents[at].parent_of(row, col, self.rows, self.cols) {
                    Some((parent_row, parent_col)) => at = parent_row * self.cols + parent_col,
                    // Only the root has no parent, and it has been validated.
                    None => break,
                }
            }
            for &pixel in &chain {
                marks[pixel] = Mark::ReachesRoot;
            }
        }
        Ok(())
    }

    /// The comb path: down the first column, then along each row.
    ///
    /// This is the classic raster integration, and the viewer's fallback when
    /// no path is supplied with an unwrapping.
    pub fn comb(rows: usize, cols: usize) -> Self {
        let mut parents = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                parents.push(match (row, col) {
                    (0, 0) => ParentDirection::Root,
                    // The first column hangs off the pixel above it, …
                    (_, 0) => ParentDirection::Up,
                    // … and every other pixel off the one to its left.
                    _ => ParentDirection::Left,
                });
            }
        }
        Self {
            parents,
            rows,
            cols,
            root: (0, 0),
        }
    }

    /// Number of rows the path covers.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns the path covers.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Where the integration started.
    pub fn root(&self) -> (usize, usize) {
        self.root
    }

    /// The direction stored for one pixel.
    pub fn parent_direction(&self, row: usize, col: usize) -> Option<ParentDirection> {
        (row < self.rows && col < self.cols).then(|| self.parents[row * self.cols + col])
    }

    /// The bytes this path is stored as, one per pixel, row-major.
    pub fn to_codes(&self) -> Vec<u8> {
        self.parents.iter().map(|parent| parent.code()).collect()
    }

    /// How the integration crossed `edge`, or [`Traversal::Cut`] if it never
    /// did.
    ///
    /// An edge is in the path exactly when one of its endpoints names the other
    /// as its parent.
    pub fn traversal(&self, edge: EdgeId) -> Traversal {
        let ((source_row, source_col), (target_row, target_col)) = edge.endpoints();
        if target_row >= self.rows || target_col >= self.cols {
            return Traversal::Cut;
        }

        // The target was reached from the source: the walk ran along the edge's
        // own naming.
        if let Some(direction) = self.parent_direction(target_row, target_col)
            && direction.parent_of(target_row, target_col, self.rows, self.cols)
                == Some((source_row, source_col))
        {
            return Traversal::Forward;
        }

        // The source was reached from the target: the walk ran against it.
        if let Some(direction) = self.parent_direction(source_row, source_col)
            && direction.parent_of(source_row, source_col, self.rows, self.cols)
                == Some((target_row, target_col))
        {
            return Traversal::Backward;
        }

        Traversal::Cut
    }
}

/// Why an [`Unwrapping`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnwrappingError {
    /// The candidate has different dimensions from the wrapped field.
    SizeMismatch {
        /// Dimensions of the wrapped field, as `(rows, cols)`.
        wrapped: (usize, usize),
        /// Dimensions of the candidate, as `(rows, cols)`.
        unwrapped: (usize, usize),
    },
    /// The field has no samples, so there is no graph to analyse.
    Empty,
    /// An edge names a pixel pair that does not exist.
    EdgeOutOfBounds(EdgeId),
    /// The integration path does not cover the same pixels as the field.
    PathSizeMismatch {
        /// Dimensions of the field, as `(rows, cols)`.
        field: (usize, usize),
        /// Dimensions the path covers, as `(rows, cols)`.
        path: (usize, usize),
    },
}

impl std::fmt::Display for UnwrappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::SizeMismatch { wrapped, unwrapped } => write!(
                f,
                "candidate is {} × {}, but the wrapped field is {} × {}",
                unwrapped.0, unwrapped.1, wrapped.0, wrapped.1
            ),
            Self::Empty => write!(f, "the field is empty"),
            Self::EdgeOutOfBounds(edge) => {
                let ((ar, ac), (br, bc)) = edge.endpoints();
                write!(f, "edge ({ar}, {ac}) -> ({br}, {bc}) leaves the field")
            }
            Self::PathSizeMismatch { field, path } => write!(
                f,
                "the integration path covers {} × {}, but the field is {} × {}",
                path.0, path.1, field.0, field.1
            ),
        }
    }
}

impl std::error::Error for UnwrappingError {}

/// Counts worth showing next to the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnwrappingStats {
    /// Edges in the integration path: always `mn - 1`.
    pub tree_edges: usize,
    /// Edges outside it: always `(m-1)(n-1)`.
    pub cut_edges: usize,
    /// Edges whose integration delta is not the wrapped delta.
    pub inconsistent_edges: usize,
    /// Inner corners with residue `+1`.
    pub positive_residues: usize,
    /// Inner corners with residue `-1`.
    pub negative_residues: usize,
}

/// A candidate unwrapping, together with the integration path that claims to
/// produce it and everything the viewer derives from the pair.
#[derive(Clone, Debug)]
pub struct Unwrapping {
    unwrapped: PhaseField,
    /// Row-major, `rows × (cols - 1)`.
    horizontal: Vec<EdgeState>,
    /// Row-major, `(rows - 1) × cols`.
    vertical: Vec<EdgeState>,
    /// Row-major, `(rows - 1) × (cols - 1)`, each in `-1..=1`.
    residues: Vec<i8>,
    /// The walk that produced `unwrapped`, when it is known.
    path: Option<IntegrationPath>,
    stats: UnwrappingStats,
}

impl Unwrapping {
    /// Analyses `unwrapped` as a candidate unwrapping of `wrapped`.
    ///
    /// `path` is the walk that is claimed to have produced it. It is optional,
    /// and its absence costs less than it might seem: the residues come from
    /// `wrapped` alone and the disagreeing edges from the two fields together,
    /// so everything the highlight shows survives without it. What is lost is
    /// the cut/tree distinction between walls, the arrows in the node view, and
    /// the edge counts. These all describe a path, and cannot be invented when
    /// none was given.
    ///
    /// # Errors
    ///
    /// Returns [`UnwrappingError`] if the fields disagree on size, if the field
    /// is empty, or if `path` covers a different shape.
    pub fn new(
        wrapped: &PhaseField,
        unwrapped: PhaseField,
        path: Option<IntegrationPath>,
    ) -> Result<Self, UnwrappingError> {
        let (rows, cols) = (wrapped.rows(), wrapped.cols());
        if (unwrapped.rows(), unwrapped.cols()) != (rows, cols) {
            return Err(UnwrappingError::SizeMismatch {
                wrapped: (rows, cols),
                unwrapped: (unwrapped.rows(), unwrapped.cols()),
            });
        }
        if rows == 0 || cols == 0 {
            return Err(UnwrappingError::Empty);
        }
        if let Some(path) = path.as_ref()
            && (path.rows(), path.cols()) != (rows, cols)
        {
            return Err(UnwrappingError::PathSizeMismatch {
                field: (rows, cols),
                path: (path.rows(), path.cols()),
            });
        }

        let mut horizontal = vec![EdgeState::default(); rows * cols.saturating_sub(1)];
        let mut vertical = vec![EdgeState::default(); rows.saturating_sub(1) * cols];

        for row in 0..rows {
            for col in 0..cols.saturating_sub(1) {
                let state = &mut horizontal[row * (cols - 1) + col];
                state.jump = jump(wrapped, &unwrapped, (row, col), (row, col + 1));
                state.traversal = path
                    .as_ref()
                    .map(|path| path.traversal(EdgeId::horizontal(row, col)));
            }
        }
        for row in 0..rows.saturating_sub(1) {
            for col in 0..cols {
                let state = &mut vertical[row * cols + col];
                state.jump = jump(wrapped, &unwrapped, (row, col), (row + 1, col));
                state.traversal = path
                    .as_ref()
                    .map(|path| path.traversal(EdgeId::vertical(row, col)));
            }
        }

        let residues = residue_field(wrapped);

        let tree_edges = horizontal
            .iter()
            .chain(&vertical)
            .filter(|state| state.traversal.is_some_and(Traversal::is_in_tree))
            .count();

        let stats = UnwrappingStats {
            tree_edges,
            cut_edges: (horizontal.len() + vertical.len()) - tree_edges,
            inconsistent_edges: horizontal
                .iter()
                .chain(&vertical)
                .filter(|state| !state.is_consistent())
                .count(),
            positive_residues: residues.iter().filter(|&&q| q > 0).count(),
            negative_residues: residues.iter().filter(|&&q| q < 0).count(),
        };

        Ok(Self {
            unwrapped,
            horizontal,
            vertical,
            residues,
            path,
            stats,
        })
    }

    /// The walk that produced the candidate, if one was supplied.
    pub fn path(&self) -> Option<&IntegrationPath> {
        self.path.as_ref()
    }

    /// `true` when an integration path is known, so the walls can distinguish
    /// cut edges from the path and the node view can draw arrows.
    pub fn has_path(&self) -> bool {
        self.path.is_some()
    }

    /// The candidate unwrapped phase.
    pub fn unwrapped(&self) -> &PhaseField {
        &self.unwrapped
    }

    /// Counts for the sidebar.
    pub fn stats(&self) -> UnwrappingStats {
        self.stats
    }

    /// Number of rows of pixels.
    pub fn rows(&self) -> usize {
        self.unwrapped.rows()
    }

    /// Number of columns of pixels.
    pub fn cols(&self) -> usize {
        self.unwrapped.cols()
    }

    /// The state of the edge between `(row, col)` and `(row, col + 1)`.
    pub fn horizontal_edge(&self, row: usize, col: usize) -> Option<EdgeState> {
        if row < self.rows() && col + 1 < self.cols() {
            self.horizontal.get(row * (self.cols() - 1) + col).copied()
        } else {
            None
        }
    }

    /// The state of the edge between `(row, col)` and `(row + 1, col)`.
    pub fn vertical_edge(&self, row: usize, col: usize) -> Option<EdgeState> {
        if row + 1 < self.rows() && col < self.cols() {
            self.vertical.get(row * self.cols() + col).copied()
        } else {
            None
        }
    }

    /// Packs the per-edge state for the GPU: one RGBA8 texel per pixel, holding
    /// `[right traversal, right |jump|, down traversal, down |jump|]`.
    ///
    /// The last column has no right edge and the last row has no bottom edge;
    /// those slots carry [`EDGE_NO_ROLE`] and a zero jump.
    pub fn edge_texels(&self) -> Vec<u8> {
        let (rows, cols) = (self.rows(), self.cols());
        let mut texels = Vec::with_capacity(rows * cols * 4);
        for row in 0..rows {
            for col in 0..cols {
                for state in [self.horizontal_edge(row, col), self.vertical_edge(row, col)] {
                    let (traversal, jump) = state.map_or((EDGE_NO_ROLE, 0), |state| {
                        (
                            state.traversal.map_or(EDGE_NO_ROLE, Traversal::code),
                            u8::try_from(state.jump.unsigned_abs()).unwrap_or(u8::MAX),
                        )
                    });
                    texels.push(traversal);
                    texels.push(jump);
                }
            }
        }
        texels
    }

    /// Packs the residues for the GPU: one R8 texel per inner corner, holding
    /// `charge + 1`, so `0` is `-1`, `1` is neutral and `2` is `+1`.
    pub fn residue_texels(&self) -> Vec<u8> {
        self.residues
            .iter()
            .map(|&q| u8::try_from(q + 1).unwrap_or(1))
            .collect()
    }

    /// The residue at the inner corner shared by pixels `(row, col)`,
    /// `(row, col+1)`, `(row+1, col+1)` and `(row+1, col)`.
    pub fn residue(&self, row: usize, col: usize) -> Option<i8> {
        if row + 1 < self.rows() && col + 1 < self.cols() {
            self.residues.get(row * (self.cols() - 1) + col).copied()
        } else {
            None
        }
    }
}

/// `(delta phi - delta psi) / 2pi` on the edge from `a` to `b`, rounded.
///
/// Non-finite samples yield `0`: nothing can be said about a masked edge, and
/// reporting it as inconsistent would drown the real signal.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the quotient of two finite f32 phases is far inside i32 range"
)]
fn jump(wrapped: &PhaseField, unwrapped: &PhaseField, a: (usize, usize), b: (usize, usize)) -> i32 {
    let (Some(psi_a), Some(psi_b)) = (wrapped.get(a.0, a.1), wrapped.get(b.0, b.1)) else {
        return 0;
    };
    let (Some(phi_a), Some(phi_b)) = (unwrapped.get(a.0, a.1), unwrapped.get(b.0, b.1)) else {
        return 0;
    };

    let expected = phase::wrap(psi_b - psi_a);
    let actual = phi_b - phi_a;
    let turns = (actual - expected) / TAU;
    if turns.is_finite() {
        turns.round() as i32
    } else {
        0
    }
}

/// The residue of every inner corner, row-major, `(rows-1) × (cols-1)`.
fn residue_field(wrapped: &PhaseField) -> Vec<i8> {
    let (rows, cols) = (wrapped.rows(), wrapped.cols());
    let mut residues = Vec::with_capacity(rows.saturating_sub(1) * cols.saturating_sub(1));
    for row in 0..rows.saturating_sub(1) {
        for col in 0..cols.saturating_sub(1) {
            residues.push(residue_at(wrapped, row, col));
        }
    }
    residues
}

/// The residue of the loop around the inner corner below-right of `(row, col)`.
///
/// Sums the wrapped differences clockwise on screen (right, down, left, up)
/// and divides by `2pi`. Each term lies in `(-pi, pi]`, so the sum is strictly
/// inside `(-4pi, 4pi)` for real data and the charge is always `-1`, `0` or `+1`.
///
/// A loop touching a non-finite sample has no meaningful residue and reports
/// `0`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the result is clamped to -1..=1 before narrowing"
)]
fn residue_at(wrapped: &PhaseField, row: usize, col: usize) -> i8 {
    let corner = |r: usize, c: usize| wrapped.get(r, c);
    let (Some(a), Some(b), Some(d), Some(c)) = (
        corner(row, col),
        corner(row, col + 1),
        corner(row + 1, col + 1),
        corner(row + 1, col),
    ) else {
        return 0;
    };

    let sum = phase::wrap(b - a) + phase::wrap(d - b) + phase::wrap(c - d) + phase::wrap(a - c);
    if !sum.is_finite() {
        return 0;
    }
    (sum / TAU).round().clamp(-1.0, 1.0) as i8
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    use crate::demo::integrate;

    fn ramp(rows: usize, cols: usize) -> PhaseField {
        PhaseField::linear_gradient(rows, cols, 1.5, 0.75)
    }

    /// Counts the edges the path actually uses.
    fn count_tree_edges(path: &IntegrationPath, rows: usize, cols: usize) -> usize {
        let mut count = 0;
        for row in 0..rows {
            for col in 0..cols {
                if col + 1 < cols && path.traversal(EdgeId::horizontal(row, col)).is_in_tree() {
                    count += 1;
                }
                if row + 1 < rows && path.traversal(EdgeId::vertical(row, col)).is_in_tree() {
                    count += 1;
                }
            }
        }
        count
    }

    fn wrapped_of(field: &PhaseField) -> PhaseField {
        let data = field.as_slice().iter().copied().map(phase::wrap).collect();
        PhaseField::new(data, field.rows(), field.cols()).expect("same dimensions")
    }

    #[test]
    fn a_spanning_tree_has_the_edge_count_the_algebra_predicts() {
        for (rows, cols) in [(5, 4), (1, 7), (9, 1), (2, 2)] {
            let path = IntegrationPath::comb(rows, cols);
            let tree_edges = count_tree_edges(&path, rows, cols);
            assert_eq!(
                tree_edges,
                rows * cols - 1,
                "{rows}×{cols}: a spanning tree has mn - 1 edges"
            );

            let total = rows * (cols - 1) + cols * (rows - 1);
            assert_eq!(
                total,
                2 * rows * cols - rows - cols,
                "{rows}×{cols}: e_total = 2mn - m - n"
            );
            assert_eq!(
                total - tree_edges,
                (rows - 1) * (cols - 1),
                "{rows}×{cols}: e_non-tree = (m-1)(n-1)"
            );
        }
    }

    #[test]
    fn integrating_along_the_tree_leaves_every_tree_edge_consistent() {
        let (rows, cols) = (9, 7);
        let truth = ramp(rows, cols);
        let psi = wrapped_of(&truth);
        let path = IntegrationPath::comb(rows, cols);

        let phi = integrate(&psi, &path).expect("the comb is a spanning tree");
        let unwrapping =
            Unwrapping::new(&psi, phi, Some(path.clone())).expect("the comb is a spanning tree");

        for row in 0..rows {
            for col in 0..cols {
                for edge in [EdgeId::horizontal(row, col), EdgeId::vertical(row, col)] {
                    if !path.traversal(edge).is_in_tree() {
                        continue;
                    }
                    let state = match edge.axis {
                        Axis::Horizontal => unwrapping.horizontal_edge(edge.row, edge.col),
                        Axis::Vertical => unwrapping.vertical_edge(edge.row, edge.col),
                    }
                    .expect("the edge is inside the field");
                    assert!(
                        state.is_consistent(),
                        "tree edge {edge:?} must move the phase by exactly the wrapped delta"
                    );
                    assert_eq!(
                        state.traversal.map(Traversal::is_in_tree),
                        Some(true),
                        "tree edge {edge:?} must be marked as part of the path"
                    );
                }
            }
        }
    }

    #[test]
    fn a_residue_free_field_unwraps_with_no_inconsistent_edges() {
        let (rows, cols) = (11, 13);
        // A gentle ramp: consecutive samples are far less than pi apart, so no
        // loop can accumulate a full turn and there are no residues.
        let truth = PhaseField::linear_gradient(rows, cols, 1.0, 0.5);
        let psi = wrapped_of(&truth);
        let path = IntegrationPath::comb(rows, cols);

        let phi = integrate(&psi, &path).expect("spanning tree");
        let unwrapping = Unwrapping::new(&psi, phi, Some(path.clone())).expect("spanning tree");
        let stats = unwrapping.stats();

        assert_eq!(
            (stats.positive_residues, stats.negative_residues),
            (0, 0),
            "a smooth ramp has no residues"
        );
        assert_eq!(
            stats.inconsistent_edges, 0,
            "with no residues the integration is path independent, so nothing disagrees"
        );
        assert_eq!(stats.tree_edges, rows * cols - 1, "mn - 1 tree edges");
        assert_eq!(
            stats.cut_edges,
            (rows - 1) * (cols - 1),
            "(m-1)(n-1) cut edges"
        );
    }

    /// The lower bound from the theory: at least `max(N+, N-)` edges must
    /// disagree, because every residue has to be enclosed by a defect.
    #[test]
    fn a_single_residue_forces_at_least_one_inconsistent_edge() {
        // A 2 × 2 loop carrying one full turn: the four wrapped differences
        // around it sum to 2pi, so the corner has charge +1.
        let quarter = PI / 2.0 + 0.2;
        let psi = PhaseField::new(
            vec![
                phase::wrap(0.0),
                phase::wrap(quarter),
                phase::wrap(3.0 * quarter),
                phase::wrap(2.0 * quarter),
            ],
            2,
            2,
        )
        .expect("4 == 2 × 2");

        let residue = residue_at(&psi, 0, 0);
        assert_eq!(residue, 1, "this loop encircles one full turn");

        let (rows, cols) = (2, 2);
        let path = IntegrationPath::comb(rows, cols);
        let phi = integrate(&psi, &path).expect("spanning tree");
        let unwrapping = Unwrapping::new(&psi, phi, Some(path.clone())).expect("spanning tree");
        let stats = unwrapping.stats();

        assert_eq!(stats.positive_residues, 1, "one +1 charge");
        assert!(
            stats.inconsistent_edges >= stats.positive_residues.max(stats.negative_residues),
            "at least max(N+, N-) edges must disagree, got {}",
            stats.inconsistent_edges
        );
        assert!(
            stats.inconsistent_edges <= (rows - 1) * (cols - 1),
            "and at most (m-1)(n-1) of them, got {}",
            stats.inconsistent_edges
        );
    }

    /// Clockwise circulation of the wrapped differences around the outer
    /// perimeter of the pixel grid: right along the top, down the right edge,
    /// left along the bottom, up the left edge.
    fn boundary_circulation(psi: &PhaseField) -> f32 {
        let (rows, cols) = (psi.rows(), psi.cols());
        let at = |r: usize, c: usize| psi.get(r, c).expect("inside the field");

        let mut sum = 0.0;
        for col in 0..cols - 1 {
            sum += phase::wrap(at(0, col + 1) - at(0, col));
        }
        for row in 0..rows - 1 {
            sum += phase::wrap(at(row + 1, cols - 1) - at(row, cols - 1));
        }
        for col in (0..cols - 1).rev() {
            sum += phase::wrap(at(rows - 1, col) - at(rows - 1, col + 1));
        }
        for row in (0..rows - 1).rev() {
            sum += phase::wrap(at(row, 0) - at(row + 1, 0));
        }
        sum
    }

    /// Summing every face's residue makes each interior edge appear twice, once
    /// in each direction, so they all cancel and only the outer boundary
    /// survives. The total charge is therefore the boundary circulation, which
    /// is the charge on the dual vertex `O` and *not* zero in general.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the circulation of a finite field is a small multiple of 2pi"
    )]
    #[test]
    fn total_residue_equals_the_boundary_circulation() {
        for (rows, cols, noise, seed) in [(24, 24, 0.9, 7), (17, 31, 1.3, 99), (9, 9, 0.5, 3)] {
            let truth = crate::demo::noisy_ramp(rows, cols, 3.0, 1.5, noise, seed);
            let psi = wrapped_of(&truth);
            let residues = residue_field(&psi);

            let total: i32 = residues.iter().map(|&q| i32::from(q)).sum();
            let expected = (boundary_circulation(&psi) / TAU).round() as i32;
            assert_eq!(
                total, expected,
                "{rows}×{cols}: interior edges must cancel, leaving the boundary circulation"
            );
            assert!(
                residues.iter().any(|&q| q != 0),
                "{rows}×{cols}: the noisy field should contain residues, or this proves nothing"
            );
            assert!(
                residues.iter().all(|&q| (-1..=1).contains(&q)),
                "{rows}×{cols}: a single loop can only carry one turn"
            );
        }
    }

    #[test]
    fn the_comb_walks_every_edge_along_its_own_naming() {
        let (rows, cols) = (4, 3);
        let path = IntegrationPath::comb(rows, cols);
        assert_eq!(path.root(), (0, 0), "the comb starts at the first pixel");

        for row in 0..rows {
            for col in 0..cols {
                for edge in [EdgeId::horizontal(row, col), EdgeId::vertical(row, col)] {
                    let traversal = path.traversal(edge);
                    assert_ne!(
                        traversal,
                        Traversal::Backward,
                        "the comb is built from (0,0) outwards, so nothing is walked backwards: {edge:?}"
                    );
                }
            }
        }
    }

    /// A walk that reaches a pixel from below or the right runs against the
    /// edge's own naming, and has to be reported as such or the node view draws
    /// its arrows the wrong way round.
    #[test]
    fn a_walk_against_the_grain_is_reported_backwards() {
        // A 1 × 3 row rooted at its right-hand end, so the walk runs leftwards.
        let path = IntegrationPath::from_codes(
            &[
                ParentDirection::Right.code(),
                ParentDirection::Right.code(),
                ParentDirection::Root.code(),
            ],
            1,
            3,
        )
        .expect("a chain rooted at one end is a valid walk");

        assert_eq!(path.root(), (0, 2), "the root is where it was put");
        assert_eq!(
            path.traversal(EdgeId::horizontal(0, 0)),
            Traversal::Backward,
            "reached from the right, so against the edge's naming"
        );
        assert_eq!(
            path.traversal(EdgeId::horizontal(0, 1)),
            Traversal::Backward,
            "and so is the next one along"
        );
    }

    #[test]
    fn a_parent_array_must_describe_a_single_walk() {
        // Two roots: two walks, not one.
        assert_eq!(
            IntegrationPath::from_codes(&[0, 0], 1, 2).expect_err("two roots"),
            PathError::RootCount { found: 2 },
            "a walk starts in exactly one place"
        );
        assert_eq!(
            IntegrationPath::from_codes(
                &[ParentDirection::Right.code(), ParentDirection::Left.code()],
                1,
                2
            )
            .expect_err("no root"),
            PathError::RootCount { found: 0 },
            "and it has to start somewhere"
        );

        // A pair pointing at each other never reaches the root.
        let cycle = IntegrationPath::from_codes(
            &[
                ParentDirection::Root.code(),
                ParentDirection::Right.code(),
                ParentDirection::Left.code(),
            ],
            1,
            3,
        );
        assert!(
            matches!(cycle, Err(PathError::Cycle { .. })),
            "a loop is not a walk, got {cycle:?}"
        );

        assert_eq!(
            IntegrationPath::from_codes(&[0, 9], 1, 2).expect_err("9 is not a direction"),
            PathError::BadDirection {
                row: 0,
                col: 1,
                code: 9
            },
            "only 0..=4 name a neighbour"
        );
        assert_eq!(
            IntegrationPath::from_codes(&[0, ParentDirection::Right.code()], 1, 2)
                .expect_err("the last pixel has nothing to its right"),
            PathError::ParentOutOfBounds { row: 0, col: 1 },
            "a parent outside the field is not a parent"
        );
        assert_eq!(
            IntegrationPath::from_codes(&[0], 2, 2).expect_err("one entry for four pixels"),
            PathError::LengthMismatch {
                rows: 2,
                cols: 2,
                len: 1
            },
            "there must be exactly one direction per pixel"
        );
    }

    /// Residues and disagreeing edges never needed a path, so they are
    /// unchanged without one. Only the edges' roles become unknown.
    #[test]
    fn an_unwrapping_without_a_path_still_measures_everything_it_can() {
        let (rows, cols) = (12, 10);
        let psi = wrapped_of(&crate::demo::noisy_ramp(rows, cols, 3.0, 1.5, 0.9, 11));
        let path = IntegrationPath::comb(rows, cols);
        let phi = integrate(&psi, &path).expect("the comb spans the grid");

        let with = Unwrapping::new(&psi, phi.clone(), Some(path)).expect("valid");
        let without = Unwrapping::new(&psi, phi, None).expect("a path is optional");

        assert!(with.has_path(), "one was given");
        assert!(!without.has_path(), "and one was not");

        assert_eq!(
            with.stats().inconsistent_edges,
            without.stats().inconsistent_edges,
            "the disagreeing edges come from psi and phi alone"
        );
        assert_eq!(
            (
                with.stats().positive_residues,
                with.stats().negative_residues
            ),
            (
                without.stats().positive_residues,
                without.stats().negative_residues
            ),
            "and the residues from psi alone"
        );

        assert_eq!(
            without.stats().tree_edges,
            0,
            "with no path there is no path to count"
        );
        assert_eq!(
            without
                .horizontal_edge(0, 0)
                .expect("the edge exists")
                .traversal,
            None,
            "and no edge can be said to have a role in one"
        );
        assert!(
            with.stats().tree_edges > 0,
            "whereas a supplied path does have edges"
        );
    }

    #[test]
    fn a_path_of_the_wrong_shape_is_rejected() {
        let psi = wrapped_of(&ramp(3, 3));
        let phi = ramp(3, 3);

        let wrong_shape = Unwrapping::new(&psi, phi, Some(IntegrationPath::comb(4, 4)))
            .expect_err("a 4 × 4 path cannot describe a 3 × 3 field");
        assert_eq!(
            wrong_shape,
            UnwrappingError::PathSizeMismatch {
                field: (3, 3),
                path: (4, 4)
            },
            "a path has to cover the pixels it claims to walk"
        );
    }

    #[test]
    fn a_size_mismatch_is_rejected() {
        let psi = wrapped_of(&ramp(3, 3));
        let wrong = ramp(3, 4);
        assert_eq!(
            Unwrapping::new(&psi, wrong, Some(IntegrationPath::comb(3, 3)))
                .expect_err("the candidate is a different shape"),
            UnwrappingError::SizeMismatch {
                wrapped: (3, 3),
                unwrapped: (3, 4)
            },
            "the candidate must cover the same pixels"
        );
    }

    #[test]
    fn accessors_reject_edges_and_corners_outside_the_field() {
        let psi = wrapped_of(&ramp(3, 4));
        let phi = integrate(&psi, &IntegrationPath::comb(3, 4)).expect("spanning tree");
        let unwrapping =
            Unwrapping::new(&psi, phi, Some(IntegrationPath::comb(3, 4))).expect("spanning tree");

        assert!(
            unwrapping.horizontal_edge(0, 2).is_some(),
            "the last horizontal edge of a row is at col = cols - 2"
        );
        assert!(
            unwrapping.horizontal_edge(0, 3).is_none(),
            "there is no edge leaving the last column to the right"
        );
        assert!(
            unwrapping.vertical_edge(1, 0).is_some(),
            "the last vertical edge of a column is at row = rows - 2"
        );
        assert!(
            unwrapping.vertical_edge(2, 0).is_none(),
            "there is no edge leaving the last row downwards"
        );
        assert!(
            unwrapping.residue(1, 2).is_some(),
            "inner corners are (rows-1) × (cols-1)"
        );
        assert!(
            unwrapping.residue(2, 0).is_none(),
            "there is no inner corner below the last row"
        );
    }
}

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
//! `G*` has one vertex per inner corner — `(m-1) × (n-1)` of them — plus a
//! single vertex `O` standing for the whole outer boundary. Each inner corner
//! carries the residue of the four pixels around it.
//!
//! # What gets highlighted
//!
//! For every edge, integration *should* move the phase by the wrapped
//! difference `wrap(ψ_b - ψ_a)`. Along the tree that holds by construction, but
//! a candidate unwrapping is only a candidate: this module measures
//! `(Δφ - Δψ) / 2π` on *every* edge rather than assuming it, and a non-zero
//! result is what the viewer draws in the highlight colour.

use std::collections::VecDeque;
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

/// Texel code meaning "no edge here": the right edge of the last column and
/// the bottom edge of the last row. The shader treats those positions as the
/// outer boundary instead.
pub const EDGE_ABSENT: u8 = 3;

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
    /// The edge's role in the integration path.
    pub traversal: Traversal,

    /// `(Δφ - Δψ) / 2π`, rounded.
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
    /// A spanning tree of `mn` pixels has exactly `mn - 1` edges.
    WrongEdgeCount {
        /// How many edges a spanning tree would need.
        expected: usize,
        /// How many were supplied.
        got: usize,
    },
    /// The edges do not connect every pixel to the seed, so they are not a
    /// spanning tree (they contain a cycle, or leave the graph disconnected).
    NotSpanning {
        /// How many pixels the walk reached.
        reached: usize,
        /// How many pixels there are.
        total: usize,
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
            Self::WrongEdgeCount { expected, got } => {
                write!(f, "a spanning tree needs {expected} edges, got {got}")
            }
            Self::NotSpanning { reached, total } => write!(
                f,
                "the integration path reaches {reached} of {total} pixels, so it is not a spanning tree"
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
    stats: UnwrappingStats,
}

impl Unwrapping {
    /// Analyses `unwrapped` as a candidate unwrapping of `wrapped`, integrated
    /// along `tree` starting from pixel `(0, 0)`.
    ///
    /// # Errors
    ///
    /// Returns [`UnwrappingError`] if the two fields disagree on size, if the
    /// field is empty, or if `tree` is not a spanning tree of the pixel graph.
    pub fn new(
        wrapped: &PhaseField,
        unwrapped: PhaseField,
        tree: &[EdgeId],
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

        let mut horizontal = vec![EdgeState::default(); rows * cols.saturating_sub(1)];
        let mut vertical = vec![EdgeState::default(); rows.saturating_sub(1) * cols];

        let traversals = walk_tree(rows, cols, tree)?;
        for (edge, traversal) in tree.iter().zip(traversals) {
            match edge.axis {
                Axis::Horizontal => {
                    horizontal[edge.row * (cols - 1) + edge.col].traversal = traversal;
                }
                Axis::Vertical => {
                    vertical[edge.row * cols + edge.col].traversal = traversal;
                }
            }
        }

        for row in 0..rows {
            for col in 0..cols.saturating_sub(1) {
                horizontal[row * (cols - 1) + col].jump =
                    jump(wrapped, &unwrapped, (row, col), (row, col + 1));
            }
        }
        for row in 0..rows.saturating_sub(1) {
            for col in 0..cols {
                vertical[row * cols + col].jump =
                    jump(wrapped, &unwrapped, (row, col), (row + 1, col));
            }
        }

        let residues = residue_field(wrapped);

        let stats = UnwrappingStats {
            tree_edges: tree.len(),
            cut_edges: horizontal.len() + vertical.len() - tree.len(),
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
            stats,
        })
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
    /// those slots carry [`EDGE_ABSENT`].
    pub fn edge_texels(&self) -> Vec<u8> {
        let (rows, cols) = (self.rows(), self.cols());
        let mut texels = Vec::with_capacity(rows * cols * 4);
        for row in 0..rows {
            for col in 0..cols {
                for state in [self.horizontal_edge(row, col), self.vertical_edge(row, col)] {
                    let (traversal, jump) = state.map_or((EDGE_ABSENT, 0), |state| {
                        (
                            state.traversal.code(),
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

/// Walks the supplied edges from pixel `(0, 0)`, returning how each one is
/// traversed, in the order the edges were given.
///
/// This doubles as the spanning-tree check: with exactly `mn - 1` edges, a walk
/// that reaches every pixel can only have been over a tree.
fn walk_tree(rows: usize, cols: usize, tree: &[EdgeId]) -> Result<Vec<Traversal>, UnwrappingError> {
    let pixels = rows * cols;
    let expected = pixels - 1;
    if tree.len() != expected {
        return Err(UnwrappingError::WrongEdgeCount {
            expected,
            got: tree.len(),
        });
    }

    // Adjacency, carrying the index of the edge that produced each link so the
    // walk can report the traversal of the caller's own edge list.
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); pixels];
    for (index, edge) in tree.iter().enumerate() {
        let ((ar, ac), (br, bc)) = edge.endpoints();
        if br >= rows || bc >= cols {
            return Err(UnwrappingError::EdgeOutOfBounds(*edge));
        }
        let (a, b) = (ar * cols + ac, br * cols + bc);
        adjacency[a].push((b, index));
        adjacency[b].push((a, index));
    }

    let mut traversals = vec![Traversal::Cut; tree.len()];
    let mut visited = vec![false; pixels];
    let mut queue = VecDeque::new();

    visited[0] = true;
    queue.push_back(0usize);
    let mut reached = 1;

    while let Some(pixel) = queue.pop_front() {
        for &(neighbour, index) in &adjacency[pixel] {
            if visited[neighbour] {
                continue;
            }
            visited[neighbour] = true;
            reached += 1;
            // The walk entered `neighbour` from `pixel`; the edge is forward
            // when `pixel` is also its canonical source.
            let ((ar, ac), _) = tree[index].endpoints();
            traversals[index] = if ar * cols + ac == pixel {
                Traversal::Forward
            } else {
                Traversal::Backward
            };
            queue.push_back(neighbour);
        }
    }

    if reached != pixels {
        return Err(UnwrappingError::NotSpanning {
            reached,
            total: pixels,
        });
    }
    Ok(traversals)
}

/// `(Δφ - Δψ) / 2π` on the edge from `a` to `b`, rounded.
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
/// Sums the wrapped differences clockwise on screen — right, down, left, up —
/// and divides by `2π`. Each term lies in `(-π, π]`, so the sum is strictly
/// inside `(-4π, 4π)` for real data and the charge is always `-1`, `0` or `+1`.
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

    use crate::demo::{comb_tree as comb, integrate};

    fn ramp(rows: usize, cols: usize) -> PhaseField {
        PhaseField::linear_gradient(rows, cols, 1.5, 0.75)
    }

    fn wrapped_of(field: &PhaseField) -> PhaseField {
        let data = field.as_slice().iter().copied().map(phase::wrap).collect();
        PhaseField::new(data, field.rows(), field.cols()).expect("same dimensions")
    }

    #[test]
    fn a_spanning_tree_has_the_edge_count_the_algebra_predicts() {
        for (rows, cols) in [(5, 4), (1, 7), (9, 1), (2, 2)] {
            let tree = comb(rows, cols);
            assert_eq!(
                tree.len(),
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
                total - tree.len(),
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
        let tree = comb(rows, cols);

        let phi = integrate(&psi, &tree).expect("the comb is a spanning tree");
        let unwrapping = Unwrapping::new(&psi, phi, &tree).expect("the comb is a spanning tree");

        for edge in &tree {
            let state = match edge.axis {
                Axis::Horizontal => unwrapping.horizontal_edge(edge.row, edge.col),
                Axis::Vertical => unwrapping.vertical_edge(edge.row, edge.col),
            }
            .expect("the edge is inside the field");
            assert!(
                state.is_consistent(),
                "tree edge {edge:?} must move the phase by exactly the wrapped delta"
            );
            assert!(
                state.traversal.is_in_tree(),
                "tree edge {edge:?} must be marked as part of the path"
            );
        }
    }

    #[test]
    fn a_residue_free_field_unwraps_with_no_inconsistent_edges() {
        let (rows, cols) = (11, 13);
        // A gentle ramp: consecutive samples are far less than π apart, so no
        // loop can accumulate a full turn and there are no residues.
        let truth = PhaseField::linear_gradient(rows, cols, 1.0, 0.5);
        let psi = wrapped_of(&truth);
        let tree = comb(rows, cols);

        let phi = integrate(&psi, &tree).expect("spanning tree");
        let unwrapping = Unwrapping::new(&psi, phi, &tree).expect("spanning tree");
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
        // around it sum to 2π, so the corner has charge +1.
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
        let tree = comb(rows, cols);
        let phi = integrate(&psi, &tree).expect("spanning tree");
        let unwrapping = Unwrapping::new(&psi, phi, &tree).expect("spanning tree");
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
    /// survives. The total charge is therefore the boundary circulation — the
    /// charge that the dual vertex `O` carries — and *not* zero in general.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the circulation of a finite field is a small multiple of 2π"
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
    fn the_walk_orients_every_tree_edge_away_from_the_seed() {
        let (rows, cols) = (4, 3);
        let tree = comb(rows, cols);
        let traversals = walk_tree(rows, cols, &tree).expect("the comb spans the grid");

        assert_eq!(traversals.len(), tree.len(), "one traversal per edge");
        assert!(
            traversals.iter().all(|t| *t == Traversal::Forward),
            "the comb is built from (0,0) outwards, so every edge is walked forwards"
        );
    }

    #[test]
    fn a_tree_given_against_the_grain_is_walked_backwards() {
        // A 1 × 3 row whose only spanning tree is the two horizontal edges,
        // seeded at (0, 0): both are walked forwards.
        let forwards = walk_tree(1, 3, &[EdgeId::horizontal(0, 0), EdgeId::horizontal(0, 1)])
            .expect("spans the row");
        assert_eq!(
            forwards,
            vec![Traversal::Forward, Traversal::Forward],
            "walking rightwards from the seed follows the canonical orientation"
        );

        // A 3 × 1 column seeded at (0, 0) walks downwards, which is also
        // canonical; to get a backward edge the seed must be reached from below.
        let column = walk_tree(1, 3, &[EdgeId::horizontal(0, 1), EdgeId::horizontal(0, 0)])
            .expect("spans the row");
        assert_eq!(
            column,
            vec![Traversal::Forward, Traversal::Forward],
            "edge order must not change how the walk orients them"
        );
    }

    #[test]
    fn a_non_tree_is_rejected_rather_than_half_analysed() {
        let psi = wrapped_of(&ramp(3, 3));
        let phi = ramp(3, 3);

        let too_few = Unwrapping::new(&psi, phi.clone(), &[EdgeId::horizontal(0, 0)])
            .expect_err("one edge cannot span nine pixels");
        assert_eq!(
            too_few,
            UnwrappingError::WrongEdgeCount {
                expected: 8,
                got: 1
            },
            "a 3 × 3 spanning tree needs 8 edges"
        );

        // Eight edges, but one component is cut off and another has a cycle.
        let disconnected = vec![
            EdgeId::horizontal(0, 0),
            EdgeId::horizontal(0, 1),
            EdgeId::vertical(0, 0),
            EdgeId::vertical(0, 1),
            EdgeId::horizontal(1, 0),
            EdgeId::horizontal(1, 1),
            EdgeId::horizontal(2, 0),
            EdgeId::horizontal(2, 1),
        ];
        assert!(
            matches!(
                Unwrapping::new(&psi, phi.clone(), &disconnected),
                Err(UnwrappingError::NotSpanning { .. })
            ),
            "the bottom row is unreachable, so this is not a spanning tree"
        );

        let out_of_bounds = {
            let mut tree = comb(3, 3);
            tree[0] = EdgeId::horizontal(0, 9);
            tree
        };
        assert_eq!(
            Unwrapping::new(&psi, phi, &out_of_bounds).expect_err("the edge leaves the field"),
            UnwrappingError::EdgeOutOfBounds(EdgeId::horizontal(0, 9)),
            "an edge leaving the field must be rejected by name"
        );
    }

    #[test]
    fn a_size_mismatch_is_rejected() {
        let psi = wrapped_of(&ramp(3, 3));
        let wrong = ramp(3, 4);
        assert_eq!(
            Unwrapping::new(&psi, wrong, &comb(3, 3))
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
        let phi = integrate(&psi, &comb(3, 4)).expect("spanning tree");
        let unwrapping = Unwrapping::new(&psi, phi, &comb(3, 4)).expect("spanning tree");

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

//! Synthetic data, so the viewer has something to show.
//!
//! The viewer itself ingests a candidate unwrapping and an integration path
//! from outside; this module only manufactures a plausible pair. It contains
//! the one piece of unwrapping *algorithm* in the crate — [`integrate`], the
//! naive path integration — which exists to produce a candidate worth looking
//! at, not as the project's answer to the problem.

use std::f32::consts::TAU;

use crate::graph::{Axis, EdgeId, Unwrapping, UnwrappingError};
use crate::phase::{self, PhaseField};

/// A small deterministic generator, so a given seed always paints the same
/// field and a screenshot can be reproduced. Not suitable for anything but
/// making pictures.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Run the seed through the SplitMix64 finalizer first. Seeding the
        // xorshift state directly would make adjacent seeds produce closely
        // related streams, and `| 1` alone would collapse 42 and 43 onto the
        // same state entirely. The final `| 1` only rules out zero, which is
        // xorshift's fixed point.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*, Vigna 2016.
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        // Top 24 bits: exactly the mantissa an f32 can hold without rounding.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Roughly standard normal, by the Irwin–Hall construction. Close enough
    /// for speckle, and it needs no transcendental functions.
    fn next_normal(&mut self) -> f32 {
        let sum: f32 = (0..12).map(|_| self.next_unit()).sum();
        sum - 6.0
    }
}

/// A linear phase ramp with additive noise.
///
/// The ramp is what an interferogram over flat terrain looks like; the noise is
/// what turns it into a field with residues, and so into one where the
/// integration path matters.
pub fn noisy_ramp(
    rows: usize,
    cols: usize,
    cycles_x: f32,
    cycles_y: f32,
    noise: f32,
    seed: u64,
) -> PhaseField {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let v = (row as f32 + 0.5) / rows as f32;
        for col in 0..cols {
            let u = (col as f32 + 0.5) / cols as f32;
            let ramp = TAU * cycles_x.mul_add(u, cycles_y * v);
            data.push(noise.mul_add(rng.next_normal(), ramp));
        }
    }
    PhaseField::new(data, rows, cols).expect("the field was built row by row")
}

/// Wraps every sample of `field` into `(-π, π]`.
pub fn wrap_field(field: &PhaseField) -> PhaseField {
    let data = field.as_slice().iter().copied().map(phase::wrap).collect();
    PhaseField::new(data, field.rows(), field.cols()).expect("wrapping preserves the shape")
}

/// The comb spanning tree: down column 0, then across every row.
///
/// This is the classic raster integration path, and a spanning tree for any
/// `m × n` — it has `(m-1) + m(n-1) = mn - 1` edges and reaches every pixel.
pub fn comb_tree(rows: usize, cols: usize) -> Vec<EdgeId> {
    let mut tree = Vec::with_capacity((rows * cols).saturating_sub(1));
    for row in 0..rows.saturating_sub(1) {
        tree.push(EdgeId::vertical(row, 0));
    }
    for row in 0..rows {
        for col in 0..cols.saturating_sub(1) {
            tree.push(EdgeId::horizontal(row, col));
        }
    }
    tree
}

/// Integrates `wrapped` along `tree`, seeded at pixel `(0, 0)`.
///
/// Every step adds the wrapped difference across one edge, which is exactly
/// what makes each tree edge consistent by construction — and leaves the cut
/// edges free to disagree wherever a residue is enclosed.
///
/// # Errors
///
/// Returns [`UnwrappingError`] if `tree` is not a spanning tree of the field.
pub fn integrate(wrapped: &PhaseField, tree: &[EdgeId]) -> Result<PhaseField, UnwrappingError> {
    let (rows, cols) = (wrapped.rows(), wrapped.cols());
    if rows == 0 || cols == 0 {
        return Err(UnwrappingError::Empty);
    }
    if tree.len() != rows * cols - 1 {
        return Err(UnwrappingError::WrongEdgeCount {
            expected: rows * cols - 1,
            got: tree.len(),
        });
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); rows * cols];
    for edge in tree {
        let ((ar, ac), (br, bc)) = edge.endpoints();
        if br >= rows || bc >= cols {
            return Err(UnwrappingError::EdgeOutOfBounds(*edge));
        }
        let (a, b) = (ar * cols + ac, br * cols + bc);
        adjacency[a].push(b);
        adjacency[b].push(a);
    }

    let mut phase_of = vec![f32::NAN; rows * cols];
    let mut visited = vec![false; rows * cols];
    let mut stack = vec![0usize];

    phase_of[0] = wrapped.as_slice()[0];
    visited[0] = true;
    let mut reached = 1;

    while let Some(pixel) = stack.pop() {
        for &neighbour in &adjacency[pixel] {
            if visited[neighbour] {
                continue;
            }
            visited[neighbour] = true;
            reached += 1;

            let from = wrapped.as_slice()[pixel];
            let to = wrapped.as_slice()[neighbour];
            phase_of[neighbour] = phase_of[pixel] + phase::wrap(to - from);
            stack.push(neighbour);
        }
    }

    if reached != rows * cols {
        return Err(UnwrappingError::NotSpanning {
            reached,
            total: rows * cols,
        });
    }

    Ok(PhaseField::new(phase_of, rows, cols)
        .expect("the integral has one sample per pixel, by construction"))
}

/// Everything one demo needs: the underlying phase, what an instrument would
/// observe, and a candidate unwrapping of it.
#[derive(Clone, Debug)]
pub struct Scene {
    /// The phase before wrapping. Not observable in practice — it is here so
    /// the reconstruction can be compared against the thing it is reconstructing.
    pub truth: PhaseField,
    /// `wrap(truth)`: the observable.
    pub wrapped: PhaseField,
    /// A candidate unwrapping of `wrapped`, with its integration path.
    pub unwrapping: Unwrapping,
}

/// Settings behind the demo, so the sidebar can rebuild it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SceneSettings {
    /// Rows of pixels.
    pub rows: usize,
    /// Columns of pixels.
    pub cols: usize,
    /// Fringes across the field, horizontally.
    pub cycles_x: f32,
    /// Fringes down the field, vertically.
    pub cycles_y: f32,
    /// Standard deviation of the additive noise, in radians. Zero gives a
    /// residue-free field that unwraps perfectly.
    pub noise: f32,
    /// Which field a given noise level produces.
    pub seed: u64,
}

impl Default for SceneSettings {
    fn default() -> Self {
        Self {
            rows: 128,
            cols: 160,
            cycles_x: 4.0,
            cycles_y: 1.5,
            noise: 0.55,
            seed: 20_260_920,
        }
    }
}

impl Scene {
    /// Builds a demo scene.
    ///
    /// # Errors
    ///
    /// Returns [`UnwrappingError`] only if the settings describe an empty
    /// field; every other input is constructed here and is valid by
    /// construction.
    pub fn new(settings: SceneSettings) -> Result<Self, UnwrappingError> {
        let truth = noisy_ramp(
            settings.rows,
            settings.cols,
            settings.cycles_x,
            settings.cycles_y,
            settings.noise,
            settings.seed,
        );
        let wrapped = wrap_field(&truth);
        let tree = comb_tree(settings.rows, settings.cols);
        let candidate = integrate(&wrapped, &tree)?;
        let unwrapping = Unwrapping::new(&wrapped, candidate, &tree)?;

        Ok(Self {
            truth,
            wrapped,
            unwrapping,
        })
    }
}

/// The edge of `G` that the wall at a grid line corresponds to.
///
/// Walls sit on integer grid lines: the vertical wall at `x = line` between
/// rows `row` and `row + 1` separates pixels `(row, line - 1)` and
/// `(row, line)`. Lines `0` and `cols` are the outer boundary and have no edge.
pub fn wall_edge(
    rows: usize,
    cols: usize,
    axis: Axis,
    line: usize,
    along: usize,
) -> Option<EdgeId> {
    match axis {
        // A vertical wall separates two pixels side by side.
        Axis::Horizontal => {
            (line >= 1 && line < cols && along < rows).then(|| EdgeId::horizontal(along, line - 1))
        }
        // A horizontal wall separates two pixels one above the other.
        Axis::Vertical => {
            (line >= 1 && line < rows && along < cols).then(|| EdgeId::vertical(line - 1, along))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_comb_is_a_spanning_tree() {
        for (rows, cols) in [(5, 4), (1, 6), (6, 1), (2, 2), (17, 13)] {
            let tree = comb_tree(rows, cols);
            assert_eq!(
                tree.len(),
                rows * cols - 1,
                "{rows}×{cols}: a spanning tree has mn - 1 edges"
            );
            // `integrate` fails unless the edges really do reach every pixel.
            let field = noisy_ramp(rows, cols, 1.0, 1.0, 0.0, 1);
            assert!(
                integrate(&wrap_field(&field), &tree).is_ok(),
                "{rows}×{cols}: the comb must reach every pixel"
            );
        }
    }

    #[test]
    fn integration_reproduces_a_field_that_was_never_wrapped() {
        // Steps well under π, so wrapping loses nothing and the integral must
        // return the original ramp up to the constant it was seeded with.
        let (rows, cols) = (8, 9);
        let truth = noisy_ramp(rows, cols, 0.5, 0.25, 0.0, 1);
        let wrapped = wrap_field(&truth);
        let recovered = integrate(&wrapped, &comb_tree(rows, cols)).expect("spanning tree");

        let offset = recovered.as_slice()[0] - truth.as_slice()[0];
        for (i, (&got, &want)) in recovered
            .as_slice()
            .iter()
            .zip(truth.as_slice())
            .enumerate()
        {
            assert!(
                (got - (want + offset)).abs() < 1e-3,
                "sample {i}: expected {} got {got}",
                want + offset
            );
        }
    }

    #[test]
    fn noise_creates_residues_and_zero_noise_does_not() {
        let clean = Scene::new(SceneSettings {
            rows: 32,
            cols: 32,
            noise: 0.0,
            ..SceneSettings::default()
        })
        .expect("non-empty");
        let clean_stats = clean.unwrapping.stats();
        assert_eq!(
            (clean_stats.positive_residues, clean_stats.negative_residues),
            (0, 0),
            "a noise-free ramp has no residues"
        );
        assert_eq!(
            clean_stats.inconsistent_edges, 0,
            "and so nothing for the viewer to highlight"
        );

        let noisy = Scene::new(SceneSettings {
            rows: 32,
            cols: 32,
            noise: 0.9,
            ..SceneSettings::default()
        })
        .expect("non-empty");
        let noisy_stats = noisy.unwrapping.stats();
        assert!(
            noisy_stats.positive_residues > 0,
            "noise should produce residues, or the demo shows nothing"
        );
        assert!(
            noisy_stats.inconsistent_edges > 0,
            "and residues force edges to disagree"
        );
    }

    /// The bounds quoted in the theory, checked on real generated data.
    #[test]
    fn inconsistent_edges_respect_both_bounds() {
        for noise in [0.2, 0.5, 0.9, 1.4] {
            let scene = Scene::new(SceneSettings {
                rows: 40,
                cols: 36,
                noise,
                ..SceneSettings::default()
            })
            .expect("non-empty");
            let stats = scene.unwrapping.stats();

            assert!(
                stats.inconsistent_edges <= (40 - 1) * (36 - 1),
                "noise {noise}: at most (m-1)(n-1) edges can disagree, got {}",
                stats.inconsistent_edges
            );
            assert!(
                stats.inconsistent_edges >= stats.positive_residues.max(stats.negative_residues),
                "noise {noise}: at least max(N+, N-) must disagree, got {} with {}/{} charges",
                stats.inconsistent_edges,
                stats.positive_residues,
                stats.negative_residues
            );
        }
    }

    #[test]
    fn the_generator_is_reproducible() {
        let a = noisy_ramp(8, 8, 1.0, 1.0, 0.7, 42);
        let b = noisy_ramp(8, 8, 1.0, 1.0, 0.7, 42);
        assert_eq!(a, b, "the same seed must paint the same field");

        let c = noisy_ramp(8, 8, 1.0, 1.0, 0.7, 43);
        assert_ne!(a, c, "a different seed must not");
    }

    #[test]
    fn wrapping_lands_every_sample_in_the_interval() {
        use std::f32::consts::PI;
        let wrapped = wrap_field(&noisy_ramp(16, 16, 3.0, 3.0, 1.0, 5));
        for &value in wrapped.as_slice() {
            assert!(-PI < value && value <= PI, "{value} escaped (-π, π]");
        }
    }

    #[test]
    fn wall_edges_map_grid_lines_to_the_pixels_they_separate() {
        let (rows, cols) = (5, 4);

        assert_eq!(
            wall_edge(rows, cols, Axis::Horizontal, 1, 0),
            Some(EdgeId::horizontal(0, 0)),
            "the wall at x = 1 in row 0 separates (0,0) from (0,1)"
        );
        assert_eq!(
            wall_edge(rows, cols, Axis::Horizontal, 0, 0),
            None,
            "x = 0 is the outer boundary, not an edge of G"
        );
        assert_eq!(
            wall_edge(rows, cols, Axis::Horizontal, cols, 0),
            None,
            "x = cols is the outer boundary too"
        );

        assert_eq!(
            wall_edge(rows, cols, Axis::Vertical, 2, 3),
            Some(EdgeId::vertical(1, 3)),
            "the wall at y = 2 in column 3 separates (1,3) from (2,3)"
        );
        assert_eq!(
            wall_edge(rows, cols, Axis::Vertical, rows, 0),
            None,
            "y = rows is the bottom boundary"
        );

        // Every interior grid line must name an edge, and the count must match
        // the algebra.
        let mut named = 0;
        for line in 0..=cols {
            for along in 0..rows {
                named +=
                    usize::from(wall_edge(rows, cols, Axis::Horizontal, line, along).is_some());
            }
        }
        for line in 0..=rows {
            for along in 0..cols {
                named += usize::from(wall_edge(rows, cols, Axis::Vertical, line, along).is_some());
            }
        }
        assert_eq!(
            named,
            2 * rows * cols - rows - cols,
            "the walls must name every edge of G exactly once"
        );
    }
}

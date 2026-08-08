//! Inv 1 — conserved identity. The index is the agent's self-graph: vertices
//! are documents, edge weights are floored shared-vocabulary affinities. We
//! compute two things:
//!
//!   * a **fingerprint** — a blake3 digest over a canonical (relabelling-
//!     invariant) encoding of the graph. This is the "computed invariant
//!     unchanged under relabelling" the blueprint requires. Cheap and always
//!     exact.
//!   * **chi** = Char(A), the least-weight bipartition cut (Stoer-Wagner). This
//!     is the paper's character invariant proper; used for display and the
//!     `>= floor > 0` conformance check.
//!
//! Vertices are keyed by document `content_hash`, so reordering documents,
//! renumbering ids, or permuting scenes never changes the encoding.

use crate::index::schema::{Document, Identity};

/// Positive weight floor c > 0 (paper's beta).
///
/// This is an **imposed construction parameter, not a measured property of the
/// corpus**. Adding it to every edge is what makes the object a contact graph
/// in the first place: a weighted graph permitting `wt(e) -> 0` is not one
/// (`instantiation-of-finite-weighted-graphs.tex`, rem:floor-content), so
/// without the floor there is no guarantee the min cut stays positive.
///
/// Read the consequence precisely. Because every edge carries FLOOR, the graph
/// is complete and *every* cut weighs at least FLOOR by arithmetic. So
/// `chi >= floor` is **true by construction and cannot fail on any input** — it
/// is an axiom the builder enforces, not a fact about your documents. An
/// earlier version of this comment claimed the floor made "chi >= floor hold
/// exactly (thm:identity(i))", which reads as though the inequality were a
/// verified result; that phrasing invited a tautological conformance check.
///
/// What *does* carry corpus signal is the weight distribution above the floor —
/// see [`EdgeStats::frac_at_floor`], which reports the fraction of document
/// pairs sitting at exactly FLOOR (i.e. sharing no vocabulary at all).
pub const FLOOR: f64 = 1e-6;

/// Quantisation for edge weights in the fingerprint, so floating-point noise
/// does not change the digest while genuine structural change does.
const WEIGHT_QUANTUM: f64 = 1e-4;

/// The self-graph derived from an index.
pub struct SelfGraph {
    /// Vertex content-hash keys, in canonical (sorted) order.
    pub verts: Vec<String>,
    /// Upper-triangular edges (i < j) with weight, i/j into `verts`.
    pub edges: Vec<(usize, usize, f64)>,
}

/// Cosine-free affinity: floored sum over shared vocab of min term frequency,
/// aggregated per document (summed across its passages). Keeps the graph small
/// (one vertex per document) while reflecting real vocabulary overlap.
fn doc_term_vector(doc: &Document) -> Vec<(u32, u32)> {
    // Merge passage term vectors into a per-document bag.
    let mut bag: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for p in &doc.passages {
        for &(tid, tf) in &p.terms {
            *bag.entry(tid).or_insert(0) += tf;
        }
    }
    bag.into_iter().collect()
}

/// Shared-vocab affinity between two sorted term bags: sum of min tf over the
/// intersection. Both inputs are sorted by term id.
fn affinity(a: &[(u32, u32)], b: &[(u32, u32)]) -> f64 {
    let mut i = 0;
    let mut j = 0;
    let mut acc = 0u64;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                acc += a[i].1.min(b[j].1) as u64;
                i += 1;
                j += 1;
            }
        }
    }
    acc as f64
}

/// Build the self-graph from documents. Complete graph on the floor; genuine
/// vocabulary overlap raises individual edges above it.
pub fn build_self_graph(documents: &[Document]) -> SelfGraph {
    // Canonical vertex order: sort document indices by content_hash.
    let mut order: Vec<usize> = (0..documents.len()).collect();
    order.sort_by(|&a, &b| documents[a].content_hash.cmp(&documents[b].content_hash));

    let verts: Vec<String> = order
        .iter()
        .map(|&i| documents[i].content_hash.clone())
        .collect();
    let vectors: Vec<Vec<(u32, u32)>> = order.iter().map(|&i| doc_term_vector(&documents[i])).collect();

    let n = verts.len();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let w = FLOOR + affinity(&vectors[i], &vectors[j]);
            edges.push((i, j, w));
        }
    }
    SelfGraph { verts, edges }
}

/// Canonical fingerprint: blake3 over sorted vertex hashes then the sorted,
/// quantised edge multiset keyed by the endpoint hashes (not indices). Any
/// relabelling that preserves separations and weights yields the same bytes.
pub fn fingerprint(g: &SelfGraph) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"spraypaint-selfgraph-v1");

    // Vertices, already in canonical (sorted-hash) order.
    hasher.update(&(g.verts.len() as u64).to_le_bytes());
    for v in &g.verts {
        hasher.update(&(v.len() as u64).to_le_bytes());
        hasher.update(v.as_bytes());
    }

    // Edges keyed by endpoint hashes so they are index-independent.
    let mut edge_keys: Vec<(String, String, u64)> = g
        .edges
        .iter()
        .map(|&(i, j, w)| {
            let (a, b) = if g.verts[i] <= g.verts[j] {
                (g.verts[i].clone(), g.verts[j].clone())
            } else {
                (g.verts[j].clone(), g.verts[i].clone())
            };
            let q = (w / WEIGHT_QUANTUM).round() as u64;
            (a, b, q)
        })
        .collect();
    edge_keys.sort();
    hasher.update(&(edge_keys.len() as u64).to_le_bytes());
    for (a, b, q) in &edge_keys {
        hasher.update(a.as_bytes());
        hasher.update(b.as_bytes());
        hasher.update(&q.to_le_bytes());
    }

    format!("b3:{}", hasher.finalize().to_hex())
}

/// Char(A) = global minimum cut of the weighted self-graph (Stoer-Wagner).
/// For n < 2 there is no bipartition; return the floor as a degenerate value.
pub fn char_invariant(g: &SelfGraph) -> f64 {
    let n = g.verts.len();
    if n < 2 {
        return FLOOR;
    }
    // Dense weight matrix.
    let mut w = vec![vec![0.0f64; n]; n];
    for &(i, j, weight) in &g.edges {
        w[i][j] = weight;
        w[j][i] = weight;
    }

    // Stoer-Wagner global min cut.
    let mut vertices: Vec<usize> = (0..n).collect();
    let mut best = f64::INFINITY;
    // Merged-weight working matrix.
    let mut mat = w;
    let mut active = vertices.len();
    // Map of "super-vertex" membership is implicit via matrix shrink using a
    // present[] mask.
    let mut present = vec![true; n];

    while active > 1 {
        // Minimum cut phase.
        let mut a_added = vec![false; n];
        let mut weights = vec![0.0f64; n];
        let mut prev = usize::MAX;
        let mut last = usize::MAX;
        for _ in 0..active {
            // Pick the most tightly connected not-yet-added present vertex.
            let mut sel = usize::MAX;
            let mut sel_w = -1.0;
            for v in 0..n {
                if present[v] && !a_added[v] && weights[v] > sel_w {
                    sel_w = weights[v];
                    sel = v;
                }
            }
            if sel == usize::MAX {
                break;
            }
            a_added[sel] = true;
            prev = last;
            last = sel;
            for v in 0..n {
                if present[v] && !a_added[v] {
                    weights[v] += mat[sel][v];
                }
            }
        }
        // cut-of-the-phase = weight of `last` into the rest.
        if last != usize::MAX {
            best = best.min(weights[last]);
        }
        // Merge `last` into `prev`.
        if prev != usize::MAX && last != usize::MAX {
            for v in 0..n {
                if present[v] && v != prev && v != last {
                    mat[prev][v] += mat[last][v];
                    mat[v][prev] += mat[v][last];
                }
            }
            present[last] = false;
            active -= 1;
        } else {
            break;
        }
    }
    let _ = &mut vertices;
    if best.is_finite() {
        best
    } else {
        FLOOR
    }
}

/// Assemble the full `Identity` block for storage.
///
/// The output shape is deliberately frozen: it is what the fingerprint is
/// computed over and what `load()` re-checks, so adding a field here changes
/// every existing index's identity. [`EdgeStats`] and [`Degeneracy`] are
/// computed on demand by `verify` instead, and are not stored.
pub fn compute_identity(documents: &[Document]) -> Identity {
    let g = build_self_graph(documents);
    let fp = fingerprint(&g);
    let chi = char_invariant(&g);
    Identity {
        fingerprint: fp,
        char_invariant: chi,
        floor: FLOOR,
        n_vertices: g.verts.len() as u32,
        n_edges: g.edges.len() as u32,
    }
}

/// Descriptive statistics over the self-graph's edge weights.
///
/// These exist because `chi >= floor` is an arithmetic certainty (see [`FLOOR`])
/// and therefore says nothing about the corpus. The *distribution* of weights
/// above the floor does. This is the honest replacement for an asserted
/// infimum: rather than claiming a lower bound the construction already
/// guarantees, report how much of the graph sits at that bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeStats {
    pub n_edges: usize,
    pub min_w: f64,
    pub mean_w: f64,
    pub max_w: f64,
    /// Fraction of edges at exactly FLOOR — document pairs sharing *zero*
    /// vocabulary. At `1.0` the graph is the floor and nothing else, so chi is
    /// FLOOR by fiat and carries no corpus signal whatsoever.
    pub frac_at_floor: f64,
}

/// Edge-weight statistics for a self-graph. An edgeless graph (n < 2) reports
/// zeroes and `frac_at_floor = 1.0` — there is no weight mass above the floor
/// because there is no weight mass at all, which is the degenerate reading.
pub fn edge_stats(g: &SelfGraph) -> EdgeStats {
    let n = g.edges.len();
    if n == 0 {
        return EdgeStats {
            n_edges: 0,
            min_w: 0.0,
            mean_w: 0.0,
            max_w: 0.0,
            frac_at_floor: 1.0,
        };
    }
    let mut min_w = f64::INFINITY;
    let mut max_w = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut at_floor = 0usize;
    for &(_, _, w) in &g.edges {
        min_w = min_w.min(w);
        max_w = max_w.max(w);
        sum += w;
        // `affinity` accumulates integer term counts, so an edge is at the
        // floor exactly when the shared-vocabulary sum was 0 — an exact
        // comparison is correct here, not a tolerance question.
        if w == FLOOR {
            at_floor += 1;
        }
    }
    EdgeStats {
        n_edges: n,
        min_w,
        mean_w: sum / n as f64,
        max_w,
        frac_at_floor: at_floor as f64 / n as f64,
    }
}

/// Regimes in which a PASS from `verify` would not be evidence of anything.
///
/// Each of these makes some invariant vacuously true rather than verified, so
/// reporting a bare PASS would overstate what was checked (`prin:refusal` — a
/// framework with no refusals has an empty defined class). `verify` surfaces
/// these as NOT-APPLICABLE and exits 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Degeneracy {
    /// An empty index: no graph exists, so every graph invariant is vacuous.
    NoDocuments,
    /// One document: `char_invariant` returns FLOOR without ever examining the
    /// graph, because there is no bipartition to cut.
    SingleDocument,
    /// Every edge sits at FLOOR: no pair of documents shares any vocabulary, so
    /// chi is the floor by fiat and reflects no corpus structure.
    FloorOnlyGraph,
}

impl Degeneracy {
    /// A short human-readable reason, used verbatim in `verify` output.
    pub fn reason(self) -> &'static str {
        match self {
            Degeneracy::NoDocuments => "index contains no documents: every graph invariant is vacuous",
            Degeneracy::SingleDocument => {
                "index contains a single document: chi is the floor by definition (no bipartition exists)"
            }
            Degeneracy::FloorOnlyGraph => {
                "every edge is at the floor: no two documents share vocabulary, so chi carries no corpus signal"
            }
        }
    }
}

/// Classify a self-graph's degenerate regime, if it is in one.
///
/// Ordered most-degenerate first, and they nest: an empty graph is also
/// floor-only under [`edge_stats`]'s convention, so returning the *strongest*
/// applicable description is what makes the message useful.
pub fn classify(g: &SelfGraph) -> Option<Degeneracy> {
    match g.verts.len() {
        0 => Some(Degeneracy::NoDocuments),
        1 => Some(Degeneracy::SingleDocument),
        _ => {
            if edge_stats(g).frac_at_floor == 1.0 {
                Some(Degeneracy::FloorOnlyGraph)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n: usize, edges: &[(usize, usize, f64)]) -> SelfGraph {
        SelfGraph {
            verts: (0..n).map(|i| format!("v{i}")).collect(),
            edges: edges.to_vec(),
        }
    }

    /// Exhaustive min cut by enumerating every bipartition. Exponential, so
    /// only usable for tiny n — which is exactly what makes it a trustworthy
    /// oracle: it has no algorithmic cleverness to get wrong.
    fn bruteforce_mincut(g: &SelfGraph) -> f64 {
        let n = g.verts.len();
        assert!(n >= 2 && n <= 7, "oracle is exponential; keep n small");
        let mut best = f64::INFINITY;
        // Fix vertex 0 on the left to avoid counting each cut twice; the empty
        // and full sets are excluded because a cut needs both sides non-empty.
        for mask in 0u32..(1 << (n - 1)) {
            let side = |v: usize| -> bool {
                if v == 0 { true } else { mask & (1 << (v - 1)) != 0 }
            };
            if (1..n).all(side) {
                continue; // right side empty
            }
            let mut w = 0.0;
            for &(i, j, weight) in &g.edges {
                if side(i) != side(j) {
                    w += weight;
                }
            }
            best = best.min(w);
        }
        best
    }

    /// Cross-check Stoer-Wagner against exhaustive enumeration.
    ///
    /// This is a *second, structurally independent* algorithm, which is the
    /// point: `rem:selfconsistency` warns that an optimality check is only
    /// informative if it is not the optimality condition the implementation was
    /// built from. Re-running Stoer-Wagner and comparing to itself would prove
    /// nothing. Enumeration shares no code path with the merge-and-contract
    /// loop, so agreement is real evidence.
    #[test]
    fn stoer_wagner_matches_bruteforce_mincut() {
        // Hand-picked shapes plus a deterministic pseudo-random sweep. The
        // cases matter more than the count: a bridge, a lopsided star, and a
        // near-uniform clique exercise different phase-merge orders.
        let mut cases: Vec<SelfGraph> = vec![
            // Two triangles joined by one light bridge: min cut is the bridge.
            graph(6, &[
                (0, 1, 5.0), (0, 2, 5.0), (1, 2, 5.0),
                (3, 4, 5.0), (3, 5, 5.0), (4, 5, 5.0),
                (2, 3, 0.25),
            ]),
            // Star: min cut isolates the cheapest leaf.
            graph(5, &[(0, 1, 3.0), (0, 2, 1.5), (0, 3, 9.0), (0, 4, 4.0)]),
            // Complete graph, all equal: any single vertex cuts at (n-1)*w.
            graph(4, &[
                (0, 1, 2.0), (0, 2, 2.0), (0, 3, 2.0),
                (1, 2, 2.0), (1, 3, 2.0), (2, 3, 2.0),
            ]),
            // Minimal case.
            graph(2, &[(0, 1, 0.75)]),
            // Floor-only complete graph, as build_self_graph produces for a
            // corpus with no shared vocabulary.
            graph(4, &[
                (0, 1, FLOOR), (0, 2, FLOOR), (0, 3, FLOOR),
                (1, 2, FLOOR), (1, 3, FLOOR), (2, 3, FLOOR),
            ]),
        ];

        // Deterministic sweep over complete graphs with varied weights. A fixed
        // LCG, not rand: the test must fail reproducibly.
        let mut seed = 0x2545F491_4F6CDD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for n in 2..=7usize {
            for _ in 0..12 {
                let mut edges = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        let w = FLOOR + (next() % 40) as f64;
                        edges.push((i, j, w));
                    }
                }
                cases.push(graph(n, &edges));
            }
        }

        for (idx, g) in cases.iter().enumerate() {
            let sw = char_invariant(g);
            let bf = bruteforce_mincut(g);
            assert!(
                (sw - bf).abs() <= 1e-9 * bf.abs().max(1.0),
                "case {idx} (n={}): stoer-wagner gave {sw}, enumeration gave {bf}",
                g.verts.len()
            );
        }
    }

    #[test]
    fn edge_stats_reports_floor_fraction() {
        // Two edges at the floor, one above it.
        let g = graph(3, &[(0, 1, FLOOR), (0, 2, FLOOR), (1, 2, FLOOR + 4.0)]);
        let s = edge_stats(&g);
        assert_eq!(s.n_edges, 3);
        assert_eq!(s.min_w, FLOOR);
        assert_eq!(s.max_w, FLOOR + 4.0);
        assert!((s.frac_at_floor - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(classify(&g), None, "a graph with real overlap is not degenerate");
    }

    #[test]
    fn classify_names_each_degenerate_regime() {
        assert_eq!(classify(&graph(0, &[])), Some(Degeneracy::NoDocuments));
        assert_eq!(classify(&graph(1, &[])), Some(Degeneracy::SingleDocument));
        let floor_only = graph(3, &[(0, 1, FLOOR), (0, 2, FLOOR), (1, 2, FLOOR)]);
        assert_eq!(classify(&floor_only), Some(Degeneracy::FloorOnlyGraph));
        assert_eq!(edge_stats(&floor_only).frac_at_floor, 1.0);
    }

    /// The tautology, stated as a test so nobody "fixes" verify by re-asserting
    /// it. Every cut of a floor-complete graph exceeds the floor by arithmetic,
    /// so `chi >= floor` cannot discriminate a good index from a bad one.
    #[test]
    fn chi_is_at_least_floor_by_construction_on_every_shape() {
        for n in 2..=6usize {
            let mut edges = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    edges.push((i, j, FLOOR));
                }
            }
            let g = graph(n, &edges);
            assert!(char_invariant(&g) >= FLOOR);
        }
    }
}

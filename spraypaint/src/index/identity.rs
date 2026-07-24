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

/// Positive weight floor c > 0 (paper's beta). Added to every edge so the graph
/// is complete and chi >= floor holds exactly (thm:identity(i)).
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

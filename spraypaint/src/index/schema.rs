//! On-disk index schema. The index doubles as the agent's *self-graph*
//! (Inv 1): documents are vertices, and the corpus statistics plus the
//! identity block are computed from a canonical form so the fingerprint is
//! invariant under relabelling (file reorder, id renumbering, scene permute).
//!
//! Crucially, passages store term ids + line ranges but **no raw body**
//! (Inv 3: search-not-fetch). Snippets are re-read from disk at query time.

use serde::{Deserialize, Serialize};

/// Bump when the on-disk shape changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub schema_version: u32,
    /// Absolute root path (provenance; excluded from the fingerprint).
    pub root: String,
    /// Build time, unix seconds (provenance; excluded from the fingerprint).
    pub built_unix: u64,
    /// Scene groups over documents (water-filling substrate).
    pub scenes: Vec<SceneGroup>,
    /// Documents = vertices of the self-graph.
    pub documents: Vec<Document>,
    /// Corpus-wide statistics (global IDF lives here).
    pub stats: CorpusStats,
    /// Conserved-identity block (Inv 1).
    pub identity: Identity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGroup {
    pub name: String,
    /// Document ids belonging to this scene.
    pub doc_ids: Vec<u32>,
    pub stats: SceneStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneStats {
    /// Mean passage length (in tokens) within this scene — per-scene length
    /// normalisation for BM25.
    pub avg_passage_len: f32,
    pub passage_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: u32,
    /// Repo-relative path, forward-slash normalised.
    pub path: String,
    /// blake3 of the file body — change detection and the fingerprint key.
    pub content_hash: String,
    pub passages: Vec<Passage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub start_line: u32,
    pub end_line: u32,
    /// Token count (BM25 length normalisation).
    pub len: u32,
    /// (term_id, term_frequency), sorted by term_id (canonical form).
    pub terms: Vec<(u32, u32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    /// term_id -> term string, sorted lexicographically (canonical).
    pub vocab: Vec<String>,
    /// Document frequency per term_id (how many documents contain the term).
    pub global_df: Vec<u32>,
    pub total_docs: u32,
    pub avg_passage_len: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// blake3 over the canonical self-graph — the conserved invariant (Inv 1).
    pub fingerprint: String,
    /// Char(A): least-weight bipartition cut of the self-graph (paper's chi).
    pub char_invariant: f64,
    /// Positive weight floor c > 0.
    pub floor: f64,
    pub n_vertices: u32,
    pub n_edges: u32,
}

impl CorpusStats {
    /// IDF of a term id (Okapi BM25 idf with +0.5 smoothing, floored at 0).
    pub fn idf(&self, term_id: usize) -> f64 {
        let n = self.total_docs as f64;
        let df = self.global_df.get(term_id).copied().unwrap_or(0) as f64;
        // Standard BM25 idf; clamp to >= 0 so common terms never go negative.
        (((n - df + 0.5) / (df + 0.5)) + 1.0).ln().max(0.0)
    }

    /// Resolve a query term string to its term id, if present in the vocab.
    /// Vocab is sorted, so binary search.
    pub fn term_id(&self, term: &str) -> Option<usize> {
        self.vocab.binary_search_by(|v| v.as_str().cmp(term)).ok()
    }
}

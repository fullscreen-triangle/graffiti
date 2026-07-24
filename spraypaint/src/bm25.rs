//! Okapi BM25 scoring. IDF is **global** (corpus-wide) so per-passage scores
//! across scenes live on one comparable scale — a prerequisite for the single
//! water-filling price to be a meaningful cross-scene threshold. Length
//! normalisation is **per-scene** so a scene of long files is not penalised.

use crate::index::schema::{CorpusStats, Document, Index, SceneGroup};

/// BM25 free parameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Bm25Params { k1: 1.2, b: 0.75 }
    }
}

/// A scored passage: enough to render a result and to feed the allocator.
#[derive(Debug, Clone)]
pub struct Scored {
    pub doc_id: u32,
    pub passage_idx: u32,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f64,
}

/// Score every passage in `scene` against the query term ids, returning results
/// sorted by descending score (ties broken deterministically). Only passages
/// with a positive score are returned — the marginal-gain sequence for this
/// scene's water-filling profile.
pub fn score_scene(
    scene: &SceneGroup,
    idx: &Index,
    query_terms: &[usize],
    params: &Bm25Params,
) -> Vec<Scored> {
    if query_terms.is_empty() {
        return Vec::new();
    }
    let stats: &CorpusStats = &idx.stats;
    let avg_len = if scene.stats.avg_passage_len > 0.0 {
        scene.stats.avg_passage_len as f64
    } else {
        stats.avg_passage_len.max(1.0) as f64
    };
    let k1 = params.k1 as f64;
    let b = params.b as f64;

    // Precompute idf per query term once.
    let idfs: Vec<f64> = query_terms.iter().map(|&t| stats.idf(t)).collect();

    let mut out = Vec::new();
    for &doc_id in &scene.doc_ids {
        let doc: &Document = match idx.documents.get(doc_id as usize) {
            Some(d) if d.id == doc_id => d,
            // Fall back to a linear find if ids are not array-aligned.
            _ => match idx.documents.iter().find(|d| d.id == doc_id) {
                Some(d) => d,
                None => continue,
            },
        };
        for (p_idx, passage) in doc.passages.iter().enumerate() {
            let len = passage.len as f64;
            let norm = 1.0 - b + b * (len / avg_len);
            let mut score = 0.0;
            for (qi, &term_id) in query_terms.iter().enumerate() {
                let idf = idfs[qi];
                if idf <= 0.0 {
                    continue;
                }
                // tf lookup: passage.terms is sorted by term_id.
                let tf = passage
                    .terms
                    .binary_search_by(|(tid, _)| (*tid as usize).cmp(&term_id))
                    .ok()
                    .map(|pos| passage.terms[pos].1 as f64)
                    .unwrap_or(0.0);
                if tf > 0.0 {
                    score += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
                }
            }
            if score > 0.0 {
                out.push(Scored {
                    doc_id,
                    passage_idx: p_idx as u32,
                    start_line: passage.start_line,
                    end_line: passage.end_line,
                    score,
                });
            }
        }
    }

    // Sort descending by score; deterministic tie-break by (doc_id, passage_idx).
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.doc_id.cmp(&b.doc_id))
            .then(a.passage_idx.cmp(&b.passage_idx))
    });
    out
}

//! The COMMITMENT phase (Inv 4): answer a query by a fresh search over the
//! current index (Inv 3, search-not-fetch). No answer is stored; snippets are
//! re-read from disk at query time. The pipeline is:
//!
//!   query -> term ids -> per-scene BM25 (bm25.rs) -> per-scene gain profiles
//!         -> water-filling allocation across scenes (waterfill.rs)
//!         -> take each scene's top-N passages -> re-read snippet bodies.

use std::path::Path;

use anyhow::Result;

use crate::bm25::{self, Bm25Params, Scored};
use crate::chunk;
use crate::index::schema::Index;
use crate::waterfill::{self, Allocation};

/// A single returned passage, with its freshly-read snippet.
pub struct Result_ {
    pub scene: String,
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f64,
    pub snippet: String,
}

/// Everything an `ask` produces — the results plus the diagnostics the
/// blueprint exposes (price, per-scene allocation).
pub struct AskOutcome {
    pub query_terms: Vec<String>,
    /// Term ids that resolved against the vocab (kept for JSON/diagnostic use).
    #[allow(dead_code)]
    pub matched_term_ids: Vec<usize>,
    pub price: f64,
    /// (scene_name, allocated_count, available_count)
    pub allocation: Vec<(String, usize, usize)>,
    pub results: Vec<Result_>,
}

/// Query-word stopwords, mirroring the spirit of `purpose`'s question-word list.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "for", "on", "at", "to", "is", "are", "and", "or", "how",
    "what", "which", "where", "who", "why", "when", "do", "does", "did", "i", "find", "show",
    "me", "get", "tell", "about",
];

fn query_terms(query: &str) -> Vec<String> {
    chunk::tokenize(query)
        .into_iter()
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Run the search. `scene_filter`, if non-empty, restricts to named scenes.
/// Pure and deterministic given (index, query, budget, scene_filter).
pub fn run(
    root_dir: &Path,
    index: &Index,
    query: &str,
    budget: usize,
    scene_filter: &[String],
) -> Result<AskOutcome> {
    let params = Bm25Params::default();
    let terms = query_terms(query);

    // Resolve terms to ids against the canonical vocab.
    let matched_term_ids: Vec<usize> = terms
        .iter()
        .filter_map(|t| index.stats.term_id(t))
        .collect();

    // Select scenes (optionally filtered).
    let scenes: Vec<&crate::index::schema::SceneGroup> = index
        .scenes
        .iter()
        .filter(|s| scene_filter.is_empty() || scene_filter.iter().any(|f| f == &s.name))
        .collect();

    // Per-scene BM25 -> descending score lists (the gain profiles).
    let per_scene_scored: Vec<Vec<Scored>> = scenes
        .iter()
        .map(|s| bm25::score_scene(s, index, &matched_term_ids, &params))
        .collect();
    let scene_scores: Vec<Vec<f64>> = per_scene_scored
        .iter()
        .map(|v| v.iter().map(|s| s.score).collect())
        .collect();

    // Water-filling across scenes.
    let alloc: Allocation = waterfill::water_fill(&scene_scores, budget, 1e-9);

    // Assemble results: each scene contributes its top per_scene[i] passages.
    let mut results = Vec::new();
    let mut allocation = Vec::new();
    for (i, scene) in scenes.iter().enumerate() {
        let take = alloc.per_scene[i];
        allocation.push((scene.name.clone(), take, per_scene_scored[i].len()));
        for scored in per_scene_scored[i].iter().take(take) {
            let doc = &index.documents[scored.doc_id as usize];
            let snippet = read_snippet(root_dir, &doc.path, scored.start_line, scored.end_line);
            results.push(Result_ {
                scene: scene.name.clone(),
                path: doc.path.clone(),
                start_line: scored.start_line,
                end_line: scored.end_line,
                score: scored.score,
                snippet,
            });
        }
    }

    // Default presentation groups by scene then descending score; the caller can
    // re-sort for --flat.
    results.sort_by(|a, b| {
        a.scene
            .cmp(&b.scene)
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });

    Ok(AskOutcome {
        query_terms: terms,
        matched_term_ids,
        price: alloc.price,
        allocation,
        results,
    })
}

/// Re-read a passage body from disk (search-not-fetch: bodies are never stored
/// in the index). Returns a first-line-anchored snippet, trimmed.
fn read_snippet(root_dir: &Path, rel_path: &str, start: u32, end: u32) -> String {
    let full = root_dir.join(rel_path);
    let body = match std::fs::read_to_string(&full) {
        Ok(b) => b,
        Err(_) => return String::new(),
    };
    let lines: Vec<&str> = body.lines().collect();
    let s = (start.saturating_sub(1)) as usize;
    let e = (end as usize).min(lines.len());
    if s >= lines.len() {
        return String::new();
    }
    // First non-empty line of the passage as the display snippet, capped.
    lines[s..e]
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().chars().take(160).collect())
        .unwrap_or_default()
}

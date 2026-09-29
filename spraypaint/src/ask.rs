//! The COMMITMENT phase (Inv 4): answer a query by a fresh search over the
//! current index (Inv 3, search-not-fetch). No answer is stored; snippets are
//! re-read from disk at query time. The pipeline is:
//!
//!   query -> term ids -> per-scene BM25 (bm25.rs) -> drop overlapping windows
//!         -> per-scene gain profiles
//!         -> water-filling allocation across scenes (waterfill.rs)
//!         -> take each scene's top-N passages -> re-read their evidence lines
//!         -> coverage verdict (coverage.rs).

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::bm25::{self, Bm25Params, Scored};
use crate::chunk;
use crate::coverage::{self, Coverage, TermCoverage};
use crate::evidence;
use crate::index::schema::Index;
use crate::waterfill::{self, Allocation};

/// A single returned passage, with its freshly-read evidence.
pub struct Result_ {
    pub scene: String,
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f64,
    /// The evidence lines (see `evidence.rs`), joined with `\n`. Falls back to
    /// the passage's first non-empty line only when no line matches — which
    /// means the file changed since it was indexed.
    pub snippet: String,
    /// Line range of `snippet` within the file, 1-based inclusive.
    pub evidence_start: u32,
    pub evidence_end: u32,
    /// Query terms this passage contains, in query order.
    pub matched_terms: Vec<String>,
}

/// Per-scene score statistics over **every** scoring passage.
///
/// These exist because they cannot be recovered downstream. `results` carries
/// only the *allocated* passages, so a consumer computing a median from it is
/// taking the median of a top-k slice — a number that says more about the
/// budget than about the scene. `best`/`median` here are taken from
/// `per_scene_scored` before truncation, so they describe the scene.
///
/// `available` is the count of passages scoring **above zero for this query**,
/// after overlapping windows are dropped — not the scene's passage count. The scene total is a property of the
/// index, not of a query, and is reported by `scenes`.
pub struct SceneStats {
    pub scene: String,
    pub allocated: usize,
    pub available: usize,
    /// Highest score in the scene, or `None` when nothing scored.
    pub best: Option<f64>,
    /// Median over all scoring passages (lower of the two middles when even).
    pub median: Option<f64>,
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
    /// Score distribution per scene, computed before truncation.
    pub scene_stats: Vec<SceneStats>,
    pub results: Vec<Result_>,
    /// Whether the corpus contains what the query is about (coverage.rs).
    pub coverage: Coverage,
}

/// Query-word stopwords, mirroring the spirit of `purpose`'s question-word list.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "for", "on", "at", "to", "is", "are", "and", "or", "how",
    "what", "which", "where", "who", "why", "when", "do", "does", "did", "i", "find", "show",
    "me", "get", "tell", "about",
];

/// Query terms after stopword removal, deduplicated in first-seen order. A
/// repeated word would otherwise be scored twice by BM25 and counted twice in
/// coverage, as if the query had said two things.
fn query_terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in chunk::tokenize(query) {
        if !STOPWORDS.contains(&t.as_str()) && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// Keep a passage only if it does not overlap a higher-scoring passage of the
/// same document. Windows overlap by construction (`window` 40, `overlap` 10),
/// so without this the same lines can come back twice and be counted twice by
/// the allocator. `scored` must be in descending score order, as
/// `bm25::score_scene` returns it; the higher-scoring window wins.
fn drop_overlaps(scored: Vec<Scored>) -> Vec<Scored> {
    let mut taken: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut kept = Vec::with_capacity(scored.len());
    for s in scored {
        let spans = taken.entry(s.doc_id).or_default();
        if spans.iter().any(|&(a, b)| s.start_line <= b && a <= s.end_line) {
            continue;
        }
        spans.push((s.start_line, s.end_line));
        kept.push(s);
    }
    kept
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

    // Resolve terms to ids against the canonical vocab, and weight every term —
    // including ones the corpus has never seen, which coverage needs most.
    let term_ids: Vec<Option<usize>> = terms.iter().map(|t| index.stats.term_id(t)).collect();
    let term_dfs: Vec<u32> = term_ids
        .iter()
        .map(|id| id.and_then(|i| index.stats.global_df.get(i).copied()).unwrap_or(0))
        .collect();
    let weighted: Vec<(String, f64)> = terms
        .iter()
        .zip(&term_dfs)
        .map(|(t, &df)| (t.clone(), index.stats.idf_of_df(df)))
        .collect();
    let matched_term_ids: Vec<usize> = term_ids.iter().flatten().copied().collect();

    // Select scenes (optionally filtered).
    let scenes: Vec<&crate::index::schema::SceneGroup> = index
        .scenes
        .iter()
        .filter(|s| scene_filter.is_empty() || scene_filter.iter().any(|f| f == &s.name))
        .collect();

    // Per-scene BM25 -> descending score lists (the gain profiles).
    let per_scene_scored: Vec<Vec<Scored>> = scenes
        .iter()
        .map(|s| drop_overlaps(bm25::score_scene(s, index, &matched_term_ids, &params)))
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
    let mut scene_stats = Vec::new();
    for (i, scene) in scenes.iter().enumerate() {
        let take = alloc.per_scene[i];
        allocation.push((scene.name.clone(), take, per_scene_scored[i].len()));

        // Computed here, against the full list, because after the `take` below
        // the tail is gone and this can never be recovered.
        let all = &per_scene_scored[i];
        scene_stats.push(SceneStats {
            scene: scene.name.clone(),
            allocated: take,
            available: all.len(),
            // `score_scene` returns descending order, so the head is the max and
            // the midpoint is the median without a second sort.
            best: all.first().map(|s| s.score),
            median: if all.is_empty() {
                None
            } else {
                Some(all[all.len() / 2].score)
            },
        });

        for scored in all.iter().take(take) {
            let doc = &index.documents[scored.doc_id as usize];
            let passage = &doc.passages[scored.passage_idx as usize];
            // From the index, not the re-read text: these are the terms the
            // passage was scored on.
            let matched_terms: Vec<String> = terms
                .iter()
                .zip(&term_ids)
                .filter(|(_, id)| {
                    id.is_some_and(|id| {
                        passage
                            .terms
                            .binary_search_by(|(tid, _)| (*tid as usize).cmp(&id))
                            .is_ok()
                    })
                })
                .map(|(t, _)| t.clone())
                .collect();
            let ev = read_evidence(root_dir, &doc.path, scored.start_line, scored.end_line, &weighted);
            results.push(Result_ {
                scene: scene.name.clone(),
                path: doc.path.clone(),
                start_line: scored.start_line,
                end_line: scored.end_line,
                score: scored.score,
                snippet: ev.text,
                evidence_start: ev.start_line,
                evidence_end: ev.end_line,
                matched_terms,
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

    // Coverage is judged on what was returned, not on the whole corpus: a term
    // that exists but fell outside the budget is reported as such, so the
    // caller knows raising `-k` might help where adding a source would not.
    let coverage_terms: Vec<TermCoverage> = terms
        .iter()
        .zip(&term_dfs)
        .zip(&weighted)
        .map(|((t, &df), (_, w))| TermCoverage {
            term: t.clone(),
            df,
            weight: *w,
            in_results: results.iter().any(|r| r.matched_terms.contains(t)),
        })
        .collect();
    let complete = results
        .iter()
        .filter(|r| !terms.is_empty() && r.matched_terms.len() == terms.len())
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        .map(|r| format!("{}:{}-{}", r.path, r.start_line, r.end_line));
    let coverage = coverage::assess(coverage_terms, complete);

    Ok(AskOutcome {
        query_terms: terms,
        matched_term_ids,
        price: alloc.price,
        allocation,
        scene_stats,
        results,
        coverage,
    })
}

/// Re-read a passage from disk (search-not-fetch: bodies are never stored in
/// the index) and return its evidence lines.
///
/// When no line matches — the file changed since indexing — the fallback is
/// the passage's first non-empty line, as the snippet used to be. It shows
/// where the passage is, not why it matched; `matched_terms`, taken from the
/// index, still says what it was scored on.
fn read_evidence(
    root_dir: &Path,
    rel_path: &str,
    start: u32,
    end: u32,
    terms: &[(String, f64)],
) -> evidence::Evidence {
    let empty = evidence::Evidence {
        start_line: start,
        end_line: start,
        text: String::new(),
    };
    let body = match std::fs::read_to_string(root_dir.join(rel_path)) {
        Ok(b) => b,
        Err(_) => return empty,
    };
    let lines: Vec<&str> = body.lines().collect();
    let s = (start.saturating_sub(1)) as usize;
    let e = (end as usize).min(lines.len());
    if s >= e {
        return empty;
    }
    if let Some(ev) = evidence::select(&lines[s..e], start, terms) {
        return ev;
    }
    lines[s..e]
        .iter()
        .enumerate()
        .find(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| evidence::Evidence {
            start_line: start + i as u32,
            end_line: start + i as u32,
            text: l.trim().chars().take(160).collect(),
        })
        .unwrap_or(empty)
}

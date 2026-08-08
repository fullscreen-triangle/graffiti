//! Rendering. Human output mirrors `purpose`'s `file:line  [scene] path` shape;
//! `--json` emits the agent-facing structure documented in the plan.

use crate::ask::AskOutcome;
use crate::index::schema::Index;

/// Human-readable `ask` output, grouped by scene (so cross-scene presence is
/// visible). `flat` re-sorts globally by score.
pub fn ask_human(outcome: &AskOutcome, count: u64, fingerprint: &str, flat: bool) -> String {
    let mut buf = String::new();
    if outcome.results.is_empty() {
        buf.push_str("no matching passages.\n");
        return buf;
    }
    buf.push_str(&format!(
        "{} passage(s)  |  price p*={:.4}  |  committed act #{}  |  {}\n\n",
        outcome.results.len(),
        outcome.price,
        count,
        &fingerprint[..fingerprint.len().min(14)]
    ));

    let mut rows: Vec<&crate::ask::Result_> = outcome.results.iter().collect();
    if flat {
        rows.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut last_scene = String::new();
    for r in rows {
        if !flat && r.scene != last_scene {
            buf.push_str(&format!("[{}]\n", r.scene));
            last_scene = r.scene.clone();
        }
        buf.push_str(&format!(
            "  {}:{}-{}  (score {:.3})\n      {}\n",
            r.path, r.start_line, r.end_line, r.score, r.snippet
        ));
    }
    buf
}

/// JSON `ask` output for machine consumers (the primary audience).
///
/// Carries `"dry_run": false` even though this function is only ever reached on
/// a committed act. The marker exists so a consumer can branch on the *payload*
/// rather than on which URL or flag produced it. Emitting it from only one of
/// the two shapes would be worse than emitting it from neither: `undefined` is
/// falsy in JavaScript, so a preview whose payload lacked the key would read as
/// `dry_run == false` and render as a committed answer — silently inverting the
/// one distinction the dry-run path exists to preserve.
pub fn ask_json(outcome: &AskOutcome, budget: usize, count: u64, fingerprint: &str) -> String {
    let results: Vec<serde_json::Value> = outcome
        .results
        .iter()
        .map(|r| {
            serde_json::json!({
                "scene": r.scene,
                "path": r.path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "score": r.score,
                "snippet": r.snippet,
            })
        })
        .collect();
    let allocation: Vec<serde_json::Value> = outcome
        .scene_stats
        .iter()
        .map(|s| {
            // `best_score`/`median_score` are computed over every scoring
            // passage, not over `results` — which holds only the allocated
            // ones. A consumer cannot derive these from `results`, so omitting
            // them would force it to fabricate a distribution from a top-k
            // slice. `null` when the scene scored nothing at all.
            serde_json::json!({
                "scene": s.scene,
                "allocated": s.allocated,
                "available": s.available,
                "best_score": s.best,
                "median_score": s.median,
            })
        })
        .collect();
    let v = serde_json::json!({
        "dry_run": false,
        "query_terms": outcome.query_terms,
        "budget": budget,
        "price": outcome.price,
        "committed_count": count,
        "identity_fingerprint": fingerprint,
        "allocation": allocation,
        "results": results,
    });
    serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".to_string())
}

/// Dry-run diagnostics for `ask` — explicitly labelled NON-answers (Inv 3: a
/// zero-act read-out emits no answer, only diagnostics; count is not touched).
pub fn ask_dry_run(outcome: &AskOutcome) -> String {
    let mut buf = String::new();
    buf.push_str("[dry-run: diagnostics only, no answer emitted, count unchanged]\n");
    buf.push_str(&format!("query terms: {:?}\n", outcome.query_terms));
    buf.push_str(&format!("clearing price p*: {:.4}\n", outcome.price));
    buf.push_str("per-scene allocation (allocated/available):\n");
    for (name, took, avail) in &outcome.allocation {
        buf.push_str(&format!("  {name}: {took}/{avail}\n"));
    }
    buf
}

/// JSON dry-run diagnostics — the machine-readable sibling of [`ask_dry_run`].
///
/// Deliberately shaped like [`ask_json`] so a consumer can render a preview and
/// a committed answer with the same code path, with two differences that are
/// the whole point of a dry run and must stay visible in the payload:
///
///   * `"dry_run": true` — an explicit marker, so a preview can never be
///     mistaken for an answer by a consumer that ignores which URL it called.
///   * `committed_count` is **absent**, not zero. A dry run does not touch the
///     counter (Inv 3), and emitting `0` would assert a count that is almost
///     certainly wrong. Absent means "this act did not happen".
///
/// `results` *is* included. The interactive UI previews on every gesture, and
/// a preview with no passages could only show allocation and price — which
/// would push the UI toward committing on each drag to see actual results,
/// permanently inflating a monotone counter that has no decrement path.
pub fn ask_dry_run_json(outcome: &AskOutcome, budget: usize, fingerprint: &str) -> String {
    let results: Vec<serde_json::Value> = outcome
        .results
        .iter()
        .map(|r| {
            serde_json::json!({
                "scene": r.scene,
                "path": r.path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "score": r.score,
                "snippet": r.snippet,
            })
        })
        .collect();
    let allocation: Vec<serde_json::Value> = outcome
        .scene_stats
        .iter()
        .map(|s| {
            // `best_score`/`median_score` are computed over every scoring
            // passage, not over `results` — which holds only the allocated
            // ones. A consumer cannot derive these from `results`, so omitting
            // them would force it to fabricate a distribution from a top-k
            // slice. `null` when the scene scored nothing at all.
            serde_json::json!({
                "scene": s.scene,
                "allocated": s.allocated,
                "available": s.available,
                "best_score": s.best,
                "median_score": s.median,
            })
        })
        .collect();
    let v = serde_json::json!({
        "dry_run": true,
        "query_terms": outcome.query_terms,
        "budget": budget,
        "price": outcome.price,
        "identity_fingerprint": fingerprint,
        "allocation": allocation,
        "results": results,
    });
    serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".to_string())
}

/// `identity` command output.
pub fn identity_human(index: &Index) -> String {
    let id = &index.identity;
    format!(
        "fingerprint: {}\nchar_invariant (chi): {:.6}\nfloor: {:.2e}\nvertices: {}\nedges: {}\n",
        id.fingerprint, id.char_invariant, id.floor, id.n_vertices, id.n_edges
    )
}

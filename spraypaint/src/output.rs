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
        .allocation
        .iter()
        .map(|(name, took, avail)| {
            serde_json::json!({ "scene": name, "allocated": took, "available": avail })
        })
        .collect();
    let v = serde_json::json!({
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

/// `identity` command output.
pub fn identity_human(index: &Index) -> String {
    let id = &index.identity;
    format!(
        "fingerprint: {}\nchar_invariant (chi): {:.6}\nfloor: {:.2e}\nvertices: {}\nedges: {}\n",
        id.fingerprint, id.char_invariant, id.floor, id.n_vertices, id.n_edges
    )
}

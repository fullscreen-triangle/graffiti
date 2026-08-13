//! JSON API handlers.
//!
//! Every handler delegates to [`crate::actions`], so the server and the CLI
//! share one invariant-critical ordering. The work here is parameter parsing,
//! status-code mapping, and the index cache.

use std::sync::Arc;
use std::time::SystemTime;

use tiny_http::{Method, Request};

use crate::actions::{self, AskRequest};
use crate::config::SprayConfig;
use crate::index::schema::Index;
use crate::output;

use super::{json, read_body, ServerState};

/// An index plus the file identity it was loaded from.
pub struct CachedIndex {
    pub index: Arc<Index>,
    mtime: Option<SystemTime>,
    len: u64,
}

/// Stable error codes. The UI switches on these, so they are part of the API.
fn err(code: &str, message: &str) -> String {
    serde_json::json!({ "error": { "code": code, "message": message } }).to_string()
}

/// Map an `anyhow` error to an HTTP status and a stable code.
///
/// The mapping is by *meaning*, not by convenience:
///
///   * a missing index is a 404 — the resource genuinely is not there yet;
///   * a fingerprint mismatch is a 409 Conflict, because it is **actionable**:
///     the index on disk disagrees with itself and re-indexing fixes it. A 500
///     would tell the user only that something broke;
///   * anything else is a 500.
fn map_error(e: &anyhow::Error) -> (u16, String) {
    let msg = format!("{e:#}");
    if msg.contains("no index at") {
        (404, err("no_index", &msg))
    } else if msg.contains("fingerprint mismatch") {
        (409, err("identity_mismatch", &msg))
    } else if msg.contains("schema version mismatch") {
        (409, err("schema_mismatch", &msg))
    } else {
        (500, err("internal", &msg))
    }
}

/// Percent-decode a query-string value (`+` is a space).
fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => {
                let hex = |c: u8| -> Option<u8> {
                    match c {
                        b'0'..=b'9' => Some(c - b'0'),
                        b'a'..=b'f' => Some(c - b'a' + 10),
                        b'A'..=b'F' => Some(c - b'A' + 10),
                        _ => None,
                    }
                };
                match (hex(b[i + 1]), hex(b[i + 2])) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    // A malformed escape is kept literally rather than dropped,
                    // so a query containing a bare '%' still searches for it.
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parse `?a=1&b=2` into pairs.
fn query_pairs(url: &str) -> Vec<(String, String)> {
    let qs = match url.split_once('?') {
        Some((_, q)) => q,
        None => return Vec::new(),
    };
    qs.split('&')
        .filter(|s| !s.is_empty())
        .map(|kv| match kv.split_once('=') {
            Some((k, v)) => (urldecode(k), urldecode(v)),
            None => (urldecode(kv), String::new()),
        })
        .collect()
}

fn get<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Build an [`AskRequest`] from query params or a JSON body.
///
/// `budget` is clamped to at least 1 rather than rejected: `k=0` is a
/// well-formed request for nothing, and the water-filling allocator has no
/// meaningful behaviour there. Clamping keeps the UI's slider total.
fn parse_ask(pairs: &[(String, String)], body: &str) -> Result<AskRequest, String> {
    if !body.trim().is_empty() {
        let v: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| format!("body is not valid JSON: {e}"))?;
        let query = v
            .get("query")
            .and_then(|q| q.as_str())
            .ok_or("missing required field: query")?
            .to_string();
        let budget = v
            .get("budget")
            .and_then(|b| b.as_u64())
            .unwrap_or(12)
            .max(1) as usize;
        let scenes = v
            .get("scenes")
            .and_then(|s| s.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        return Ok(AskRequest::new(query, budget, scenes));
    }

    let query = get(pairs, "q")
        .or_else(|| get(pairs, "query"))
        .ok_or("missing required parameter: q")?
        .to_string();
    if query.trim().is_empty() {
        return Err("q must not be empty".to_string());
    }
    let budget = get(pairs, "k")
        .or_else(|| get(pairs, "budget"))
        .map(|s| s.parse::<usize>().map_err(|_| format!("k is not a number: {s}")))
        .transpose()?
        .unwrap_or(12)
        .max(1);
    let scenes: Vec<String> = get(pairs, "scenes")
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Ok(AskRequest::new(query, budget, scenes))
}

/// Load the index, reusing the cached copy when `index.json` is unchanged.
///
/// The freshness test (mtime + length) is performed **while holding the shared
/// phase lock**, which is what makes it sound: an index build holds the
/// exclusive lock across its write-and-rename, so a reader that holds the
/// shared lock cannot observe the file mid-swap. Without the lock this would be
/// a benign-looking TOCTOU.
///
/// Caveat, deliberately documented rather than hidden: some filesystems have
/// 1-second mtime granularity, so two builds within the same second *and*
/// producing exactly the same file length would look identical. Every real edit
/// changes the length. `--no-cache` opts out entirely.
pub fn load_cached(state: &ServerState) -> anyhow::Result<Arc<Index>> {
    if state.no_cache {
        return Ok(Arc::new(actions::load_index(&state.root)?));
    }

    let _guard = crate::phase::PhaseGuard::commitment(&state.root)?;
    let path = crate::root::index_path(&state.root);
    let (mtime, len) = match std::fs::metadata(&path) {
        Ok(m) => (m.modified().ok(), m.len()),
        Err(_) => (None, 0),
    };

    if let Ok(g) = state.cache.lock() {
        if let Some(c) = g.as_ref() {
            if c.len == len && c.mtime == mtime {
                return Ok(Arc::clone(&c.index));
            }
        }
    }

    // `load` re-verifies the fingerprint (Inv 1) — the cache skips repeated
    // blake3 work on an unchanged file, never the verification of a new one.
    let idx = Arc::new(actions::load_index(&state.root)?);
    if let Ok(mut g) = state.cache.lock() {
        *g = Some(CachedIndex {
            index: Arc::clone(&idx),
            mtime,
            len,
        });
    }
    Ok(idx)
}

/// Route an `/api/...` request.
pub fn route(state: &ServerState, mut req: Request, path: &str, url: &str) {
    let method = req.method().clone();
    let pairs = query_pairs(url);

    // Read the body before matching so a POST handler never has to care whether
    // an earlier arm consumed the reader.
    let body = if method == Method::Post {
        match read_body(&mut req) {
            Ok(b) => b,
            Err(e) => {
                json(state, req, 413, err("bad_request", &format!("{e:#}")));
                return;
            }
        }
    } else {
        String::new()
    };

    match (&method, path) {
        (Method::Get, "/api/health") => health(state, req),
        (Method::Get, "/api/ask") | (Method::Post, "/api/ask") => ask(state, req, &pairs, &body),
        (Method::Get, "/api/dry-run") | (Method::Post, "/api/dry-run") => {
            dry_run(state, req, &pairs, &body)
        }
        (Method::Get, "/api/identity") => identity(state, req),
        (Method::Get, "/api/count") => count(state, req),
        (Method::Get, "/api/scenes") => scenes(state, req),
        (Method::Get, "/api/verify") => verify(state, req),
        (Method::Post, "/api/index") => start_index(state, req),
        (Method::Get, "/api/index/status") => index_status(state, req),
        (Method::Get, _) | (Method::Post, _) => {
            json(state, req, 404, err("not_found", &format!("no such endpoint: {path}")))
        }
        _ => json(state, req,
            405,
            err("method_not_allowed", "use GET or POST"),
        ),
    }
}

/// Liveness plus enough context for the UI to render a useful empty state.
///
/// Takes no lock and does not load the index — deliberately. This must answer
/// while an index build holds the exclusive lock, or the UI cannot distinguish
/// "server busy building" from "server dead".
fn health(state: &ServerState, req: Request) {
    let has_index = crate::root::index_path(&state.root).exists();
    let v = serde_json::json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "root": state.root.display().to_string(),
        "has_index": has_index,
        "allow_index": state.allow_index,
        "indexing": state.job.is_running(),
    });
    json(state, req, 200, v.to_string());
}

/// Commit one search act. **This increments the monotone count.**
fn ask(state: &ServerState, req: Request, pairs: &[(String, String)], body: &str) {
    let r = match parse_ask(pairs, body) {
        Ok(r) => r,
        Err(m) => return json(state, req, 400, err("bad_request", &m)),
    };
    match actions::ask(&state.root, &r) {
        // Byte-for-byte the same renderer the CLI uses, so `--json` output and
        // the API response cannot drift apart.
        Ok(resp) => json(state, req,
            200,
            output::ask_json(&resp.outcome, r.budget, resp.count, &resp.fingerprint),
        ),
        Err(e) => {
            let (code, b) = map_error(&e);
            json(state, req, code, b)
        }
    }
}

/// Diagnostics with no answer committed. **This is what interactive previews
/// must use** — the count is monotone with no decrement path, so previewing via
/// `/api/ask` would permanently inflate it on every slider drag.
fn dry_run(state: &ServerState, req: Request, pairs: &[(String, String)], body: &str) {
    let r = match parse_ask(pairs, body) {
        Ok(r) => r,
        Err(m) => return json(state, req, 400, err("bad_request", &m)),
    };
    match actions::dry_run(&state.root, &r) {
        Ok(resp) => json(state, req,
            200,
            output::ask_dry_run_json(&resp.outcome, r.budget, &resp.fingerprint),
        ),
        Err(e) => {
            let (code, b) = map_error(&e);
            json(state, req, code, b)
        }
    }
}

fn identity(state: &ServerState, req: Request) {
    match load_cached(state) {
        Ok(idx) => match serde_json::to_string(&idx.identity) {
            Ok(s) => json(state, req, 200, s),
            Err(e) => json(state, req, 500, err("internal", &e.to_string())),
        },
        Err(e) => {
            let (code, b) = map_error(&e);
            json(state, req, code, b)
        }
    }
}

fn count(state: &ServerState, req: Request) {
    match actions::count(&state.root) {
        Ok(c) => json(state, req, 200, serde_json::json!({ "committed_count": c }).to_string()),
        Err(e) => {
            let (code, b) = map_error(&e);
            json(state, req, code, b)
        }
    }
}

fn scenes(state: &ServerState, req: Request) {
    match actions::scenes(&state.root) {
        Ok(list) => {
            let v: Vec<_> = list
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "name": s.name,
                        "documents": s.documents,
                        "passages": s.passages,
                    })
                })
                .collect();
            json(state, req, 200, serde_json::to_string(&v).unwrap_or_else(|_| "[]".into()))
        }
        Err(e) => {
            let (code, b) = map_error(&e);
            json(state, req, code, b)
        }
    }
}

/// Re-check the invariants.
///
/// **Always HTTP 200, including when an invariant is breached.** A breach is a
/// successful answer to the question that was asked — the check ran and found
/// something. Returning 500 would conflate "the index is corrupt" with "the
/// server could not tell you", and a UI written against that cannot show the
/// user the one thing they need to see. `actions::verify` returns a report
/// rather than a `Result` for exactly this reason.
///
/// The payload mirrors `spraypaint verify --json` field for field, including
/// the three-state `status` and the top-level strict `pass`.
fn verify(state: &ServerState, req: Request) {
    let rep = actions::verify(&state.root);
    let inv = |r: &actions::InvariantReport| {
        serde_json::json!({
            "title": r.title,
            "status": r.status().as_str(),
            "pass": r.status() == actions::Status::Pass,
            "checks": r.checks.iter().map(|c| serde_json::json!({
                "name": c.name,
                "status": c.status.as_str(),
                "detail": c.detail,
            })).collect::<Vec<_>>(),
        })
    };
    let v = serde_json::json!({
        "pass": rep.pass(),
        "overall": rep.status().as_str(),
        "degeneracies": rep.degeneracies,
        "inv1_identity": inv(&rep.inv1),
        "inv2_count": inv(&rep.inv2),
        "inv3_search_not_fetch": inv(&rep.inv3),
        "inv4_phases": inv(&rep.inv4),
    });
    json(state, req, 200, v.to_string());
}

/// Start a background index build. Gated four ways — see the module docs.
fn start_index(state: &ServerState, req: Request) {
    if !state.allow_index {
        return json(state, req,
            403,
            err(
                "index_disabled",
                "indexing from the browser is disabled; restart with `spraypaint serve --allow-index`",
            ),
        );
    }
    if state.job.is_running() {
        return json(state, req, 409, err("index_running", "an index build is already running"));
    }

    // Preflight the lock without blocking: another *process* (a CLI `index`)
    // may hold it, which our own job flag cannot see. Blocking here would tie
    // up a worker thread for the duration of someone else's build.
    match crate::phase::PhaseGuard::try_construction(&state.root) {
        Ok(Some(guard)) => drop(guard),
        Ok(None) => {
            return json(state, req,
                409,
                err("index_running", "another process holds the construction lock"),
            )
        }
        Err(e) => return json(state, req, 500, err("internal", &format!("{e:#}"))),
    }

    if state.job.start(state.root.clone(), SprayConfig::default()) {
        // 202 Accepted: the work is queued, not finished. The UI polls
        // /api/index/status rather than waiting on this response.
        json(state, req,
            202,
            serde_json::json!({ "status": "running" }).to_string(),
        )
    } else {
        json(state, req, 409, err("index_running", "an index build is already running"))
    }
}

fn index_status(state: &ServerState, req: Request) {
    let s = state.job.state();
    let v = serde_json::json!({
        "status": s.as_str(),
        "running": state.job.is_running(),
        "detail": s.detail(),
    });
    json(state, req, 200, v.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_pairs_decodes_percent_and_plus() {
        let p = query_pairs("/api/ask?q=hello+world&k=5");
        assert_eq!(get(&p, "q"), Some("hello world"));
        assert_eq!(get(&p, "k"), Some("5"));

        let p = query_pairs("/api/ask?q=a%20b%2Bc");
        assert_eq!(get(&p, "q"), Some("a b+c"));
    }

    #[test]
    fn a_lone_percent_survives_rather_than_being_dropped() {
        // Someone searching for "100%" must not silently get "100".
        let p = query_pairs("/api/ask?q=100%");
        assert_eq!(get(&p, "q"), Some("100%"));
    }

    #[test]
    fn no_query_string_yields_no_pairs() {
        assert!(query_pairs("/api/health").is_empty());
    }

    #[test]
    fn parse_ask_reads_query_params() {
        let p = query_pairs("/api/ask?q=graph&k=7&scenes=a,b");
        let r = parse_ask(&p, "").expect("valid");
        assert_eq!(r.query, "graph");
        assert_eq!(r.budget, 7);
        assert_eq!(r.scenes, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn parse_ask_reads_a_json_body() {
        let r = parse_ask(&[], r#"{"query":"graph","budget":3,"scenes":["x"]}"#).expect("valid");
        assert_eq!(r.query, "graph");
        assert_eq!(r.budget, 3);
        assert_eq!(r.scenes, vec!["x".to_string()]);
    }

    #[test]
    fn budget_is_clamped_to_at_least_one() {
        let p = query_pairs("/api/ask?q=x&k=0");
        assert_eq!(parse_ask(&p, "").expect("valid").budget, 1);
        assert_eq!(parse_ask(&[], r#"{"query":"x","budget":0}"#).unwrap().budget, 1);
    }

    #[test]
    fn a_missing_or_empty_query_is_a_client_error() {
        assert!(parse_ask(&query_pairs("/api/ask?k=5"), "").is_err());
        assert!(parse_ask(&query_pairs("/api/ask?q="), "").is_err());
        assert!(parse_ask(&[], r#"{"budget":5}"#).is_err());
    }

    #[test]
    fn a_non_numeric_budget_is_rejected_rather_than_silently_defaulted() {
        assert!(parse_ask(&query_pairs("/api/ask?q=x&k=lots"), "").is_err());
    }

    /// The status/code mapping is part of the API contract.
    #[test]
    fn errors_map_to_meaningful_status_codes() {
        let (c, b) = map_error(&anyhow::anyhow!("no index at /tmp/x — run `spraypaint index`"));
        assert_eq!(c, 404);
        assert!(b.contains("no_index"));

        let (c, b) = map_error(&anyhow::anyhow!("identity fingerprint mismatch (Inv 1): stored a"));
        assert_eq!(c, 409, "a mismatch is actionable — re-index — so it is a conflict");
        assert!(b.contains("identity_mismatch"));

        let (c, _) = map_error(&anyhow::anyhow!("disk on fire"));
        assert_eq!(c, 500);
    }
}

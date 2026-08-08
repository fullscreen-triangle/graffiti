//! The shared core: every operation that touches the index, the phase lock, or
//! the committed count. Both the CLI (`main.rs`) and the HTTP server
//! (`serve/`) call these functions, so the invariant-critical *ordering* is
//! written down once.
//!
//! Rules for this module:
//!
//!   * It returns values. It never prints and never exits. Rendering belongs to
//!     `output.rs`, exit codes to `main.rs`, status codes to `serve/`.
//!   * The ordering inside [`ask`] is load-bearing (Inv 2/3) and is annotated
//!     inline. Do not reorder it to "tidy" the function.

use std::path::Path;

use anyhow::Result;

use crate::config::SprayConfig;
use crate::index::schema::{Identity, Index};
use crate::{ask as ask_mod, count, index, phase, root, walk};

/// A search request. Mirrors `ask`'s argv 1:1 so the CLI and the HTTP API
/// cannot diverge in what they accept.
#[derive(Debug, Clone)]
pub struct AskRequest {
    pub query: String,
    /// Water-filling budget A (`-k` / `--budget`).
    pub budget: usize,
    /// Restrict to named scenes; empty means all scenes.
    pub scenes: Vec<String>,
}

impl AskRequest {
    pub fn new(query: impl Into<String>, budget: usize, scenes: Vec<String>) -> Self {
        AskRequest {
            query: query.into(),
            budget,
            scenes,
        }
    }
}

/// A committed search: the outcome plus the count of the act that produced it.
pub struct AskResponse {
    pub outcome: ask_mod::AskOutcome,
    /// The committed count *after* this act (Inv 2).
    pub count: u64,
    /// The identity fingerprint of the index the answer came from (Inv 1).
    pub fingerprint: String,
}

/// A dry run: diagnostics only. No answer is emitted and the count is untouched.
pub struct DryRunResponse {
    pub outcome: ask_mod::AskOutcome,
    pub fingerprint: String,
}

/// One scene as reported by `scenes`.
pub struct SceneSummary {
    pub name: String,
    pub documents: usize,
    pub passages: u32,
}

/// What an index build produced.
pub struct IndexSummary {
    pub root: String,
    pub documents: usize,
    pub passages: usize,
    pub scenes: usize,
    pub identity_fingerprint: String,
}

/// What an index dry-run would have done.
pub struct IndexDryRun {
    pub root: String,
    pub would_index: usize,
}

/// Commit one search act (the COMMITMENT phase).
///
/// The ordering below is the invariant, not an implementation detail:
///
///   1. shared phase guard  — Inv 4: never overlaps an exclusive index write
///   2. `index::load`       — Inv 1: recomputes and asserts the fingerprint
///   3. `ask::run`          — Inv 3: a fresh search; snippets re-read from disk
///   4. `count::commit`     — Inv 2: exactly one act, *after* the search and
///                            *before* the answer reaches the caller, so
///                            "no answer without committing >= 1 act" holds
pub fn ask(root_dir: &Path, req: &AskRequest) -> Result<AskResponse> {
    let _guard = phase::PhaseGuard::commitment(root_dir)?;
    let idx = index::load(root_dir)?;
    let outcome = ask_mod::run(root_dir, &idx, &req.query, req.budget, &req.scenes)?;
    let count = count::commit(root_dir)?;
    Ok(AskResponse {
        outcome,
        count,
        fingerprint: idx.identity.fingerprint.clone(),
    })
}

/// Diagnostics without committing (Inv 3: a zero-act read-out emits no answer).
///
/// Identical to [`ask`] up to step 3, then returns *without* touching the count.
/// Interactive previews must use this: the count is monotone with no decrement
/// path, so a slider that committed on every drag would inflate it permanently.
pub fn dry_run(root_dir: &Path, req: &AskRequest) -> Result<DryRunResponse> {
    let _guard = phase::PhaseGuard::commitment(root_dir)?;
    let idx = index::load(root_dir)?;
    let outcome = ask_mod::run(root_dir, &idx, &req.query, req.budget, &req.scenes)?;
    Ok(DryRunResponse {
        outcome,
        fingerprint: idx.identity.fingerprint.clone(),
    })
}

/// The conserved-identity block (Inv 1). Loading verifies the fingerprint.
pub fn identity(root_dir: &Path) -> Result<Identity> {
    Ok(index::load(root_dir)?.identity)
}

/// The monotone committed count (Inv 2).
pub fn count(root_dir: &Path) -> Result<u64> {
    count::read(root_dir)
}

/// The detected (or `scenes.toml`-overridden) scene groups.
pub fn scenes(root_dir: &Path) -> Result<Vec<SceneSummary>> {
    let idx = index::load(root_dir)?;
    Ok(idx
        .scenes
        .iter()
        .map(|s| SceneSummary {
            name: s.name.clone(),
            documents: s.doc_ids.len(),
            passages: s.stats.passage_count,
        })
        .collect())
}

/// Load the index without consuming it — used where a caller needs the whole
/// structure (e.g. the server's cache) rather than one projection of it.
pub fn load_index(root_dir: &Path) -> Result<Index> {
    index::load(root_dir)
}

/// Report what an index build would cover, writing nothing.
pub fn index_dry_run(root_dir: &Path, cfg: &SprayConfig) -> IndexDryRun {
    let files = walk::walk(root_dir, cfg);
    IndexDryRun {
        root: root_dir.display().to_string(),
        would_index: files.len(),
    }
}

/// Build and persist the index (the CONSTRUCTION phase).
///
/// Takes the *exclusive* phase lock for the whole build+write (Inv 4), so no
/// commitment-phase reader can observe a half-written index.
pub fn build_index(root_dir: &Path, cfg: &SprayConfig) -> Result<IndexSummary> {
    let _guard = phase::PhaseGuard::construction(root_dir)?;
    let idx = index::build(root_dir, cfg)?;
    index::save(root_dir, &idx)?;
    Ok(IndexSummary {
        root: root_dir.display().to_string(),
        documents: idx.documents.len(),
        passages: idx.documents.iter().map(|d| d.passages.len()).sum(),
        scenes: idx.scenes.len(),
        identity_fingerprint: idx.identity.fingerprint.clone(),
    })
}

// ─────────────────────────── verify ───────────────────────────

/// The outcome of one check. Three states, not two.
///
/// The third state is the point. A check that *could not run* — because the
/// corpus is degenerate, or the counter has never been written, or another
/// process holds the lock — has not passed, but neither has it found a breach.
/// Folding it into PASS is the failure mode this whole phase exists to fix:
/// it reports "verified" for a repo where the invariant was never exercised
/// (`prin:refusal` — a framework with no refusals has an empty defined class).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    /// The check did not apply or could not be run. Never evidence of
    /// conformance; surfaces as exit code 2.
    NotApplicable,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::NotApplicable => "N/A",
        }
    }
    pub fn is_fail(self) -> bool {
        matches!(self, Status::Fail)
    }
    pub fn is_na(self) -> bool {
        matches!(self, Status::NotApplicable)
    }
}

/// The result of one named check. A breach is a *value*, not an error:
/// `verify`'s whole job is to report breaches, so returning `Err` for one would
/// conflate "the invariant is broken" with "the check could not run" — which
/// are precisely the two things [`Status`] keeps apart.
pub struct CheckResult {
    /// Stable machine-readable name, e.g. `fingerprint`, `stored_fields`.
    pub name: &'static str,
    pub status: Status,
    pub detail: String,
}

impl CheckResult {
    pub fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        CheckResult { name, status: Status::Pass, detail: detail.into() }
    }
    pub fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        CheckResult { name, status: Status::Fail, detail: detail.into() }
    }
    pub fn na(name: &'static str, detail: impl Into<String>) -> Self {
        CheckResult { name, status: Status::NotApplicable, detail: detail.into() }
    }
}

/// One invariant: a heading plus the named checks that make it up.
pub struct InvariantReport {
    /// e.g. `"Inv 1 conserved identity"`.
    pub title: &'static str,
    pub checks: Vec<CheckResult>,
}

impl InvariantReport {
    /// The invariant's status is its worst check: any FAIL fails it, otherwise
    /// any N/A makes it not-applicable.
    pub fn status(&self) -> Status {
        if self.checks.iter().any(|c| c.status.is_fail()) {
            Status::Fail
        } else if self.checks.iter().any(|c| c.status.is_na()) {
            Status::NotApplicable
        } else {
            Status::Pass
        }
    }
}

/// The four-invariant conformance report.
pub struct VerifyReport {
    pub inv1: InvariantReport,
    pub inv2: InvariantReport,
    pub inv3: InvariantReport,
    pub inv4: InvariantReport,
    /// Degenerate regimes detected in the corpus itself, each of which makes
    /// some invariant vacuous rather than verified.
    pub degeneracies: Vec<String>,
}

impl VerifyReport {
    pub fn invariants(&self) -> [&InvariantReport; 4] {
        [&self.inv1, &self.inv2, &self.inv3, &self.inv4]
    }

    /// Every individual check, in report order.
    pub fn checks(&self) -> impl Iterator<Item = &CheckResult> {
        self.invariants().into_iter().flat_map(|i| i.checks.iter())
    }

    pub fn any_fail(&self) -> bool {
        self.checks().any(|c| c.status.is_fail())
    }

    pub fn any_na(&self) -> bool {
        self.checks().any(|c| c.status.is_na())
    }

    /// Overall status. PASS requires every check to pass, none to be N/A, **and
    /// no degenerate regime to have been detected**.
    ///
    /// The third clause is not redundant with the second. A degenerate corpus
    /// makes checks pass *vacuously* rather than making them unrunnable: on a
    /// single-document repo every Inv 1 check returns PASS, because
    /// `char_invariant` returns the floor without examining a graph that has no
    /// bipartition to examine. Those PASS marks are arithmetic, not evidence.
    ///
    /// Without this clause the report contradicted itself — it printed
    /// "degenerate regimes (a PASS here would not be evidence)" immediately
    /// above `overall: PASS`. Degeneracies were detected and displayed but had
    /// no effect on the verdict, which is the same vacuous-certification defect
    /// this phase exists to remove, one level up.
    pub fn status(&self) -> Status {
        if self.any_fail() {
            Status::Fail
        } else if self.any_na() || !self.degeneracies.is_empty() {
            Status::NotApplicable
        } else {
            Status::Pass
        }
    }

    /// Backwards-compatible boolean for existing JSON consumers.
    ///
    /// Retained so a parser reading the top-level `pass` field keeps working
    /// across this release. It is deliberately *strict*: N/A is not `true`.
    pub fn pass(&self) -> bool {
        self.status() == Status::Pass
    }

    /// The process exit code this report implies. See `main.rs`.
    pub fn exit_code(&self, allow_degenerate: bool) -> i32 {
        match self.status() {
            Status::Pass => 0,
            Status::Fail => 1,
            Status::NotApplicable => {
                if allow_degenerate {
                    0
                } else {
                    2
                }
            }
        }
    }
}

/// Relative tolerance for comparing a recomputed f64 against a stored one.
/// The stored value made a round trip through JSON, so bit equality is the
/// wrong test; anything looser than this would start hiding real tampering.
const CHI_REL_TOL: f64 = 1e-9;

/// Recompute all four invariants. Never returns `Err` — see [`CheckResult`].
pub fn verify(root_dir: &Path) -> VerifyReport {
    let mut degeneracies: Vec<String> = Vec::new();

    let inv1 = verify_identity(root_dir, &mut degeneracies);
    let inv2 = verify_count(root_dir);
    let inv3 = InvariantReport {
        title: "Inv 3 search-not-fetch",
        checks: vec![verify_no_answer_cache(root_dir)],
    };
    let inv4 = verify_phases(root_dir);

    // Scene degeneracy is not an Inv-1 property but it does make the
    // water-filling allocation vacuous: with fewer than two scenes there is
    // nothing to allocate *across*, so a price of zero is arithmetic rather
    // than evidence.
    if let Ok(idx) = index::load(root_dir) {
        if idx.scenes.len() < 2 {
            degeneracies.push(format!(
                "single scene ({}): water-filling has nothing to allocate across",
                idx.scenes.len()
            ));
        }
    }

    VerifyReport { inv1, inv2, inv3, inv4, degeneracies }
}

/// Inv 1, as four separately-named checks.
///
/// Splitting it matters because the old single check reported one PASS covering
/// a genuinely strong test (the fingerprint) and a vacuous one (`chi >= floor`),
/// so the strong result was indistinguishable from the empty one.
fn verify_identity(root_dir: &Path, degeneracies: &mut Vec<String>) -> InvariantReport {
    let title = "Inv 1 conserved identity";

    // Check 1 — fingerprint. `index::load` recomputes the self-graph digest
    // from the stored documents and rejects a mismatch. This is the strongest
    // check the crate has and it is kept verbatim.
    let idx = match index::load(root_dir) {
        Ok(i) => i,
        Err(e) => {
            // Nothing downstream can run without an index, so report the one
            // failure rather than four copies of it.
            return InvariantReport {
                title,
                checks: vec![CheckResult::fail("fingerprint", format!("{e}"))],
            };
        }
    };
    let mut checks = vec![CheckResult::pass(
        "fingerprint",
        format!(
            "recomputed self-graph digest matches stored {}",
            short_fp(&idx.identity.fingerprint)
        ),
    )];

    let g = index::identity::build_self_graph(&idx.documents);
    let chi = index::identity::char_invariant(&g);
    let stats = index::identity::edge_stats(&g);

    // Check 2 — chi_floor. Kept as a PASS criterion by explicit decision, but
    // its detail must not overclaim: the inequality is arithmetic, not
    // evidence. Reporting frac_at_floor beside it is what makes the line
    // informative — that number *can* vary with the corpus.
    let floor_ok = chi >= idx.identity.floor && idx.identity.floor > 0.0;
    checks.push(if floor_ok {
        CheckResult::pass(
            "chi_floor",
            format!(
                "chi={:.6} >= floor={:.2e} (holds by construction: every edge carries the floor, \
                 so this cannot fail); {:.1}% of edges are at the floor",
                chi,
                idx.identity.floor,
                stats.frac_at_floor * 100.0
            ),
        )
    } else {
        // Unreachable given build_self_graph, and that is exactly why it is
        // worth reporting loudly: it would mean the graph builder changed.
        CheckResult::fail(
            "chi_floor",
            format!(
                "chi={:.6} < floor={:.2e} — impossible for a floor-complete graph; \
                 the self-graph builder is not adding the floor",
                chi, idx.identity.floor
            ),
        )
    });

    // Check 3 — stored_fields. The one here that can genuinely fail. `load()`
    // verifies the fingerprint, which is computed from `documents` alone, so
    // char_invariant / n_vertices / n_edges are stored but never re-checked: a
    // tampered chi passes today entirely unnoticed.
    let mut mismatches: Vec<String> = Vec::new();
    let stored_chi = idx.identity.char_invariant;
    if (stored_chi - chi).abs() > CHI_REL_TOL * chi.abs().max(1.0) {
        mismatches.push(format!("char_invariant stored={stored_chi} recomputed={chi}"));
    }
    if idx.identity.n_vertices as usize != g.verts.len() {
        mismatches.push(format!(
            "n_vertices stored={} recomputed={}",
            idx.identity.n_vertices,
            g.verts.len()
        ));
    }
    if idx.identity.n_edges as usize != g.edges.len() {
        mismatches.push(format!(
            "n_edges stored={} recomputed={}",
            idx.identity.n_edges,
            g.edges.len()
        ));
    }
    if idx.identity.floor != index::identity::FLOOR {
        mismatches.push(format!(
            "floor stored={} expected={}",
            idx.identity.floor,
            index::identity::FLOOR
        ));
    }
    checks.push(if mismatches.is_empty() {
        CheckResult::pass(
            "stored_fields",
            format!(
                "char_invariant, floor, n_vertices={}, n_edges={} all match recomputation",
                g.verts.len(),
                g.edges.len()
            ),
        )
    } else {
        CheckResult::fail(
            "stored_fields",
            format!("identity block does not match recomputation: {}", mismatches.join("; ")),
        )
    });

    // Check 4 — schema_version. Verified inside `load()`; reaching here means
    // it matched. Surfaced as its own line so the check is visible rather than
    // implicit in a successful load.
    checks.push(CheckResult::pass(
        "schema_version",
        format!(
            "index schema v{} matches this build",
            idx.schema_version
        ),
    ));

    // Degeneracy: recorded once, here, where the graph is already built.
    if let Some(d) = index::identity::classify(&g) {
        degeneracies.push(d.reason().to_string());
    }

    InvariantReport { title, checks }
}

/// Inv 2 — the counter is readable and monotone.
fn verify_count(root_dir: &Path) -> InvariantReport {
    let title = "Inv 2 never-resetting count";
    // `count::read` maps both "absent" and "unparseable" to Ok(0), which would
    // report a corrupt counter as a healthy zero. `read_strict` separates them:
    // absent is a repo that has committed nothing (N/A — nothing to verify),
    // unparseable is a breach (the monotone value is gone).
    let check = match count::read_strict(root_dir) {
        Ok(Some(c)) => CheckResult::pass(
            "count_readable",
            format!("committed count = {c}; u64 with no decrement path in the code"),
        ),
        Ok(None) => CheckResult::na(
            "count_readable",
            "no counter file yet: nothing has been committed, so monotonicity is untested",
        ),
        Err(e) => CheckResult::fail("count_readable", format!("{e}")),
    };
    InvariantReport { title, checks: vec![check] }
}

/// Inv 4 — the phase lock is operational.
fn verify_phases(root_dir: &Path) -> InvariantReport {
    let title = "Inv 4 exclusive phases";
    // Non-blocking: the blocking variant would hang for the duration of a
    // concurrent index build, which is the one situation where a user is most
    // likely to run `verify` to find out what is going on.
    let check = match phase::PhaseGuard::try_construction(root_dir) {
        Ok(Some(_g)) => CheckResult::pass(
            "lock_operational",
            "exclusive construction lock acquired and released",
        ),
        Ok(None) => CheckResult::na(
            "lock_operational",
            "another process holds the phase lock (an index build is likely running); \
             exclusion could not be tested from here",
        ),
        Err(e) => CheckResult::fail("lock_operational", format!("{e}")),
    };
    InvariantReport { title, checks: vec![check] }
}

/// First 12 hex chars of a `b3:`-prefixed digest, for readable output.
fn short_fp(fp: &str) -> String {
    let hex = fp.strip_prefix("b3:").unwrap_or(fp);
    format!("b3:{}…", &hex[..hex.len().min(12)])
}

/// Structural Inv 3 check: the persisted index must contain no stored answer
/// bodies. Passages hold `terms`/line ranges; we confirm no answer-bearing
/// *field key* leaked into the schema. We parse the JSON and inspect object
/// keys rather than scanning raw text — otherwise indexed source code that
/// merely *mentions* "snippet"/"body" (e.g. spraypaint's own source) would
/// trip a substring scan. The distinction matters: a value that contains the
/// word is fine; a key that stores an answer is not.
fn verify_no_answer_cache(root_dir: &Path) -> CheckResult {
    const NAME: &str = "no_answer_cache";
    let path = root::index_path(root_dir);
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(e) => return CheckResult::fail(NAME, format!("{e}")),
    };
    let value: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => return CheckResult::fail(NAME, format!("parse: {e}")),
    };
    const BANNED_KEYS: &[&str] = &["snippet", "body", "answer", "cached", "text", "content"];
    if let Some(bad) = find_banned_key(&value, BANNED_KEYS) {
        return CheckResult::fail(
            NAME,
            format!("index leaks stored answers: object key '{bad}'"),
        );
    }
    CheckResult::pass(
        NAME,
        "index stores no answer fields; snippets re-read at query time",
    )
}

/// Recursively search a JSON value for any object key in `banned`.
fn find_banned_key(value: &serde_json::Value, banned: &[&str]) -> Option<String> {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if banned.iter().any(|b| b.eq_ignore_ascii_case(k)) {
                    return Some(k.clone());
                }
                if let Some(found) = find_banned_key(v, banned) {
                    return Some(found);
                }
            }
            None
        }
        serde_json::Value::Array(arr) => arr.iter().find_map(|v| find_banned_key(v, banned)),
        _ => None,
    }
}

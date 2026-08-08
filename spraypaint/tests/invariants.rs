//! End-to-end conformance tests for the four blueprint invariants.
//!
//! These shell out to the compiled binary rather than calling the library
//! directly, deliberately: the invariants are properties of a *process* — the
//! phase lock, the on-disk counter, the persisted index — and several of them
//! (notably Inv 2 under concurrency and Inv 4's exclusion) are only observable
//! across process boundaries. In-process unit tests live beside their modules.
//!
//! Run with `cargo test`. Each block maps to one invariant.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build a small fixture repo under a unique temp dir; return its root.
fn make_fixture(tag: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("spraypaint-test-{tag}"));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(base.join("alpha")).unwrap();
    fs::create_dir_all(base.join("beta")).unwrap();
    // A .git marker so root detection stops here.
    fs::create_dir_all(base.join(".git")).unwrap();

    fs::write(
        base.join("alpha").join("waterfill.rs"),
        "fn water_fill(scenes: &[Vec<f64>]) -> usize {\n    // allocate attention across scenes by a single price\n    scenes.len()\n}\n",
    )
    .unwrap();
    fs::write(
        base.join("beta").join("notes.md"),
        "# Attention\n\nThe agent divides attention across concurrent scenes by water filling.\n",
    )
    .unwrap();
    fs::write(
        base.join("readme.md"),
        "# Fixture\n\nA tiny corpus about attention and scenes.\n",
    )
    .unwrap();
    base
}

fn bin() -> PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for integration tests.
    PathBuf::from(env!("CARGO_BIN_EXE_spraypaint"))
}

fn run(root: &Path, args: &[&str]) -> (String, String, i32) {
    let out = Command::new(bin())
        .args(args)
        .arg("--root")
        .arg(root)
        .output()
        .expect("spawn spraypaint");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

fn index(root: &Path) {
    let (_o, e, code) = run(root, &["index"]);
    assert_eq!(code, 0, "index failed: {e}");
}

#[test]
fn inv1_identity_conserved_under_reindex() {
    let root = make_fixture("inv1");
    index(&root);
    let (fp1, _, _) = run(&root, &["identity", "--json"]);
    // Re-index (content unchanged) must yield the same fingerprint.
    index(&root);
    let (fp2, _, _) = run(&root, &["identity", "--json"]);
    assert_eq!(fp1, fp2, "fingerprint changed across re-index");
    // chi >= floor > 0.
    assert!(fp1.contains("char_invariant"));
}

#[test]
fn inv2_count_monotone_never_resets() {
    let root = make_fixture("inv2");
    index(&root);
    let start = read_count(&root);
    for _ in 0..3 {
        let (_o, e, code) = run(&root, &["ask", "attention scenes"]);
        assert_eq!(code, 0, "ask failed: {e}");
    }
    assert_eq!(read_count(&root), start + 3, "count did not advance by 3");
    // Re-index must NOT reset the count.
    index(&root);
    assert_eq!(read_count(&root), start + 3, "re-index reset the count");
    // Dry-run must NOT increment.
    let (_o, _e, code) = run(&root, &["ask", "attention", "--dry-run"]);
    assert_eq!(code, 0);
    assert_eq!(read_count(&root), start + 3, "dry-run incremented the count");
}

/// Inv 2 under concurrency — the property a single-threaded loop cannot show.
///
/// `ask` holds only a SHARED phase lock, precisely so several can run at once.
/// If the counter's read-modify-write is not separately serialised, two of them
/// read N and both write N+1: an increment is lost, and because the counter has
/// no decrement path there is nothing that later restores it. `serve` answering
/// concurrent requests makes this routine rather than theoretical.
///
/// Structured as several rounds rather than one large batch. A single round
/// only collides when two processes happen to interleave inside the
/// read-modify-write window, which on a small fixture is a minority of runs —
/// measured at roughly 40% against the unlocked implementation. That would be a
/// test which usually goes green on broken code, i.e. worse than none. Rounds
/// are independent trials, so the miss probability falls off geometrically:
/// at ~0.6 per round, six rounds leave well under 5%. The assertion is checked
/// per round so a failure names the round that lost the increment.
#[test]
fn inv2_count_survives_concurrent_asks() {
    const PER_ROUND: usize = 8;
    const ROUNDS: usize = 6;
    let root = make_fixture("inv2conc");
    index(&root);
    let mut expected = read_count(&root);

    for round in 0..ROUNDS {
        // Spawn the whole round first, *then* wait. Sequential spawn-and-wait
        // would never overlap and would pass even with the race present.
        let children: Vec<_> = (0..PER_ROUND)
            .map(|_| {
                Command::new(bin())
                    .args(["ask", "attention scenes", "--json"])
                    .arg("--root")
                    .arg(&root)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .expect("spawn spraypaint")
            })
            .collect();
        for mut c in children {
            let st = c.wait().expect("wait");
            assert!(st.success(), "a concurrent ask failed in round {round}: {st:?}");
        }
        expected += PER_ROUND as u64;
        assert_eq!(
            read_count(&root),
            expected,
            "lost increment in round {round}: {PER_ROUND} concurrent asks \
             did not advance the count by {PER_ROUND}"
        );
    }
}

fn read_count(root: &Path) -> u64 {
    let (o, _, _) = run(root, &["count", "--json"]);
    // crude parse of {"committed_count":N}
    o.split(':')
        .nth(1)
        .and_then(|s| s.trim().trim_matches(|c: char| !c.is_ascii_digit()).parse().ok())
        .unwrap_or(0)
}

#[test]
fn inv3_search_not_fetch_deterministic_and_no_cache() {
    let root = make_fixture("inv3");
    index(&root);
    // Same (index, query) -> identical results (JSON, minus the volatile count).
    let (a, _, _) = run(&root, &["ask", "water filling attention", "--json"]);
    let (b, _, _) = run(&root, &["ask", "water filling attention", "--json"]);
    let strip = |s: &str| {
        s.lines()
            .filter(|l| !l.contains("committed_count"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&a), strip(&b), "identical query not reproducible");
    // The index file stores no answer bodies.
    let idx = fs::read_to_string(root.join(".spraypaint").join("index.json")).unwrap();
    for banned in ["\"snippet\"", "\"body\"", "\"answer\""] {
        assert!(!idx.contains(banned), "index leaked stored answers: {banned}");
    }
}

#[test]
fn inv3_results_change_after_content_edit() {
    let root = make_fixture("inv3edit");
    index(&root);
    let (before, _, _) = run(&root, &["ask", "kuramoto synchronisation", "--json"]);
    // Add new content mentioning the term, re-index (fresh search, not fetch).
    fs::write(
        root.join("beta").join("sync.md"),
        "# Sync\n\nKuramoto synchronisation drives dispersed phases to lock.\n",
    )
    .unwrap();
    index(&root);
    let (after, _, _) = run(&root, &["ask", "kuramoto synchronisation", "--json"]);
    assert_ne!(before, after, "results did not reflect new content");
}

/// A freshly-indexed repo with nothing committed is NOT verified.
///
/// This test previously asserted PASS/exit-0 here, and that assertion was the
/// bug in miniature: `index` alone never touches the counter, so Inv 2 was
/// reported green on a fixture that had not exercised it once. Exit 2 is the
/// honest answer — the check could not run. Asserting it here is what stops
/// the vacuous PASS from being reinstated as "fixing a failing test".
#[test]
fn verify_is_not_applicable_before_anything_is_committed() {
    let root = make_fixture("inv4fresh");
    index(&root);
    let (o, _e, code) = run(&root, &["verify"]);
    assert_eq!(code, 2, "expected NOT-APPLICABLE (exit 2), got {code}: {o}");
    assert!(o.contains("overall: N/A"), "expected overall N/A: {o}");
    assert!(
        o.contains("no counter file yet"),
        "N/A should name the untested invariant: {o}"
    );
    // The invariants that *can* be checked without a commit still are — N/A is
    // scoped to the one check that lacks evidence, not smeared over the report.
    assert!(o.contains("Inv 1 conserved identity     [PASS]"), "{o}");
    assert!(o.contains("Inv 4 exclusive phases       [PASS]"), "{o}");
}

/// All four invariants pass once the repo has actually exercised them.
#[test]
fn inv4_verify_passes() {
    let root = make_fixture("inv4");
    index(&root);
    // Commit one act so Inv 2 has something to read. Without this the counter
    // file does not exist and the run is legitimately N/A (see above).
    run(&root, &["ask", "attention scenes"]);
    let (o, _e, code) = run(&root, &["verify"]);
    assert_eq!(code, 0, "verify failed: {o}");
    assert!(o.contains("overall: PASS"), "verify not PASS: {o}");
    // Every check individually, so a future N/A cannot hide inside an overall
    // PASS if the aggregation rule is ever loosened.
    for check in [
        "fingerprint",
        "chi_floor",
        "stored_fields",
        "schema_version",
        "count_readable",
        "no_answer_cache",
        "lock_operational",
    ] {
        assert!(o.contains(check), "missing check {check} in report: {o}");
    }
    assert!(!o.contains("[FAIL]"), "unexpected FAIL: {o}");
    assert!(!o.contains("[N/A]"), "unexpected N/A: {o}");
}

/// `--allow-degenerate` maps exit 2 to 0 without suppressing anything else.
#[test]
fn allow_degenerate_flag_maps_two_to_zero() {
    let root = make_fixture("inv4allowdeg");
    index(&root);
    let (_, _, plain) = run(&root, &["verify"]);
    assert_eq!(plain, 2, "fixture should be N/A without the flag");
    let (o, _, code) = run(&root, &["verify", "--allow-degenerate"]);
    assert_eq!(code, 0, "flag should map 2 -> 0: {o}");
    // The flag changes the exit code, not the report: the N/A is still stated
    // so the operator can see what was waived rather than what was verified.
    assert!(o.contains("overall: N/A"), "flag must not rewrite the report: {o}");
}

#[test]
fn scenes_are_top_level_dirs() {
    let root = make_fixture("scenes");
    index(&root);
    let (o, _, _) = run(&root, &["scenes"]);
    assert!(o.contains("alpha"), "missing alpha scene: {o}");
    assert!(o.contains("beta"), "missing beta scene: {o}");
}

// ---------------------------------------------------------------------------
// Negative controls.
//
// Everything above asserts that a correct index is accepted. That is only half
// a conformance suite, and the weaker half: a `verify` whose every check was
// replaced by `true` would pass all of it. These tests corrupt an index in one
// specific way each and require the *matching* check to object.
//
// The invariants are claims about what the tool refuses (`prin:refusal` — a
// framework with no refusals has an empty defined class), so the refusals are
// what has to be tested.
// ---------------------------------------------------------------------------

fn index_json_path(root: &Path) -> PathBuf {
    root.join(".spraypaint").join("index.json")
}

fn read_index_json(root: &Path) -> serde_json::Value {
    let raw = fs::read_to_string(index_json_path(root)).expect("read index.json");
    serde_json::from_str(&raw).expect("parse index.json")
}

fn write_index_json(root: &Path, v: &serde_json::Value) {
    fs::write(index_json_path(root), serde_json::to_string_pretty(v).unwrap())
        .expect("write index.json");
}

/// Assert a command fails *and says why*.
///
/// The substring is not decoration. Asserting only `code != 0` would keep this
/// suite green if a check were swapped for one that rejects everything, or if
/// the binary started failing for an unrelated reason (a panic, a missing
/// file) — the test would pass while testing nothing. Naming the expected
/// message ties the failure to the specific check under test.
fn expect_fail(root: &Path, args: &[&str], needle: &str) {
    let (o, e, code) = run(root, args);
    let combined = format!("{o}{e}");
    assert_ne!(code, 0, "expected failure but exit was 0: {combined}");
    assert!(
        combined.contains(needle),
        "failed for the wrong reason — expected {needle:?} in output:\n{combined}"
    );
}

/// Tampering with a stored document hash must break the fingerprint.
///
/// This is `index::load`'s recompute-and-compare, the crate's strongest single
/// check, which had no coverage at all before this test.
#[test]
fn verify_fails_on_mutated_content_hash() {
    let root = make_fixture("negcontenthash");
    index(&root);
    let mut v = read_index_json(&root);
    let h = v["documents"][0]["content_hash"].as_str().unwrap().to_string();
    // Flip one hex digit: minimal edit, and it keeps the field well-formed so
    // we are testing the identity check rather than the JSON parser.
    let flipped = format!(
        "{}{}",
        &h[..h.len() - 1],
        if h.ends_with('a') { 'b' } else { 'a' }
    );
    v["documents"][0]["content_hash"] = serde_json::Value::String(flipped);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "fingerprint mismatch");
}

/// Tampering with the *stored* fingerprint must also be caught — the same
/// comparison from the other side.
#[test]
fn verify_fails_on_flipped_fingerprint() {
    let root = make_fixture("negfingerprint");
    index(&root);
    let mut v = read_index_json(&root);
    let fp = v["identity"]["fingerprint"].as_str().unwrap().to_string();
    let flipped = format!(
        "{}{}",
        &fp[..fp.len() - 1],
        if fp.ends_with('a') { 'b' } else { 'a' }
    );
    v["identity"]["fingerprint"] = serde_json::Value::String(flipped);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "fingerprint mismatch");
}

/// The load-bearing test for the new `stored_fields` check.
///
/// `index::load` recomputes the fingerprint from `documents` alone, so nothing
/// in the identity block other than `fingerprint` was ever re-derived. A
/// tampered `char_invariant` therefore passed verification silently: the tool
/// would print a chi it had not computed and call it verified. This test fails
/// against any build without `stored_fields`.
#[test]
fn verify_fails_on_tampered_char_invariant() {
    let root = make_fixture("negchi");
    index(&root);
    let mut v = read_index_json(&root);
    v["identity"]["char_invariant"] = serde_json::json!(999.0);
    write_index_json(&root, &v);
    // The fingerprint still matches — that check is computed over `documents`
    // and is blind to this edit. Only `stored_fields` can catch it.
    let (o, _, _) = run(&root, &["verify"]);
    assert!(
        o.contains("fingerprint      [PASS]"),
        "fingerprint should be unaffected by an identity-block edit: {o}"
    );
    expect_fail(&root, &["verify"], "char_invariant stored=999");
}

#[test]
fn verify_fails_on_tampered_n_vertices() {
    let root = make_fixture("negnverts");
    index(&root);
    let mut v = read_index_json(&root);
    v["identity"]["n_vertices"] = serde_json::json!(4242);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "n_vertices stored=4242");
}

#[test]
fn verify_fails_on_tampered_n_edges() {
    let root = make_fixture("negnedges");
    index(&root);
    let mut v = read_index_json(&root);
    v["identity"]["n_edges"] = serde_json::json!(4242);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "n_edges stored=4242");
}

/// The floor is a construction parameter, so an index claiming a different one
/// was not built by this binary.
#[test]
fn verify_fails_on_tampered_floor() {
    let root = make_fixture("negfloor");
    index(&root);
    let mut v = read_index_json(&root);
    v["identity"]["floor"] = serde_json::json!(0.5);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "floor stored=0.5");
}

/// `SCHEMA_VERSION` was written on every save and never read back, so an index
/// from an incompatible build was parsed best-effort and its differences
/// silently misinterpreted.
#[test]
fn verify_fails_on_tampered_schema_version() {
    let root = make_fixture("negschema");
    index(&root);
    let mut v = read_index_json(&root);
    v["schema_version"] = serde_json::json!(99);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "schema version mismatch");
}

/// Reordering passages within a document must NOT change the fingerprint.
///
/// Written as a PASS deliberately. Order-independence is the design, not an
/// oversight: `doc_term_vector` merges a document's passages into a single
/// `BTreeMap` bag and `build_self_graph` canonicalises documents by sorting on
/// `content_hash`. That is relabelling invariance (`thm:identity`(ii)), the
/// property the fingerprint exists to have. A test asserting failure here would
/// encode a misunderstanding of the invariant and then "fix" the code to match.
#[test]
fn passage_reorder_is_relabelling_and_passes() {
    let root = make_fixture("negreorder");
    // The fixture files are a few lines each and the default window is 40, so
    // every document is a single passage — nothing to reorder. Write one long
    // enough to span several windows, and index with a small window so the
    // split does not depend on the default staying 40.
    let long: String = (0..60)
        .map(|i| format!("line {i} about attention scenes and water filling\n"))
        .collect();
    fs::write(root.join("alpha").join("long.md"), long).unwrap();
    let (_o, e, code) = run(&root, &["index", "--window", "10", "--overlap", "2"]);
    assert_eq!(code, 0, "index failed: {e}");
    // Commit an act so Inv 2 is exercised; otherwise the run is N/A for an
    // unrelated reason and this test would not be measuring identity at all.
    run(&root, &["ask", "attention scenes"]);

    let before = read_index_json(&root)["identity"]["fingerprint"]
        .as_str()
        .unwrap()
        .to_string();

    let mut v = read_index_json(&root);
    // Find a document with at least two passages and reverse them.
    let mut reordered = false;
    for doc in v["documents"].as_array_mut().unwrap() {
        let ps = doc["passages"].as_array_mut().unwrap();
        if ps.len() >= 2 {
            ps.reverse();
            reordered = true;
            break;
        }
    }
    assert!(reordered, "fixture has no multi-passage document to reorder");
    write_index_json(&root, &v);

    // Plain verify, no escape hatch: this fixture has two scenes and a
    // committed act, so a non-zero exit here would be the identity check
    // objecting to the reorder — which is the thing under test.
    let (o, _e, code) = run(&root, &["verify"]);
    assert_eq!(code, 0, "passage order must not affect identity: {o}");
    let after = read_index_json(&root)["identity"]["fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(before, after, "fingerprint should be order-invariant");
}

/// The pair to the test above: an edit that changes the *self-graph* is not a
/// relabelling and must change the fingerprint. Without this, order-invariance
/// could be satisfied by a fingerprint that ignores passage content entirely.
///
/// Removing a term is the right edit to make here. See the sibling test below
/// for why raising a term frequency is not.
#[test]
fn removing_a_term_does_change_the_fingerprint() {
    let root = make_fixture("negretermdrop");
    index(&root);
    // Commit an act first. Without one, Inv 2 is N/A and `verify` exits 2
    // before the identity reason is visible — `expect_fail` would then see a
    // non-zero exit for the wrong reason, which is exactly what it guards.
    run(&root, &["ask", "attention scenes"]);
    let mut v = read_index_json(&root);
    let terms = v["documents"][0]["passages"][0]["terms"]
        .as_array_mut()
        .expect("passage terms");
    assert!(!terms.is_empty(), "fixture passage has no terms");
    terms.remove(0);
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "fingerprint mismatch");
}

/// Raising a term frequency on the *higher* side of a pair does NOT change the
/// fingerprint — and that is correct, not a gap.
///
/// The fingerprint digests the self-graph, not the raw passage bytes. Edge
/// weight is `FLOOR + sum over shared terms of min(tf_a, tf_b)`, so raising the
/// larger of the two frequencies leaves every `min` — and therefore every edge,
/// and therefore the whole graph — untouched. Inv 1 is conservation of
/// *identity*, and identity is the graph.
///
/// This is pinned as its own test because the insensitivity looks like a bug
/// until you follow the arithmetic, and the tempting "fix" is to digest raw
/// passage bytes instead. That would break `passage_reorder_is_relabelling`
/// above: the two properties are in tension, and this pair marks the boundary
/// the design deliberately draws between them.
#[test]
fn raising_a_term_frequency_leaves_the_graph_unchanged() {
    let root = make_fixture("negretfraise");
    index(&root);
    run(&root, &["ask", "attention scenes"]);
    let before = read_index_json(&root)["identity"]["fingerprint"]
        .as_str()
        .unwrap()
        .to_string();

    let mut v = read_index_json(&root);
    // Every fixture document has tf=1 for its terms, so *raising* one keeps
    // min(1, 1+7) = 1 on every edge it participates in.
    let terms = v["documents"][0]["passages"][0]["terms"]
        .as_array_mut()
        .expect("passage terms");
    let old = terms[0][1].as_u64().unwrap();
    terms[0][1] = serde_json::json!(old + 7);
    write_index_json(&root, &v);

    let (o, _e, code) = run(&root, &["verify"]);
    assert_eq!(code, 0, "min-preserving edit must not disturb identity: {o}");
    let after = read_index_json(&root)["identity"]["fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        before, after,
        "edge weights are min-based; raising the larger side changes no edge"
    );
}

/// First-ever coverage of `verify_no_answer_cache` (Inv 3).
///
/// The index must store *no* answer. If a cached snippet ever appeared in it,
/// `ask` would be fetching a previous answer rather than searching, which is
/// precisely the failure Inv 3 names.
#[test]
fn verify_fails_on_injected_banned_key() {
    let root = make_fixture("negbanned");
    index(&root);
    let mut v = read_index_json(&root);
    v["documents"][0]["snippet"] = serde_json::json!("a cached answer");
    write_index_json(&root, &v);
    expect_fail(&root, &["verify"], "snippet");
}

/// An unparseable counter is corruption, and corruption is a FAIL — not a
/// silent zero. `count::read` maps both absent and unparseable to `Ok(0)`,
/// which is why `read_strict` exists.
#[test]
fn verify_fails_on_corrupt_count() {
    let root = make_fixture("negcount");
    index(&root);
    run(&root, &["ask", "attention scenes"]);
    fs::write(root.join(".spraypaint").join("count"), "not-a-number").unwrap();
    expect_fail(&root, &["verify"], "count");
}

/// A single-document corpus cannot exercise the graph invariants: there is no
/// bipartition, so `char_invariant` returns the floor without examining
/// anything. Reporting PASS there would certify a check that never ran.
#[test]
fn verify_not_applicable_on_single_document_repo() {
    let base = std::env::temp_dir().join("spraypaint-test-negsingledoc");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(base.join(".git")).unwrap();
    fs::create_dir_all(base.join("only")).unwrap();
    fs::write(
        base.join("only").join("one.md"),
        "# Only\n\nA single document about attention.\n",
    )
    .unwrap();
    index(&base);
    run(&base, &["ask", "attention"]);

    let (o, _e, code) = run(&base, &["verify"]);
    assert_eq!(code, 2, "single-document repo should be N/A: {o}");
    assert!(
        o.contains("single document"),
        "should name the degenerate regime: {o}"
    );
    // And the escape hatch works on exactly this case.
    let (_, _, code) = run(&base, &["verify", "--allow-degenerate"]);
    assert_eq!(code, 0);
}

/// A corpus whose documents share no vocabulary produces a graph that is the
/// floor and nothing else, so chi is FLOOR by fiat rather than by measurement.
#[test]
fn verify_not_applicable_on_floor_only_graph() {
    let base = std::env::temp_dir().join("spraypaint-test-negflooronly");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(base.join(".git")).unwrap();
    fs::create_dir_all(base.join("aa")).unwrap();
    fs::create_dir_all(base.join("bb")).unwrap();
    // Disjoint vocabularies: no term occurs in both files.
    fs::write(base.join("aa").join("a.md"), "alpha bravo charlie delta\n").unwrap();
    fs::write(base.join("bb").join("b.md"), "xi omicron rho sigma\n").unwrap();
    index(&base);
    run(&base, &["ask", "alpha"]);

    let (o, _e, code) = run(&base, &["verify"]);
    assert_eq!(code, 2, "floor-only graph should be N/A: {o}");
    assert!(
        o.contains("every edge is at the floor"),
        "should name the floor-only regime: {o}"
    );
    // And the honest number is surfaced rather than hidden behind a PASS mark.
    assert!(o.contains("100.0% of edges are at the floor"), "{o}");
}

/// One scene means water-filling has nothing to allocate *across*, so the
/// allocation is arithmetic rather than evidence.
#[test]
fn verify_not_applicable_on_single_scene_repo() {
    let base = std::env::temp_dir().join("spraypaint-test-negsinglescene");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(base.join(".git")).unwrap();
    fs::create_dir_all(base.join("solo")).unwrap();
    fs::write(
        base.join("solo").join("a.md"),
        "# A\n\nAttention across scenes by water filling.\n",
    )
    .unwrap();
    fs::write(
        base.join("solo").join("b.md"),
        "# B\n\nWater filling allocates attention to scenes.\n",
    )
    .unwrap();
    index(&base);
    run(&base, &["ask", "attention"]);

    let (o, _e, code) = run(&base, &["verify"]);
    assert_eq!(code, 2, "single-scene repo should be N/A: {o}");
    assert!(o.contains("single scene"), "should name the regime: {o}");
}

/// `scene.rs`'s `scenes.toml` override had no coverage.
#[test]
fn scenes_toml_override_is_honoured() {
    let root = make_fixture("scenestoml");
    // The override lives in `.spraypaint/`, not at the repo root (root.rs's
    // `scenes_path`). Create the dir since nothing has indexed here yet.
    let spray = root.join(".spraypaint");
    fs::create_dir_all(&spray).unwrap();
    fs::write(
        spray.join("scenes.toml"),
        "[scenes]\nengine = [\"alpha\"]\nprose = [\"beta\", \"readme.md\"]\n",
    )
    .unwrap();
    index(&root);
    let (o, _, code) = run(&root, &["scenes"]);
    assert_eq!(code, 0, "scenes failed: {o}");
    // Compare scene *names* (first column) rather than substrings of the whole
    // line: the override's prefixes are the directory names, so a naive
    // `contains("alpha")` could match a prefix and pass either way.
    let names: Vec<String> = o
        .lines()
        .filter_map(|l| l.split_whitespace().next().map(str::to_string))
        .collect();
    assert!(names.iter().any(|n| n == "engine"), "override scene 'engine' missing: {o}");
    assert!(names.iter().any(|n| n == "prose"), "override scene 'prose' missing: {o}");
    assert!(
        !names.iter().any(|n| n == "alpha" || n == "beta" || n == "(root)"),
        "directory-derived scenes should be replaced by the override, not merged: {o}"
    );
}

//! End-to-end conformance tests for the four blueprint invariants, driving the
//! installed `spraypaint` behaviour through the library's public build/ask path
//! is not possible (binary crate), so these tests shell out to the compiled
//! binary via `assert`-style checks over a temp fixture repo.
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

#[test]
fn inv4_verify_passes() {
    let root = make_fixture("inv4");
    index(&root);
    let (o, _e, code) = run(&root, &["verify"]);
    assert_eq!(code, 0, "verify failed: {o}");
    assert!(o.contains("overall: PASS"), "verify not PASS: {o}");
}

#[test]
fn scenes_are_top_level_dirs() {
    let root = make_fixture("scenes");
    index(&root);
    let (o, _, _) = run(&root, &["scenes"]);
    assert!(o.contains("alpha"), "missing alpha scene: {o}");
    assert!(o.contains("beta"), "missing beta scene: {o}");
}

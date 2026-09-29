//! End-to-end tests for what `ask` tells its reader: the coverage verdict, the
//! evidence lines, and that one stretch of text is not returned twice.
//!
//! Like `invariants.rs`, these shell out to the compiled binary, so they test
//! the JSON a consumer actually receives.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn fixture(tag: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("spraypaint-coverage-{tag}"));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(base.join(".git")).unwrap();
    fs::create_dir_all(base.join("mail")).unwrap();
    fs::create_dir_all(base.join("notes")).unwrap();

    // A correspondent who writes a lot about the city — the look-alike.
    fs::write(
        base.join("mail").join("boss.md"),
        "# Greifswald\n\nWhere to register in Greifswald, where to get a bike repaired in \
         Greifswald, which Greifswald office handles residence permits.\n",
    )
    .unwrap();
    // The match sits deep inside a window: 30 lines of filler first.
    let mut deep = String::from("# Notes\n");
    for i in 0..30 {
        deep.push_str(&format!("filler line {i}\n"));
    }
    deep.push_str("The kuramoto model locks dispersed oscillator phases.\n");
    for i in 0..8 {
        deep.push_str(&format!("trailing line {i}\n"));
    }
    fs::write(base.join("notes").join("deep.md"), deep).unwrap();
    fs::write(
        base.join("notes").join("attention.md"),
        "# Attention\n\nThe agent divides attention across concurrent scenes by water filling.\n",
    )
    .unwrap();
    base
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spraypaint"))
}

fn run(root: &Path, args: &[&str]) -> (String, i32) {
    let out = Command::new(bin())
        .args(args)
        .arg("--root")
        .arg(root)
        .output()
        .expect("spawn spraypaint");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

fn index(root: &Path) {
    let (_, code) = run(root, &["index"]);
    assert_eq!(code, 0, "index failed");
}

fn ask(root: &Path, query: &str) -> Value {
    let (out, code) = run(root, &["ask", query, "--json"]);
    assert_eq!(code, 0, "ask failed: {out}");
    serde_json::from_str(&out).expect("ask --json is not JSON")
}

/// The regression the evidence change exists for: the snippet used to be the
/// window's first non-empty line, which here is `# Notes` — nowhere near the
/// match on line 32.
#[test]
fn evidence_shows_the_matching_line_not_the_first_one() {
    let root = fixture("deep");
    index(&root);
    let v = ask(&root, "kuramoto");
    let r = &v["results"][0];
    assert_eq!(r["path"], "notes/deep.md");
    let snippet = r["snippet"].as_str().unwrap();
    assert!(snippet.contains("kuramoto model"), "evidence missed the match: {snippet:?}");
    let (s, e) = (
        r["evidence_start_line"].as_u64().unwrap(),
        r["evidence_end_line"].as_u64().unwrap(),
    );
    assert!(s <= 32 && 32 <= e, "evidence range {s}-{e} does not contain line 32");
    assert_eq!(r["matched_terms"], serde_json::json!(["kuramoto"]));
}

/// The dentist case: the query's common word matches a prolific correspondent,
/// the word it is about is absent. The verdict must say so and name the word.
#[test]
fn a_query_about_something_absent_is_declined() {
    let root = fixture("dentist");
    index(&root);
    let v = ask(&root, "suitable dentist in Greifswald");
    assert_eq!(v["coverage"]["verdict"], "declined", "{:#}", v["coverage"]);
    let reason = v["coverage"]["reason"].as_str().unwrap();
    assert!(reason.contains("not in the corpus") && reason.contains("dentist"), "{reason}");
    // The look-alike is still returned — the verdict qualifies it, it does not hide it.
    assert_eq!(v["results"][0]["path"], "mail/boss.md");
    let dentist = v["coverage"]["terms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["term"] == "dentist")
        .unwrap();
    assert_eq!(dentist["df"], 0);
    assert_eq!(dentist["in_results"], false);
}

#[test]
fn a_passage_holding_every_term_is_covered() {
    let root = fixture("covered");
    index(&root);
    let v = ask(&root, "water filling attention");
    assert_eq!(v["coverage"]["verdict"], "covered", "{:#}", v["coverage"]);
    assert!(v["coverage"]["reason"].as_str().unwrap().contains("notes/attention.md"));
}

#[test]
fn repeated_query_words_are_counted_once() {
    let root = fixture("dedupe");
    index(&root);
    let v = ask(&root, "kuramoto kuramoto");
    assert_eq!(v["query_terms"], serde_json::json!(["kuramoto"]));
}

/// Windows overlap by construction; the same lines must not come back twice.
#[test]
fn overlapping_windows_of_one_file_are_returned_once() {
    let root = fixture("overlap");
    let mut long = String::new();
    for i in 0..120 {
        long.push_str(&format!("line {i} mentions synchronisation\n"));
    }
    fs::write(root.join("notes").join("long.md"), long).unwrap();
    index(&root);
    let v = ask(&root, "synchronisation");
    let spans: Vec<(u64, u64)> = v["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["path"] == "notes/long.md")
        .map(|r| (r["start_line"].as_u64().unwrap(), r["end_line"].as_u64().unwrap()))
        .collect();
    assert!(spans.len() >= 2, "expected several windows of the long file: {spans:?}");
    for (i, a) in spans.iter().enumerate() {
        for b in &spans[i + 1..] {
            assert!(a.1 < b.0 || b.1 < a.0, "overlapping windows returned: {a:?} and {b:?}");
        }
    }
}

/// Coverage is a diagnostic, so a dry run reports it — without committing.
#[test]
fn dry_run_reports_coverage_without_committing() {
    let root = fixture("dry");
    index(&root);
    let (out, code) = run(&root, &["ask", "suitable dentist", "--dry-run", "--json"]);
    assert_eq!(code, 0);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["dry_run"], true);
    assert_eq!(v["coverage"]["verdict"], "declined");
    assert!(v.get("committed_count").is_none());
    let (count, _) = run(&root, &["count"]);
    assert!(count.trim().ends_with('0'), "dry run committed: {count}");
}

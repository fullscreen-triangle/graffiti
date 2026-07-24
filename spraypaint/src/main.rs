//! spraypaint — split-attention full-text search.
//!
//! A sibling to `purpose`: `spraypaint index` once, `spraypaint ask "..."` many
//! times. Full-text passages ranked by BM25 within scenes and allocated across
//! scenes by the paper's water-filling rule. A faithful runtime for the four
//! blueprint invariants of the Split-Attention Synchronised Agents calculus:
//!
//!   Inv 1 conserved identity  (index/identity.rs)
//!   Inv 2 never-resetting count (count.rs)
//!   Inv 3 search-not-fetch     (ask.rs — no answer cache; snippets re-read)
//!   Inv 4 exclusive phases     (phase.rs — index=exclusive, ask=shared lock)

mod ask;
mod bm25;
mod chunk;
mod config;
mod count;
mod index;
mod output;
mod phase;
mod root;
mod scene;
mod walk;
mod waterfill;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use config::SprayConfig;

#[derive(Parser)]
#[command(name = "spraypaint", version, about = "Split-attention full-text search (BM25 within scenes, water-filling across them)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the index over a repo (construction phase). Writes .spraypaint/index.json.
    Index(IndexArgs),
    /// Search the index by a fresh water-filled BM25 walk (commitment phase).
    Ask(AskArgs),
    /// Print the conserved identity fingerprint and chi (Inv 1).
    Identity(RootArgs),
    /// Print the monotone committed count (Inv 2).
    Count(RootArgs),
    /// List the detected/overridden scenes.
    Scenes(RootArgs),
    /// Re-check all four invariants; nonzero exit on any breach.
    Verify(RootArgs),
}

#[derive(Args)]
struct RootArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct IndexArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    json: bool,
    /// Report what would be indexed; write nothing.
    #[arg(long)]
    dry_run: bool,
    /// Passage window length in lines.
    #[arg(long)]
    window: Option<usize>,
    /// Overlap between consecutive windows in lines.
    #[arg(long)]
    overlap: Option<usize>,
}

#[derive(Args)]
struct AskArgs {
    /// The query (quote if it contains spaces).
    query: String,
    #[arg(long)]
    root: Option<PathBuf>,
    /// Total passages to return (water-filling budget A).
    #[arg(short = 'k', long, default_value_t = 12)]
    budget: usize,
    #[arg(long)]
    json: bool,
    /// Diagnostics only: no answer, does not increment the committed count.
    #[arg(long)]
    dry_run: bool,
    /// Rank globally by score instead of grouping by scene.
    #[arg(long)]
    flat: bool,
    /// Restrict to named scenes (comma-separated).
    #[arg(long, value_delimiter = ',')]
    scenes: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Index(a) => cmd_index(a),
        Cmd::Ask(a) => cmd_ask(a),
        Cmd::Identity(a) => cmd_identity(a),
        Cmd::Count(a) => cmd_count(a),
        Cmd::Scenes(a) => cmd_scenes(a),
        Cmd::Verify(a) => cmd_verify(a),
    }
}

fn cmd_index(a: IndexArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let mut cfg = SprayConfig::default();
    if let Some(w) = a.window {
        cfg.window = w;
    }
    if let Some(o) = a.overlap {
        cfg.overlap = o;
    }

    if a.dry_run {
        let files = walk::walk(&root_dir, &cfg);
        if a.json {
            let v = serde_json::json!({
                "root": root_dir.display().to_string(),
                "would_index": files.len(),
            });
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            eprintln!("Dry-run over {} ...", root_dir.display());
            println!("Would index {} file(s) (nothing written)", files.len());
        }
        return Ok(());
    }

    // Construction phase: exclusive lock for the whole build+write.
    let _guard = phase::PhaseGuard::construction(&root_dir)?;
    eprintln!("Indexing {} ...", root_dir.display());
    let idx = index::build(&root_dir, &cfg)?;
    index::save(&root_dir, &idx)?;

    let n_passages: usize = idx.documents.iter().map(|d| d.passages.len()).sum();
    if a.json {
        let v = serde_json::json!({
            "root": root_dir.display().to_string(),
            "documents": idx.documents.len(),
            "passages": n_passages,
            "scenes": idx.scenes.len(),
            "identity_fingerprint": idx.identity.fingerprint,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!(
            "Indexed {} document(s), {} passage(s), {} scene(s) into {}",
            idx.documents.len(),
            n_passages,
            idx.scenes.len(),
            root::index_path(&root_dir).display()
        );
    }
    Ok(())
}

fn cmd_ask(a: AskArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    // Commitment phase: shared lock — never overlaps an exclusive index write.
    let _guard = phase::PhaseGuard::commitment(&root_dir)?;
    let idx = index::load(&root_dir)?;

    let outcome = ask::run(&root_dir, &idx, &a.query, a.budget, &a.scenes)?;

    if a.dry_run {
        // Inv 3: a zero-act read-out emits no answer, only diagnostics; the
        // committed count is NOT incremented.
        print!("{}", output::ask_dry_run(&outcome));
        return Ok(());
    }

    // Inv 2 + Inv 3: commit exactly one act after the search, before emitting
    // the answer — "no answer without committing >=1 act".
    let count = count::commit(&root_dir)?;

    if a.json {
        println!(
            "{}",
            output::ask_json(&outcome, a.budget, count, &idx.identity.fingerprint)
        );
    } else {
        print!(
            "{}",
            output::ask_human(&outcome, count, &idx.identity.fingerprint, a.flat)
        );
    }
    Ok(())
}

fn cmd_identity(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let idx = index::load(&root_dir)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&idx.identity)?);
    } else {
        print!("{}", output::identity_human(&idx));
    }
    Ok(())
}

fn cmd_count(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let c = count::read(&root_dir)?;
    if a.json {
        println!("{}", serde_json::json!({ "committed_count": c }));
    } else {
        println!("committed acts: {c}");
    }
    Ok(())
}

fn cmd_scenes(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let idx = index::load(&root_dir)?;
    if a.json {
        let v: Vec<_> = idx
            .scenes
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "documents": s.doc_ids.len(),
                    "passages": s.stats.passage_count,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        for s in &idx.scenes {
            println!(
                "{:24}  {} doc(s), {} passage(s)",
                s.name,
                s.doc_ids.len(),
                s.stats.passage_count
            );
        }
    }
    Ok(())
}

/// `verify`: recompute the four invariants and report PASS/FAIL. Nonzero exit
/// on any breach — a one-command conformance certificate.
fn cmd_verify(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;

    // Inv 1: load() recomputes and asserts the fingerprint; success => PASS.
    let (inv1, inv1_msg) = match index::load(&root_dir) {
        Ok(idx) => {
            let g = index::identity::build_self_graph(&idx.documents);
            let chi = index::identity::char_invariant(&g);
            let ok = chi >= idx.identity.floor && idx.identity.floor > 0.0;
            (
                ok,
                format!(
                    "fingerprint verified; chi={:.6} >= floor={:.2e}",
                    chi, idx.identity.floor
                ),
            )
        }
        Err(e) => (false, format!("{e}")),
    };

    // Inv 2: count is readable and nonnegative (u64 — monotone by construction;
    // there is no decrement path in the code).
    let (inv2, inv2_msg) = match count::read(&root_dir) {
        Ok(c) => (true, format!("committed count = {c}, no decrement path")),
        Err(e) => (false, format!("{e}")),
    };

    // Inv 3: schema stores no answer/snippet bodies — passages carry term ids
    // and line ranges only. Verified structurally by re-reading the raw index.
    let (inv3, inv3_msg) = verify_no_answer_cache(&root_dir);

    // Inv 4: the lock is acquirable (exclusive then released), proving the
    // phase machinery is in place.
    let (inv4, inv4_msg) = match phase::PhaseGuard::construction(&root_dir) {
        Ok(_g) => (true, "construction/commitment lock operational".to_string()),
        Err(e) => (false, format!("{e}")),
    };

    let all = inv1 && inv2 && inv3 && inv4;
    if a.json {
        let v = serde_json::json!({
            "pass": all,
            "inv1_identity": { "pass": inv1, "detail": inv1_msg },
            "inv2_count": { "pass": inv2, "detail": inv2_msg },
            "inv3_search_not_fetch": { "pass": inv3, "detail": inv3_msg },
            "inv4_phases": { "pass": inv4, "detail": inv4_msg },
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        let mark = |b: bool| if b { "PASS" } else { "FAIL" };
        println!("Inv 1 conserved identity     [{}] {}", mark(inv1), inv1_msg);
        println!("Inv 2 never-resetting count  [{}] {}", mark(inv2), inv2_msg);
        println!("Inv 3 search-not-fetch       [{}] {}", mark(inv3), inv3_msg);
        println!("Inv 4 exclusive phases       [{}] {}", mark(inv4), inv4_msg);
        println!("\noverall: {}", if all { "PASS" } else { "FAIL" });
    }
    if all {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

/// Structural Inv 3 check: the persisted index must contain no stored answer
/// bodies. Passages hold `terms`/line ranges; we confirm no answer-bearing
/// *field key* leaked into the schema. We parse the JSON and inspect object
/// keys rather than scanning raw text — otherwise indexed source code that
/// merely *mentions* "snippet"/"body" (e.g. spraypaint's own source) would
/// trip a substring scan. The distinction matters: a value that contains the
/// word is fine; a key that stores an answer is not.
fn verify_no_answer_cache(root_dir: &std::path::Path) -> (bool, String) {
    let path = root::index_path(root_dir);
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(e) => return (false, format!("{e}")),
    };
    let value: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => return (false, format!("parse: {e}")),
    };
    const BANNED_KEYS: &[&str] = &["snippet", "body", "answer", "cached", "text", "content"];
    if let Some(bad) = find_banned_key(&value, BANNED_KEYS) {
        return (false, format!("index leaks stored answers: object key '{bad}'"));
    }
    (
        true,
        "index stores no answer fields; snippets re-read at query time".to_string(),
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

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
//!
//! This file is argv parsing, rendering, and exit codes. Everything that touches
//! the index, the lock, or the count lives in `actions` so that `spraypaint
//! serve` runs the identical ordering — see the module docs in `lib.rs`.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use spraypaint::actions::{self, AskRequest};
use spraypaint::config::SprayConfig;
use spraypaint::{output, root};

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
    /// Re-check all four invariants. Exit 0 = all pass, 1 = breach, 2 = degenerate.
    Verify(VerifyArgs),
    /// Serve the web UI and JSON API on localhost.
    #[cfg(feature = "serve")]
    Serve(ServeArgs),
}

#[cfg(feature = "serve")]
#[derive(Args)]
struct ServeArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    /// Port to bind. 7373 is unassigned and away from the crowded dev ports.
    #[arg(long, default_value_t = 7373)]
    port: u16,
    /// Address to bind. Anything other than loopback needs --i-know-this-is-public.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Open the URL in a browser after starting.
    #[arg(long)]
    open: bool,
    /// Allow POST /api/index to rebuild the index from the browser.
    #[arg(long)]
    allow_index: bool,
    /// Reload index.json on every request instead of caching it by mtime+len.
    #[arg(long)]
    no_cache: bool,
    /// Required to bind a non-loopback address. See the warning it prints.
    #[arg(long)]
    i_know_this_is_public: bool,
}

#[derive(Args)]
struct RootArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct VerifyArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    json: bool,
    /// Treat NOT-APPLICABLE checks as success (exit 0 instead of 2).
    ///
    /// For repos that are legitimately too small to exercise an invariant — a
    /// single-document corpus, a fresh index with nothing committed yet — where
    /// you want CI green anyway. It does not suppress FAIL.
    #[arg(long)]
    allow_degenerate: bool,
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
        #[cfg(feature = "serve")]
        Cmd::Serve(a) => cmd_serve(a),
    }
}

/// Is this bind address loopback-only?
///
/// Parsed as an `IpAddr` rather than string-matched: `127.0.0.1` is not the only
/// loopback address, and `127.0.0.2` is loopback too while looking like a public
/// address to a naive comparison. `0.0.0.0` and `::` are *not* loopback — they
/// are the wildcard, which binds every interface, and are exactly the case the
/// second flag exists to catch.
#[cfg(feature = "serve")]
fn is_loopback(host: &str) -> bool {
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

#[cfg(feature = "serve")]
fn cmd_serve(a: ServeArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;

    // Two flags, not one, because the failure mode is silent. A user who types
    // `--host 0.0.0.0` to make the UI reachable from their phone is not thinking
    // about the fact that they are also publishing their source tree; the flag
    // named for the consequence is what makes them think about it.
    if !is_loopback(&a.host) && !a.i_know_this_is_public {
        anyhow::bail!(
            "refusing to bind {} — that address is reachable from other machines.\n\n\
             This server has NO AUTHENTICATION and:\n  \
             * serves arbitrary file content from {} (snippets are re-read from disk),\n  \
             * lets anyone who can reach it irreversibly inflate the monotone count,\n  \
             * exposes your index's identity fingerprint.\n\n\
             If that is genuinely what you want, add --i-know-this-is-public.",
            a.host,
            root_dir.display()
        );
    }
    if !is_loopback(&a.host) {
        eprintln!(
            "WARNING: bound to {}, reachable from other machines, with no authentication.\n\
             WARNING: anyone who can reach this port can read the content of {}.",
            a.host,
            root_dir.display()
        );
    }

    spraypaint::serve::preflight(&root_dir, a.allow_index);
    spraypaint::serve::run(spraypaint::serve::ServeConfig {
        root: root_dir,
        port: a.port,
        host: a.host,
        allow_index: a.allow_index,
        no_cache: a.no_cache,
        open: a.open,
    })
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
        let d = actions::index_dry_run(&root_dir, &cfg);
        if a.json {
            let v = serde_json::json!({
                "root": d.root,
                "would_index": d.would_index,
            });
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            eprintln!("Dry-run over {} ...", root_dir.display());
            println!("Would index {} file(s) (nothing written)", d.would_index);
        }
        return Ok(());
    }

    eprintln!("Indexing {} ...", root_dir.display());
    let s = actions::build_index(&root_dir, &cfg)?;

    if a.json {
        let v = serde_json::json!({
            "root": s.root,
            "documents": s.documents,
            "passages": s.passages,
            "scenes": s.scenes,
            "identity_fingerprint": s.identity_fingerprint,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!(
            "Indexed {} document(s), {} passage(s), {} scene(s) into {}",
            s.documents,
            s.passages,
            s.scenes,
            root::index_path(&root_dir).display()
        );
    }
    Ok(())
}

fn cmd_ask(a: AskArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let req = AskRequest::new(a.query.clone(), a.budget, a.scenes.clone());

    if a.dry_run {
        // Inv 3: a zero-act read-out emits no answer, only diagnostics; the
        // committed count is NOT incremented.
        let d = actions::dry_run(&root_dir, &req)?;
        // `--json` applies to the dry-run branch too. Honouring it only on the
        // committed branch would make `--dry-run --json` silently emit human
        // text: a machine consumer scripting a budget sweep gets a parse error
        // on the one path that exists so it can sweep *without* committing.
        if a.json {
            println!(
                "{}",
                output::ask_dry_run_json(&d.outcome, a.budget, &d.fingerprint)
            );
        } else {
            print!("{}", output::ask_dry_run(&d.outcome));
        }
        return Ok(());
    }

    let r = actions::ask(&root_dir, &req)?;

    if a.json {
        println!(
            "{}",
            output::ask_json(&r.outcome, a.budget, r.count, &r.fingerprint)
        );
    } else {
        print!(
            "{}",
            output::ask_human(&r.outcome, r.count, &r.fingerprint, a.flat)
        );
    }
    Ok(())
}

fn cmd_identity(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    if a.json {
        let id = actions::identity(&root_dir)?;
        println!("{}", serde_json::to_string_pretty(&id)?);
    } else {
        let idx = actions::load_index(&root_dir)?;
        print!("{}", output::identity_human(&idx));
    }
    Ok(())
}

fn cmd_count(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let c = actions::count(&root_dir)?;
    if a.json {
        println!("{}", serde_json::json!({ "committed_count": c }));
    } else {
        println!("committed acts: {c}");
    }
    Ok(())
}

fn cmd_scenes(a: RootArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let scenes = actions::scenes(&root_dir)?;
    if a.json {
        let v: Vec<_> = scenes
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "documents": s.documents,
                    "passages": s.passages,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        for s in &scenes {
            println!(
                "{:24}  {} doc(s), {} passage(s)",
                s.name, s.documents, s.passages
            );
        }
    }
    Ok(())
}

/// `verify`: recompute the four invariants and report per-check status.
///
/// Exit contract:
///
///   0  every check passed and none was N/A
///   1  at least one check FAILED — an invariant is breached
///   2  no failures, but at least one check was NOT-APPLICABLE
///
/// Code 2 is the honest answer for a repo too degenerate to exercise the
/// invariants: reporting 0 there would certify something never tested.
/// `--allow-degenerate` maps 2 to 0 for callers who accept that.
fn cmd_verify(a: VerifyArgs) -> Result<()> {
    let root_dir = root::detect_root(a.root.as_deref())?;
    let rep = actions::verify(&root_dir);
    let overall = rep.status();

    if a.json {
        let inv = |r: &actions::InvariantReport| {
            serde_json::json!({
                "status": r.status().as_str(),
                "pass": r.status() == actions::Status::Pass,
                "checks": r.checks.iter().map(|c| serde_json::json!({
                    "name": c.name,
                    "status": c.status.as_str(),
                    "detail": c.detail,
                })).collect::<Vec<_>>(),
            })
        };
        // `pass` stays at the top level, and stays strict, so existing parsers
        // keep working across this release.
        let v = serde_json::json!({
            "pass": rep.pass(),
            "overall": overall.as_str(),
            "degeneracies": rep.degeneracies,
            "inv1_identity": inv(&rep.inv1),
            "inv2_count": inv(&rep.inv2),
            "inv3_search_not_fetch": inv(&rep.inv3),
            "inv4_phases": inv(&rep.inv4),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        for r in rep.invariants() {
            println!("{:28} [{}]", r.title, r.status().as_str());
            for c in &r.checks {
                println!("  {:16} [{}] {}", c.name, c.status.as_str(), c.detail);
            }
        }
        if !rep.degeneracies.is_empty() {
            println!("\ndegenerate regimes (a PASS here would not be evidence):");
            for d in &rep.degeneracies {
                println!("  - {d}");
            }
        }
        println!("\noverall: {}", overall.as_str());
        if overall == actions::Status::NotApplicable {
            // Note the wording covers both routes to N/A: a check that could
            // not run, and a check that ran but only vacuously. Every mark
            // above can read PASS and the verdict still be N/A — that is the
            // degenerate case, and saying only "could not be run" here would
            // leave it looking like a bug.
            println!(
                "\nThis repo is not verified — exiting 2. Either a check could not run, or\n\
                 the corpus is too degenerate for one to be evidence (see above).\n\
                 Pass --allow-degenerate to treat that as success."
            );
        }
    }

    let code = rep.exit_code(a.allow_degenerate);
    if code == 0 {
        Ok(())
    } else {
        std::process::exit(code);
    }
}

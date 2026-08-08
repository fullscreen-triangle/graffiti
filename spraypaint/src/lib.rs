//! spraypaint — split-attention full-text search, as a library.
//!
//! The crate ships both a binary (`src/main.rs`) and this library. They are not
//! two implementations: the binary is a thin argv-and-printing shell over
//! [`actions`], which holds every operation that touches the index, the lock, or
//! the committed count.
//!
//! That split exists for one reason. `spraypaint serve` runs the same operations
//! in-process to answer HTTP requests, and the four blueprint invariants are
//! properties of *ordering* — commitment guard before load, commit after search
//! and before the answer. If the server re-implemented that ordering it would
//! drift from the CLI, and a breach would show up in one and not the other.
//! Both callers go through [`actions`] so there is exactly one ordering to audit.
//!
//!   Inv 1 conserved identity   (index/identity.rs)
//!   Inv 2 never-resetting count (count.rs)
//!   Inv 3 search-not-fetch      (ask.rs — no answer cache; snippets re-read)
//!   Inv 4 exclusive phases      (phase.rs — index=exclusive, ask=shared lock)

pub mod actions;
pub mod ask;
pub mod bm25;
pub mod chunk;
pub mod config;
pub mod count;
pub mod index;
pub mod output;
pub mod phase;
pub mod root;
pub mod scene;
pub mod walk;
pub mod waterfill;

/// The local HTTP server. Behind the default-on `serve` feature so that
/// `--no-default-features` still builds a CLI with no HTTP stack in the tree.
#[cfg(feature = "serve")]
pub mod serve;

/// The embedded web UI. Gated with `serve`, since `rust-embed` is only pulled
/// in by that feature and there is nothing to serve the assets to without it.
#[cfg(feature = "serve")]
pub mod ui;

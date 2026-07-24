//! Inv 4 — exclusive construction/commitment phases. Index-update (construction)
//! and query-serve (commitment) must never overlap. Enforced by an advisory
//! file lock on `.spraypaint/.lock`:
//!
//!   * `index` takes an EXCLUSIVE lock for the whole build+write.
//!   * `ask` takes a SHARED lock — many concurrent reads, but never while an
//!     exclusive write is in progress.
//!
//! Combined with `ask` holding only `&Index` (the borrow checker forbids
//! mutation during commitment), the "disjoint instants" predicate holds by
//! construction rather than by a runtime flag.

use std::fs::{File, OpenOptions};
use std::path::Path;

use anyhow::{Context, Result};
use fs2::FileExt;

use crate::root;

/// Which phase a guard represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Construction,
    Commitment,
}

/// Holds the advisory lock for the lifetime of a phase. Dropping releases it.
pub struct PhaseGuard {
    _file: File,
    /// Which phase this guard represents (introspection; the lock does the work).
    #[allow(dead_code)]
    pub phase: Phase,
}

fn open_lock(root_dir: &Path) -> Result<File> {
    let dir = root::spray_dir(root_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = root::lock_path(root_dir);
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("opening lock {}", path.display()))
}

impl PhaseGuard {
    /// Enter the construction phase (exclusive). Blocks until no reader or
    /// other writer holds the lock.
    pub fn construction(root_dir: &Path) -> Result<Self> {
        let file = open_lock(root_dir)?;
        file.lock_exclusive()
            .context("acquiring exclusive (construction) lock")?;
        Ok(PhaseGuard {
            _file: file,
            phase: Phase::Construction,
        })
    }

    /// Enter the commitment phase (shared). Blocks only while a writer holds
    /// the exclusive lock.
    pub fn commitment(root_dir: &Path) -> Result<Self> {
        let file = open_lock(root_dir)?;
        file.lock_shared()
            .context("acquiring shared (commitment) lock")?;
        Ok(PhaseGuard {
            _file: file,
            phase: Phase::Commitment,
        })
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        // Best-effort unlock; the OS also releases on close.
        let _ = self._file.unlock();
    }
}

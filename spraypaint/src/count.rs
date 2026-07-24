//! Inv 2 — never-resetting committed count. A monotone counter incremented by
//! one on every committed act (a non-dry-run `ask`), persisted across sessions,
//! never decremented. There is deliberately no reset or decrement path.

use std::path::Path;

use anyhow::{Context, Result};

use crate::root;

/// Read the current committed count (0 if the file is absent).
pub fn read(root_dir: &Path) -> Result<u64> {
    let path = root::count_path(root_dir);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s.trim().parse::<u64>().unwrap_or(0)),
        Err(_) => Ok(0),
    }
}

/// Commit one act: read, add one, persist atomically, return the new count.
/// Called exactly once per successful `ask`, after the search and before the
/// answer is emitted (so "no answer without committing >=1 act" holds).
pub fn commit(root_dir: &Path) -> Result<u64> {
    let dir = root::spray_dir(root_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let current = read(root_dir)?;
    let next = current.saturating_add(1);
    let path = root::count_path(root_dir);
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, next.to_string()).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(next)
}

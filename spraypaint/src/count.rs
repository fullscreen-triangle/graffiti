//! Inv 2 — never-resetting committed count. A monotone counter incremented by
//! one on every committed act (a non-dry-run `ask`), persisted across sessions,
//! never decremented. There is deliberately no reset or decrement path.

use std::fs::OpenOptions;
use std::path::Path;

use anyhow::{Context, Result};
use fs2::FileExt;

use crate::root;

/// Read the current committed count (0 if the file is absent).
///
/// Lenient by design for display paths. `verify` uses [`read_strict`] instead,
/// which distinguishes "absent" from "corrupt" — folding both to 0 here would
/// let a truncated counter pass an invariant check silently.
pub fn read(root_dir: &Path) -> Result<u64> {
    let path = root::count_path(root_dir);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s.trim().parse::<u64>().unwrap_or(0)),
        Err(_) => Ok(0),
    }
}

/// Read the count, distinguishing the three states [`read`] collapses:
///
///   * `Ok(None)`    — no counter file yet (nothing has been committed)
///   * `Ok(Some(n))` — a well-formed count
///   * `Err(_)`      — the file exists but does not parse: Inv 2 is unverifiable
///
/// A monotone counter that cannot be read is not a monotone counter that
/// happens to be zero, and `verify` must not report it as one.
pub fn read_strict(root_dir: &Path) -> Result<Option<u64>> {
    let path = root::count_path(root_dir);
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
    };
    let trimmed = raw.trim();
    trimmed
        .parse::<u64>()
        .map(Some)
        .map_err(|e| anyhow::anyhow!("count file {} is corrupt: {e} (contents: {trimmed:?})", path.display()))
}

/// Commit one act: read, add one, persist atomically, return the new count.
/// Called exactly once per successful `ask`, after the search and before the
/// answer is emitted (so "no answer without committing >=1 act" holds).
///
/// The read-modify-write is serialised by an exclusive lock on a *dedicated*
/// file (see [`root::count_lock_path`]). Without it, two concurrent `ask`
/// processes — which both hold only a SHARED phase lock, precisely so they can
/// run concurrently — can each read N and each write N+1, losing an increment
/// and silently breaking Inv 2. `serve` makes that concurrency routine rather
/// than theoretical, and the counter has no decrement path with which to
/// correct itself afterwards.
pub fn commit(root_dir: &Path) -> Result<u64> {
    let dir = root::spray_dir(root_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let lock_path = root::count_lock_path(root_dir);
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("opening count lock {}", lock_path.display()))?;
    lock.lock_exclusive()
        .with_context(|| format!("locking {}", lock_path.display()))?;

    // Everything below is under the exclusive count lock.
    let result = (|| -> Result<u64> {
        let current = read(root_dir)?;
        let next = current.saturating_add(1);
        let path = root::count_path(root_dir);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, next.to_string())
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming into {}", path.display()))?;
        Ok(next)
    })();

    let _ = lock.unlock();
    result
}

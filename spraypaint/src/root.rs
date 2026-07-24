//! Repo-root detection. Mirrors `purpose`: walk up from cwd (or an explicit
//! `--root`) until a directory contains `.git` or `.spraypaint`; else fall back
//! to the starting directory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The directory that holds a project's `.spraypaint/` index.
pub fn detect_root(explicit: Option<&Path>) -> Result<PathBuf> {
    let start = match explicit {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir().context("cannot read current directory")?,
    };
    let start = start
        .canonicalize()
        .with_context(|| format!("path does not exist: {}", start.display()))?;

    let mut cur: &Path = &start;
    loop {
        if cur.join(".git").exists() || cur.join(".spraypaint").exists() {
            return Ok(cur.to_path_buf());
        }
        match cur.parent() {
            Some(parent) => cur = parent,
            None => break,
        }
    }
    // No marker found — the start directory is the root.
    Ok(start)
}

/// `<root>/.spraypaint`
pub fn spray_dir(root: &Path) -> PathBuf {
    root.join(".spraypaint")
}

/// `<root>/.spraypaint/index.json`
pub fn index_path(root: &Path) -> PathBuf {
    spray_dir(root).join("index.json")
}

/// `<root>/.spraypaint/count`
pub fn count_path(root: &Path) -> PathBuf {
    spray_dir(root).join("count")
}

/// `<root>/.spraypaint/.lock`
pub fn lock_path(root: &Path) -> PathBuf {
    spray_dir(root).join(".lock")
}

/// `<root>/.spraypaint/scenes.toml`
pub fn scenes_path(root: &Path) -> PathBuf {
    spray_dir(root).join("scenes.toml")
}

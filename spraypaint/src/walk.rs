//! Tree walk producing the set of indexable files. Uses the `ignore` crate so
//! `.gitignore` is honoured for free, plus a hard skip list from `SprayConfig`
//! and size/binary filtering.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

use crate::config::{FileKind, SprayConfig};

/// A file selected for indexing, with its content class.
pub struct WalkedFile {
    pub path: PathBuf,
    /// Content class; reserved for kind-aware chunking/ranking. Not yet read.
    #[allow(dead_code)]
    pub kind: FileKind,
}

/// Walk `root`, returning indexable files in a deterministic (sorted) order so
/// the corpus is reproducible regardless of filesystem enumeration order.
pub fn walk(root: &Path, cfg: &SprayConfig) -> Vec<WalkedFile> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(false) // we do our own dot-dir filtering; don't blanket-skip dotfiles
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .parents(true);

    let skip_dirs = cfg.skip_dirs.clone();
    builder.filter_entry(move |entry| {
        // Skip a directory (and its subtree) if its name is in the skip list.
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            if let Some(name) = entry.file_name().to_str() {
                if skip_dirs.iter().any(|d| d == name) {
                    return false;
                }
            }
        }
        true
    });

    let mut files = Vec::new();
    for result in builder.build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => continue,
        };
        let kind = match cfg.kind_for_ext(ext) {
            Some(k) => k,
            None => continue,
        };
        // Size filter.
        if let Ok(meta) = entry.metadata() {
            if meta.len() > cfg.max_file_bytes {
                continue;
            }
        }
        files.push(WalkedFile {
            path: path.to_path_buf(),
            kind,
        });
    }

    // Deterministic order.
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files
}

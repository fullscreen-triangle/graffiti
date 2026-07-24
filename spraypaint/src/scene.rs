//! Scene partitioning. Scenes are the water-filling substrate: the finite query
//! budget is divided across them by a single price so no scene monopolises the
//! result slots. By default each top-level directory under the root is a scene
//! (loose root files form the `(root)` scene). A `.spraypaint/scenes.toml`
//! override maps custom scene names to path prefixes.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::index::schema::Document;
use crate::root;

#[derive(Debug, Deserialize)]
struct ScenesFile {
    /// name -> list of path prefixes (repo-relative, forward-slash).
    scenes: BTreeMap<String, Vec<String>>,
}

/// Return `(scene_name, doc_ids)` groups. Deterministic: names sorted, ids sorted.
pub fn detect(root_dir: &Path, documents: &[Document]) -> Result<Vec<(String, Vec<u32>)>> {
    let override_path = root::scenes_path(root_dir);
    if override_path.exists() {
        let text = std::fs::read_to_string(&override_path)
            .with_context(|| format!("reading {}", override_path.display()))?;
        let parsed: ScenesFile = toml::from_str(&text)
            .with_context(|| format!("parsing {}", override_path.display()))?;
        return Ok(partition_by_prefixes(documents, &parsed.scenes));
    }
    Ok(partition_by_toplevel(documents))
}

/// Default: first path segment is the scene name.
fn partition_by_toplevel(documents: &[Document]) -> Vec<(String, Vec<u32>)> {
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for doc in documents {
        let name = match doc.path.split('/').next() {
            Some(seg) if doc.path.contains('/') => seg.to_string(),
            _ => "(root)".to_string(),
        };
        groups.entry(name).or_default().push(doc.id);
    }
    finalize(groups)
}

/// Override: assign each doc to the first scene whose any prefix matches. Docs
/// matching no prefix fall into `(unscoped)`.
fn partition_by_prefixes(
    documents: &[Document],
    scenes: &BTreeMap<String, Vec<String>>,
) -> Vec<(String, Vec<u32>)> {
    // Deterministic scene evaluation order: sorted names.
    let ordered: Vec<(&String, &Vec<String>)> = scenes.iter().collect();
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for doc in documents {
        let mut placed = false;
        for (name, prefixes) in &ordered {
            if prefixes.iter().any(|p| path_matches(&doc.path, p)) {
                groups.entry((*name).clone()).or_default().push(doc.id);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.entry("(unscoped)".to_string()).or_default().push(doc.id);
        }
    }
    finalize(groups)
}

/// A path matches a prefix if it equals it or sits beneath it as a directory.
fn path_matches(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    path == prefix || path.starts_with(&format!("{}/", prefix))
}

fn finalize(groups: BTreeMap<String, Vec<u32>>) -> Vec<(String, Vec<u32>)> {
    let mut out: Vec<(String, Vec<u32>)> = groups
        .into_iter()
        .map(|(name, mut ids)| {
            ids.sort_unstable();
            (name, ids)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

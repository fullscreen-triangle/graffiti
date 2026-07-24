//! Index construction and persistence. This is the CONSTRUCTION phase (Inv 4):
//! it builds the corpus + self-graph and writes `.spraypaint/index.json`. It
//! never produces a ranked answer.

pub mod identity;
pub mod schema;

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};

use crate::config::SprayConfig;
use crate::scene;
use crate::{chunk, root, walk};

use schema::{
    CorpusStats, Document, Index, Passage, SceneGroup, SceneStats, SCHEMA_VERSION,
};

/// Build an index over `root_dir` using `cfg`. Pure/deterministic given inputs.
pub fn build(root_dir: &Path, cfg: &SprayConfig) -> Result<Index> {
    let files = walk::walk(root_dir, cfg);

    // First pass: read files, chunk, tokenise, collect global vocabulary and
    // document frequencies. We build the vocab first (sorted) so term ids are
    // canonical, then re-encode passages against it.
    struct StagedDoc {
        path: String,
        content_hash: String,
        // passages as (start,end, token strings)
        passages: Vec<(u32, u32, Vec<String>)>,
    }

    let mut staged: Vec<StagedDoc> = Vec::new();
    // term -> document frequency (count of documents containing it)
    let mut df: BTreeMap<String, u32> = BTreeMap::new();

    for f in &files {
        let body = match std::fs::read_to_string(&f.path) {
            Ok(b) => b,
            Err(_) => continue, // unreadable/binary — skip
        };
        let rel = f
            .path
            .strip_prefix(root_dir)
            .unwrap_or(&f.path)
            .to_string_lossy()
            .replace('\\', "/");
        let content_hash = format!("b3:{}", blake3::hash(body.as_bytes()).to_hex());

        let raw_passages = chunk::chunk_file(&body, cfg);
        if raw_passages.is_empty() {
            continue;
        }

        let mut doc_terms_seen: std::collections::BTreeSet<String> = Default::default();
        let mut passages = Vec::new();
        for rp in raw_passages {
            let toks = chunk::tokenize(&rp.text);
            for t in &toks {
                doc_terms_seen.insert(t.clone());
            }
            passages.push((rp.start_line, rp.end_line, toks));
        }
        for t in doc_terms_seen {
            *df.entry(t).or_insert(0) += 1;
        }

        staged.push(StagedDoc {
            path: rel,
            content_hash,
            passages,
        });
    }

    // Canonical vocab: sorted term list; id = position.
    let vocab: Vec<String> = df.keys().cloned().collect();
    let term_id: BTreeMap<&str, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i as u32))
        .collect();
    let global_df: Vec<u32> = vocab.iter().map(|t| df[t]).collect();

    // Second pass: encode passages against the vocab.
    let mut documents = Vec::new();
    let mut total_passage_len: u64 = 0;
    let mut total_passages: u64 = 0;
    for (id, sd) in staged.into_iter().enumerate() {
        let mut passages = Vec::new();
        for (start, end, toks) in sd.passages {
            let len = toks.len() as u32;
            total_passage_len += len as u64;
            total_passages += 1;
            // term -> tf within this passage
            let mut tf: BTreeMap<u32, u32> = BTreeMap::new();
            for t in &toks {
                if let Some(&tid) = term_id.get(t.as_str()) {
                    *tf.entry(tid).or_insert(0) += 1;
                }
            }
            let terms: Vec<(u32, u32)> = tf.into_iter().collect(); // sorted by term_id
            passages.push(Passage {
                start_line: start,
                end_line: end,
                len,
                terms,
            });
        }
        documents.push(Document {
            id: id as u32,
            path: sd.path,
            content_hash: sd.content_hash,
            passages,
        });
    }

    let avg_passage_len = if total_passages > 0 {
        total_passage_len as f32 / total_passages as f32
    } else {
        0.0
    };

    let stats = CorpusStats {
        vocab,
        global_df,
        total_docs: documents.len() as u32,
        avg_passage_len,
    };

    // Scenes (top-level dir partition, or scenes.toml override).
    let scene_defs = scene::detect(root_dir, &documents)?;
    let scenes: Vec<SceneGroup> = scene_defs
        .into_iter()
        .map(|(name, doc_ids)| {
            let (sum, count) = doc_ids
                .iter()
                .flat_map(|&d| documents[d as usize].passages.iter())
                .fold((0u64, 0u32), |(s, c), p| (s + p.len as u64, c + 1));
            let avg = if count > 0 { sum as f32 / count as f32 } else { 0.0 };
            SceneGroup {
                name,
                doc_ids,
                stats: SceneStats {
                    avg_passage_len: avg,
                    passage_count: count,
                },
            }
        })
        .collect();

    let identity = identity::compute_identity(&documents);

    let built_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Ok(Index {
        schema_version: SCHEMA_VERSION,
        root: root_dir.display().to_string(),
        built_unix,
        scenes,
        documents,
        stats,
        identity,
    })
}

/// Persist an index atomically (tmp write + rename) so a concurrent reader never
/// observes a half-written file.
pub fn save(root_dir: &Path, index: &Index) -> Result<()> {
    let dir = root::spray_dir(root_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = root::index_path(root_dir);
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(index)?;
    std::fs::write(&tmp, json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}

/// Load an index and re-verify its identity fingerprint (Inv 1): recompute the
/// self-graph fingerprint from the stored documents and assert it matches the
/// stored value. A mismatch means the index was tampered with or corrupted.
pub fn load(root_dir: &Path) -> Result<Index> {
    let path = root::index_path(root_dir);
    let data = std::fs::read_to_string(&path).map_err(|_| {
        anyhow!(
            "no index at {} — run `spraypaint index` in this project first",
            path.display()
        )
    })?;
    let index: Index = serde_json::from_str(&data)
        .with_context(|| format!("parsing {}", path.display()))?;

    let g = identity::build_self_graph(&index.documents);
    let recomputed = identity::fingerprint(&g);
    if recomputed != index.identity.fingerprint {
        return Err(anyhow!(
            "identity fingerprint mismatch (Inv 1): stored {}, recomputed {} — index corrupt; re-run `spraypaint index`",
            index.identity.fingerprint,
            recomputed
        ));
    }
    Ok(index)
}

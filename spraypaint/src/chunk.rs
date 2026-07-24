//! Chunking: a file body becomes overlapping fixed line-windows, each window
//! nudged to start at a structural boundary (blank line, heading, or a
//! `fn`/`def`/`class`/`\section` line) when one is within a small slack.
//! Deterministic and language-agnostic. Also the tokeniser both index and
//! query share.

use unicode_segmentation::UnicodeSegmentation;

use crate::config::SprayConfig;

/// A passage carved from a file, before term extraction.
pub struct RawPassage {
    /// 1-based inclusive line range.
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

/// How far (in lines) a window start may shift to land on a structural boundary.
const BOUNDARY_SLACK: usize = 4;

/// Tokenise text into lowercase alphanumeric terms. Shared by index and query
/// so the same string produces the same terms on both sides.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.unicode_words() {
        let t: String = word.to_lowercase();
        // `unicode_words` already strips most punctuation; guard length.
        if !t.is_empty() && t.len() <= 64 {
            out.push(t);
        }
    }
    out
}

/// True if a line looks like a natural passage boundary.
fn is_boundary(line: &str) -> bool {
    let t = line.trim_start();
    if t.is_empty() {
        return true;
    }
    // Markdown / LaTeX headings.
    if t.starts_with('#') || t.starts_with("\\section") || t.starts_with("\\subsection") {
        return true;
    }
    // Common definition openers across the indexed languages.
    for kw in [
        "fn ", "pub fn", "def ", "class ", "struct ", "enum ", "trait ", "impl ",
        "function ", "func ", "type ", "interface ", "module ",
    ] {
        if t.starts_with(kw) {
            return true;
        }
    }
    false
}

/// Carve `text` into passages according to `cfg`'s window/overlap, nudging each
/// window start to a nearby structural boundary.
pub fn chunk_file(text: &str, cfg: &SprayConfig) -> Vec<RawPassage> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    if n == 0 {
        return Vec::new();
    }

    let window = cfg.window.max(1);
    let stride = cfg.stride();

    let mut passages = Vec::new();
    let mut start = 0usize;

    while start < n {
        // Nudge `start` backward to the nearest boundary within slack, so a
        // window prefers to begin at a heading/def/blank rather than mid-block.
        let mut s = start;
        let lower = start.saturating_sub(BOUNDARY_SLACK);
        for cand in (lower..=start).rev() {
            if is_boundary(lines[cand]) {
                s = cand;
                break;
            }
        }

        let end = (s + window).min(n);
        let body = lines[s..end].join("\n");
        if !body.trim().is_empty() {
            passages.push(RawPassage {
                start_line: (s as u32) + 1,
                end_line: end as u32,
                text: body,
            });
        }

        if end >= n {
            break;
        }
        // Advance from the *original* start by the stride, not from the nudged
        // position, so nudging cannot stall or skip coverage.
        start += stride;
    }

    passages
}

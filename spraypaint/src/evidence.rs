//! Evidence: the lines inside a returned passage that actually carry the query.
//!
//! A passage is a 40-line window, which is the right unit for scoring and the
//! wrong one for reading. The snippet used to be the window's *first non-empty
//! line* — a line chosen by where the chunker happened to cut, so a reader saw
//! `}` or half a LaTeX sentence while the matched terms sat thirty lines down.
//! Evidence is instead the short run of lines where the query's terms are
//! densest, weighted by IDF so that a line holding the rare term beats a line
//! holding several common ones.
//!
//! Still re-read from disk at query time (Inv 3): nothing here is stored.

use std::collections::BTreeSet;

use crate::chunk;

/// How many lines of evidence to show per passage.
pub const EVIDENCE_LINES: usize = 5;

/// Lines longer than this are cut, so one minified line cannot flood a result.
const MAX_LINE_CHARS: usize = 200;

/// A run of lines from a passage, with its 1-based inclusive line range.
pub struct Evidence {
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

/// Choose the evidence inside `lines`, the passage's lines, whose first line is
/// `first_line` in the file. `terms` are the query terms with their weights.
///
/// Every window of [`EVIDENCE_LINES`] consecutive lines is scored by the total
/// weight of the *distinct* query terms it contains, ties going to the window
/// with more matching lines and then to the earliest. Returns `None` when no
/// line contains any term — possible only if the file changed since indexing —
/// so the caller can fall back rather than show an arbitrary line as evidence.
pub fn select(lines: &[&str], first_line: u32, terms: &[(String, f64)]) -> Option<Evidence> {
    if lines.is_empty() || terms.is_empty() {
        return None;
    }

    // Per line: the indices of the query terms it contains.
    let hits: Vec<BTreeSet<usize>> = lines
        .iter()
        .map(|line| {
            let tokens = chunk::tokenize(line);
            terms
                .iter()
                .enumerate()
                .filter(|(_, (t, _))| tokens.iter().any(|tok| tok == t))
                .map(|(i, _)| i)
                .collect()
        })
        .collect();

    let width = EVIDENCE_LINES.min(lines.len());
    let mut best: Option<(usize, f64, usize)> = None; // (start, weight, hit_lines)
    for start in 0..=(lines.len() - width) {
        let window = &hits[start..start + width];
        let distinct: BTreeSet<usize> = window.iter().flatten().copied().collect();
        let weight: f64 = distinct.iter().map(|&i| terms[i].1).sum();
        let hit_lines = window.iter().filter(|h| !h.is_empty()).count();
        let better = match best {
            None => true,
            Some((_, w, n)) => weight > w || (weight == w && hit_lines > n),
        };
        if better {
            best = Some((start, weight, hit_lines));
        }
    }

    let (start, weight, _) = best?;
    if weight <= 0.0 {
        return None;
    }

    // Centre the window on its matching lines. The earliest-wins tie-break
    // otherwise pushes a lone match to the window's last line, with all the
    // context above it. The matches span fewer than `width` lines, so a window
    // of the same width centred on them still contains every one.
    let window_hits = (start..start + width).filter(|&i| !hits[i].is_empty());
    let first = window_hits.clone().next().unwrap_or(start);
    let last = window_hits.last().unwrap_or(start);
    let start = ((first + last) / 2)
        .saturating_sub(width / 2)
        .min(lines.len() - width);

    // Trim blank lines off both ends; the window keeps at least one matching
    // line, so this never empties it.
    let mut s = start;
    let mut e = start + width; // exclusive
    while s < e && lines[s].trim().is_empty() {
        s += 1;
    }
    while e > s && lines[e - 1].trim().is_empty() {
        e -= 1;
    }

    let text = lines[s..e]
        .iter()
        .map(|l| clip(l.trim_end()))
        .collect::<Vec<_>>()
        .join("\n");

    Some(Evidence {
        start_line: first_line + s as u32,
        end_line: first_line + e as u32 - 1,
        text,
    })
}

fn clip(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        line.to_string()
    } else {
        let mut s: String = line.chars().take(MAX_LINE_CHARS).collect();
        s.push('…');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(ts: &[(&str, f64)]) -> Vec<(String, f64)> {
        ts.iter().map(|(t, w)| (t.to_string(), *w)).collect()
    }

    #[test]
    fn finds_the_match_deep_in_the_window() {
        let mut lines = vec!["}"; 30];
        lines.push("A good dentist is hard to find.");
        lines.extend(["filler"; 9]);
        let ev = select(&lines, 100, &terms(&[("dentist", 2.0)])).unwrap();
        assert!(ev.text.contains("dentist"), "{}", ev.text);
        // Centred: two lines of context either side, not four above.
        assert_eq!((ev.start_line, ev.end_line), (128, 132));
    }

    #[test]
    fn a_match_at_the_edge_keeps_the_full_width() {
        let mut lines = vec!["dentist"];
        lines.extend(["filler"; 9]);
        let ev = select(&lines, 1, &terms(&[("dentist", 1.0)])).unwrap();
        assert_eq!((ev.start_line, ev.end_line), (1, 5));
    }

    #[test]
    fn rare_term_beats_several_common_ones() {
        let lines = vec![
            "the city the city the city",
            "",
            "",
            "",
            "",
            "",
            "",
            "the dentist",
        ];
        let ts = terms(&[("city", 0.1), ("the", 0.05), ("dentist", 3.0)]);
        let ev = select(&lines, 1, &ts).unwrap();
        assert!(ev.text.contains("dentist"), "{}", ev.text);
    }

    #[test]
    fn no_matching_line_yields_none() {
        let lines = vec!["nothing", "here"];
        assert!(select(&lines, 1, &terms(&[("dentist", 1.0)])).is_none());
    }

    #[test]
    fn blank_edges_are_trimmed() {
        let lines = vec!["", "", "dentist", "", ""];
        let ev = select(&lines, 10, &terms(&[("dentist", 1.0)])).unwrap();
        assert_eq!((ev.start_line, ev.end_line), (12, 12));
        assert_eq!(ev.text, "dentist");
    }
}

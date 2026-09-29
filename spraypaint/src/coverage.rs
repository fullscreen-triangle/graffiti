//! Coverage: whether the corpus contains what the query is about.
//!
//! A lexical engine cannot know whether a passage *answers* a question. It can
//! know which of the query's words the corpus contains at all, and which of
//! them the returned passages contain. Without that, every search returns its
//! best look-alike with the same confidence: asked for a dentist, a corpus with
//! no dentist in it returns whoever writes most about the city, and a caller —
//! a person or an agent — cannot tell that from a real hit.
//!
//! So every `ask` ends with a verdict, computed from the index alone:
//!
//!   * `covered`  — one returned passage contains every query term.
//!   * `partial`  — returned passages carry at least half of the query's
//!                  weight, but no single passage contains every term.
//!   * `declined` — they carry less than half: what the query is specifically
//!                  about is not in what came back.
//!
//! Terms are weighted by IDF, with a term the corpus has never seen given the
//! maximum (`df = 0`). That is what makes `declined` bite on the right queries:
//! the absent word is usually the one the query is about, and a match on the
//! query's common words cannot make up for it.
//!
//! `covered` is not "answered". It says the words are there together, which is
//! necessary for an answer and not sufficient for one.

/// Below this share of the query's weight, the verdict is `declined`.
///
/// A convention, not a derived bound: "most of what the query is about is
/// missing from the results". It is reported alongside the share itself so a
/// caller can apply its own threshold.
pub const DECLINE_BELOW: f64 = 0.5;

/// One query term, as the index sees it.
pub struct TermCoverage {
    pub term: String,
    /// Documents containing the term; 0 means the corpus has never seen it.
    pub df: u32,
    /// IDF weight (maximal when `df == 0`).
    pub weight: f64,
    /// Whether at least one *returned* passage contains it.
    pub in_results: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Covered,
    Partial,
    Declined,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Covered => "covered",
            Verdict::Partial => "partial",
            Verdict::Declined => "declined",
        }
    }
}

pub struct Coverage {
    pub verdict: Verdict,
    /// One sentence saying why, naming the terms responsible.
    pub reason: String,
    /// Query terms in query order.
    pub terms: Vec<TermCoverage>,
    /// Share of the query's total weight carried by terms that appear in at
    /// least one returned passage. In `[0, 1]`.
    pub weight_share: f64,
}

/// Assess coverage. `complete` names the highest-scoring returned passage that
/// contains every query term, as `path:start-end`, if there is one.
pub fn assess(terms: Vec<TermCoverage>, complete: Option<String>) -> Coverage {
    if terms.is_empty() {
        return Coverage {
            verdict: Verdict::Declined,
            reason: "no searchable terms: every word of the query is a stopword".to_string(),
            terms,
            weight_share: 0.0,
        };
    }

    let total: f64 = terms.iter().map(|t| t.weight).sum();
    let carried: f64 = terms.iter().filter(|t| t.in_results).map(|t| t.weight).sum();
    let weight_share = if total > 0.0 { carried / total } else { 0.0 };

    let verdict = if complete.is_some() {
        Verdict::Covered
    } else if weight_share < DECLINE_BELOW {
        Verdict::Declined
    } else {
        Verdict::Partial
    };

    let reason = match (&complete, verdict) {
        (Some(at), _) => format!("{at} contains every query term"),
        _ => {
            // Most specific first: the heaviest missing term is the one the
            // reader most needs to see.
            let mut missing: Vec<&TermCoverage> = terms.iter().filter(|t| !t.in_results).collect();
            missing.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
            let absent: Vec<&str> = missing.iter().filter(|t| t.df == 0).map(|t| t.term.as_str()).collect();
            let unreturned: Vec<&str> = missing.iter().filter(|t| t.df > 0).map(|t| t.term.as_str()).collect();

            let mut parts = Vec::new();
            if carried == 0.0 {
                parts.push("no returned passage contains any query term".to_string());
            } else {
                parts.push(format!(
                    "returned passages carry {:.0}% of the query's weight",
                    weight_share * 100.0
                ));
            }
            if !absent.is_empty() {
                parts.push(format!("not in the corpus: {}", absent.join(", ")));
            }
            if !unreturned.is_empty() {
                parts.push(format!("in the corpus but not in any returned passage: {}", unreturned.join(", ")));
            }
            if verdict == Verdict::Partial {
                parts.push("no single passage contains every term".to_string());
            }
            parts.join("; ")
        }
    };

    Coverage {
        verdict,
        reason,
        terms,
        weight_share,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(term: &str, df: u32, weight: f64, in_results: bool) -> TermCoverage {
        TermCoverage {
            term: term.to_string(),
            df,
            weight,
            in_results,
        }
    }

    #[test]
    fn a_complete_passage_is_covered() {
        let c = assess(vec![t("water", 1, 1.0, true), t("filling", 1, 1.0, true)], Some("a.md:1-40".into()));
        assert_eq!(c.verdict, Verdict::Covered);
        assert!(c.reason.contains("a.md:1-40"));
    }

    /// The dentist case: the common word matched, the word the query is about
    /// is absent from the corpus. Must decline, and must name the absent word.
    #[test]
    fn matching_only_the_common_word_declines() {
        let c = assess(vec![t("suitable", 40, 1.2, true), t("dentist", 0, 7.0, false)], None);
        assert_eq!(c.verdict, Verdict::Declined);
        assert!(c.reason.contains("not in the corpus: dentist"), "{}", c.reason);
    }

    #[test]
    fn most_weight_carried_but_split_is_partial() {
        let c = assess(vec![t("kuramoto", 2, 3.0, true), t("phase", 30, 1.0, false)], None);
        assert_eq!(c.verdict, Verdict::Partial);
        assert!(c.reason.contains("not in any returned passage: phase"), "{}", c.reason);
    }

    #[test]
    fn no_terms_declines() {
        assert_eq!(assess(vec![], None).verdict, Verdict::Declined);
    }

    #[test]
    fn missing_terms_are_listed_heaviest_first() {
        let c = assess(
            vec![t("a", 0, 1.0, false), t("b", 0, 5.0, false), t("c", 3, 0.5, true)],
            None,
        );
        assert!(c.reason.contains("not in the corpus: b, a"), "{}", c.reason);
    }
}

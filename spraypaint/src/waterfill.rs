//! Water-filling attention allocator — Algorithm 1 of the Split-Attention
//! Synchronised Agents paper (docs/sources/split-attention-agents.tex, L1541).
//!
//! Each scene `i` presents a nonincreasing sequence of BM25 scores
//! `s_{i,1} >= s_{i,2} >= ...` — the marginal gain of taking its j-th best
//! passage. Define the concave gain profile `gamma_i(a) = sum_{j<=a} s_{i,j}`,
//! so `gamma_i'(a) = s_{i,a+1}`. The value-maximising division of a budget of
//! `k` passages solves `max sum_i gamma_i(a_i)  s.t. sum_i a_i <= k`, which by
//! thm:waterfill is characterised by a single price `p*`: take passages while
//! their marginal gain exceeds `p*`, drop scenes whose best is below it. We
//! find `p*` by bisection (demand is nonincreasing in price), then reconcile
//! the discrete integer budget with a deterministic remainder-fill.

/// The result of an allocation: how many passages each scene contributes, and
/// the clearing price.
#[derive(Debug, Clone, PartialEq)]
pub struct Allocation {
    /// per_scene[i] = number of leading passages taken from scene i.
    pub per_scene: Vec<usize>,
    /// The clearing price p* (a relevance threshold).
    pub price: f64,
}

/// Count of passages in a single scene whose marginal gain strictly exceeds `p`.
/// Scores are sorted descending, so this is a prefix length.
fn scene_demand(scores: &[f64], p: f64) -> usize {
    scores.iter().take_while(|&&x| x > p).count()
}

/// Total demand across scenes at price `p` — nonincreasing in `p`.
fn total_demand(scene_scores: &[Vec<f64>], p: f64) -> usize {
    scene_scores.iter().map(|s| scene_demand(s, p)).sum()
}

/// Allocate `budget` passages across scenes by water-filling.
///
/// `scene_scores[i]` must be sorted descending (the marginal-gain sequence).
pub fn water_fill(scene_scores: &[Vec<f64>], budget: usize, eps: f64) -> Allocation {
    let n_scenes = scene_scores.len();
    if budget == 0 || n_scenes == 0 {
        return Allocation {
            per_scene: vec![0; n_scenes],
            price: f64::INFINITY,
        };
    }

    let total_available: usize = scene_scores.iter().map(|s| s.len()).sum();
    if total_available <= budget {
        // Budget not exhausted at the optimum => price is 0 (thm:waterfill).
        return Allocation {
            per_scene: scene_scores.iter().map(|s| s.len()).collect(),
            price: 0.0,
        };
    }

    // Bisection on price. demand(lo) is large, demand(hi) is 0.
    let mut lo = 0.0f64;
    let mut hi = scene_scores
        .iter()
        .filter_map(|s| s.first().copied())
        .fold(0.0f64, f64::max);

    // Guard: if hi is 0 all scores are 0 -> nothing to allocate.
    if hi <= 0.0 {
        return Allocation {
            per_scene: vec![0; n_scenes],
            price: 0.0,
        };
    }

    while hi - lo > eps {
        let p = 0.5 * (lo + hi);
        if total_demand(scene_scores, p) > budget {
            lo = p; // too many passages clear the price -> raise it
        } else {
            hi = p;
        }
    }
    let price = hi;

    let mut per_scene: Vec<usize> = scene_scores
        .iter()
        .map(|s| scene_demand(s, price))
        .collect();

    fill_remainder(&mut per_scene, scene_scores, budget);

    Allocation { per_scene, price }
}

/// Discrete slack reconciliation: bisection may land a passage or two under
/// budget at ties. Fill remaining slots by global next-best marginal gain,
/// deterministic tie-break (higher score, then lower scene idx, then lower
/// passage idx). Never exceed budget.
fn fill_remainder(per_scene: &mut [usize], scene_scores: &[Vec<f64>], budget: usize) {
    loop {
        let used: usize = per_scene.iter().sum();
        if used >= budget {
            break;
        }
        // Best next passage across scenes: scene i's next is scores[i][per_scene[i]].
        let mut best: Option<(f64, usize)> = None;
        for (i, scores) in scene_scores.iter().enumerate() {
            if let Some(&next) = scores.get(per_scene[i]) {
                if next <= 0.0 {
                    continue;
                }
                match best {
                    None => best = Some((next, i)),
                    Some((bw, _)) if next > bw => best = Some((next, i)),
                    _ => {}
                }
            }
        }
        match best {
            Some((_, i)) => per_scene[i] += 1,
            None => break, // no positive passage left
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force the optimal integer allocation maximising sum of taken scores
    /// subject to the budget — since each scene's profile is concave (sorted
    /// desc), the greedy "take globally largest marginals" is optimal, which is
    /// what we compare against.
    fn brute_optimal_value(scene_scores: &[Vec<f64>], budget: usize) -> f64 {
        let mut all: Vec<f64> = scene_scores.iter().flatten().copied().collect();
        all.sort_by(|a, b| b.partial_cmp(a).unwrap());
        all.iter().take(budget).filter(|&&x| x > 0.0).sum()
    }

    fn value_of(alloc: &Allocation, scene_scores: &[Vec<f64>]) -> f64 {
        alloc
            .per_scene
            .iter()
            .enumerate()
            .map(|(i, &a)| scene_scores[i].iter().take(a).sum::<f64>())
            .sum()
    }

    #[test]
    fn matches_brute_force_optimum() {
        let scenes = vec![
            vec![9.0, 7.0, 2.0, 1.0],
            vec![8.0, 6.0, 5.0],
            vec![3.0],
        ];
        for budget in 1..=8 {
            let alloc = water_fill(&scenes, budget, 1e-9);
            let got = value_of(&alloc, &scenes);
            let want = brute_optimal_value(&scenes, budget);
            assert!(
                (got - want).abs() < 1e-6,
                "budget {budget}: got {got}, want {want}, alloc {:?}",
                alloc
            );
            assert!(alloc.per_scene.iter().sum::<usize>() <= budget);
        }
    }

    #[test]
    fn anti_monopoly_small_scenes_beat_the_tail() {
        // The real guarantee: a scene contributes exactly the passages whose
        // marginal gain clears the price. A big scene cannot take slots its
        // *tail* does not earn. Here the giant's scores decay fast, so its tail
        // falls below the small scenes' entries and the small scenes win those
        // slots — water-filling does NOT let the giant monopolise on volume.
        let giant: Vec<f64> = (0..100).map(|i| 10.0 - (i as f64) * 0.5).collect();
        // giant: 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, ... (only a few are high)
        let small_a = vec![9.2, 8.8];
        let small_b = vec![9.4];
        let scenes = vec![giant, small_a, small_b];
        let alloc = water_fill(&scenes, 6, 1e-9);
        // small_b (9.4) beats giant[2]=9.0; small_a (9.2) beats giant[2] too.
        assert!(alloc.per_scene[1] >= 1, "small_a starved: {:?}", alloc);
        assert!(alloc.per_scene[2] >= 1, "small_b starved: {:?}", alloc);
        // And the giant does NOT take all 6.
        assert!(alloc.per_scene[0] < 6, "giant monopolised: {:?}", alloc);
        assert_eq!(alloc.per_scene.iter().sum::<usize>(), 6);
        // Sanity: the chosen set equals the global top-6 by score (epsilon —
        // the two sums add the same values in different orders).
        assert!((value_of(&alloc, &scenes) - brute_optimal_value(&scenes, 6)).abs() < 1e-6);
    }

    #[test]
    fn dense_scene_legitimately_wins_when_it_earns_it() {
        // Conversely, when a big scene's passages genuinely out-score the small
        // ones (dense high-quality region), it SHOULD take the budget. This is
        // value-maximisation, not a bug — the test pins the honest behaviour.
        let giant: Vec<f64> = (0..100).map(|i| 10.0 - (i as f64) * 0.01).collect();
        let small_a = vec![9.5, 9.4];
        let small_b = vec![9.6];
        let scenes = vec![giant, small_a, small_b];
        let alloc = water_fill(&scenes, 12, 1e-9);
        // giant's top-12 (>=9.89) all beat 9.6/9.5/9.4, so it earns all 12.
        assert_eq!(alloc.per_scene[0], 12, "value-max allocation changed: {:?}", alloc);
        assert!((value_of(&alloc, &scenes) - brute_optimal_value(&scenes, 12)).abs() < 1e-6);
    }

    #[test]
    fn budget_not_exhausted_price_zero() {
        let scenes = vec![vec![5.0, 4.0], vec![3.0]];
        let alloc = water_fill(&scenes, 10, 1e-9);
        assert_eq!(alloc.price, 0.0);
        assert_eq!(alloc.per_scene, vec![2, 1]);
    }

    #[test]
    fn deterministic() {
        let scenes = vec![vec![9.0, 7.0, 2.0], vec![8.0, 6.0], vec![3.0]];
        let a = water_fill(&scenes, 4, 1e-9);
        let b = water_fill(&scenes, 4, 1e-9);
        assert_eq!(a, b);
    }
}

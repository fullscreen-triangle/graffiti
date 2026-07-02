/**
 * Scheduler priority and soundness.
 *
 * Implements Definition 10.4 (Live Seek; Residue Descent), Definition 10.5
 * (Scheduler Priority), and Theorem 10.6 (Scheduler Soundness) of
 * semantic-causal-propagation.tex.
 */

export interface LiveSeek {
  name: string;
  /** sigma(x_s, x*_s) recorded over ticks; last element is current. */
  residueHistory: number[];
  floor: number;
  closed: boolean;
}

export function createLiveSeek(name: string, floor: number): LiveSeek {
  return { name, residueHistory: [], floor, closed: false };
}

export function descentRate(seek: LiveSeek): number {
  const h = seek.residueHistory;
  if (h.length < 2) return 0;
  const prev = h[h.length - 2]!;
  const curr = h[h.length - 1]!;
  return prev - curr;
}

export function currentResidue(seek: LiveSeek): number {
  const h = seek.residueHistory;
  return h.length > 0 ? h[h.length - 1]! : seek.floor;
}

/** Definition 10.5. */
export function priority(seek: LiveSeek): number {
  if (seek.closed) return Infinity;
  const delta = descentRate(seek);
  const denom = Math.max(currentResidue(seek) - seek.floor, seek.floor);
  if (delta <= 0) return 0;
  return delta / denom;
}

/** argmax_s P(s); returns null if all priorities are 0 (all stalled). */
export function selectNext(liveSeeks: readonly LiveSeek[]): LiveSeek | null {
  if (liveSeeks.length === 0) return null;
  let best: LiveSeek | null = null;
  let bestPriority = -Infinity;
  for (const s of liveSeeks) {
    const p = priority(s);
    if (p > bestPriority) {
      bestPriority = p;
      best = s;
    }
  }
  if (best === null || priority(best) <= 0) return null;
  return best;
}

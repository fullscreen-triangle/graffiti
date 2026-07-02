/** Test 10: Scheduler Soundness (Theorem 10.6). */

import { Checker } from "./harness";
import { Rng } from "../core/graph";
import { createLiveSeek, priority, selectNext, type LiveSeek } from "../core/scheduler";

export function run() {
  const c = new Checker("10", "Theorem 10.6 (Scheduler Soundness)");
  const rng = new Rng(37);

  for (let trial = 0; trial < 500; trial++) {
    const nSeeks = rng.int(1, 5);
    const seeks: LiveSeek[] = [];
    for (let i = 0; i < nSeeks; i++) {
      const kind = rng.choice(["stalled", "descending", "closed"] as const);
      const floor = 0.01;
      if (kind === "stalled") {
        const seek = createLiveSeek(`s${i}`, floor);
        seek.residueHistory = [1.0, 0.5, 0.5, 0.5];
        seeks.push(seek);
      } else if (kind === "descending") {
        const start = rng.float(0.5, 5.0);
        const seek = createLiveSeek(`s${i}`, floor);
        seek.residueHistory = [start, start * 0.8, start * 0.6];
        seeks.push(seek);
      } else {
        const seek = createLiveSeek(`s${i}`, floor);
        seek.residueHistory = [floor];
        seek.closed = true;
        seeks.push(seek);
      }
    }

    for (const s of seeks) {
      const descRate = s.residueHistory.length >= 2 ? s.residueHistory[s.residueHistory.length - 2]! - s.residueHistory[s.residueHistory.length - 1]! : 0;
      if (descRate <= 0 && !s.closed) {
        c.assert(priority(s) === 0, "stalled seek gets exactly zero priority");
      }
      if (s.closed) {
        c.assert(priority(s) === Infinity, "closed seek gets +infinity priority");
      }
    }

    const selected = selectNext(seeks);
    const closedSeeks = seeks.filter((s) => s.closed);
    if (closedSeeks.length > 0) {
      c.assert(selected !== null && selected.closed, "a closed seek is selected over any finite-priority seek");
    } else {
      const descending = seeks.filter((s) => {
        const h = s.residueHistory;
        return h.length >= 2 && h[h.length - 2]! - h[h.length - 1]! > 0;
      });
      if (descending.length > 0) {
        c.assert(selected !== null && priority(selected) > 0, "a descending seek is selected when no closed seek exists");
      } else {
        c.assert(selected === null, "no seek is selected when all are stalled");
      }
    }
  }

  return c.result();
}

"use client";

// Water-filling allocation across scenes, drawn from `/api/ask`'s `allocation`
// array. Two panels:
//
//   Top    — allocated vs available, per scene, with a draggable budget handle.
//   Bottom — the score distribution the allocation was computed from.
//
// Three things this drawing is careful about, because the obvious version of
// each is wrong:
//
//   * `p*` is an **output** of water-filling, found by bisection in
//     `waterfill.rs`. No endpoint accepts a price. The draggable handle is
//     therefore labelled "budget k"; `p*` is shown as the value that *resulted*,
//     and its line snaps to wherever the run put it.
//   * `available` is the count of passages scoring above zero **for this
//     query** — not the scene's size. The scene total comes from `/api/scenes`
//     and is drawn as a separate, fainter bar. Conflating them would make the
//     allocation look far more complete than it is.
//   * `best_score`/`median_score` come from the binary, computed over every
//     scoring passage before truncation. Deriving them from `results[]` would
//     take a median of a top-k slice — a number describing the budget rather
//     than the scene.

import { useCallback, useEffect, useRef, useState } from "react";
import * as d3 from "d3";

import type { AskResponse, SceneInfo } from "@/lib/api";
import { MAX_BUDGET, MIN_BUDGET } from "@/lib/gestures";

interface Props {
  result: AskResponse | null;
  scenes: SceneInfo[];
  budget: number;
  stale: boolean;
  /** Live drag feedback — debounced preview, never a commit. */
  onBudget: (k: number) => void;
}

export default function SceneAllocationTab({
  result,
  scenes,
  budget,
  stale,
  onBudget,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);

  const draw = useCallback(() => {
    const container = containerRef.current;
    if (!svgRef.current || !container || !result) return;

    const width = container.clientWidth;
    const height = container.clientHeight - 28; // header strip
    if (width <= 0 || height <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    const alloc = result.allocation;
    if (alloc.length === 0) return;

    const margin = { top: 22, right: 30, bottom: 56, left: 58 };
    const gap = 46;
    const halfH = Math.max(40, (height - margin.top - margin.bottom - gap) / 2);
    const w = Math.max(40, width - margin.left - margin.right);

    const totals = new Map(scenes.map((s) => [s.name, s.passages]));

    // ── Top: allocated / available / indexed ──
    const g1 = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(alloc.map((a) => a.scene))
      .range([0, w])
      .padding(0.25);

    // Log scale: `available` routinely runs into the hundreds while `allocated`
    // is single digits. On a linear axis every allocation bar collapses to a
    // sliver and the panel shows nothing.
    const maxCount = Math.max(
      1,
      d3.max(alloc, (a) => Math.max(a.available, totals.get(a.scene) ?? 0)) ?? 1
    );
    const y1 = d3.scaleLog().domain([0.8, maxCount * 1.2]).range([halfH, 0]).clamp(true);

    g1.append("g")
      .attr("transform", `translate(0,${halfH})`)
      .call(d3.axisBottom(x))
      .call((g) => g.selectAll("text").attr("fill", "#ccc").attr("font-size", "11px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g1.append("g")
      .call(d3.axisLeft(y1).ticks(4, "~s"))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g1.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -halfH / 2)
      .attr("y", -42)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("passages (log)");

    const bw = x.bandwidth();
    for (const a of alloc) {
      const bx = x(a.scene)!;
      const indexed = totals.get(a.scene) ?? 0;

      // Widest, faintest: everything in the index for this scene.
      if (indexed > 0) {
        g1.append("rect")
          .attr("x", bx)
          .attr("y", y1(indexed))
          .attr("width", bw)
          .attr("height", Math.max(0, halfH - y1(indexed)))
          .attr("fill", "#2a2a2a")
          .attr("rx", 2);
      }
      // Middle: passages that scored for this query.
      if (a.available > 0) {
        g1.append("rect")
          .attr("x", bx + bw * 0.12)
          .attr("y", y1(a.available))
          .attr("width", bw * 0.76)
          .attr("height", Math.max(0, halfH - y1(a.available)))
          .attr("fill", "#3c5a70")
          .attr("rx", 2);
      }
      // Narrowest, brightest: what the budget actually bought.
      if (a.allocated > 0) {
        g1.append("rect")
          .attr("x", bx + bw * 0.28)
          .attr("y", y1(a.allocated))
          .attr("width", bw * 0.44)
          .attr("height", Math.max(0, halfH - y1(a.allocated)))
          .attr("fill", "#4fc1ff")
          .attr("opacity", stale ? 0.45 : 0.95)
          .attr("rx", 2);

        g1.append("text")
          .attr("x", bx + bw / 2)
          .attr("y", y1(a.allocated) - 4)
          .attr("text-anchor", "middle")
          .attr("fill", "#e0e0e0")
          .attr("font-size", "11px")
          .attr("font-weight", "600")
          .text(a.allocated);
      }

      g1.append("rect")
        .attr("x", bx)
        .attr("y", 0)
        .attr("width", bw)
        .attr("height", halfH)
        .attr("fill", "transparent")
        .append("title")
        .text(
          `${a.scene}\n` +
            `allocated: ${a.allocated}\n` +
            `scoring for this query: ${a.available}\n` +
            `indexed in this scene: ${indexed}\n` +
            (a.best_score !== null ? `best score: ${a.best_score.toFixed(4)}\n` : "") +
            (a.median_score !== null ? `median score: ${a.median_score.toFixed(4)}` : "")
        );
    }

    // Legend, because three nested bars are not self-explanatory.
    const legend = g1.append("g").attr("transform", `translate(0,${-14})`);
    const items: [string, string][] = [
      ["#4fc1ff", "allocated"],
      ["#3c5a70", "scoring"],
      ["#2a2a2a", "indexed"],
    ];
    let lx = 0;
    for (const [color, label] of items) {
      legend.append("rect").attr("x", lx).attr("y", -8).attr("width", 9).attr("height", 9).attr("fill", color).attr("rx", 2);
      legend
        .append("text")
        .attr("x", lx + 13)
        .attr("y", 0)
        .attr("fill", "#858585")
        .attr("font-size", "10px")
        .text(label);
      lx += 20 + label.length * 6;
    }

    // ── Bottom: score distribution, with p* as the achieved cut ──
    const g2 = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top + halfH + gap})`);

    const h2 = Math.max(30, halfH - 18);
    const maxScore = Math.max(
      result.price,
      d3.max(alloc, (a) => a.best_score ?? 0) ?? 1
    );
    const y2 = d3.scaleLinear().domain([0, maxScore * 1.12]).range([h2, 0]).nice();

    g2.append("g")
      .attr("transform", `translate(0,${h2})`)
      .call(d3.axisBottom(x))
      .call((g) => g.selectAll("text").attr("fill", "#ccc").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g2.append("g")
      .call(d3.axisLeft(y2).ticks(4))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g2.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -h2 / 2)
      .attr("y", -42)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("BM25 score");

    for (const a of alloc) {
      if (a.best_score === null || a.median_score === null) continue;
      const bx = x(a.scene)!;
      const above = a.allocated > 0;

      g2.append("rect")
        .attr("x", bx + bw * 0.15)
        .attr("y", y2(a.best_score))
        .attr("width", bw * 0.7)
        .attr("height", Math.max(1, y2(a.median_score) - y2(a.best_score)))
        .attr("fill", above ? "#4fc1ff33" : "#3c3c3c33")
        .attr("stroke", above ? "#4fc1ff" : "#555")
        .attr("rx", 2)
        .append("title")
        .text(
          `${a.scene}\nbest ${a.best_score.toFixed(4)} → median ${a.median_score.toFixed(4)}\n` +
            `over all ${a.available} scoring passages, not just the ${a.allocated} returned`
        );

      g2.append("line")
        .attr("x1", bx + bw * 0.15)
        .attr("x2", bx + bw * 0.85)
        .attr("y1", y2(a.median_score))
        .attr("y2", y2(a.median_score))
        .attr("stroke", "#ccc")
        .attr("stroke-width", 2);
    }

    // p*: where water-filling actually cut. Drawn last so it sits on top, and
    // explicitly *not* draggable — nothing in the API accepts a price.
    const py = y2(result.price);
    g2.append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", py)
      .attr("y2", py)
      .attr("stroke", "#cca700")
      .attr("stroke-dasharray", "6,3")
      .attr("stroke-width", 1.5);
    g2.append("text")
      .attr("x", w - 2)
      .attr("y", py - 5)
      .attr("text-anchor", "end")
      .attr("fill", "#cca700")
      .attr("font-size", "10px")
      .text(`p* = ${result.price.toFixed(4)} (result of k=${result.budget})`);

    if (stale) {
      svg
        .append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "#1e1e1e")
        .attr("opacity", 0.25)
        .attr("pointer-events", "none");
    }
  }, [result, scenes, stale]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  if (!result) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-neutral-500">
        Run a query to see the allocation.
      </div>
    );
  }

  const totalAllocated = result.allocation.reduce((n, a) => n + a.allocated, 0);
  const totalAvailable = result.allocation.reduce((n, a) => n + a.available, 0);

  return (
    <div ref={containerRef} className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-3 border-b border-neutral-800 bg-neutral-900/50 px-3 py-1 text-[11px] text-neutral-400">
        <span>
          {totalAllocated} of {totalAvailable} scoring passages allocated
        </span>
        <label className="ml-auto flex items-center gap-2">
          <span className="uppercase tracking-wider text-neutral-500">budget k</span>
          <input
            type="range"
            min={MIN_BUDGET}
            max={MAX_BUDGET}
            value={budget}
            onMouseDown={() => setDragging(true)}
            onMouseUp={() => setDragging(false)}
            onChange={(e) => onBudget(Number(e.target.value))}
            className="w-36 accent-sky-500"
          />
          <span className="w-7 font-mono tabular-nums text-neutral-200">{budget}</span>
          {dragging && (
            // Reassurance that dragging is free. The count is monotone with no
            // decrement path, so it matters that the user can see it is not moving.
            <span className="text-[10px] uppercase tracking-wide text-amber-500">
              preview
            </span>
          )}
        </label>
      </div>
      <svg ref={svgRef} className="flex-1" />
    </div>
  );
}

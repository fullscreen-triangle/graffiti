"use client";

import React, { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import { ExecutionState, invertCoherenceEdit, ScriptDiff } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
  onCrossfilter: (diff: ScriptDiff) => void;
  stale: boolean;
}

export default function CoherenceTab({ state, onCrossfilter, stale }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    if (!svgRef.current || !state || state.seeks.length === 0) return;

    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    // use first seek's catalysts for the matrix
    const seek = state.seeks[0];
    if (!seek) return;

    const cats = seek.catalysts;
    const n = cats.length;
    if (n === 0) return;

    const margin = { top: 60, right: 40, bottom: 40, left: 100 };
    const matSize = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom - 80, n * 60);
    const cellSize = matSize / n;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // color scale: red (negative) → neutral → green (positive)
    const colorScale = d3
      .scaleLinear<string>()
      .domain([-1, 0, 0.5, 1])
      .range(["#f14c4c", "#3c3c3c", "#4ec9b0", "#89d185"]);

    // build support matrix
    const matrix: number[][] = Array.from({ length: n }, () =>
      Array(n).fill(0)
    );
    for (let i = 0; i < n; i++) {
      for (const edge of cats[i].supportEdges) {
        const j = cats.findIndex((c) => c.name === edge.target);
        if (j >= 0 && j < n) {
          matrix[i][j] = edge.strength;
        }
      }
    }

    // draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const val = matrix[i][j];

        g.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 2)
          .attr("height", cellSize - 2)
          .attr("fill", colorScale(val))
          .attr("opacity", stale ? 0.4 : 0.9)
          .attr("rx", 2)
          .attr("cursor", "pointer")
          .on("click", () => {
            // toggle support edge → check coherence
            matrix[i][j] = matrix[i][j] > 0.5 ? 0.1 : 0.8;
            // check if any 3-cycle exists
            let hasCycle = false;
            for (let a = 0; a < n && !hasCycle; a++) {
              for (let b = 0; b < n && !hasCycle; b++) {
                for (let c = 0; c < n && !hasCycle; c++) {
                  if (a !== b && b !== c && a !== c) {
                    if (
                      matrix[a][b] > 0.5 &&
                      matrix[b][c] > 0.5 &&
                      matrix[c][a] > 0.5
                    ) {
                      hasCycle = true;
                    }
                  }
                }
              }
            }
            const diff = invertCoherenceEdit("", hasCycle);
            if (diff) onCrossfilter(diff);
            draw(); // redraw
          })
          .append("title")
          .text(
            `${cats[i].name} → ${cats[j].name}\nsupport: ${val.toFixed(2)}\nclick to toggle`
          );

        // value label
        if (cellSize > 30) {
          g.append("text")
            .attr("x", j * cellSize + (cellSize - 2) / 2)
            .attr("y", i * cellSize + (cellSize - 2) / 2 + 4)
            .attr("text-anchor", "middle")
            .attr("fill", val > 0.5 ? "#1e1e1e" : "#cccccc")
            .attr("font-size", "10px")
            .attr("font-family", "monospace")
            .attr("pointer-events", "none")
            .text(val.toFixed(2));
        }
      }

      // diagonal
      g.append("rect")
        .attr("x", i * cellSize)
        .attr("y", i * cellSize)
        .attr("width", cellSize - 2)
        .attr("height", cellSize - 2)
        .attr("fill", "#2a2d2e")
        .attr("rx", 2);
    }

    // row/column labels
    for (let i = 0; i < n; i++) {
      g.append("text")
        .attr("x", -8)
        .attr("y", i * cellSize + (cellSize - 2) / 2 + 4)
        .attr("text-anchor", "end")
        .attr("fill", "#cccccc")
        .attr("font-size", "11px")
        .text(cats[i].name);

      g.append("text")
        .attr("x", i * cellSize + (cellSize - 2) / 2)
        .attr("y", -8)
        .attr("text-anchor", "middle")
        .attr("fill", "#cccccc")
        .attr("font-size", "11px")
        .attr("transform", `rotate(-35, ${i * cellSize + (cellSize - 2) / 2}, -8)`)
        .text(cats[i].name);
    }

    // coherence verdict
    let hasCycle = false;
    for (let a = 0; a < n && !hasCycle; a++) {
      for (let b = 0; b < n && !hasCycle; b++) {
        for (let c = 0; c < n && !hasCycle; c++) {
          if (a !== b && b !== c && a !== c) {
            if (
              matrix[a][b] > 0.5 &&
              matrix[b][c] > 0.5 &&
              matrix[c][a] > 0.5
            ) {
              hasCycle = true;
            }
          }
        }
      }
    }

    const verdictY = n * cellSize + 30;
    g.append("rect")
      .attr("x", 0)
      .attr("y", verdictY)
      .attr("width", 12)
      .attr("height", 12)
      .attr("rx", 2)
      .attr("fill", hasCycle ? "#89d185" : "#f14c4c");

    g.append("text")
      .attr("x", 18)
      .attr("y", verdictY + 10)
      .attr("fill", "#cccccc")
      .attr("font-size", "12px")
      .text(
        hasCycle
          ? "coherent — support cycle ≥ 3 found"
          : "warning — no support cycle ≥ 3"
      );

    if (stale) {
      svg
        .append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "#1e1e1e")
        .attr("opacity", 0.2)
        .attr("pointer-events", "none");
    }
  }, [state, stale, onCrossfilter]);

  useEffect(() => {
    draw();
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-[#858585] text-sm">
        Run a script to see the coherence map
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      <div className="flex items-center gap-2 px-3 py-1 bg-[#252526] border-b border-[#1e1e1e] text-[11px] text-[#858585] shrink-0">
        Support adjacency matrix — click cells to toggle support edges
      </div>
      <svg ref={svgRef} className="flex-1" />
    </div>
  );
}

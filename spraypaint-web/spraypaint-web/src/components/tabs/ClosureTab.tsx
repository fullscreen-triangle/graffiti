"use client";

import React, { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import { ExecutionState, invertClosureDrag, ScriptDiff } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
  onCrossfilter: (diff: ScriptDiff) => void;
  stale: boolean;
}

export default function ClosureTab({ state, onCrossfilter, stale }: Props) {
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

    const margin = { top: 20, right: 30, bottom: 50, left: 55 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const seek = state.seeks[0];
    if (!seek) return;

    const maxStep = seek.closureSteps + 2;
    const x = d3.scaleLinear().domain([0, maxStep]).range([0, w]);
    const bandH = 24;

    // axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(maxStep).tickFormat(d3.format("d")))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g.append("text")
      .attr("x", w / 2)
      .attr("y", h + 36)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("committed step M");

    // equivalence class bands
    const classes = seek.equivalenceClasses;
    const classColors = ["#4fc1ff", "#ce9178", "#c586c0", "#89d185"];

    for (let c = 0; c < classes; c++) {
      // simulate when this class appeared
      const appearStep = c === 0 ? 1 : seek.confidenceStep + 1;

      g.append("rect")
        .attr("x", x(appearStep))
        .attr("y", 30 + c * (bandH + 6))
        .attr("width", x(seek.closureSteps) - x(appearStep))
        .attr("height", bandH)
        .attr("fill", classColors[c % classColors.length])
        .attr("opacity", stale ? 0.2 : 0.25)
        .attr("rx", 3);

      g.append("text")
        .attr("x", x(appearStep) + 6)
        .attr("y", 30 + c * (bandH + 6) + bandH / 2 + 4)
        .attr("fill", classColors[c % classColors.length])
        .attr("font-size", "11px")
        .attr("opacity", stale ? 0.4 : 1)
        .text(`class ${c + 1}${c === 0 ? " (primary)" : " (alternative)"}`);
    }

    // confidence threshold line
    const confX = x(seek.confidenceStep);
    g.append("line")
      .attr("x1", confX)
      .attr("x2", confX)
      .attr("y1", 0)
      .attr("y2", h)
      .attr("stroke", "#cca700")
      .attr("stroke-dasharray", "6,3")
      .attr("stroke-width", 1.5);

    g.append("text")
      .attr("x", confX + 4)
      .attr("y", 16)
      .attr("fill", "#cca700")
      .attr("font-size", "10px")
      .text("confidence θ met");

    // closure marker (draggable)
    const closureX = x(seek.closureSteps);
    g.append("line")
      .attr("x1", closureX)
      .attr("x2", closureX)
      .attr("y1", 0)
      .attr("y2", h)
      .attr("stroke", "#89d185")
      .attr("stroke-width", 2);

    g.append("polygon")
      .attr(
        "points",
        `${closureX - 6},0 ${closureX + 6},0 ${closureX},10`
      )
      .attr("fill", "#89d185")
      .attr("cursor", "ew-resize")
      .call(
        d3.drag<SVGPolygonElement, unknown>().on("end", (event) => {
          const newStep = Math.round(x.invert(event.x));
          const wantEarlier = newStep < seek.closureSteps;
          const diff = invertClosureDrag("", wantEarlier);
          if (diff) onCrossfilter(diff);
        }) as any
      );

    g.append("text")
      .attr("x", closureX + 4)
      .attr("y", h - 8)
      .attr("fill", "#89d185")
      .attr("font-size", "10px")
      .text("closure");

    // gap annotation
    if (seek.closureSteps > seek.confidenceStep + 1) {
      const gapX0 = x(seek.confidenceStep);
      const gapX1 = x(seek.closureSteps);

      g.append("rect")
        .attr("x", gapX0)
        .attr("y", h - 50)
        .attr("width", gapX1 - gapX0)
        .attr("height", 20)
        .attr("fill", "#f14c4c11")
        .attr("stroke", "#f14c4c44")
        .attr("stroke-dasharray", "3,3")
        .attr("rx", 3);

      g.append("text")
        .attr("x", (gapX0 + gapX1) / 2)
        .attr("y", h - 36)
        .attr("text-anchor", "middle")
        .attr("fill", "#f14c4c")
        .attr("font-size", "9px")
        .text("confidence-closure gap");
    }

    // status badge
    const statusY = 30 + classes * (bandH + 6) + 20;
    const statusColor = seek.status === "converged" ? "#89d185" : "#f14c4c";
    const statusText =
      seek.status === "converged"
        ? `convergent closure — 1 class`
        : `contested closure — ${classes} classes — DECLINED`;

    g.append("rect")
      .attr("x", 0)
      .attr("y", statusY)
      .attr("width", 12)
      .attr("height", 12)
      .attr("rx", 2)
      .attr("fill", statusColor);

    g.append("text")
      .attr("x", 18)
      .attr("y", statusY + 10)
      .attr("fill", "#cccccc")
      .attr("font-size", "12px")
      .text(statusText);

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
        Run a script to see closure status
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      <div className="flex items-center gap-2 px-3 py-1 bg-[#252526] border-b border-[#1e1e1e] text-[11px] text-[#858585] shrink-0">
        Closure timeline — drag the green marker to adjust stopping rule
      </div>
      <svg ref={svgRef} className="flex-1" />
    </div>
  );
}

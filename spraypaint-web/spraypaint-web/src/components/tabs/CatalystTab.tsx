"use client";

import React, { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import { ExecutionState, ScriptDiff } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
  onCrossfilter: (diff: ScriptDiff) => void;
  stale: boolean;
}

const NS_COLORS: Record<string, string> = {
  local: "#89d185",
  remote: "#4fc1ff",
  inference: "#c586c0",
  composite: "#dcdcaa",
};

export default function CatalystTab({ state, onCrossfilter, stale }: Props) {
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

    const margin = { top: 20, right: 20, bottom: 50, left: 120 };
    const halfH = (height - margin.top - margin.bottom - 30) / 2;

    // ── Top: stacked bar (composite power per seek) ──
    const g1 = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const w = width - margin.left - margin.right;

    const y1 = d3
      .scaleBand()
      .domain(state.seeks.map((s) => s.name))
      .range([0, halfH])
      .padding(0.3);

    const x1 = d3.scaleLinear().domain([0, 1]).range([0, w]);

    // axis
    g1.append("g")
      .call(d3.axisLeft(y1))
      .call((g) => g.selectAll("text").attr("fill", "#cccccc").attr("font-size", "11px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g1.append("g")
      .attr("transform", `translate(0,${halfH})`)
      .call(d3.axisBottom(x1).ticks(5).tickFormat(d3.format(".0%")))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g1.append("text")
      .attr("x", w / 2)
      .attr("y", halfH + 34)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("composite power 1 − ∏(1 − κᵢ)");

    // stacked bars per seek
    for (const seek of state.seeks) {
      let cumX = 0;
      const barY = y1(seek.name)!;
      const barH = y1.bandwidth();

      for (const cat of seek.catalysts) {
        const barW = x1(cat.marginalGain);
        g1.append("rect")
          .attr("x", x1(cumX))
          .attr("y", barY)
          .attr("width", Math.max(0, barW))
          .attr("height", barH)
          .attr("fill", NS_COLORS[cat.namespace] ?? "#cccccc")
          .attr("opacity", stale ? 0.4 : 0.85)
          .attr("stroke", "#1e1e1e")
          .attr("stroke-width", 1)
          .attr("cursor", "pointer")
          .append("title")
          .text(`${cat.name} (${cat.namespace})\nκ = ${cat.power.toFixed(3)}\nΔ = ${cat.marginalGain.toFixed(3)}`);

        // label inside if wide enough
        if (barW > 30) {
          g1.append("text")
            .attr("x", x1(cumX) + barW / 2)
            .attr("y", barY + barH / 2 + 4)
            .attr("text-anchor", "middle")
            .attr("fill", "#1e1e1e")
            .attr("font-size", "9px")
            .attr("font-weight", "600")
            .attr("pointer-events", "none")
            .text(cat.name);
        }

        cumX += cat.marginalGain;
      }
    }

    // ── Bottom: marginal gain bar chart ──
    const g2 = svg
      .append("g")
      .attr(
        "transform",
        `translate(${margin.left},${margin.top + halfH + 50})`
      );

    // use the first seek's catalysts for the marginal-gain view
    const cats = state.seeks[0]?.catalysts ?? [];

    const x2 = d3
      .scaleBand()
      .domain(cats.map((c) => c.name))
      .range([0, w])
      .padding(0.2);

    const maxGain = d3.max(cats, (c) => c.marginalGain) ?? 0.5;
    const y2 = d3.scaleLinear().domain([0, maxGain * 1.1]).range([halfH - 20, 0]);

    g2.append("g")
      .attr("transform", `translate(0,${halfH - 20})`)
      .call(d3.axisBottom(x2))
      .call((g) => g.selectAll("text").attr("fill", "#cccccc").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g2.append("g")
      .call(d3.axisLeft(y2).ticks(4))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g2.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -(halfH - 20) / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("Δκ marginal gain");

    for (const cat of cats) {
      const barX = x2(cat.name)!;
      const barH = halfH - 20 - y2(cat.marginalGain);

      g2.append("rect")
        .attr("x", barX)
        .attr("y", y2(cat.marginalGain))
        .attr("width", x2.bandwidth())
        .attr("height", barH)
        .attr("fill", NS_COLORS[cat.namespace] ?? "#cccccc")
        .attr("opacity", stale ? 0.4 : 0.85)
        .attr("rx", 2);
    }

    // stale overlay
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
        Run a script to see catalyst breakdown
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      <div className="flex items-center gap-4 px-3 py-1 bg-[#252526] border-b border-[#1e1e1e] text-[11px] shrink-0">
        {Object.entries(NS_COLORS).map(([ns, color]) => (
          <span key={ns} className="flex items-center gap-1">
            <span
              className="w-2 h-2 rounded-sm inline-block"
              style={{ backgroundColor: color }}
            />
            <span className="text-[#858585]">{ns}</span>
          </span>
        ))}
        <span className="ml-auto text-[#858585]">
          hover bars for details
        </span>
      </div>
      <svg ref={svgRef} className="flex-1" />
    </div>
  );
}

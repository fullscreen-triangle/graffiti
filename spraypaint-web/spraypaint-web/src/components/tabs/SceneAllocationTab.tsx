"use client";

import React, { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import { ExecutionState, invertSceneAllocation, ScriptDiff } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
  onCrossfilter: (diff: ScriptDiff) => void;
  stale: boolean;
}

export default function SceneAllocationTab({ state, onCrossfilter, stale }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    if (!svgRef.current || !state) return;

    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    const scenes = state.scenes;
    if (scenes.length === 0) return;

    const margin = { top: 20, right: 30, bottom: 60, left: 55 };
    const halfH = (height - margin.top - margin.bottom - 40) / 2;
    const w = width - margin.left - margin.right;

    // ── Top: allocation bars ──
    const g1 = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(scenes.map((s) => s.scene))
      .range([0, w])
      .padding(0.25);

    const maxAlloc = d3.max(scenes, (s) => s.allocated) ?? 5;
    const y1 = d3
      .scaleLinear()
      .domain([0, maxAlloc + 1])
      .range([halfH, 0]);

    g1.append("g")
      .attr("transform", `translate(0,${halfH})`)
      .call(d3.axisBottom(x))
      .call((g) => g.selectAll("text").attr("fill", "#cccccc").attr("font-size", "11px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g1.append("g")
      .call(d3.axisLeft(y1).ticks(maxAlloc + 1).tickFormat(d3.format("d")))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g1.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -halfH / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("allocated slots");

    const clearingPrice = scenes[0]?.clearingPrice ?? 0;

    for (const scene of scenes) {
      const barX = x(scene.scene)!;
      const barH = halfH - y1(scene.allocated);
      const abovePrice = scene.bestScore >= clearingPrice;

      g1.append("rect")
        .attr("x", barX)
        .attr("y", y1(scene.allocated))
        .attr("width", x.bandwidth())
        .attr("height", Math.max(0, barH))
        .attr("fill", abovePrice ? "#4fc1ff" : "#3c3c3c")
        .attr("opacity", stale ? 0.4 : 0.85)
        .attr("rx", 2)
        .attr("cursor", "pointer")
        .on("click", () => {
          // toggle scene inclusion
          const enabled = scenes
            .filter((s) => s.scene !== scene.scene && s.allocated > 0)
            .map((s) => s.scene);
          if (scene.allocated === 0) enabled.push(scene.scene);

          const diff = invertSceneAllocation("", enabled);
          if (diff) onCrossfilter(diff);
        })
        .append("title")
        .text(
          `${scene.scene}\nallocated: ${scene.allocated}\nbest BM25: ${scene.bestScore.toFixed(3)}\nmedian: ${scene.medianScore.toFixed(3)}\nclearing price p*: ${clearingPrice.toFixed(3)}\nclick to toggle`
        );

      // allocation count label
      if (scene.allocated > 0) {
        g1.append("text")
          .attr("x", barX + x.bandwidth() / 2)
          .attr("y", y1(scene.allocated) - 4)
          .attr("text-anchor", "middle")
          .attr("fill", "#cccccc")
          .attr("font-size", "11px")
          .attr("font-weight", "600")
          .text(scene.allocated);
      }
    }

    // clearing price line
    // (mapped to allocation space as a reference)
    g1.append("text")
      .attr("x", w - 4)
      .attr("y", 12)
      .attr("text-anchor", "end")
      .attr("fill", "#cca700")
      .attr("font-size", "10px")
      .text(`p* = ${clearingPrice.toFixed(3)}`);

    // ── Bottom: BM25 score box plots ──
    const g2 = svg
      .append("g")
      .attr(
        "transform",
        `translate(${margin.left},${margin.top + halfH + 50})`
      );

    const maxScore = d3.max(scenes, (s) => s.bestScore) ?? 2;
    const y2 = d3
      .scaleLinear()
      .domain([0, maxScore * 1.1])
      .range([halfH - 20, 0]);

    g2.append("g")
      .attr("transform", `translate(0,${halfH - 20})`)
      .call(d3.axisBottom(x))
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
      .text("BM25 score");

    // clearing price line
    g2.append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", y2(clearingPrice))
      .attr("y2", y2(clearingPrice))
      .attr("stroke", "#cca700")
      .attr("stroke-dasharray", "6,3")
      .attr("stroke-width", 1.5);

    for (const scene of scenes) {
      const barX = x(scene.scene)!;
      const bw = x.bandwidth();

      // box from median to best
      g2.append("rect")
        .attr("x", barX + bw * 0.15)
        .attr("y", y2(scene.bestScore))
        .attr("width", bw * 0.7)
        .attr("height", y2(scene.medianScore) - y2(scene.bestScore))
        .attr("fill", scene.bestScore >= clearingPrice ? "#4fc1ff33" : "#3c3c3c33")
        .attr("stroke", scene.bestScore >= clearingPrice ? "#4fc1ff" : "#555")
        .attr("stroke-width", 1)
        .attr("rx", 2);

      // median line
      g2.append("line")
        .attr("x1", barX + bw * 0.15)
        .attr("x2", barX + bw * 0.85)
        .attr("y1", y2(scene.medianScore))
        .attr("y2", y2(scene.medianScore))
        .attr("stroke", "#cccccc")
        .attr("stroke-width", 2);
    }

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
        Run a script to see scene allocation
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      <div className="flex items-center gap-2 px-3 py-1 bg-[#252526] border-b border-[#1e1e1e] text-[11px] text-[#858585] shrink-0">
        Water-filling allocation — click bars to toggle scenes
      </div>
      <svg ref={svgRef} className="flex-1" />
    </div>
  );
}

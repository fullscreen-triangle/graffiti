"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import { ExecutionState, invertTrajectoryDrag, ScriptDiff } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
  onCrossfilter: (diff: ScriptDiff) => void;
  stale: boolean;
}

export default function TrajectoryTab({ state, onCrossfilter, stale }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedSeek, setSelectedSeek] = useState<string | null>(null);
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);

  const draw = useCallback(() => {
    if (!svgRef.current || !state || state.seeks.length === 0) return;

    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth;
    const height = container.clientHeight - 40;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    const margin = { top: 20, right: 30, bottom: 40, left: 55 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const seeks = selectedSeek
      ? state.seeks.filter((s) => s.name === selectedSeek)
      : state.seeks;

    const allPoints = seeks.flatMap((s) => s.trajectory);
    const maxStep = d3.max(allPoints, (d) => d.step) ?? 10;

    const x = d3.scaleLinear().domain([1, maxStep]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

    // grid lines
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(y.ticks(5))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d))
      .attr("stroke", "#333333")
      .attr("stroke-dasharray", "2,4");

    // floor line
    const floor = state.seeks[0]?.floor ?? 0.02;
    const floorY = y(floor);
    g.append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", floorY)
      .attr("y2", floorY)
      .attr("stroke", "#cca700")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "6,3");

    g.append("text")
      .attr("x", w - 4)
      .attr("y", floorY - 6)
      .attr("text-anchor", "end")
      .attr("fill", "#cca700")
      .attr("font-size", "10px")
      .attr("font-family", "monospace")
      .text(`β = ${floor}`);

    // axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(Math.min(maxStep, 10)).tickFormat(d3.format("d")))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    g.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444444"));

    // axis labels
    g.append("text")
      .attr("x", w / 2)
      .attr("y", h + 34)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("committed step M");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -h / 2)
      .attr("y", -42)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("alignment a(x, x*)");

    const colors = ["#4fc1ff", "#89d185", "#ce9178", "#c586c0", "#dcdcaa"];

    // draw each seek's trajectory
    seeks.forEach((seek, si) => {
      const color = colors[si % colors.length];
      const line = d3
        .line<(typeof seek.trajectory)[0]>()
        .x((d) => x(d.step))
        .y((d) => y(Math.min(1, Math.max(0, d.alignment))))
        .curve(d3.curveMonotoneX);

      g.append("path")
        .datum(seek.trajectory)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("opacity", stale ? 0.4 : 1);

      // dots
      g.selectAll(`.dot-${si}`)
        .data(seek.trajectory)
        .enter()
        .append("circle")
        .attr("cx", (d) => x(d.step))
        .attr("cy", (d) => y(Math.min(1, Math.max(0, d.alignment))))
        .attr("r", 3)
        .attr("fill", color)
        .attr("opacity", stale ? 0.4 : 0.8)
        .attr("cursor", "pointer")
        .on("mouseover", function (event, d) {
          d3.select(this).attr("r", 5);
          setHoveredStep(d.step);
        })
        .on("mouseout", function () {
          d3.select(this).attr("r", 3);
          setHoveredStep(null);
        });

      // confidence step marker
      const confStep = seek.confidenceStep;
      const confPoint = seek.trajectory.find((t) => t.step === confStep);
      if (confPoint) {
        g.append("line")
          .attr("x1", x(confStep))
          .attr("x2", x(confStep))
          .attr("y1", 0)
          .attr("y2", h)
          .attr("stroke", "#858585")
          .attr("stroke-dasharray", "3,3")
          .attr("opacity", 0.5);

        g.append("text")
          .attr("x", x(confStep) + 4)
          .attr("y", 12)
          .attr("fill", "#858585")
          .attr("font-size", "9px")
          .text("θ met");
      }

      // closure marker
      const closureStep = seek.closureSteps;
      g.append("line")
        .attr("x1", x(closureStep))
        .attr("x2", x(closureStep))
        .attr("y1", 0)
        .attr("y2", h)
        .attr("stroke", "#89d185")
        .attr("stroke-dasharray", "3,3")
        .attr("opacity", 0.7);

      g.append("text")
        .attr("x", x(closureStep) + 4)
        .attr("y", 24)
        .attr("fill", "#89d185")
        .attr("font-size", "9px")
        .text("closed");
    });

    // brush for crossfilter: vertical band on y-axis
    const brush = d3
      .brushY()
      .extent([
        [0, 0],
        [w, h],
      ])
      .on("end", (event) => {
        if (!event.selection) return;
        const [y0, y1] = event.selection as [number, number];
        const targetResidual = y.invert(y0); // top of brush = desired residual
        const currentFloor = state.seeks[0]?.floor ?? 0.02;

        // invert chart selection to script diff
        const diff = invertTrajectoryDrag("", targetResidual, currentFloor);
        if (diff) {
          onCrossfilter(diff);
        }

        // clear brush after applying
        g.select<SVGGElement>(".brush").call(brush.move as any, null);
      });

    g.append("g").attr("class", "brush").call(brush);

    // stale overlay
    if (stale) {
      svg
        .append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "#1e1e1e")
        .attr("opacity", 0.3)
        .attr("pointer-events", "none");
    }
  }, [state, selectedSeek, stale, onCrossfilter]);

  useEffect(() => {
    draw();
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-[#858585] text-sm">
        Run a script to see the convergence trajectory
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      {/* seek selector */}
      <div className="flex items-center gap-2 px-3 py-1 bg-[#252526] border-b border-[#1e1e1e] text-[12px] shrink-0">
        <span className="text-[#858585]">Seek:</span>
        <button
          className={`px-2 py-0.5 rounded ${!selectedSeek ? "bg-[#094771] text-[#cccccc]" : "text-[#858585] hover:text-[#cccccc]"}`}
          onClick={() => setSelectedSeek(null)}
        >
          all
        </button>
        {state.seeks.map((s) => (
          <button
            key={s.name}
            className={`px-2 py-0.5 rounded ${selectedSeek === s.name ? "bg-[#094771] text-[#cccccc]" : "text-[#858585] hover:text-[#cccccc]"}`}
            onClick={() => setSelectedSeek(s.name)}
          >
            {s.name}
          </button>
        ))}
        {hoveredStep !== null && (
          <span className="ml-auto text-[#4fc1ff]">step {hoveredStep}</span>
        )}
        <span className="ml-2 text-[10px] text-[#858585]">
          drag vertical region to adjust floor
        </span>
      </div>

      <svg ref={svgRef} className="flex-1" />
    </div>
  );
}

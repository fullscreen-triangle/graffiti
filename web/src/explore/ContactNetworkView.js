// The "network grid" view: nodes are claims (and the medium), edges are
// currently-filtered committed contacts (crossfilter's shared filter
// state, same as every other linked chart). A D3 force simulation lays
// it out; edge color encodes catalytic power, edge width encodes weight.

import React, { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";
import { currentRows } from "./crossfilterPack";

export function ContactNetworkView({ width = 640, height = 420 }) {
  const { pack } = useCrossfilter();
  const svgRef = useRef(null);
  const simRef = useRef(null);

  const draw = useCallback(() => {
    const rows = currentRows(pack);
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    if (rows.length === 0) return;

    const nodeIds = new Set();
    for (const r of rows) {
      nodeIds.add(r.from);
      nodeIds.add(r.to);
    }
    const nodes = Array.from(nodeIds, (id) => ({ id }));
    const links = rows.map((r) => ({ source: r.from, target: r.to, weight: r.weight, power: r.power }));

    if (simRef.current) simRef.current.stop();
    const sim = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(70).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-140))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(18));
    simRef.current = sim;

    const powerColor = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    const link = svg
      .append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) => (d.power == null ? "#475569" : powerColor(d.power)))
      .attr("stroke-width", (d) => Math.max(1, Math.min(6, d.weight)))
      .attr("stroke-opacity", 0.7);

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => (d.id === "__medium__" ? 10 : 6))
      .attr("fill", (d) => (d.id === "__medium__" ? "#f97316" : "#38bdf8"))
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    node.append("title").text((d) => d.id);

    sim.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });
  }, [pack, width, height]);

  useChartRedraw(draw);
  useEffect(() => {
    draw();
    return () => simRef.current?.stop();
  }, [draw]);

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-white/50 font-mono">contact network (drag nodes; orange = medium)</span>
      <svg ref={svgRef} width={width} height={height} className="bg-black/40 rounded" />
    </div>
  );
}

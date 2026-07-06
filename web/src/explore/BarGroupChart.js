// A bar chart over a crossfilter group, redrawn from raw D3 on every
// tick of the shared redraw bus -- one of the linked views. Clicking a
// bar toggles a filter on that dimension's value, which triggers
// redrawAll(), which every other chart (including this one) picks up.

import React, { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";
import { useCrossfilter, useChartRedraw } from "./CrossfilterContext";

export function BarGroupChart({ title, dimKey, groupKey, width = 320, height = 160 }) {
  const { pack, redrawAll } = useCrossfilter();
  const svgRef = useRef(null);
  const [activeKey, setActiveKey] = useState(null);

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const data = pack.groups[groupKey].top(Infinity).filter((d) => d.value > 0);
    if (data.length === 0) return;

    const margin = { top: 8, right: 8, bottom: 28, left: 28 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const x = d3
      .scaleBand()
      .domain(data.map((d) => String(d.key)))
      .range([0, innerW])
      .padding(0.15);
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value) || 1])
      .range([innerH, 0]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    g.selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", (d) => x(String(d.key)))
      .attr("y", (d) => y(d.value))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(d.value))
      .attr("fill", (d) => (activeKey !== null && String(d.key) === String(activeKey) ? "#f97316" : "#38bdf8"))
      .attr("opacity", (d) => (activeKey !== null && String(d.key) !== String(activeKey) ? 0.35 : 1))
      .style("cursor", "pointer")
      .on("click", (_event, d) => {
        const dim = pack.dims[dimKey];
        if (activeKey === d.key) {
          dim.filterAll();
          setActiveKey(null);
        } else {
          dim.filter(d.key);
          setActiveKey(d.key);
        }
        redrawAll();
      });

    g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x)).selectAll("text").style("font-size", "9px");
    g.append("g").call(d3.axisLeft(y).ticks(4)).selectAll("text").style("font-size", "9px");
  }, [pack, groupKey, dimKey, width, height, activeKey, redrawAll]);

  useChartRedraw(draw);
  useEffect(draw, [draw]);

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-white/50 font-mono">{title}</span>
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
}

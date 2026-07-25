"use client";

import React, { useRef, useEffect, useMemo } from "react";
import { ExecutionState } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
}

// We render a 2D force-directed graph via canvas since react-force-graph-3d
// requires dynamic import and WebGL context. This keeps the bundle light
// and the fallback reliable.

interface GNode {
  id: string;
  label: string;
  type: "seek" | "catalyst" | "medium";
  closed: boolean;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface GLink {
  source: string;
  target: string;
  weight: number;
  type: "catalyst" | "support" | "dependency";
}

function buildGraph(state: ExecutionState): { nodes: GNode[]; links: GLink[] } {
  const nodes: GNode[] = [];
  const links: GLink[] = [];

  // medium node
  nodes.push({
    id: "medium",
    label: "𝓜",
    type: "medium",
    closed: false,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
  });

  for (const seek of state.seeks) {
    // seek node
    nodes.push({
      id: `seek:${seek.name}`,
      label: seek.name,
      type: "seek",
      closed: seek.status === "converged",
      x: (Math.random() - 0.5) * 300,
      y: (Math.random() - 0.5) * 300,
      vx: 0,
      vy: 0,
    });

    links.push({
      source: "medium",
      target: `seek:${seek.name}`,
      weight: seek.residue,
      type: "dependency",
    });

    for (const cat of seek.catalysts) {
      const catId = `cat:${seek.name}:${cat.name}`;
      if (!nodes.find((n) => n.id === catId)) {
        nodes.push({
          id: catId,
          label: cat.name,
          type: "catalyst",
          closed: seek.status === "converged",
          x: (Math.random() - 0.5) * 300,
          y: (Math.random() - 0.5) * 300,
          vx: 0,
          vy: 0,
        });
      }
      links.push({
        source: `seek:${seek.name}`,
        target: catId,
        weight: cat.power,
        type: "catalyst",
      });

      // support edges
      for (const edge of cat.supportEdges) {
        const targetId = `cat:${seek.name}:${edge.target}`;
        links.push({
          source: catId,
          target: targetId,
          weight: edge.strength,
          type: "support",
        });
      }
    }
  }

  return { nodes, links };
}

function simulate(nodes: GNode[], links: GLink[], iterations: number) {
  for (let iter = 0; iter < iterations; iter++) {
    // repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 1;
        const force = 800 / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        nodes[i].vx -= fx;
        nodes[i].vy -= fy;
        nodes[j].vx += fx;
        nodes[j].vy += fy;
      }
    }

    // attraction along links
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    for (const link of links) {
      const s = nodeMap.get(link.source);
      const t = nodeMap.get(link.target);
      if (!s || !t) continue;
      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 1;
      const force = dist * 0.01 * (link.weight + 0.1);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      s.vx += fx;
      s.vy += fy;
      t.vx -= fx;
      t.vy -= fy;
    }

    // apply velocity with damping
    const damping = 0.85;
    for (const node of nodes) {
      if (node.type === "medium") {
        node.vx = 0;
        node.vy = 0;
        continue;
      }
      node.x += node.vx * 0.5;
      node.y += node.vy * 0.5;
      node.vx *= damping;
      node.vy *= damping;
    }
  }
}

const NS_COLORS: Record<string, string> = {
  seek: "#4fc1ff",
  catalyst: "#89d185",
  medium: "#cccccc",
};

const LINK_COLORS: Record<string, string> = {
  catalyst: "#4fc1ff44",
  support: "#89d18544",
  dependency: "#cccccc33",
};

export default function ForceGraphTab({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const graph = useMemo(() => {
    if (!state) return null;
    const g = buildGraph(state);
    simulate(g.nodes, g.links, 120);
    return g;
  }, [state]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !graph) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.parentElement?.getBoundingClientRect();
    if (rect) {
      canvas.width = rect.width * 2;
      canvas.height = rect.height * 2;
      ctx.scale(2, 2);
    }

    const w = (rect?.width ?? 600);
    const h = (rect?.height ?? 400);
    const cx = w / 2;
    const cy = h / 2;

    ctx.clearRect(0, 0, w, h);

    // draw links
    const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
    for (const link of graph.links) {
      const s = nodeMap.get(link.source);
      const t = nodeMap.get(link.target);
      if (!s || !t) continue;
      ctx.beginPath();
      ctx.moveTo(cx + s.x, cy + s.y);
      ctx.lineTo(cx + t.x, cy + t.y);
      ctx.strokeStyle = LINK_COLORS[link.type] ?? "#ffffff22";
      ctx.lineWidth = Math.max(0.5, link.weight * 2);
      ctx.stroke();
    }

    // draw nodes
    for (const node of graph.nodes) {
      const r = node.type === "medium" ? 10 : node.type === "seek" ? 7 : 4;
      ctx.beginPath();
      ctx.arc(cx + node.x, cy + node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = NS_COLORS[node.type] ?? "#cccccc";
      if (node.closed) {
        ctx.globalAlpha = 1;
      } else {
        ctx.globalAlpha = 0.6;
      }
      ctx.fill();
      ctx.globalAlpha = 1;

      // label
      ctx.fillStyle = "#cccccc";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(node.label, cx + node.x, cy + node.y + r + 12);
    }
  }, [graph]);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-[#858585] text-sm">
        Run a script to see the agent graph
      </div>
    );
  }

  return (
    <div className="h-full relative">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ imageRendering: "auto" }}
      />
      {/* legend */}
      <div className="absolute bottom-3 left-3 flex gap-4 text-[11px] text-[#858585] bg-[#1e1e1e99] px-2 py-1 rounded">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#cccccc] inline-block" />
          medium
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#4fc1ff] inline-block" />
          seek
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#89d185] inline-block" />
          catalyst
        </span>
      </div>
    </div>
  );
}

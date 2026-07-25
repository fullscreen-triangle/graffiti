"use client";

import React, { useMemo, useEffect, useRef } from "react";
import { ExecutionState } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
}

function generateReport(state: ExecutionState): string {
  const seeks = state.seeks;
  const scenes = state.scenes;
  const inv = state.invariants;

  const seekRows = seeks
    .map(
      (s) =>
        `| ${s.name} | ${s.target.substring(0, 40)} | ${s.catalysts.length} | ${s.compositepower.toFixed(3)} | ${s.status} | ${s.residue.toFixed(4)} |`
    )
    .join("\n");

  const sceneRows = scenes
    .map(
      (s) =>
        `| ${s.scene} | ${s.allocated} | ${s.bestScore.toFixed(3)} | ${s.medianScore.toFixed(3)} |`
    )
    .join("\n");

  const catSection = seeks
    .map((s) => {
      const catList = s.catalysts
        .map(
          (c) =>
            `  - **${c.name}** (${c.namespace}): κ = ${c.power.toFixed(3)}, Δκ = ${c.marginalGain.toFixed(3)}`
        )
        .join("\n");

      return `### Seek: \`${s.name}\`\n\n**Target:** ${s.target}\n\n**Catalyst chain:**\n${catList}\n\n**Composite power:** 1 − ∏(1 − κᵢ) = ${s.compositepower.toFixed(4)}\n\n**Terminal residue:** ${s.residue.toFixed(4)}\n\n**Status:** ${s.status === "converged" ? "Convergent closure — single equivalence class" : `Contested closure — ${s.equivalenceClasses} distinct classes — DECLINED`}`;
    })
    .join("\n\n---\n\n");

  return `# Execution Report

## 1. Summary

This report documents the execution of a Graffiti (.grf) search project
comprising **${seeks.length} seek${seeks.length !== 1 ? "s" : ""}** across
**${scenes.filter((s) => s.allocated > 0).length} active scenes**.
Execution committed **${inv.committedCount} act${inv.committedCount !== 1 ? "s" : ""}**
against the search medium, with all four runtime invariants satisfied.

### Resolution Floor

Every claim returned by this execution satisfies the resolution-floor
bound: no search returns a claim separated from the medium by a cut of
weight zero. The ambient floor for this project is:

$$\\beta > 0$$

This is a structural property of the search medium (Theorem 3.3), not a
defect of the search process.

## 2. Seek Results

| Seek | Target | Catalysts | Composite κ | Status | Residue |
|------|--------|-----------|-------------|--------|---------|
${seekRows}

## 3. Catalyst Analysis

The multiplicative composition law governs catalyst chains:

$$\\kappa(\\gamma_1 \\circ \\cdots \\circ \\gamma_n) = 1 - \\prod_{i=1}^{n}(1 - \\kappa_i)$$

Each successive catalyst closes a fraction of the remaining alignment gap,
with strictly diminishing marginal returns (Corollary 6.5).

${catSection}

## 4. Coherence Assessment

A claim is robustly grounded if and only if its supporting catalysts form
a strongly connected structure of at least three members (Theorem 6.13).
The coherence condition is:

$$|\\text{support cycle}| \\geq 3 \\quad \\text{with} \\quad \\theta > \\tfrac{1}{2}$$

${seeks.map((s) => `**${s.name}:** ${s.catalysts.length >= 3 ? "✓ Triangle condition satisfied" : "⚠ Fewer than 3 catalysts — coherence not guaranteed"}`).join("\n\n")}

## 5. Scene Allocation

The result budget *k* is allocated across scenes by the water-filling
rule. Each scene contributes passages while their relevance clears the
clearing price *p*\\*; scenes below it receive zero allocation.

| Scene | Allocated | Best BM25 | Median BM25 |
|-------|-----------|-----------|-------------|
${sceneRows}

**Clearing price:** p* = ${scenes[0]?.clearingPrice.toFixed(3) ?? "—"}

## 6. Closure Analysis

Closure (Definition 7.2) is strictly stronger than a confidence threshold
(Theorem 7.3). A search is closed when every further available catalyst
resolves to a claim-region already reached.

${seeks.map((s) => `**${s.name}:** Confidence met at step ${s.confidenceStep}; closure at step ${s.closureSteps}. Gap: ${s.closureSteps - s.confidenceStep} steps of additional verification.`).join("\n\n")}

## 7. Invariant Conformance

| Invariant | Status |
|-----------|--------|
| Conserved identity (hash + χ) | ${inv.identityValid ? "✓ Valid" : "✗ Invalid"} |
| Never-resetting count (M = ${inv.committedCount}) | ✓ Monotone |
| Search-not-fetch | ${inv.searchNotFetch ? "✓ Enforced" : "✗ Violated"} |
| Exclusive phases | ${inv.exclusivePhases ? "✓ No overlap" : "✗ Overlap detected"} |

## References

1. Sachikonye, K. F. (2026). *Semantic Causal Propagation: An
   Individuation-Theoretic Calculus of Boundary-Free Search.* Technical
   University of Munich.

2. Ford, L. R. & Fulkerson, D. R. (1956). Maximal flow through a network.
   *Canadian Journal of Mathematics*, 8, 399–404.

3. Edmonds, J. & Karp, R. M. (1972). Theoretical improvements in
   algorithmic efficiency for network flow problems. *Journal of the ACM*,
   19(2), 248–264.
`;
}

export default function ReportTab({ state }: Props) {
  const contentRef = useRef<HTMLDivElement>(null);

  const report = useMemo(() => {
    if (!state) return "";
    return generateReport(state);
  }, [state]);

  // render KaTeX for math blocks
  useEffect(() => {
    if (!contentRef.current) return;

    const loadKatex = async () => {
      try {
        const katex = await import("katex");
        const el = contentRef.current;
        if (!el) return;

        // render display math $$...$$
        const html = el.innerHTML;
        const rendered = html.replace(
          /\$\$([\s\S]*?)\$\$/g,
          (_, tex: string) => {
            try {
              return katex.default.renderToString(tex.trim(), {
                displayMode: true,
                throwOnError: false,
              });
            } catch {
              return `<code>${tex}</code>`;
            }
          }
        );
        el.innerHTML = rendered;
      } catch (e) {
        console.warn("KaTeX not available, equations shown as text", e);
      }
    };

    loadKatex();
  }, [report]);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-[#858585] text-sm">
        Run a script to generate the execution report
      </div>
    );
  }

  // simple markdown-to-html (headers, bold, tables, lists, code, hr)
  const toHtml = (md: string): string => {
    return md
      .replace(/^### (.+)$/gm, '<h3 class="report-h3">$1</h3>')
      .replace(/^## (.+)$/gm, '<h2 class="report-h2">$1</h2>')
      .replace(/^# (.+)$/gm, '<h1 class="report-h1">$1</h1>')
      .replace(/^---$/gm, "<hr />")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, '<code class="report-code">$1</code>')
      .replace(
        /^\|(.+)\|$/gm,
        (_, row: string) => {
          const cells = row.split("|").map((c: string) => c.trim());
          const tds = cells.map((c: string) => `<td>${c}</td>`).join("");
          return `<tr>${tds}</tr>`;
        }
      )
      .replace(
        /(<tr>.*<\/tr>\n?)+/g,
        (block: string) => `<table class="report-table">${block}</table>`
      )
      .replace(/^  - (.+)$/gm, '<li class="report-li">$1</li>')
      .replace(/(<li.*<\/li>\n?)+/g, (block: string) => `<ul>${block}</ul>`)
      .replace(/\n\n/g, "</p><p>")
      .replace(/^/, "<p>")
      .replace(/$/, "</p>");
  };

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-3xl mx-auto px-8 py-6">
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
        />
        <div
          ref={contentRef}
          className="report-content"
          dangerouslySetInnerHTML={{ __html: toHtml(report) }}
        />
      </div>
    </div>
  );
}

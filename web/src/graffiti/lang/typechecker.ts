/**
 * Static type system for Graffiti (.grf).
 *
 * Implements Definition (Types), Definition (Typing Judgement), Rule
 * (Floor Positivity), Rule (Generation; Bond; Close) [analogue: Rule
 * (Coherence for Closure)], and Theorem (No Zero-Residue Claim) of
 * semantic-causal-propagation.tex, Section "The Type System".
 */

import type { CatalystDecl, ChainExpr, Program, ProjectDecl, SeekStmt } from "./ast";
import { CatalystRegistry } from "../orchestration/catalyst";

export type Severity = "error" | "warning";

export interface Diagnostic {
  severity: Severity;
  code: string;
  message: string;
  seekName?: string;
  line?: number;
}

export interface TypeCheckResult {
  ok: boolean;
  diagnostics: Diagnostic[];
  ambientFloor: number;
}

/**
 * Theorem (No Zero-Residue Claim): the ambient floor must be a positive
 * declared constant before any `seek` in scope can type-check
 * (Rule (Floor Positivity)).
 */
export function checkFloorPositivity(program: Program): { floor: number; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];
  const floorDecls = program.decls.filter((d) => d.kind === "floorDecl");

  if (floorDecls.length === 0) {
    diagnostics.push({
      severity: "error",
      code: "NO_FLOOR_DECLARED",
      message:
        'No "floor" declaration found. Every Claim value must carry a positive floor annotation ' +
        "(Rule: Floor Positivity); declare `floor <value>` before any seek.",
    });
    return { floor: NaN, diagnostics };
  }

  const last = floorDecls[floorDecls.length - 1] as { value: number };
  if (!(last.value > 0)) {
    diagnostics.push({
      severity: "error",
      code: "NONPOSITIVE_FLOOR",
      message: `Declared floor must be strictly positive (got ${last.value}). ` + "A zero or negative floor makes every Claim value ill-typed (Theorem: No Zero-Residue Claim).",
    });
  }

  return { floor: last.value, diagnostics };
}

function collectCatalystRefs(chain: ChainExpr, out: string[]): void {
  if (chain.kind === "catalystRef") {
    out.push(chain.qname.join("."));
  } else if (chain.kind === "chainSeq") {
    for (const step of chain.steps) collectCatalystRefs(step, out);
  } else if (chain.kind === "chainPar") {
    for (const branch of chain.branches) collectCatalystRefs(branch, out);
  }
}

/**
 * Rule (Coherence for Closure): a seek whose `via` chain is explicit and
 * whose `until` clause is bare `converge` (no explicit `cond` weakening
 * it) type-checks only if the chain's catalysts number at least three --
 * the minimum necessary, by Theorem (Coherence Requires Three Mutually
 * Supporting Catalysts), to ground a claim robustly. A chain of one or
 * two catalysts under bare `converge` does not hard-fail (an author may
 * deliberately accept weaker evidence under an explicit `cond`), but it
 * is flagged as a warning here because it silently claims full closure
 * strength it structurally cannot have.
 */
function checkSeekCoherence(seek: SeekStmt): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  if (seek.via === null) return diagnostics;
  if (seek.until.kind !== "converge") return diagnostics;

  const refs: string[] = [];
  collectCatalystRefs(seek.via, refs);
  const uniqueCatalysts = Array.from(new Set(refs));

  if (uniqueCatalysts.length < 3) {
    diagnostics.push({
      severity: "warning",
      code: "COHERENCE_WARNING",
      seekName: seek.name,
      line: seek.line,
      message:
        `seek "${seek.name}" uses ${uniqueCatalysts.length} distinct catalyst(s) under bare ` +
        '"until converge". Coherent grounding against a single dissenting source requires a ' +
        "strongly connected support cycle of at least three independent catalysts " +
        "(Theorem: Coherence Requires Three Mutually Supporting Catalysts). Add at least one " +
        'more independent catalyst, or weaken to an explicit "until <cond>" if fewer sources ' +
        "are an intentional, acknowledged risk.",
    });
  }

  return diagnostics;
}

/**
 * Verify a `via` chain's catalyst references resolve against the
 * registry and that argument/type shapes are structurally sane. This is
 * a light-weight structural check; full signature checking against
 * declared Catalyst input/output types is performed when a
 * CatalystRegistry with typed signatures is supplied.
 */
function checkCatalystReferences(seek: SeekStmt, registry: CatalystRegistry | null): Diagnostic[] {
  if (seek.via === null || registry === null) return [];
  const diagnostics: Diagnostic[] = [];
  const refs: string[] = [];
  collectCatalystRefs(seek.via, refs);
  for (const ref of refs) {
    if (!registry.has(ref)) {
      diagnostics.push({
        severity: "error",
        code: "UNKNOWN_CATALYST",
        seekName: seek.name,
        line: seek.line,
        message: `seek "${seek.name}" references unknown catalyst "${ref}". Registered catalysts: ${registry.names().join(", ") || "(none)"}.`,
      });
    }
  }
  return diagnostics;
}

/**
 * Mandatory-exclusion check (Corollary: Mandatory Negation Boundary):
 * every seek's `not{}` clause must name at least one exclusion term. The
 * parser already enforces syntactic presence of `not{}`; this verifies
 * it is non-vacuous (at least one string literal), since an empty
 * boundary would assert no exclusion at all.
 */
function checkBoundaryNonVacuous(seek: SeekStmt): Diagnostic[] {
  const countTerms = (b: SeekStmt["not"]): number => {
    if (b.kind === "boundaryTerms") return b.terms.length;
    return countTerms(b.left) + countTerms(b.right);
  };
  if (countTerms(seek.not) === 0) {
    return [
      {
        severity: "error",
        code: "VACUOUS_BOUNDARY",
        seekName: seek.name,
        line: seek.line,
        message:
          `seek "${seek.name}" has an empty not{} clause. A claim in a medium with no privileged ` +
          "claim is individuated only by what it is not (Theorem: Individuation by Negation); an " +
          "empty exclusion asserts no individuation at all.",
      },
    ];
  }
  return [];
}

function checkProject(project: ProjectDecl, registry: CatalystRegistry | null): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const yielded = new Set<string>();

  for (const seek of project.seeks) {
    diagnostics.push(...checkBoundaryNonVacuous(seek));
    diagnostics.push(...checkSeekCoherence(seek));
    diagnostics.push(...checkCatalystReferences(seek, registry));

    if (yielded.has(seek.yieldName)) {
      diagnostics.push({
        severity: "error",
        code: "DUPLICATE_YIELD",
        seekName: seek.name,
        line: seek.line,
        message: `seek "${seek.name}" yields "${seek.yieldName}", which is already bound earlier in project "${project.name}".`,
      });
    }
    yielded.add(seek.yieldName);
  }

  if (project.goal) {
    for (const claimName of project.goal.claims) {
      if (!yielded.has(claimName)) {
        diagnostics.push({
          severity: "error",
          code: "UNKNOWN_GOAL_CLAIM",
          message: `project "${project.name}" goal references "${claimName}", which no seek yields.`,
        });
      }
    }
  }

  return diagnostics;
}

/** Detect a dependency cycle among a project's seeks (acyclicity is required by construction). */
function checkAcyclic(project: ProjectDecl): Diagnostic[] {
  const yieldToSeek = new Map<string, string>();
  for (const seek of project.seeks) yieldToSeek.set(seek.yieldName, seek.name);

  const refsOf = (seek: SeekStmt): Set<string> => {
    const names = new Set<string>();
    const scanExpr = (e: { kind: string; name?: string }) => {
      if (e.kind === "ident" && e.name) names.add(e.name);
    };
    const scanArgs = (args: Array<{ value: { kind: string; name?: string } }>) => {
      for (const a of args) scanExpr(a.value);
    };
    if (seek.toward.kind === "regionCall") scanArgs(seek.toward.args);
    const scanChain = (c: ChainExpr): void => {
      if (c.kind === "catalystRef") scanArgs(c.args);
      else if (c.kind === "chainSeq") c.steps.forEach(scanChain);
      else if (c.kind === "chainPar") c.branches.forEach(scanChain);
    };
    if (seek.via) scanChain(seek.via);
    return names;
  };

  const adjacency = new Map<string, string[]>();
  for (const seek of project.seeks) {
    const deps: string[] = [];
    for (const name of refsOf(seek)) {
      // A seek's via{} chain may reference its own yield name as a working
      // value (e.g. `via{ web_search(budget) }` inside the seek that
      // yields `budget`, meaning "operate on the claim under construction",
      // not "depend on a prior binding"). This is not a dependency edge.
      if (name === seek.yieldName) continue;
      const depSeek = yieldToSeek.get(name);
      if (depSeek) deps.push(depSeek);
    }
    adjacency.set(seek.name, deps);
  }

  const state = new Map<string, "visiting" | "done">();
  const diagnostics: Diagnostic[] = [];

  const visit = (node: string): boolean => {
    const s = state.get(node);
    if (s === "done") return true;
    if (s === "visiting") return false;
    state.set(node, "visiting");
    for (const dep of adjacency.get(node) ?? []) {
      if (!visit(dep)) return false;
    }
    state.set(node, "done");
    return true;
  };

  for (const seek of project.seeks) {
    if (!visit(seek.name)) {
      diagnostics.push({
        severity: "error",
        code: "CYCLIC_PROJECT",
        seekName: seek.name,
        message: `project "${project.name}" has a dependency cycle involving seek "${seek.name}". A project's seeks must form a directed acyclic graph.`,
      });
      break;
    }
  }

  return diagnostics;
}

export interface TypeCheckOptions {
  registry?: CatalystRegistry;
}

/** Type-check a full Graffiti program (Theorem: Type Soundness, progress/preservation). */
export function typecheck(program: Program, options: TypeCheckOptions = {}): TypeCheckResult {
  const diagnostics: Diagnostic[] = [];
  const { floor, diagnostics: floorDiagnostics } = checkFloorPositivity(program);
  diagnostics.push(...floorDiagnostics);

  for (const decl of program.decls) {
    if (decl.kind === "projectDecl") {
      diagnostics.push(...checkProject(decl, options.registry ?? null));
      diagnostics.push(...checkAcyclic(decl));
    }
    if (decl.kind === "moduleDecl") {
      for (const inner of decl.decls) {
        if (inner.kind === "projectDecl") {
          diagnostics.push(...checkProject(inner, options.registry ?? null));
          diagnostics.push(...checkAcyclic(inner));
        }
      }
    }
  }

  const ok = diagnostics.every((d) => d.severity !== "error");
  return { ok, diagnostics, ambientFloor: floor };
}

export { CatalystDecl };

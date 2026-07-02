/**
 * The project orchestrator: topological ordering of a project's seek DAG
 * and dispatch against the interpreter.
 *
 * Implements Definition (Project Dependency Graph) and Theorem
 * (Well-Founded Evaluation Order) of semantic-causal-propagation.tex,
 * Section "Heterogeneous Catalyst Orchestration".
 */

import type { ChainExpr, ProjectDecl, SeekStmt } from "../lang/ast";
import { Interpreter, GraffitiRuntimeError } from "../lang/interpreter";
import type { CatalystRegistry } from "./catalyst";
import type { GraffitiValue } from "../lang/values";

export class CyclicProjectError extends GraffitiRuntimeError {}

/** Names referenced by a seek's toward{} region and via{} chain arguments. */
function referencedNames(seek: SeekStmt): Set<string> {
  const names = new Set<string>();
  const scan = (e: { kind: string; name?: string }) => {
    if (e.kind === "ident" && e.name) names.add(e.name);
  };
  const scanArgs = (args: Array<{ value: { kind: string; name?: string } }>) => {
    for (const a of args) scan(a.value);
  };
  if (seek.toward.kind === "regionCall") scanArgs(seek.toward.args);
  const scanChain = (c: ChainExpr): void => {
    if (c.kind === "catalystRef") scanArgs(c.args);
    else if (c.kind === "chainSeq") c.steps.forEach(scanChain);
    else if (c.kind === "chainPar") c.branches.forEach(scanChain);
  };
  if (seek.via) scanChain(seek.via);
  return names;
}

/**
 * Definition (Project Dependency Graph): produce a topological order over
 * a project's seeks such that every seek is ordered after every seek it
 * references. Any two valid topological orders yield the same observable
 * results (Theorem: Well-Founded Evaluation Order), so this returns one
 * canonical order (stable sort by first appearance among ties).
 */
export function topologicalOrder(project: ProjectDecl): SeekStmt[] {
  const yieldToSeek = new Map<string, SeekStmt>();
  for (const seek of project.seeks) yieldToSeek.set(seek.yieldName, seek);

  const state = new Map<string, "visiting" | "done">();
  const order: SeekStmt[] = [];

  const visit = (seek: SeekStmt): void => {
    const s = state.get(seek.name);
    if (s === "done") return;
    if (s === "visiting") {
      throw new CyclicProjectError(
        `project "${project.name}" has a dependency cycle involving seek "${seek.name}".`,
      );
    }
    state.set(seek.name, "visiting");
    for (const name of referencedNames(seek)) {
      // A seek's own yield name referenced inside its own via{} chain names
      // the claim under construction, not a dependency on a prior binding.
      if (name === seek.yieldName) continue;
      const dep = yieldToSeek.get(name);
      if (dep) visit(dep);
    }
    state.set(seek.name, "done");
    order.push(seek);
  };

  for (const seek of project.seeks) visit(seek);
  return order;
}

export interface RunProjectOptions {
  registry: CatalystRegistry;
  ambientFloor: number;
}

export interface RunProjectResult {
  results: Map<string, GraffitiValue>;
  committedRecord: number;
}

/**
 * Execute a full project: order its seeks topologically, then run each
 * through the Interpreter in that order over one shared, growing contact
 * graph, so later seeks' toward{}/via{} clauses can reference earlier
 * yields.
 */
export async function runProject(project: ProjectDecl, options: RunProjectOptions): Promise<RunProjectResult> {
  const order = topologicalOrder(project);
  const interpreter = new Interpreter({ registry: options.registry, ambientFloor: options.ambientFloor });
  const results = new Map<string, GraffitiValue>();

  for (const seek of order) {
    const { value } = await interpreter.executeSeek(seek);
    results.set(seek.yieldName, value);
  }

  return { results, committedRecord: interpreter.record };
}

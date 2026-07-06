/**
 * GraffitiSession: a persistent interpreter across many runProject/runSource
 * calls, pruned between calls by @buhera/purpose (docs/sources/purpose-propagation.tex,
 * "Carry the Uncertainty, Not the Knowledge").
 *
 * Without this wrapper, orchestrator.ts's runProject builds a fresh
 * Interpreter (and therefore a fresh, empty ContactGraph) on every call --
 * correct for one project, but useless for a long-running agent session
 * that accumulates claims across many seeks over time: nothing is ever
 * pruned, so the graph grows without bound (exactly the problem
 * purpose-propagation.tex's tandem carry exists to solve).
 *
 * A GraffitiSession keeps one Interpreter alive across calls. After each
 * run, every claim the interpreter has committed is handed to
 * @buhera/purpose as a Step (id = claim name, terms = ClaimTerms recorded
 * by the interpreter, cost = the claim's serialized length, timestamp =
 * a logical clock). A goal-scoped carry() then reports which claims are
 * still necessary and fit the budget; dropped claims are evicted from the
 * live Interpreter's graph via Purpose's own Step-level accounting, not by
 * Graffiti re-deriving reachability itself -- Graffiti never decides
 * necessity, per INTERFACE.md's division of responsibility.
 */

import type { Goal, Step, StepId } from "@buhera/purpose";
import { Session as PurposeSession, type CarryResult, type SessionConfig } from "@buhera/purpose";

import { Interpreter, type InterpreterOptions } from "../lang/interpreter";
import { topologicalOrder } from "./orchestrator";
import type { ProjectDecl } from "../lang/ast";
import type { GraffitiValue } from "../lang/values";

export interface GraffitiSessionOptions extends InterpreterOptions {
  purpose?: SessionConfig;
}

export interface RunInSessionResult {
  results: Map<string, GraffitiValue>;
  committedRecord: number;
}

/** Default cost function: a claim's cost is its serialized string length. */
function defaultCostOf(claim: string): number {
  return Math.max(1, claim.length);
}

/**
 * A session-scoped wrapper around Interpreter, adding @buhera/purpose
 * accounting. Every runProject call executes against the same underlying
 * Interpreter (so later projects' toward{}/via{} clauses may reference
 * earlier yields exactly as within a single project), and carry() prunes
 * the retained claim set against a goal and a token budget between calls.
 */
export class GraffitiSession {
  private readonly interpreter: Interpreter;
  private readonly purpose: PurposeSession;
  private clock = 0;
  private costOf: (claim: string) => number = defaultCostOf;

  constructor(options: GraffitiSessionOptions) {
    this.interpreter = new Interpreter(options);
    this.purpose = new PurposeSession(options.purpose);
  }

  /** Override how a claim's carry cost is computed (default: string length). */
  setCostFunction(costOf: (claim: string) => number): void {
    this.costOf = costOf;
  }

  /** Execute one project against the session's persistent interpreter, then sync claims into Purpose. */
  async runProject(project: ProjectDecl): Promise<RunInSessionResult> {
    const order = topologicalOrder(project);
    const results = new Map<string, GraffitiValue>();
    const startRecord = this.interpreter.record;

    for (const seek of order) {
      const { value } = await this.interpreter.executeSeek(seek);
      results.set(seek.yieldName, value);
    }

    this.syncClaimsIntoPurpose();
    return { results, committedRecord: this.interpreter.record - startRecord };
  }

  /**
   * Hand every claim the interpreter has committed so far to
   * @buhera/purpose as a Step. Idempotent per (id, terms, cost) --
   * Purpose's Session.addStep no-ops on an unchanged re-registration, so
   * calling this after every runProject is safe and cheap.
   */
  private syncClaimsIntoPurpose(): void {
    for (const { claim, terms } of this.interpreter.claimTerms()) {
      const step: Step = {
        id: claim,
        terms,
        cost: this.costOf(claim),
        timestamp: this.clock++,
      };
      this.purpose.addStep(step);
    }
  }

  /**
   * Run the tandem carry (seek -> necessary -> knapsack) for a goal under
   * a token budget. Returns Purpose's verdict on which claims are still
   * load-bearing and fit the budget; this session does not act on it
   * automatically -- call evict() with the result's `dropped` list when
   * ready to actually prune the live interpreter's graph.
   */
  carry(goalTerms: Iterable<string>, budget: number): CarryResult {
    const goal: Goal = { terms: new Set(goalTerms) };
    return this.purpose.carry({ goal, budget });
  }

  /**
   * Evict a set of claims (by id) from Purpose's own step accounting.
   * This does not remove committed edges from the live Interpreter's
   * ContactGraph -- Graffiti's calculus never retracts a committed step
   * (Theorem: Monotonicity of the Committed Record) -- so eviction here
   * only affects what future carry() calls consider part of the
   * session's retained history, not the interpreter's evaluation
   * semantics for the current project run.
   */
  evict(claimIds: Iterable<StepId>): void {
    for (const id of claimIds) this.purpose.removeStep(id);
  }

  /** Number of claims currently tracked by Purpose's session accounting. */
  stepCount(): number {
    return this.purpose.stepCount();
  }

  /** The interpreter's current ambient floor, as seen by @buhera/purpose. */
  purposeFloor(): number {
    return this.purpose.floor();
  }

  /** Escape hatch to the live interpreter, e.g. for contactGraph() inspection. */
  underlyingInterpreter(): Interpreter {
    return this.interpreter;
  }
}

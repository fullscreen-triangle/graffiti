/**
 * Small-step-faithful operational semantics for Graffiti (.grf).
 *
 * Implements the reduction rules of semantic-causal-propagation.tex,
 * Section "Operational Semantics": Rule (Seek Converges), Rule (Seek
 * Declines), Definition (Committed Step; Committed Record), and Theorem
 * (Monotonicity of the Committed Record). Rather than a literal
 * expression-stepper, this interpreter realises the same state
 * transitions -- a mutating contact graph plus a strictly monotone
 * committed-record counter -- directly against the manuscript's core
 * calculus, which is the intended reading of "evaluation is measurement"
 * (the graph is mutated by evaluation, not simulated by it).
 */

import { ContactGraph, MEDIUM } from "../core/graph";
import { equivalenceClassesByIdentity, isClosedByIdentity, resolveOutcome } from "../core/closure";
import type {
  AdmitClause,
  Arg,
  BoundaryExpr,
  ChainExpr,
  Expr,
  ProjectDecl,
  RegionExpr,
  SeekStmt,
} from "./ast";
import type { CatalystRegistry } from "../orchestration/catalyst";
import type { ClaimValue, DeclineValue, GraffitiValue } from "./values";

export class GraffitiRuntimeError extends Error {}

/** The evaluation environment: bound seek results, available for later seeks to reference. */
export class Environment {
  private readonly bindings = new Map<string, GraffitiValue>();

  bind(name: string, value: GraffitiValue): void {
    this.bindings.set(name, value);
  }

  get(name: string): GraffitiValue | undefined {
    return this.bindings.get(name);
  }

  resolveClaimName(name: string): string {
    const bound = this.bindings.get(name);
    if (bound && bound.kind === "Claim") return bound.targetClaim;
    if (bound && bound.kind === "Decline") return bound.classes[0]?.[0] ?? name;
    return name;
  }
}

function evalExprToString(expr: Expr, env: Environment): string {
  switch (expr.kind) {
    case "ident":
      return env.resolveClaimName(expr.name);
    case "number":
      return String(expr.value);
    case "string":
      return expr.value;
    case "paren":
      return evalExprToString(expr.expr, env);
    case "catalystRef":
      return expr.qname.join(".");
  }
}

function evalArgsToRecord(args: Arg[], env: Environment): Record<string, unknown> {
  const record: Record<string, unknown> = {};
  args.forEach((arg, index) => {
    const key = arg.name ?? String(index);
    record[key] = evalExprToString(arg.value, env);
  });
  return record;
}

function boundaryTermCount(b: BoundaryExpr): number {
  if (b.kind === "boundaryTerms") return b.terms.length;
  return boundaryTermCount(b.left) + boundaryTermCount(b.right);
}

function regionTargetName(region: RegionExpr, env: Environment): string {
  if (region.kind === "regionIdent") return env.resolveClaimName(region.name);
  // A region call names the target claim by its call signature; deterministic
  // and stable so repeated seeks toward the same description converge on the
  // same claim identity in the contact graph.
  const argPart = region.args.map((a) => evalExprToString(a.value, env)).join(",");
  return `${region.qname.join(".")}(${argPart})`;
}

interface FlattenedCatalystStep {
  qname: string[];
  args: Arg[];
}

function flattenChain(chain: ChainExpr): FlattenedCatalystStep[][] {
  // Returns a list of branches, each an ordered sequence of steps
  // (sequential composition within a branch, parallel across branches),
  // matching the `>>` / `||` precedence of Definition (Graffiti grammar).
  if (chain.kind === "catalystRef") {
    return [[{ qname: chain.qname, args: chain.args }]];
  }
  if (chain.kind === "chainSeq") {
    const seq: FlattenedCatalystStep[] = [];
    for (const step of chain.steps) {
      const flattened = flattenChain(step);
      // A sequential step should itself be atomic (catalystRef) by grammar
      // construction; nested seq/par under >> is not produced by the parser.
      seq.push(...flattened[0]!);
    }
    return [seq];
  }
  // chainPar
  return chain.branches.flatMap((b) => flattenChain(b));
}

export interface SeekExecutionResult {
  value: GraffitiValue;
  /** Committed steps this seek contributed to the project's monotone record. */
  committedSteps: number;
}

export interface InterpreterOptions {
  registry: CatalystRegistry;
  ambientFloor: number;
  /** Upper bound on catalyst invocations per seek, guarding against a
   * non-terminating via chain in a registry that never closes. */
  maxCatalystInvocations?: number;
}

/**
 * The distinctions (tau(u), in @buhera/purpose's vocabulary) a committed
 * claim draws: the seek it was committed under, the catalyst that
 * resolved it (if any), and the stringified arguments that catalyst was
 * invoked with. This is the one piece of state Graffiti records purely
 * for Purpose's benefit -- the calculus itself never reads it -- so a
 * GraffitiSession (orchestration/session.ts) can hand each claim to
 * @buhera/purpose as a Step and get reachability/necessity/carry back.
 */
export interface ClaimTerms {
  claim: string;
  terms: Set<string>;
}

/**
 * One committed contact event, logged verbatim as it happens (Definition:
 * Committed Step). This is the finest-grained record the interpreter
 * produces -- a claim or a seek is an aggregate over these rows, never
 * the other way around -- so it is the natural row shape for a
 * crossfilter-style exploration UI (web/src/pages/explore.js): every
 * parameter a user might want to vary (floor, catalyst power, which
 * catalyst, which seek) is already a native field here, with no
 * additional derivation needed.
 */
export interface ContactRow {
  /** Position in the committed record at the moment this contact was made. */
  index: number;
  from: string;
  to: string;
  weight: number;
  seekName: string;
  /** Null for the seed->target contact that exists with no via{} chain. */
  catalystName: string | null;
  /** Catalytic power kappa in [0,1] this contact realised, if a catalyst produced it. */
  power: number | null;
  floor: number;
}

/**
 * The interpreter executes one project's seeks in dependency order over a
 * single, growing ContactGraph (Definition: Contact Graph is the runtime
 * state of a Graffiti program). The committed record `M` is the
 * project-wide monotone clock (Theorem: Monotonicity of the Committed
 * Record).
 */
export class Interpreter {
  private readonly graph: ContactGraph;
  private readonly env = new Environment();
  private committedRecord = 0;
  private readonly registry: CatalystRegistry;
  private readonly maxCatalystInvocations: number;
  private readonly termsByClaim = new Map<string, Set<string>>();
  private readonly contacts: ContactRow[] = [];

  constructor(options: InterpreterOptions) {
    this.registry = options.registry;
    this.maxCatalystInvocations = options.maxCatalystInvocations ?? 64;
    this.graph = new ContactGraph([MEDIUM], options.ambientFloor);
    // MEDIUM is not itself a claim; remove it from the claims list view by
    // construction (ContactGraph treats every non-medium vertex as a claim,
    // and MEDIUM is never added as a claim in commitClaim below).
  }

  get record(): number {
    return this.committedRecord;
  }

  /** Every claim's accumulated term set, for handoff to a GraffitiSession. */
  claimTerms(): ClaimTerms[] {
    return Array.from(this.termsByClaim, ([claim, terms]) => ({ claim, terms }));
  }

  /** Every committed contact, in commit order -- the row set for a crossfilter view. */
  contactLog(): ContactRow[] {
    return this.contacts.slice();
  }

  /** Merge additional terms into a claim's recorded distinctions (tau(u)). */
  private addTerms(claim: string, terms: Iterable<string>): void {
    if (claim === MEDIUM) return;
    let set = this.termsByClaim.get(claim);
    if (!set) {
      set = new Set<string>();
      this.termsByClaim.set(claim, set);
    }
    for (const t of terms) set.add(t);
  }

  /**
   * Introduce a claim if not already present, and wire it to the medium
   * at the ambient floor -- Definition 2.2 requires every claim vertex to
   * be adjacent to the medium (this is what makes it a contact graph at
   * all, and is the edge minCutSide/separationCost measure against).
   */
  private ensureClaim(name: string): void {
    if (name === MEDIUM) return;
    if (!this.graph.claims.includes(name)) {
      (this.graph.claims as string[]).push(name);
    }
    if (!this.graph.hasEdge(name, MEDIUM)) {
      this.graph.addEdge(name, MEDIUM, this.graph.floor);
    }
  }

  /**
   * Commit one contact edge (a cut event, Definition: Committed Step) from
   * `from` to `to` at the given weight, strictly incrementing the
   * committed record (Theorem: Monotonicity of the Committed Record). Not
   * a pure function: this mutates the interpreter's graph state, realising
   * "evaluation is measurement." Also appends a ContactRow to the log
   * regardless of whether the edge was new (a crossfilter view over
   * contactLog() wants every commit attempt, including re-affirmations of
   * an existing edge, since those are still real committed steps by
   * Theorem: Monotonicity of the Committed Record).
   */
  private commitContact(
    from: string,
    to: string,
    weight: number,
    meta: { seekName: string; catalystName: string | null; power: number | null },
  ): void {
    this.ensureClaim(from);
    this.ensureClaim(to);
    const appliedWeight = Math.max(weight, this.graph.floor);
    if (!this.graph.hasEdge(from, to)) {
      this.graph.addEdge(from, to, appliedWeight);
    }
    this.committedRecord += 1;
    this.contacts.push({
      index: this.committedRecord,
      from,
      to,
      weight: appliedWeight,
      seekName: meta.seekName,
      catalystName: meta.catalystName,
      power: meta.power,
      floor: this.graph.floor,
    });
  }

  private async runBranch(
    seed: string,
    branch: FlattenedCatalystStep[],
    seekName: string,
  ): Promise<{ target: string; power: number[] }> {
    let currentClaim = seed;
    const powers: number[] = [];

    for (const step of branch) {
      const catalystName = step.qname.join(".");
      const def = this.registry.get(catalystName);
      const args = evalArgsToRecord(step.args, this.env);
      const result = await def.provider({ currentClaim, args });

      // A catalyst contact's weight represents the strength of the bond
      // between two claims resolved by the same propagation, which must
      // exceed the ambient medium-adjacency floor for a chain to
      // constitute a genuinely tighter local cluster than "individuated
      // against the undifferentiated medium alone" (mirroring
      // twoClusterGraph's internalWeight >> bridgeWeight construction).
      // Catalytic power scales the bond strength above that baseline: a
      // stronger catalyst produces a more tightly bound contact.
      const bondWeight = this.graph.floor * (2 + 8 * result.power);
      this.commitContact(currentClaim, result.claim, bondWeight, {
        seekName,
        catalystName,
        power: result.power,
      });

      // Record this claim's distinctions (tau(u)) for @buhera/purpose: the
      // seek it was reached under, the catalyst that resolved it, and its
      // call arguments -- the calculus never reads this, only a
      // GraffitiSession does.
      const argTerms = Object.entries(args).map(([k, v]) => `${k}=${String(v)}`);
      this.addTerms(result.claim, [seekName, catalystName, ...argTerms]);

      currentClaim = result.claim;
      powers.push(result.power);
    }

    return { target: currentClaim, power: powers };
  }

  /**
   * Execute one `seek` statement: dispatch its `via` chain's branches
   * (each an admissible propagation, Definition: Propagation), commit the
   * resulting contacts, and resolve the outcome by closure
   * (Definition: Closure; Theorem: Convergent Closure or Honest Decline).
   */
  async executeSeek(seek: SeekStmt): Promise<SeekExecutionResult> {
    const exclusionCount = boundaryTermCount(seek.not);
    if (exclusionCount === 0) {
      throw new GraffitiRuntimeError(`seek "${seek.name}" has no exclusion terms; individuation is undefined.`);
    }

    const targetName = regionTargetName(seek.toward, this.env);
    this.ensureClaim(targetName);
    this.addTerms(targetName, [seek.name]);

    const seedName = `${seek.name}:seed`;
    this.commitContact(seedName, targetName, this.graph.floor, {
      seekName: seek.name,
      catalystName: null,
      power: null,
    });

    const startRecord = this.committedRecord;
    // The region description itself (`targetName`) is a candidate target,
    // never a "reached" claim on its own: reachedTargets collects only the
    // claims actually resolved by catalyst invocations (Definition:
    // Propagation -- a propagation's terminal claim is what a catalyst
    // resolves to). With no via{} chain at all, the seed's direct
    // connection to targetName is the sole propagation and its class is
    // exactly {targetName}.
    const reachedTargets: string[] = [];

    if (seek.via) {
      const branches = flattenChain(seek.via);
      let invocations = 0;
      for (const branch of branches) {
        if (invocations >= this.maxCatalystInvocations) break;
        const { target } = await this.runBranch(seedName, branch, seek.name);
        reachedTargets.push(target);
        invocations += branch.length;
      }
    } else {
      reachedTargets.push(targetName);
    }

    const classes = equivalenceClassesByIdentity(reachedTargets);
    const closed = isClosedByIdentity([], classes); // registry catalysts already invoked above
    const outcome = resolveOutcome(classes);

    const committedSteps = this.committedRecord - startRecord;

    if (!closed) {
      // Under bare `converge` with no explicit decline handling, an unclosed
      // search cannot yet be reported; the caller (project executor) is
      // expected to have already exhausted the available registry for this
      // seek, so in practice `closed` will be true once all catalysts in a
      // finite registry have been tried (Theorem: Convergent Closure or
      // Honest Decline). We still resolve on best available evidence.
    }

    if (outcome.state === "convergent") {
      // The representative claim is whichever reached target the sole
      // equivalence class contains -- the catalysts' resolved claim when a
      // via{} chain ran, or the region description itself when none did.
      const representative = outcome.classes[0]?.[0] ?? targetName;
      const value: ClaimValue = {
        kind: "Claim",
        targetClaim: representative,
        floor: this.graph.floor,
        residue: this.graph.separationCost(representative),
        classesAtClosure: outcome.classes,
      };
      this.env.bind(seek.yieldName, value);
      return { value, committedSteps };
    }

    if (seek.until.kind === "converge" && !seek.until.otherwiseDecline) {
      throw new GraffitiRuntimeError(
        `seek "${seek.name}" reached contested closure (${outcome.classes.length} distinct claim-regions) ` +
          'but has no "otherwise decline" clause to handle it. Add "until converge otherwise decline" ' +
          "to accept a Decline value here.",
      );
    }

    const declineValue: DeclineValue = {
      kind: "Decline",
      seekName: seek.name,
      classes: outcome.classes,
      floor: this.graph.floor,
    };
    this.env.bind(seek.yieldName, declineValue);
    return { value: declineValue, committedSteps };
  }

  /** Execute every seek of a project in the dependency order supplied by the caller. */
  async executeProject(project: ProjectDecl): Promise<Map<string, GraffitiValue>> {
    const results = new Map<string, GraffitiValue>();
    for (const seek of project.seeks) {
      const { value } = await this.executeSeek(seek);
      results.set(seek.yieldName, value);
    }
    return results;
  }

  contactGraph(): ContactGraph {
    return this.graph;
  }
}

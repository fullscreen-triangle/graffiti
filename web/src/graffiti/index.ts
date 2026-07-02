/**
 * Graffiti: an individuation-theoretic search calculus and the .grf
 * orchestration language, implemented per
 * docs/publications/semantic-causal-propagation/semantic-causal-propagation.tex.
 *
 * This is the public entry point. A Buhera OS integration typically only
 * needs:
 *
 *   import { CatalystRegistry, parseProgram, typecheck, runProject } from "./graffiti";
 *
 * See src/graffiti/examples for worked scripts and registrations.
 */

// --- Core calculus -----------------------------------------------------
export { FlowNetwork, type NodeId } from "./core/maxflow";
export {
  ContactGraph,
  MEDIUM,
  Rng,
  randomContactGraph,
  twoClusterGraph,
  type ContactGraphOptions,
  type RandomContactGraphOptions,
} from "./core/graph";
export { complement, doubleComplementIsIdentity, Decoder } from "./core/individuation";
export {
  sampleRepresentation,
  representationMean,
  isOnShell,
  committedRecordAfterSwitch,
} from "./core/representation";
export {
  Propagation,
  isConvergent,
  endpointInvariants,
  randomInteriorVariant,
  type EndpointInvariants,
} from "./core/propagation";
export {
  compositePower,
  repeatedPower,
  residualAfterChain,
  saturates,
  logResidual,
  SupportGraph,
  supportEdge,
  signOnlyCoherenceVerdict,
  magnitudeCoherenceVerdict,
  type CatalystId,
} from "./core/catalysis";
export {
  equivalenceClasses,
  equivalenceClassesByIdentity,
  confidenceThresholdMet,
  isClosed,
  isClosedByIdentity,
  resolveOutcome,
  type SearchOutcome,
  type SearchOutcomeState,
} from "./core/closure";
export {
  createLiveSeek,
  descentRate,
  currentResidue,
  priority,
  selectNext,
  type LiveSeek,
} from "./core/scheduler";

// --- Catalyst orchestration ---------------------------------------------
export {
  CatalystRegistry,
  type CatalystContext,
  type CatalystResult,
  type CatalystProvider,
  type CatalystDefinition,
  type CatalystNamespace,
} from "./orchestration/catalyst";
export {
  createFixtureSearchCatalyst,
  createMockInferenceCatalyst,
  createInertCatalyst,
} from "./orchestration/mock-catalysts";
export {
  topologicalOrder,
  runProject,
  CyclicProjectError,
  type RunProjectOptions,
  type RunProjectResult,
} from "./orchestration/orchestrator";

// --- The .grf language ---------------------------------------------------
export { tokenize, LexError, type Token, type TokenType } from "./lang/lexer";
export { parseProgram, ParseError } from "./lang/parser";
export * from "./lang/ast";
export {
  typecheck,
  checkFloorPositivity,
  type TypeCheckResult,
  type TypeCheckOptions,
  type Diagnostic,
  type Severity,
} from "./lang/typechecker";
export { Interpreter, Environment, GraffitiRuntimeError, type InterpreterOptions, type SeekExecutionResult } from "./lang/interpreter";
export { isClaim, isDecline, graffitiTypeOf, type ClaimValue, type DeclineValue, type GraffitiValue } from "./lang/values";

// --- Convenience: compile + run a source string in one call --------------
import { parseProgram } from "./lang/parser";
import { typecheck, type Diagnostic } from "./lang/typechecker";
import { runProject } from "./orchestration/orchestrator";
import type { CatalystRegistry } from "./orchestration/catalyst";
import type { GraffitiValue } from "./lang/values";
import type { ProjectDecl } from "./lang/ast";

export interface CompileResult {
  ok: boolean;
  diagnostics: Diagnostic[];
  ambientFloor: number;
  projects: ProjectDecl[];
}

/** Parse and type-check a .grf source string without executing it. */
export function compile(source: string, registry?: CatalystRegistry): CompileResult {
  const program = parseProgram(source);
  const check = typecheck(program, registry ? { registry } : {});
  const projects = program.decls.filter((d): d is ProjectDecl => d.kind === "projectDecl");
  return { ok: check.ok, diagnostics: check.diagnostics, ambientFloor: check.ambientFloor, projects };
}

export interface RunSourceResult {
  compile: CompileResult;
  projectResults: Map<string, Map<string, GraffitiValue>>;
}

/**
 * Parse, type-check, and execute every project declared in a .grf source
 * string against the given catalyst registry. Throws if the source fails
 * to parse or fails type-checking with any error-severity diagnostic.
 */
export async function runSource(source: string, registry: CatalystRegistry): Promise<RunSourceResult> {
  const result = compile(source, registry);
  if (!result.ok) {
    const messages = result.diagnostics
      .filter((d) => d.severity === "error")
      .map((d) => d.message)
      .join("\n");
    throw new Error(`Graffiti type-check failed:\n${messages}`);
  }

  const projectResults = new Map<string, Map<string, GraffitiValue>>();
  for (const project of result.projects) {
    const { results } = await runProject(project, { registry, ambientFloor: result.ambientFloor });
    projectResults.set(project.name, results);
  }

  return { compile: result, projectResults };
}

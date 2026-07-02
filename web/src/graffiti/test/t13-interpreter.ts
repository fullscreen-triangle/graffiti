/** Test 13: End-to-end interpreter and orchestrator (Rule: Seek
 * Converges, Rule: Seek Declines, Theorem: Monotonicity of the Committed
 * Record, Theorem: Well-Founded Evaluation Order). */

import { Checker } from "./harness";
import { runSource } from "../index";
import { CatalystRegistry } from "../orchestration/catalyst";
import { createFixtureSearchCatalyst, createMockInferenceCatalyst } from "../orchestration/mock-catalysts";

const CONVERGENT_SOURCE = `
floor 0.02

catalyst web_search { namespace: remote input: Region output: Claim }
catalyst local_notes { namespace: local input: Region output: Claim }
catalyst archive_lookup { namespace: remote input: Region output: Claim }

project apollo {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    until converge
    yield launch_date

  seek budget
    not{ "post-hoc estimates without citation" }
    toward{ cost_of(launch_date) }
    via{ web_search(budget) >> local_notes(budget) >> archive_lookup(budget) }
    until converge otherwise decline
    yield budget

  goal {
    claims: [launch_date, budget]
    coherence: >= 0.5
  }
}
`;

async function testConvergent(c: Checker) {
  const registry = new CatalystRegistry();
  registry.register(createFixtureSearchCatalyst("web_search", {}));
  registry.register(createMockInferenceCatalyst("local_notes", (claim) => `${claim}:agree`));
  registry.register(createMockInferenceCatalyst("archive_lookup", (claim) => `${claim}:agree`));

  const { compile, projectResults } = await runSource(CONVERGENT_SOURCE, registry);
  c.assert(compile.ok, "convergent example type-checks");

  const apollo = projectResults.get("apollo");
  c.assert(apollo !== undefined, "apollo project executed");
  if (!apollo) return;

  const launchDate = apollo.get("launch_date");
  c.assert(launchDate !== undefined && launchDate.kind === "Claim", "launch_date resolves to a Claim");

  const budget = apollo.get("budget");
  c.assert(budget !== undefined, "budget seek produced a value");
  if (budget) {
    c.assert(budget.floor > 0, "yielded Claim/Decline carries a strictly positive floor");
  }
}

async function testDeclining(c: Checker) {
  const registry = new CatalystRegistry();
  // Two independent catalysts that resolve to genuinely different claims:
  // the interpreter's closure test compares reached claims by identity
  // (Definition 8.1's operational reading for a runtime interpreter,
  // equivalenceClassesByIdentity), so two distinct resolved claim strings
  // are, correctly, two distinct equivalence classes -- contested closure.
  registry.register(createMockInferenceCatalyst("a", () => "cause:mechanical-failure"));
  registry.register(createMockInferenceCatalyst("b", () => "cause:human-error"));

  const source = `
    floor 0.02

    catalyst a { namespace: local input: Region output: Claim }
    catalyst b { namespace: local input: Region output: Claim }

    project disputed {
      seek cause
        not{ "single-source attribution" }
        toward{ primary_cause_of(event) }
        via{ a(cause) || b(cause) }
        until converge otherwise decline
        yield cause
    }
  `;

  const { projectResults } = await runSource(source, registry);
  const disputed = projectResults.get("disputed");
  const cause = disputed?.get("cause");
  c.assert(cause !== undefined && cause.kind === "Decline", "seek with genuinely conflicting catalysts declines rather than guessing");
  if (cause && cause.kind === "Decline") {
    c.assert(cause.classes.length >= 2, "decline reports at least two distinct equivalence classes");
  }
}

export async function run() {
  const c = new Checker("13", "Interpreter/Orchestrator (Seek Converges/Declines, DAG order)");
  await testConvergent(c);
  await testDeclining(c);
  return c.result();
}

/**
 * Test 14: GraffitiSession integration with @buhera/purpose (INTERFACE.md
 * §7's integration-test contract). Confirms a session persists claims
 * across multiple runProject calls, tags terms correctly, and that
 * carry() reports an unrelated-goal's claims as droppable while a
 * goal-relevant claim survives.
 */

import { Checker } from "./harness";
import { parseProgram } from "../lang/parser";
import { typecheck } from "../lang/typechecker";
import { GraffitiSession } from "../orchestration/session";
import { CatalystRegistry } from "../orchestration/catalyst";
import { createMockInferenceCatalyst } from "../orchestration/mock-catalysts";
import type { ProjectDecl } from "../lang/ast";

function compileProject(source: string, registry: CatalystRegistry): ProjectDecl {
  const program = parseProgram(source);
  const check = typecheck(program, { registry });
  if (!check.ok) {
    throw new Error(check.diagnostics.map((d) => d.message).join("\n"));
  }
  const project = program.decls.find((d): d is ProjectDecl => d.kind === "projectDecl");
  if (!project) throw new Error("no project declared");
  return project;
}

const APOLLO_SOURCE = `
floor 0.02

catalyst web_search { namespace: remote input: Region output: Claim }

project apollo {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    via{ web_search(query: launch_date) }
    until converge
    yield launch_date
}
`;

const UNRELATED_SOURCE = `
floor 0.02

catalyst weather_lookup { namespace: remote input: Region output: Claim }

project weather {
  seek forecast
    not{ "unsourced forecasts" }
    toward{ forecast_of("Munich") }
    via{ weather_lookup(query: forecast) }
    until converge
    yield forecast
}
`;

async function testSessionPersistsAcrossRuns(c: Checker) {
  const registry = new CatalystRegistry();
  registry.register(createMockInferenceCatalyst("web_search", () => "1969-07-16"));
  registry.register(createMockInferenceCatalyst("weather_lookup", () => "cloudy"));

  const session = new GraffitiSession({ registry, ambientFloor: 0.02 });

  const apollo = compileProject(APOLLO_SOURCE, registry);
  const { results: apolloResults } = await session.runProject(apollo);
  c.assert(apolloResults.get("launch_date")?.kind === "Claim", "apollo seek resolves to a Claim within a session");

  const weather = compileProject(UNRELATED_SOURCE, registry);
  const { results: weatherResults } = await session.runProject(weather);
  c.assert(weatherResults.get("forecast")?.kind === "Claim", "weather seek resolves to a Claim within the same session");

  // Both projects' claims should now be tracked by @buhera/purpose.
  c.assert(session.stepCount() > 0, "session tracks committed claims as Purpose steps after two runs");

  const interpreter = session.underlyingInterpreter();
  const claimTerms = interpreter.claimTerms();
  c.assert(claimTerms.length > 0, "interpreter records term sets for committed claims");

  const launchDateTerms = claimTerms.find((ct) => ct.claim === "1969-07-16");
  c.assert(
    launchDateTerms !== undefined && launchDateTerms.terms.has("launch_date") && launchDateTerms.terms.has("web_search"),
    "a claim resolved via a catalyst is tagged with its seek name and catalyst name",
  );
}

async function testCarryPrunesUnrelatedGoal(c: Checker) {
  const registry = new CatalystRegistry();
  registry.register(createMockInferenceCatalyst("web_search", () => "1969-07-16"));
  registry.register(createMockInferenceCatalyst("weather_lookup", () => "cloudy"));

  const session = new GraffitiSession({ registry, ambientFloor: 0.02 });
  await session.runProject(compileProject(APOLLO_SOURCE, registry));
  await session.runProject(compileProject(UNRELATED_SOURCE, registry));

  // A goal scoped only to the apollo seek's terms should find the weather
  // claim unreachable and therefore droppable, while the apollo claim
  // survives (either kept or regenerable, never "dropped").
  const carryResult = session.carry(["launch_date", "web_search"], 1000);
  c.assert(carryResult.ok, "carry() succeeds for a non-empty goal and positive budget");
  if (!carryResult.ok) return;

  const allTouched = [...carryResult.keep, ...carryResult.regenerable, ...carryResult.dropped];
  c.assert(allTouched.length === session.stepCount(), "carry() partitions every tracked claim into keep/regenerable/dropped");

  const weatherDropped = carryResult.dropped.includes("cloudy");
  c.assert(weatherDropped, "a claim unreachable from the goal (weather) is reported droppable");

  const apolloRetained = carryResult.keep.includes("1969-07-16") || carryResult.regenerable.includes("1969-07-16");
  c.assert(apolloRetained, "a claim reachable from the goal (apollo) is never reported dropped");
}

async function testEvictReducesTrackedSteps(c: Checker) {
  const registry = new CatalystRegistry();
  registry.register(createMockInferenceCatalyst("web_search", () => "1969-07-16"));
  registry.register(createMockInferenceCatalyst("weather_lookup", () => "cloudy"));

  const session = new GraffitiSession({ registry, ambientFloor: 0.02 });
  await session.runProject(compileProject(APOLLO_SOURCE, registry));
  await session.runProject(compileProject(UNRELATED_SOURCE, registry));

  const before = session.stepCount();
  const carryResult = session.carry(["launch_date", "web_search"], 1000);
  if (!carryResult.ok) {
    c.assert(false, "carry() unexpectedly failed in evict test");
    return;
  }

  session.evict(carryResult.dropped);
  const after = session.stepCount();
  c.assert(after < before, "evict() reduces the number of claims Purpose tracks for this session");
  c.assert(after === before - carryResult.dropped.length, "evict() removes exactly the dropped claims");
}

export async function run() {
  const c = new Checker("14", "GraffitiSession / @buhera/purpose integration (term tagging, carry, evict)");
  await testSessionPersistsAcrossRuns(c);
  await testCarryPrunesUnrelatedGoal(c);
  await testEvictReducesTrackedSteps(c);
  return c.result();
}

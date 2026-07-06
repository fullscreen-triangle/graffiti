// Runs a fixed demonstration .grf program against live, user-adjustable
// parameters (floor, per-catalyst power) and returns the interpreter's
// full ContactRow log -- the crossfilter dataset. This is the "writable"
// half of the explore page: moving a slider re-executes Graffiti for
// real, it does not just re-filter a frozen dataset.

import * as graffiti from "@/graffiti";

export const EXPLORE_SOURCE = `
floor __FLOOR__

catalyst web_search     { namespace: remote input: Region output: Claim }
catalyst local_notes    { namespace: local  input: Region output: Claim }
catalyst archive_lookup { namespace: remote input: Region output: Claim }

project apollo_budget {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    until converge
    yield launch_date

  seek budget_overrun
    not{ "post-hoc estimates without primary-source citation" }
    toward{ cost_vs_authorization(launch_date) }
    via{ web_search(query: budget_overrun)
           >> local_notes(query: budget_overrun)
           >> archive_lookup(query: budget_overrun) }
    until converge otherwise decline
    yield budget_overrun

  goal {
    claims: [launch_date, budget_overrun]
    coherence: >= 0.5
  }
}
`;

export const DEFAULT_PARAMS = {
  floor: 0.02,
  webSearchPower: 0.6,
  localNotesPower: 0.5,
  archiveLookupPower: 0.5,
};

/** Run the demo program with the given parameters; returns { rows, outcome }. */
export async function runExploreProgram(params) {
  const source = EXPLORE_SOURCE.replace("__FLOOR__", String(params.floor));

  const registry = new graffiti.CatalystRegistry();
  registry.register(
    graffiti.createFixtureSearchCatalyst(
      "web_search",
      { budget_overrun: "budget:on-authorization" },
      params.webSearchPower,
    ),
  );
  registry.register(
    graffiti.createMockInferenceCatalyst(
      "local_notes",
      (c) => `${c}:corroborated`,
      params.localNotesPower,
    ),
  );
  registry.register(
    graffiti.createMockInferenceCatalyst(
      "archive_lookup",
      (c) => `${c}:corroborated`,
      params.archiveLookupPower,
    ),
  );

  // Build+run through a fresh Interpreter directly (rather than
  // runSource, which discards the Interpreter after the call) so the
  // full ContactRow log can be read back for the crossfilter view.
  const program = graffiti.parseProgram(source);
  const check = graffiti.typecheck(program, { registry });
  if (!check.ok) {
    const messages = check.diagnostics.filter((d) => d.severity === "error").map((d) => d.message);
    throw new Error(messages.join("\n") || "type-check failed");
  }

  const project = program.decls.find((d) => d.kind === "projectDecl");
  const interpreter = new graffiti.Interpreter({ registry, ambientFloor: check.ambientFloor });
  const results = await interpreter.executeProject(project);

  return {
    rows: interpreter.contactLog(),
    results,
  };
}

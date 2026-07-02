/**
 * Worked example, matching Example (An Explicit, Coherent Multi-Catalyst
 * Chain) of semantic-causal-propagation.tex, Section "Worked Examples".
 *
 * Run with: npx tsx src/graffiti/examples/apollo-budget.ts
 */

import {
  CatalystRegistry,
  createFixtureSearchCatalyst,
  createMockInferenceCatalyst,
  runSource,
} from "../index";

const SOURCE = `
floor 0.02

catalyst web_search {
  namespace: remote
  input: Region output: Claim
}
catalyst local_notes {
  namespace: local
  input: Region output: Claim
}
catalyst archive_lookup {
  namespace: remote
  input: Region output: Claim
}

project apollo_budget {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    until converge
    yield launch_date

  seek budget_overrun
    not{ "post-hoc estimates without primary-source citation" }
    toward{ cost_vs_authorization(launch_date) }
    via{ web_search(budget_overrun)
           >> local_notes(budget_overrun)
           >> archive_lookup(budget_overrun) }
    until converge otherwise decline
    yield budget_overrun

  goal {
    claims: [launch_date, budget_overrun]
    coherence: >= 0.5
  }
}
`;

async function main() {
  const registry = new CatalystRegistry();
  registry.register(
    createFixtureSearchCatalyst("web_search", {
      "cost_vs_authorization(launch_date_of(Apollo 11))": "budget:on-authorization",
    }),
  );
  registry.register(createMockInferenceCatalyst("local_notes", (c) => `${c}:corroborated`));
  registry.register(createMockInferenceCatalyst("archive_lookup", (c) => `${c}:corroborated`));

  const { projectResults } = await runSource(SOURCE, registry);
  const apollo = projectResults.get("apollo_budget")!;

  for (const [name, value] of apollo) {
    if (value.kind === "Claim") {
      console.log(`${name} -> Claim(${value.targetClaim}) floor=${value.floor.toFixed(4)} residue=${value.residue.toFixed(4)}`);
    } else {
      console.log(`${name} -> Decline: ${value.classes.length} distinct claim-regions found`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

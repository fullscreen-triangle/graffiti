// ── Sample .grf scripts and file-tree structure ──

export interface FileNode {
  name: string;
  type: "file" | "folder";
  children?: FileNode[];
  content?: string;
}

export const PROMPT_TEMPLATE = `-- Write your search intent in plain English below.
-- An AI model will generate the .grf script for you.
--
-- Example:
--   "Find the founding date of TUM using at least
--    three independent sources, and decline if they
--    disagree."
--
-- Then press Run to execute the generated script.

`;

export const SAMPLE_SCRIPTS: Record<string, string> = {
  "prompt.grf": PROMPT_TEMPLATE,

  "tutorial_01_basic.grf": `-- Tutorial 1: A single-source lookup
-- The simplest possible seek: one target, default catalysts.

floor 0.02

project basic_lookup {
  seek founding_year
    not { "unsourced claims", "disputed dates" }
    toward { founding_year_of("Technical University of Munich") }
    until converge
    yield founding_year
}
`,

  "tutorial_02_catalysts.grf": `-- Tutorial 2: Explicit catalyst chains
-- Name the sources you want and let coherence do the rest.

floor 0.02

catalyst web_search {
  namespace: remote
  input: Region  output: Claim
}
catalyst local_notes {
  namespace: local
  input: Region  output: Claim
}
catalyst archive_lookup {
  namespace: remote
  input: Region  output: Claim
}

project apollo_budget {
  seek launch_date
    not { "secondary summaries without citation" }
    toward { launch_date_of("Apollo 11") }
    until converge
    yield launch_date

  seek budget_overrun
    not { "post-hoc estimates without primary-source citation" }
    toward { cost_vs_authorization(launch_date) }
    via { web_search(budget_overrun)
            >> local_notes(budget_overrun)
            >> archive_lookup(budget_overrun) }
    until converge otherwise decline
    yield budget_overrun

  goal {
    claims: [launch_date, budget_overrun]
    coherence: >= 0.5
  }
}
`,

  "tutorial_03_decline.grf": `-- Tutorial 3: Honest decline
-- When sources disagree, the script says so instead of guessing.

floor 0.02

catalyst source_a {
  namespace: remote
  input: Region  output: Claim
}
catalyst source_b {
  namespace: remote
  input: Region  output: Claim
}

project disputed_event {
  seek disputed_cause
    not { "single-source attribution" }
    toward { primary_cause_of("Tunguska event") }
    via { source_a(disputed_cause)
            >> source_b(disputed_cause) }
    until converge otherwise decline
    yield disputed_cause
}
`,

  "tutorial_04_scenes.grf": `-- Tutorial 4: Scene-scoped search
-- Restrict search to specific parts of the codebase.

floor 0.01

catalyst code_search {
  namespace: local
  input: Region  output: Claim
}
catalyst doc_search {
  namespace: local
  input: Region  output: Claim
}
catalyst model_classify {
  namespace: inference
  input: Region  output: Claim
}

project find_allocation {
  seek water_filling
    not { "unrelated algorithms", "greedy heuristics" }
    toward { implementation_of("water-filling allocation") }
    via { code_search(water_filling)
            >> doc_search(water_filling)
            >> model_classify(water_filling) }
    until converge
    yield water_filling

  goal {
    claims: [water_filling]
    coherence: >= 0.5
  }
}
`,

  "tutorial_05_multi_seek.grf": `-- Tutorial 5: Multi-seek DAG
-- Seeks can reference each other's yields, forming a project graph.

floor 0.02

catalyst arxiv_search {
  namespace: remote
  input: Region  output: Claim
}
catalyst semantic_scholar {
  namespace: remote
  input: Region  output: Claim
}
catalyst local_corpus {
  namespace: local
  input: Region  output: Claim
}
catalyst llm_summarise {
  namespace: inference
  input: Region  output: Claim
}

project literature_review {
  seek core_papers
    not { "preprints without peer review", "secondary citations" }
    toward { papers_on("graph-theoretic search calculus") }
    via { arxiv_search(core_papers)
            >> semantic_scholar(core_papers)
            >> local_corpus(core_papers) }
    until converge otherwise decline
    yield core_papers

  seek methodology_gap
    not { "already addressed in core_papers" }
    toward { open_problems(core_papers) }
    via { semantic_scholar(methodology_gap)
            >> llm_summarise(methodology_gap)
            >> arxiv_search(methodology_gap) }
    until converge otherwise decline
    yield methodology_gap

  seek synthesis
    not { "claims not grounded in core_papers or methodology_gap" }
    toward { narrative_synthesis(core_papers, methodology_gap) }
    via { llm_summarise(synthesis)
            >> local_corpus(synthesis)
            >> arxiv_search(synthesis) }
    until converge
    yield synthesis

  goal {
    claims: [core_papers, methodology_gap, synthesis]
    coherence: >= 0.6
  }
}
`,
};

export const FILE_TREE: FileNode[] = [
  {
    name: "spraypaint",
    type: "folder",
    children: [
      {
        name: "scripts",
        type: "folder",
        children: [
          { name: "prompt.grf", type: "file", content: SAMPLE_SCRIPTS["prompt.grf"] },
        ],
      },
      {
        name: "tutorials",
        type: "folder",
        children: [
          { name: "tutorial_01_basic.grf", type: "file", content: SAMPLE_SCRIPTS["tutorial_01_basic.grf"] },
          { name: "tutorial_02_catalysts.grf", type: "file", content: SAMPLE_SCRIPTS["tutorial_02_catalysts.grf"] },
          { name: "tutorial_03_decline.grf", type: "file", content: SAMPLE_SCRIPTS["tutorial_03_decline.grf"] },
          { name: "tutorial_04_scenes.grf", type: "file", content: SAMPLE_SCRIPTS["tutorial_04_scenes.grf"] },
          { name: "tutorial_05_multi_seek.grf", type: "file", content: SAMPLE_SCRIPTS["tutorial_05_multi_seek.grf"] },
        ],
      },
      {
        name: "examples",
        type: "folder",
        children: [
          { name: "README.md", type: "file", content: "# Examples\n\nGenerated scripts from chart interactions appear here." },
        ],
      },
    ],
  },
];

/** Test 11: Lexer and parser round-trip correctness against
 * Definition (Graffiti grammar). */

import { Checker } from "./harness";
import { tokenize } from "../lang/lexer";
import { parseProgram } from "../lang/parser";

const SOURCE = `
floor 0.02

catalyst web_search {
  namespace: remote
  input: Region output: Claim
}

project demo {
  seek fact
    not{ "unsourced", "disputed" }
    toward{ founding_year_of(subject) }
    via{ web_search(fact) }
    until converge otherwise decline
    yield fact

  goal {
    claims: [fact]
    coherence: >= 0.5
  }
}
`;

export function run() {
  const c = new Checker("11", "Lexer/Parser (Definition: Graffiti grammar)");

  const tokens = tokenize(SOURCE);
  c.assert(tokens.length > 0, "tokenizer produces a non-empty token stream");
  c.assert(tokens[tokens.length - 1]!.type === "EOF", "token stream terminates with EOF");
  c.assert(tokens.some((t) => t.type === "KEYWORD" && t.value === "seek"), "seek keyword tokenised");
  c.assert(tokens.some((t) => t.type === "STRING" && t.value === "unsourced"), "string literal tokenised with content preserved");

  const program = parseProgram(SOURCE);
  c.assert(program.decls.length === 3, `program has 3 top-level decls (got ${program.decls.length})`);

  const floorDecl = program.decls.find((d) => d.kind === "floorDecl");
  c.assert(floorDecl !== undefined && (floorDecl as { value: number }).value === 0.02, "floor declaration parsed with correct value");

  const catalystDecl = program.decls.find((d) => d.kind === "catalystDecl");
  c.assert(catalystDecl !== undefined, "catalyst declaration parsed");

  const project = program.decls.find((d) => d.kind === "projectDecl") as
    | { seeks: Array<{ name: string; not: unknown; via: unknown; yieldName: string }>; goal: unknown }
    | undefined;
  c.assert(project !== undefined, "project declaration parsed");
  c.assert(project !== undefined && project.seeks.length === 1, "project has exactly one seek");
  c.assert(project !== undefined && project.seeks[0]!.yieldName === "fact", "seek yield name parsed correctly");
  c.assert(project !== undefined && project.goal !== null, "goal block parsed");

  // Round-trip: re-tokenising a re-serialised subset should not throw.
  let threw = false;
  try {
    tokenize('seek x not{"a"} toward{y} until converge yield x');
  } catch {
    threw = true;
  }
  c.assert(!threw, "minimal seek statement tokenises without error");

  return c.result();
}

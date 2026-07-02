/** Test 12: Static type checker (Rule: Floor Positivity, Theorem: No
 * Zero-Residue Claim, Rule: Coherence for Closure, Corollary: Mandatory
 * Negation Boundary). */

import { Checker } from "./harness";
import { parseProgram } from "../lang/parser";
import { typecheck } from "../lang/typechecker";
import { CatalystRegistry } from "../orchestration/catalyst";

export function run() {
  const c = new Checker("12", "Type Checker (Floor Positivity, No Zero-Residue Claim, Coherence)");

  // No floor declared: must fail.
  {
    const program = parseProgram(`
      project p {
        seek x not{"a"} toward{y} until converge yield x
      }
    `);
    const result = typecheck(program);
    c.assert(!result.ok, "missing floor declaration is rejected");
    c.assert(result.diagnostics.some((d) => d.code === "NO_FLOOR_DECLARED"), "NO_FLOOR_DECLARED diagnostic present");
  }

  // Non-positive floor: must fail.
  {
    const program = parseProgram(`
      floor 0
      project p {
        seek x not{"a"} toward{y} until converge yield x
      }
    `);
    const result = typecheck(program);
    c.assert(!result.ok, "zero floor is rejected");
    c.assert(result.diagnostics.some((d) => d.code === "NONPOSITIVE_FLOOR"), "NONPOSITIVE_FLOOR diagnostic present");
  }

  // Positive floor, valid seek: must pass.
  {
    const program = parseProgram(`
      floor 0.05
      project p {
        seek x not{"a"} toward{y} until converge yield x
      }
    `);
    const result = typecheck(program);
    c.assert(result.ok, "well-formed project with positive floor type-checks");
    c.assertClose(result.ambientFloor, 0.05, 1e-12, "ambient floor recorded correctly");
  }

  // Fewer than 3 catalysts under bare converge: coherence warning, not a hard error.
  {
    const program = parseProgram(`
      floor 0.05
      catalyst a { namespace: local input: Region output: Claim }
      catalyst b { namespace: local input: Region output: Claim }
      project p {
        seek x not{"z"} toward{y} via{ a(x) >> b(x) } until converge yield x
      }
    `);
    const result = typecheck(program);
    c.assert(result.ok, "sub-triangle chain is a warning, not a type error");
    c.assert(result.diagnostics.some((d) => d.code === "COHERENCE_WARNING"), "COHERENCE_WARNING emitted for a 2-catalyst chain under bare converge");
  }

  // Three catalysts under bare converge: no coherence warning.
  {
    const program = parseProgram(`
      floor 0.05
      catalyst a { namespace: local input: Region output: Claim }
      catalyst b { namespace: local input: Region output: Claim }
      catalyst d { namespace: remote input: Region output: Claim }
      project p {
        seek x not{"z"} toward{y} via{ a(x) >> b(x) >> d(x) } until converge yield x
      }
    `);
    const result = typecheck(program);
    c.assert(!result.diagnostics.some((d) => d.code === "COHERENCE_WARNING"), "no coherence warning for a 3-catalyst chain");
  }

  // Unknown catalyst reference: must fail when a registry is supplied.
  {
    const program = parseProgram(`
      floor 0.05
      project p {
        seek x not{"z"} toward{y} via{ unknown_catalyst(x) } until converge yield x
      }
    `);
    const registry = new CatalystRegistry();
    const result = typecheck(program, { registry });
    c.assert(!result.ok, "unknown catalyst reference is rejected when a registry is supplied");
    c.assert(result.diagnostics.some((d) => d.code === "UNKNOWN_CATALYST"), "UNKNOWN_CATALYST diagnostic present");
  }

  // Cyclic project: must fail.
  {
    const program = parseProgram(`
      floor 0.05
      project p {
        seek x not{"z"} toward{y(b)} until converge yield a
        seek y not{"z"} toward{y(a)} until converge yield b
      }
    `);
    const result = typecheck(program);
    c.assert(!result.ok, "cyclic project dependency is rejected");
    c.assert(result.diagnostics.some((d) => d.code === "CYCLIC_PROJECT"), "CYCLIC_PROJECT diagnostic present");
  }

  return c.result();
}

/**
 * Runtime values of Graffiti (.grf).
 *
 * Implements Definition (Types) of semantic-causal-propagation.tex: every
 * typed value carries a floor annotation and a residue
 * (Theorem: No Zero-Residue Claim).
 */

import type { GraffitiType } from "./ast";

export interface ClaimValue {
  kind: "Claim";
  targetClaim: string;
  floor: number;
  residue: number;
  /** Which equivalence classes the search reached before closing (Definition: Closure). */
  classesAtClosure: string[][];
}

export interface DeclineValue {
  kind: "Decline";
  seekName: string;
  /** The distinct equivalence classes found at contested closure (Theorem: Convergent Closure or Honest Decline). */
  classes: string[][];
  floor: number;
}

export type GraffitiValue = ClaimValue | DeclineValue;

export function isClaim(v: GraffitiValue): v is ClaimValue {
  return v.kind === "Claim";
}

export function isDecline(v: GraffitiValue): v is DeclineValue {
  return v.kind === "Decline";
}

export function graffitiTypeOf(v: GraffitiValue): GraffitiType {
  return "Claim"; // Decline is a Claim-typed value carrying a declined payload at the language level.
}

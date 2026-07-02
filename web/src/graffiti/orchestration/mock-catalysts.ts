/**
 * Reference mock catalysts.
 *
 * Two trivial, in-memory catalysts so this library's own tests and
 * examples can run standalone without network or model access. Real
 * Buhera OS integrations supply real catalysts (an actual web search, an
 * actual Ollama or Hugging Face inference call, an actual local
 * filesystem scan) implementing the same CatalystDefinition contract --
 * nothing in the calculus or the language distinguishes a mock from a
 * real catalyst (Theorem "Namespace Neutrality").
 */

import type { CatalystDefinition, CatalystResult } from "./catalyst";

/**
 * A fixture-backed "search" catalyst: a fixed lookup table standing in
 * for a real search engine or document store. Namespace `local` since it
 * reads only in-memory fixture data.
 */
export function createFixtureSearchCatalyst(
  name: string,
  fixtures: Record<string, string>,
  power = 0.6,
): CatalystDefinition {
  return {
    name,
    namespace: "local",
    provider: async (ctx): Promise<CatalystResult> => {
      const query = String(ctx.args["query"] ?? ctx.currentClaim);
      const claim = fixtures[query] ?? `${name}:unresolved:${query}`;
      return { claim, power };
    },
  };
}

/**
 * A deterministic pseudo-inference catalyst standing in for a locally
 * hosted or externally retrieved machine-learned model. It applies a
 * fixed, injectable transform to the current claim rather than performing
 * any real inference, so tests are reproducible without a model runtime.
 * Namespace `inference`.
 */
export function createMockInferenceCatalyst(
  name: string,
  transform: (claim: string) => string,
  power = 0.5,
): CatalystDefinition {
  return {
    name,
    namespace: "inference",
    provider: async (ctx): Promise<CatalystResult> => {
      return { claim: transform(ctx.currentClaim), power };
    },
  };
}

/**
 * A no-op catalyst (power 0, "inert" in the sense of Definition 7.2) --
 * useful as a negative control in tests of the scheduler and the
 * coherence checker.
 */
export function createInertCatalyst(name: string): CatalystDefinition {
  return {
    name,
    namespace: "local",
    provider: async (ctx): Promise<CatalystResult> => {
      return { claim: ctx.currentClaim, power: 0 };
    },
  };
}

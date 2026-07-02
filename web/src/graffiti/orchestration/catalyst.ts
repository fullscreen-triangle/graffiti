/**
 * The catalyst registry.
 *
 * Implements Definition (Catalyst Registry Entry) and Theorem (Namespace
 * Neutrality) of semantic-causal-propagation.tex, Section "Heterogeneous
 * Catalyst Orchestration".
 *
 * A catalyst is any computational resource a seek's `via` chain can
 * invoke: a local file scan, a remote API call, or an inference from a
 * locally hosted or externally retrieved machine-learned model. The
 * theory is proved namespace-neutral (Theorem "Namespace Neutrality"):
 * every result of the catalytic algebra (multiplicative composition,
 * coherence, saturation) holds identically regardless of which namespace
 * a catalyst belongs to. The namespace tag below exists purely as
 * scheduling metadata (expected latency, resource contention, cost) and
 * carries no semantic weight in the calculus.
 */

export type CatalystNamespace = "local" | "remote" | "inference" | "composite";

export interface CatalystContext {
  /** The claim currently held by the propagation invoking this catalyst. */
  currentClaim: string;
  /** Free-form arguments supplied at the call site (e.g. web_search(query)). */
  args: Record<string, unknown>;
}

export interface CatalystResult {
  /** The claim this catalyst invocation resolves to. */
  claim: string;
  /** The catalytic power kappa in [0,1] this invocation realised (Definition 7.2). */
  power: number;
}

/**
 * A catalyst provider: the partial function pi of Definition (Catalyst
 * Registry Entry) implementing the catalyst's effect on the current
 * claim. Providers are async because remote and inference namespaces are
 * inherently asynchronous; local providers may resolve synchronously
 * wrapped in a resolved promise.
 */
export type CatalystProvider = (ctx: CatalystContext) => Promise<CatalystResult>;

export interface CatalystDefinition {
  name: string;
  namespace: CatalystNamespace;
  provider: CatalystProvider;
}

/**
 * A finite map from catalyst names to their definitions (Definition
 * "Catalyst Registry Entry"). The orchestrator resolves each `via` chain
 * reference against this registry.
 */
export class CatalystRegistry {
  private readonly catalysts = new Map<string, CatalystDefinition>();

  register(def: CatalystDefinition): this {
    if (this.catalysts.has(def.name)) {
      throw new Error(`Catalyst "${def.name}" is already registered`);
    }
    this.catalysts.set(def.name, def);
    return this;
  }

  get(name: string): CatalystDefinition {
    const def = this.catalysts.get(name);
    if (!def) {
      throw new Error(`Unknown catalyst "${name}"`);
    }
    return def;
  }

  has(name: string): boolean {
    return this.catalysts.has(name);
  }

  names(): string[] {
    return Array.from(this.catalysts.keys());
  }

  /** All catalysts of a given namespace -- scheduling metadata only, per Theorem (Namespace Neutrality). */
  byNamespace(namespace: CatalystNamespace): CatalystDefinition[] {
    return Array.from(this.catalysts.values()).filter((c) => c.namespace === namespace);
  }
}

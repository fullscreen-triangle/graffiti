/**
 * Individuation by negation and the recognition/search identity.
 *
 * Implements Theorem 4.2 (Individuation by Negation) and Theorem 4.5
 * (Recognition/Search Identity) of semantic-causal-propagation.tex.
 */

/** co(U) = V \ U, computed over string-identified claims. */
export function complement<T>(whole: ReadonlySet<T>, subset: ReadonlySet<T>): Set<T> {
  const result = new Set<T>();
  for (const item of whole) {
    if (!subset.has(item)) result.add(item);
  }
  return result;
}

function setEquals<T>(a: ReadonlySet<T>, b: ReadonlySet<T>): boolean {
  if (a.size !== b.size) return false;
  for (const item of a) {
    if (!b.has(item)) return false;
  }
  return true;
}

/** Verify co(co(U)) === U (Theorem 4.2 involution). */
export function doubleComplementIsIdentity<T>(whole: ReadonlySet<T>, subset: ReadonlySet<T>): boolean {
  return setEquals(complement(whole, complement(whole, subset)), subset);
}

/**
 * A finite decoder Dec: Q -> V and its fibre map Proj (Definition 4.4 /
 * Theorem 4.5). `mapping` gives Dec directly; Proj(v) is derived as the
 * fibre { q : Dec(q) = v }, exactly as Theorem 4.5 requires -- recognition
 * and search are proved to be inverse readings of this single relation.
 */
export class Decoder {
  private readonly mapping: Map<string, string>;

  constructor(mapping: Map<string, string> | Record<string, string>) {
    this.mapping = mapping instanceof Map ? new Map(mapping) : new Map(Object.entries(mapping));
  }

  decode(query: string): string {
    const claim = this.mapping.get(query);
    if (claim === undefined) {
      throw new Error(`No claim registered for query "${query}"`);
    }
    return claim;
  }

  /** Proj(v) := Dec^{-1}({v}) -- the fibre of queries decoding to `claim`. */
  fibre(claim: string): Set<string> {
    const result = new Set<string>();
    for (const [q, v] of this.mapping) {
      if (v === claim) result.add(q);
    }
    return result;
  }

  entries(): Array<[string, string]> {
    return Array.from(this.mapping.entries());
  }

  /**
   * Reconstruct Dec from the fibre partition {Proj(v)}_v, per the proof of
   * Theorem 4.5: a function and its fibre map determine each other on a
   * finite domain.
   */
  recoverFromFibres(): Map<string, string> {
    const claims = new Set(this.mapping.values());
    const rebuilt = new Map<string, string>();
    for (const v of claims) {
      for (const q of this.fibre(v)) {
        rebuilt.set(q, v);
      }
    }
    return rebuilt;
  }

  matchesMapping(other: Map<string, string>): boolean {
    if (other.size !== this.mapping.size) return false;
    for (const [k, v] of this.mapping) {
      if (other.get(k) !== v) return false;
    }
    return true;
  }
}

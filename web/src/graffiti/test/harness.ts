/**
 * Minimal, dependency-free assertion harness for the Graffiti test suite,
 * matching the self-contained-suite discipline of the manuscript's
 * validation appendix (no external test framework required to run
 * `npm run test:graffiti`).
 */

export interface CheckResult {
  category: string;
  theorem: string;
  passed: number;
  failed: number;
  failures: string[];
}

export class Checker {
  private passed = 0;
  private failed = 0;
  private readonly failures: string[] = [];

  constructor(readonly category: string, readonly theorem: string) {}

  assert(condition: boolean, message: string): void {
    if (condition) {
      this.passed++;
    } else {
      this.failed++;
      this.failures.push(message);
    }
  }

  assertClose(actual: number, expected: number, tolerance: number, message: string): void {
    this.assert(Math.abs(actual - expected) <= tolerance, `${message} (actual=${actual}, expected=${expected}, tol=${tolerance})`);
  }

  result(): CheckResult {
    return { category: this.category, theorem: this.theorem, passed: this.passed, failed: this.failed, failures: this.failures };
  }
}

export function reportAll(results: CheckResult[]): boolean {
  let totalPassed = 0;
  let totalChecks = 0;
  let allOk = true;

  for (const r of results) {
    const total = r.passed + r.failed;
    totalPassed += r.passed;
    totalChecks += total;
    const status = r.failed === 0 ? "PASS" : "FAIL";
    if (r.failed > 0) allOk = false;
    console.log(`[${r.category}] ${r.theorem.padEnd(60)} ${String(r.passed).padStart(5)}/${String(total).padEnd(5)} ${status}`);
    for (const f of r.failures.slice(0, 5)) {
      console.log(`      - ${f}`);
    }
  }

  console.log();
  console.log(`Pass rate: ${results.filter((r) => r.failed === 0).length}/${results.length} categories, ${totalPassed}/${totalChecks} individual checks`);
  return allOk;
}

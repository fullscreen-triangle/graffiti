/**
 * Run the complete Graffiti TypeScript test suite.
 *
 * Usage: npx tsx src/graffiti/test/run-all.ts
 * (or: npm run test:graffiti, from web/)
 */

import { reportAll, type CheckResult } from "./harness";
import * as t01 from "./t01-floor";
import * as t02 from "./t02-individuation";
import * as t03 from "./t03-recognition-search";
import * as t04 from "./t04-representation-mobility";
import * as t05 from "./t05-path-opacity";
import * as t06 from "./t06-multiplicative-law";
import * as t07 from "./t07-saturation";
import * as t08 from "./t08-coherence";
import * as t09 from "./t09-closure";
import * as t10 from "./t10-scheduler";
import * as t11 from "./t11-lexer-parser";
import * as t12 from "./t12-typechecker";
import * as t13 from "./t13-interpreter";
import * as t14 from "./t14-purpose-session";

async function main() {
  const results: CheckResult[] = [
    t01.run(),
    t02.run(),
    t03.run(),
    t04.run(),
    t05.run(),
    t06.run(),
    t07.run(),
    t08.run(),
    t09.run(),
    t10.run(),
    t11.run(),
    t12.run(),
    await t13.run(),
    await t14.run(),
  ];

  const allOk = reportAll(results);
  process.exitCode = allOk ? 0 : 1;
}

main();

// ── Execution state, mock data generation, and crossfilter inversion ──

export interface SeekResult {
  name: string;
  target: string;
  status: "converged" | "declined";
  floor: number;
  residue: number;
  compositepower: number;
  catalysts: CatalystResult[];
  trajectory: TrajectoryPoint[];
  closureSteps: number;
  confidenceStep: number;
  equivalenceClasses: number;
}

export interface CatalystResult {
  name: string;
  namespace: "local" | "remote" | "inference" | "composite";
  power: number;
  marginalGain: number;
  supportEdges: { target: string; strength: number }[];
}

export interface TrajectoryPoint {
  step: number;
  alignment: number;
  residual: number;
  logResidual: number;
  catalyst: string;
}

export interface SceneAllocation {
  scene: string;
  allocated: number;
  totalPassages: number;
  bestScore: number;
  medianScore: number;
  clearingPrice: number;
}

export interface InvariantState {
  identityHash: string;
  chi: number;
  identityValid: boolean;
  committedCount: number;
  countHistory: number[];
  searchNotFetch: boolean;
  exclusivePhases: boolean;
  lastPhase: "idle" | "indexing" | "querying";
}

export interface ExecutionState {
  seeks: SeekResult[];
  scenes: SceneAllocation[];
  invariants: InvariantState;
  stale: boolean;
  runCount: number;
}

// ── Mock data generators ──

function generateTrajectory(
  floor: number,
  catalysts: { name: string; power: number }[],
  steps: number
): TrajectoryPoint[] {
  const omega = 1.0;
  const floorNorm = floor / omega;
  let alignment = 0.7 + Math.random() * 0.25;
  const points: TrajectoryPoint[] = [];

  for (let i = 0; i < steps; i++) {
    const catIdx = i % catalysts.length;
    const cat = catalysts[catIdx];
    const gap = alignment - floorNorm;
    const reduction = gap * cat.power * (0.7 + Math.random() * 0.6);
    alignment = Math.max(floorNorm, alignment - reduction);

    // add some noise for interior steps (path opacity)
    if (i > 0 && i < steps - 1 && Math.random() < 0.3) {
      alignment += (Math.random() - 0.5) * 0.08;
      alignment = Math.max(floorNorm, Math.min(1.0, alignment));
    }

    const residual = alignment - floorNorm;
    points.push({
      step: i + 1,
      alignment,
      residual,
      logResidual: residual > 0 ? Math.log(residual) : -10,
      catalyst: cat.name,
    });
  }

  // ensure terminal convergence
  if (points.length > 0) {
    const last = points[points.length - 1];
    last.alignment = floorNorm + floor * 0.1;
    last.residual = last.alignment - floorNorm;
    last.logResidual = Math.log(last.residual);
  }

  return points;
}

function generateCatalysts(script: string): CatalystResult[] {
  const catalystPattern = /catalyst\s+(\w+)\s*\{[^}]*namespace:\s*(\w+)/g;
  const found: CatalystResult[] = [];
  let match;

  while ((match = catalystPattern.exec(script)) !== null) {
    const name = match[1];
    const ns = match[2] as CatalystResult["namespace"];
    const power = 0.15 + Math.random() * 0.55;
    found.push({
      name,
      namespace: ns,
      power,
      marginalGain: 0,
      supportEdges: [],
    });
  }

  if (found.length === 0) {
    found.push(
      { name: "default_search", namespace: "remote", power: 0.4, marginalGain: 0, supportEdges: [] },
      { name: "default_local", namespace: "local", power: 0.3, marginalGain: 0, supportEdges: [] },
      { name: "default_infer", namespace: "inference", power: 0.25, marginalGain: 0, supportEdges: [] },
    );
  }

  // compute marginal gains from multiplicative law
  let composite = 0;
  for (const cat of found) {
    const newComposite = 1 - (1 - composite) * (1 - cat.power);
    cat.marginalGain = newComposite - composite;
    composite = newComposite;
  }

  // generate support edges (for coherence)
  for (let i = 0; i < found.length; i++) {
    for (let j = 0; j < found.length; j++) {
      if (i !== j) {
        const strength = 0.3 + Math.random() * 0.5;
        found[i].supportEdges.push({ target: found[j].name, strength });
      }
    }
  }

  return found;
}

function generateSeeks(script: string, floor: number): SeekResult[] {
  const seekPattern = /seek\s+(\w+)[^]*?toward\s*\{([^}]*)\}[^]*?until\s+(\w+)/g;
  const seeks: SeekResult[] = [];
  let match;

  while ((match = seekPattern.exec(script)) !== null) {
    const name = match[1];
    const target = match[2].trim();
    const untilType = match[3];

    const catalysts = generateCatalysts(script);
    const compositepower = 1 - catalysts.reduce((acc, c) => acc * (1 - c.power), 1);
    const hasDecline = script.includes("otherwise decline");
    const willDecline = hasDecline && Math.random() < 0.2;

    const steps = 6 + Math.floor(Math.random() * 12);
    const trajectory = generateTrajectory(
      floor,
      catalysts.map((c) => ({ name: c.name, power: c.power })),
      steps
    );

    const confidenceStep = Math.floor(steps * 0.3) + 1;
    const closureSteps = steps;

    seeks.push({
      name,
      target,
      status: willDecline ? "declined" : "converged",
      floor,
      residue: trajectory[trajectory.length - 1]?.residual ?? floor,
      compositepower,
      catalysts,
      trajectory,
      closureSteps,
      confidenceStep,
      equivalenceClasses: willDecline ? 2 : 1,
    });
  }

  if (seeks.length === 0) {
    seeks.push({
      name: "default_seek",
      target: "unspecified",
      status: "converged",
      floor,
      residue: floor * 1.1,
      compositepower: 0.75,
      catalysts: generateCatalysts(script),
      trajectory: generateTrajectory(floor, [{ name: "default", power: 0.3 }], 10),
      closureSteps: 10,
      confidenceStep: 3,
      equivalenceClasses: 1,
    });
  }

  return seeks;
}

function generateScenes(): SceneAllocation[] {
  const sceneNames = ["core", "docs", "crates", "tests", "examples", "(root)"];
  const k = 10;
  const scenes: SceneAllocation[] = [];

  // water-filling: compute clearing price
  const scores = sceneNames.map(() => Math.random() * 2 + 0.5);
  const sorted = [...scores].sort((a, b) => b - a);
  const clearingPrice = sorted[Math.min(3, sorted.length - 1)];

  let remaining = k;
  for (let i = 0; i < sceneNames.length; i++) {
    const above = scores[i] > clearingPrice;
    const alloc = above ? Math.max(1, Math.floor(remaining * (scores[i] / scores.reduce((a, b) => a + b, 0)))) : 0;
    remaining -= alloc;
    scenes.push({
      scene: sceneNames[i],
      allocated: alloc,
      totalPassages: Math.floor(Math.random() * 50) + 5,
      bestScore: scores[i],
      medianScore: scores[i] * (0.3 + Math.random() * 0.4),
      clearingPrice,
    });
  }

  return scenes;
}

function generateInvariants(runCount: number): InvariantState {
  const hash = Array.from({ length: 16 }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join("");

  return {
    identityHash: `0x${hash}`,
    chi: 0.02 + Math.random() * 0.05,
    identityValid: true,
    committedCount: runCount,
    countHistory: Array.from({ length: Math.max(1, runCount) }, (_, i) => i + 1),
    searchNotFetch: true,
    exclusivePhases: true,
    lastPhase: "idle",
  };
}

export function executeScript(script: string, prevState?: ExecutionState): ExecutionState {
  const floorMatch = script.match(/floor\s+([\d.]+)/);
  const floor = floorMatch ? parseFloat(floorMatch[1]) : 0.02;
  const runCount = (prevState?.runCount ?? 0) + 1;

  return {
    seeks: generateSeeks(script, floor),
    scenes: generateScenes(),
    invariants: generateInvariants(runCount),
    stale: false,
    runCount,
  };
}

// ── Crossfilter inversion: chart manipulation → script diff ──

export interface ScriptDiff {
  description: string;
  oldText: string;
  newText: string;
}

export function invertTrajectoryDrag(
  script: string,
  targetResidual: number,
  currentFloor: number
): ScriptDiff | null {
  const newFloor = Math.max(0.001, Math.min(targetResidual * 0.8, 0.5));
  const floorMatch = script.match(/floor\s+([\d.]+)/);
  if (!floorMatch) return null;

  return {
    description: `Adjust floor to ${newFloor.toFixed(4)} for target residual ${targetResidual.toFixed(4)}`,
    oldText: `floor ${floorMatch[1]}`,
    newText: `floor ${newFloor.toFixed(4)}`,
  };
}

export function invertCatalystPowerDrag(
  script: string,
  catalystName: string,
  desiredPower: number,
  registryCatalysts: { name: string; namespace: string; typicalPower: number }[]
): ScriptDiff | null {
  // find the catalyst in the registry closest to desired power
  const best = registryCatalysts.reduce((prev, curr) =>
    Math.abs(curr.typicalPower - desiredPower) < Math.abs(prev.typicalPower - desiredPower)
      ? curr
      : prev
  );

  if (best.name === catalystName) return null;

  const pattern = new RegExp(`(${catalystName})\\(`, "g");
  if (!pattern.test(script)) return null;

  return {
    description: `Replace ${catalystName} (κ≈${desiredPower.toFixed(2)}) with ${best.name} from registry`,
    oldText: catalystName,
    newText: best.name,
  };
}

export function invertCoherenceEdit(
  script: string,
  hasCycle: boolean
): ScriptDiff | null {
  if (hasCycle) return null;

  // if coherence broken, downgrade until clause
  const convergeMatch = script.match(/until\s+converge\b(?!\s+otherwise)/);
  if (!convergeMatch) return null;

  return {
    description: "Coherence triangle broken — downgrade to threshold convergence",
    oldText: "until converge",
    newText: "until converge otherwise decline",
  };
}

export function invertSceneAllocation(
  script: string,
  enabledScenes: string[]
): ScriptDiff | null {
  // this would add a --scenes constraint or boundary clause
  const scenesStr = enabledScenes.join(", ");
  const seekMatch = script.match(/(not\s*\{[^}]*\})/);
  if (!seekMatch) return null;

  const oldNot = seekMatch[1];
  const newNot = oldNot.replace("}", `, "results outside: ${scenesStr}" }`);

  return {
    description: `Restrict search to scenes: ${scenesStr}`,
    oldText: oldNot,
    newText: newNot,
  };
}

export function invertClosureDrag(
  script: string,
  wantEarlierClosure: boolean
): ScriptDiff | null {
  if (wantEarlierClosure) {
    const convergeMatch = script.match(/until\s+converge/);
    if (!convergeMatch) return null;
    return {
      description: "Relax closure to numeric threshold for earlier termination",
      oldText: "until converge",
      newText: "until alignment <= 0.15",
    };
  } else {
    const threshMatch = script.match(/until\s+alignment\s*<=\s*[\d.]+/);
    if (!threshMatch) return null;
    return {
      description: "Tighten to full closure (until converge)",
      oldText: threshMatch[0],
      newText: "until converge",
    };
  }
}

export function applyDiff(script: string, diff: ScriptDiff): string {
  return script.replace(diff.oldText, diff.newText);
}

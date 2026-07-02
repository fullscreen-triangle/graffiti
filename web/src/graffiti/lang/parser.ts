/**
 * Recursive-descent parser for Graffiti (.grf).
 *
 * Implements Definition (Graffiti grammar) of
 * semantic-causal-propagation.tex, Section "Lexical Structure and
 * Grammar", as a direct-encoding recursive-descent parser (Aho, Lam,
 * Sethi & Ullman 2006).
 */

import { tokenize, type Token } from "./lexer";
import type {
  AdmitClause,
  Arg,
  BoundaryExpr,
  CatalystDecl,
  ChainExpr,
  CondExpr,
  Decl,
  Expr,
  FloorDecl,
  GoalBlock,
  GraffitiType,
  ImportDecl,
  ModuleDecl,
  Program,
  ProjectDecl,
  RegionExpr,
  RelOp,
  SeekStmt,
} from "./ast";

export class ParseError extends Error {
  constructor(message: string, readonly line: number, readonly col: number) {
    super(`${message} (line ${line}, col ${col})`);
    this.name = "ParseError";
  }
}

const GRAFFITI_TYPES: ReadonlySet<string> = new Set(["Claim", "Region", "Catalyst", "Chain", "Residue"]);
const REL_OPS: ReadonlySet<string> = new Set([">", "<", ">=", "<=", "=="]);

class TokenStream {
  private pos = 0;
  constructor(private readonly tokens: Token[]) {}

  peek(offset = 0): Token {
    const idx = Math.min(this.pos + offset, this.tokens.length - 1);
    return this.tokens[idx]!;
  }

  next(): Token {
    const t = this.peek();
    if (t.type !== "EOF") this.pos++;
    return t;
  }

  atEnd(): boolean {
    return this.peek().type === "EOF";
  }

  expectKeyword(kw: string): Token {
    const t = this.peek();
    if (t.type !== "KEYWORD" || t.value !== kw) {
      throw new ParseError(`Expected keyword "${kw}", got "${t.value || "<eof>"}"`, t.line, t.col);
    }
    return this.next();
  }

  expectPunct(p: string): Token {
    const t = this.peek();
    if (t.type !== "PUNCT" || t.value !== p) {
      throw new ParseError(`Expected "${p}", got "${t.value || "<eof>"}"`, t.line, t.col);
    }
    return this.next();
  }

  expectOp(op: string): Token {
    const t = this.peek();
    if (t.type !== "OP" || t.value !== op) {
      throw new ParseError(`Expected operator "${op}", got "${t.value || "<eof>"}"`, t.line, t.col);
    }
    return this.next();
  }

  expectIdent(): Token {
    const t = this.peek();
    if (t.type !== "IDENT") {
      throw new ParseError(`Expected identifier, got "${t.value || "<eof>"}"`, t.line, t.col);
    }
    return this.next();
  }

  expectNumber(): Token {
    const t = this.peek();
    if (t.type !== "NUMBER") {
      throw new ParseError(`Expected number, got "${t.value || "<eof>"}"`, t.line, t.col);
    }
    return this.next();
  }

  expectString(): Token {
    const t = this.peek();
    if (t.type !== "STRING") {
      throw new ParseError(`Expected string literal, got "${t.value || "<eof>"}"`, t.line, t.col);
    }
    return this.next();
  }

  isKeyword(kw: string): boolean {
    const t = this.peek();
    return t.type === "KEYWORD" && t.value === kw;
  }

  isPunct(p: string): boolean {
    const t = this.peek();
    return t.type === "PUNCT" && t.value === p;
  }

  isOp(op: string): boolean {
    const t = this.peek();
    return t.type === "OP" && t.value === op;
  }
}

export function parseProgram(source: string): Program {
  const stream = new TokenStream(tokenize(source));
  const decls: Decl[] = [];
  while (!stream.atEnd()) {
    decls.push(parseDecl(stream));
  }
  return { kind: "program", decls };
}

function parseDecl(s: TokenStream): Decl {
  if (s.isKeyword("floor")) return parseFloorDecl(s);
  if (s.isKeyword("import")) return parseImportDecl(s);
  if (s.isKeyword("module")) return parseModuleDecl(s);
  if (s.isKeyword("project")) return parseProjectDecl(s);
  if (s.isKeyword("catalyst")) return parseCatalystDecl(s);

  const t = s.peek();
  throw new ParseError(`Expected a declaration, got "${t.value || "<eof>"}"`, t.line, t.col);
}

function parseFloorDecl(s: TokenStream): FloorDecl {
  s.expectKeyword("floor");
  const n = s.expectNumber();
  return { kind: "floorDecl", value: Number(n.value) };
}

function parseQName(s: TokenStream): string[] {
  const parts = [s.expectIdent().value];
  while (s.isOp(".")) {
    s.next();
    parts.push(s.expectIdent().value);
  }
  return parts;
}

function parseImportDecl(s: TokenStream): ImportDecl {
  s.expectKeyword("import");
  return { kind: "import", qname: parseQName(s) };
}

function parseModuleDecl(s: TokenStream): ModuleDecl {
  s.expectKeyword("module");
  const name = s.expectIdent().value;
  s.expectPunct("{");
  const decls: Decl[] = [];
  while (!s.isPunct("}")) {
    decls.push(parseDecl(s));
  }
  s.expectPunct("}");
  return { kind: "moduleDecl", name, decls };
}

function parseGraffitiType(s: TokenStream): GraffitiType {
  const t = s.expectIdent();
  if (!GRAFFITI_TYPES.has(t.value)) {
    throw new ParseError(
      `Unknown type "${t.value}"; expected one of Claim, Region, Catalyst, Chain, Residue`,
      t.line,
      t.col,
    );
  }
  return t.value as GraffitiType;
}

function parseCatalystDecl(s: TokenStream): CatalystDecl {
  s.expectKeyword("catalyst");
  const name = s.expectIdent().value;
  s.expectPunct("{");
  s.expectKeyword("namespace");
  s.expectOp(":");
  const namespace = s.expectIdent().value;
  s.expectKeyword("input");
  s.expectOp(":");
  const inputType = parseGraffitiType(s);
  s.expectKeyword("output");
  s.expectOp(":");
  const outputType = parseGraffitiType(s);
  s.expectPunct("}");
  return { kind: "catalystDecl", name, namespace, inputType, outputType };
}

function parseProjectDecl(s: TokenStream): ProjectDecl {
  s.expectKeyword("project");
  const name = s.expectIdent().value;
  s.expectPunct("{");
  const seeks: SeekStmt[] = [];
  let goal: GoalBlock | null = null;
  while (!s.isPunct("}")) {
    if (s.isKeyword("goal")) {
      goal = parseGoalBlock(s);
    } else {
      seeks.push(parseSeekStmt(s));
    }
  }
  s.expectPunct("}");
  return { kind: "projectDecl", name, seeks, goal };
}

function parseRelOp(s: TokenStream): RelOp {
  const t = s.peek();
  if (t.type === "OP" && REL_OPS.has(t.value)) {
    s.next();
    return t.value as RelOp;
  }
  throw new ParseError(`Expected a relational operator, got "${t.value || "<eof>"}"`, t.line, t.col);
}

function parseGoalBlock(s: TokenStream): GoalBlock {
  s.expectKeyword("goal");
  s.expectPunct("{");
  s.expectKeyword("claims");
  s.expectOp(":");
  s.expectPunct("[");
  const claims: string[] = [s.expectIdent().value];
  while (s.isPunct(",")) {
    s.next();
    claims.push(s.expectIdent().value);
  }
  s.expectPunct("]");
  s.expectKeyword("coherence");
  s.expectOp(":");
  const coherenceOp = parseRelOp(s);
  const coherenceValue = Number(s.expectNumber().value);
  s.expectPunct("}");
  return { kind: "goalBlock", claims, coherenceOp, coherenceValue };
}

/** Mandatory exclusion clause: Corollary (Mandatory Negation Boundary). */
function parseSeekStmt(s: TokenStream): SeekStmt {
  const seekTok = s.expectKeyword("seek");
  const name = s.expectIdent().value;

  s.expectKeyword("not");
  s.expectPunct("{");
  const notBoundary = parseBoundary(s);
  s.expectPunct("}");

  s.expectKeyword("toward");
  s.expectPunct("{");
  const toward = parseRegion(s);
  s.expectPunct("}");

  let via: ChainExpr | null = null;
  if (s.isKeyword("via")) {
    s.next();
    s.expectPunct("{");
    via = parseChain(s);
    s.expectPunct("}");
  }

  s.expectKeyword("until");
  const until = parseAdmit(s);

  s.expectKeyword("yield");
  const yieldName = s.expectIdent().value;

  return { kind: "seekStmt", name, not: notBoundary, toward, via, until, yieldName, line: seekTok.line };
}

function parseBoundaryAtom(s: TokenStream): BoundaryExpr {
  const terms = [s.expectString().value];
  while (s.isPunct(",")) {
    s.next();
    terms.push(s.expectString().value);
  }
  return { kind: "boundaryTerms", terms };
}

function parseBoundary(s: TokenStream): BoundaryExpr {
  let left = parseBoundaryAtom(s);
  while (s.isOp("||") || s.isOp("&&")) {
    const isOr = s.isOp("||");
    s.next();
    const right = parseBoundaryAtom(s);
    left = isOr ? { kind: "boundaryOr", left, right } : { kind: "boundaryAnd", left, right };
  }
  return left;
}

function parseArgs(s: TokenStream): Arg[] {
  const args: Arg[] = [];
  if (s.isPunct(")")) return args;
  args.push(parseArg(s));
  while (s.isPunct(",")) {
    s.next();
    args.push(parseArg(s));
  }
  return args;
}

function parseArg(s: TokenStream): Arg {
  // Disambiguate `name: expr` from a bare expression by lookahead.
  if (s.peek().type === "IDENT" && s.peek(1).type === "OP" && s.peek(1).value === ":") {
    const name = s.expectIdent().value;
    s.expectOp(":");
    return { name, value: parseExpr(s) };
  }
  return { name: null, value: parseExpr(s) };
}

function parseExpr(s: TokenStream): Expr {
  const t = s.peek();
  if (t.type === "NUMBER") {
    s.next();
    return { kind: "number", value: Number(t.value) };
  }
  if (t.type === "STRING") {
    s.next();
    return { kind: "string", value: t.value };
  }
  if (t.type === "PUNCT" && t.value === "(") {
    s.next();
    const inner = parseExpr(s);
    s.expectPunct(")");
    return { kind: "paren", expr: inner };
  }
  if (t.type === "IDENT") {
    const qname = parseQName(s);
    if (s.isPunct("(")) {
      s.next();
      const args = parseArgs(s);
      s.expectPunct(")");
      return { kind: "catalystRef", qname, args };
    }
    return { kind: "ident", name: qname.join(".") };
  }
  throw new ParseError(`Expected an expression, got "${t.value || "<eof>"}"`, t.line, t.col);
}

function parseRegion(s: TokenStream): RegionExpr {
  const qname = parseQName(s);
  if (s.isPunct("(")) {
    s.next();
    const args = parseArgs(s);
    s.expectPunct(")");
    return { kind: "regionCall", qname, args };
  }
  return { kind: "regionIdent", name: qname.join(".") };
}

/** Precedence (high to low): >> (catalysis) then || (parallel), per the grammar's operator table. */
function parseChainAtom(s: TokenStream): ChainExpr {
  const qname = parseQName(s);
  s.expectPunct("(");
  const args = parseArgs(s);
  s.expectPunct(")");
  return { kind: "catalystRef", qname, args };
}

function parseChainSeq(s: TokenStream): ChainExpr {
  const steps = [parseChainAtom(s)];
  while (s.isOp(">>")) {
    s.next();
    steps.push(parseChainAtom(s));
  }
  return steps.length === 1 ? steps[0]! : { kind: "chainSeq", steps };
}

function parseChain(s: TokenStream): ChainExpr {
  const branches = [parseChainSeq(s)];
  while (s.isOp("||")) {
    s.next();
    branches.push(parseChainSeq(s));
  }
  return branches.length === 1 ? branches[0]! : { kind: "chainPar", branches };
}

function parseCond(s: TokenStream): CondExpr {
  const left = parseExpr(s);
  const op = parseRelOp(s);
  const right = parseExpr(s);
  return { kind: "cond", left, op, right };
}

function parseAdmit(s: TokenStream): AdmitClause {
  if (s.isKeyword("converge")) {
    s.next();
    let otherwiseDecline = false;
    if (s.isKeyword("otherwise")) {
      s.next();
      s.expectKeyword("decline");
      otherwiseDecline = true;
    }
    return { kind: "converge", otherwiseDecline };
  }
  if (s.isKeyword("diverge")) {
    s.next();
    return { kind: "converge", otherwiseDecline: false };
  }
  return { kind: "cond", cond: parseCond(s) };
}

/**
 * Abstract syntax tree for Graffiti (.grf).
 *
 * Node shapes follow Definition (Graffiti grammar) of
 * semantic-causal-propagation.tex, Section "Lexical Structure and Grammar".
 */

export interface Program {
  kind: "program";
  decls: Decl[];
}

export type Decl = FloorDecl | ImportDecl | ModuleDecl | ProjectDecl | CatalystDecl;

export interface FloorDecl {
  kind: "floorDecl";
  value: number;
}

export interface ImportDecl {
  kind: "import";
  qname: string[];
}

export interface ModuleDecl {
  kind: "moduleDecl";
  name: string;
  decls: Decl[];
}

export type GraffitiType = "Claim" | "Region" | "Catalyst" | "Chain" | "Residue";

export interface CatalystDecl {
  kind: "catalystDecl";
  name: string;
  namespace: string;
  inputType: GraffitiType;
  outputType: GraffitiType;
}

export interface ProjectDecl {
  kind: "projectDecl";
  name: string;
  seeks: SeekStmt[];
  goal: GoalBlock | null;
}

export interface GoalBlock {
  kind: "goalBlock";
  claims: string[];
  coherenceOp: RelOp;
  coherenceValue: number;
}

export type RelOp = ">" | "<" | ">=" | "<=" | "==";

export interface SeekStmt {
  kind: "seekStmt";
  name: string;
  not: BoundaryExpr;
  toward: RegionExpr;
  via: ChainExpr | null;
  until: AdmitClause;
  yieldName: string;
  line: number;
}

/** Definition (Boundary): a conjunction/disjunction of exclusion terms. */
export type BoundaryExpr =
  | { kind: "boundaryTerms"; terms: string[] }
  | { kind: "boundaryAnd"; left: BoundaryExpr; right: BoundaryExpr }
  | { kind: "boundaryOr"; left: BoundaryExpr; right: BoundaryExpr };

/** A target region: either a call (e.g. founding_year_of("X")) or a bare identifier. */
export type RegionExpr =
  | { kind: "regionCall"; qname: string[]; args: Arg[] }
  | { kind: "regionIdent"; name: string };

export interface Arg {
  name: string | null;
  value: Expr;
}

export type ChainExpr =
  | { kind: "catalystRef"; qname: string[]; args: Arg[] }
  | { kind: "chainSeq"; steps: ChainExpr[] }
  | { kind: "chainPar"; branches: ChainExpr[] };

export type AdmitClause =
  | { kind: "converge"; otherwiseDecline: boolean }
  | { kind: "cond"; cond: CondExpr };

export interface CondExpr {
  kind: "cond";
  left: Expr;
  op: RelOp;
  right: Expr;
}

export type Expr =
  | { kind: "ident"; name: string }
  | { kind: "number"; value: number }
  | { kind: "string"; value: string }
  | { kind: "catalystRef"; qname: string[]; args: Arg[] }
  | { kind: "paren"; expr: Expr };

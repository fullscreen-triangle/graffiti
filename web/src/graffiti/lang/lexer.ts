/**
 * Lexical structure of Graffiti (.grf).
 *
 * Implements Definition (Token Classes) of semantic-causal-propagation.tex,
 * Section "Lexical Structure and Grammar".
 */

export type TokenType =
  | "KEYWORD"
  | "IDENT"
  | "NUMBER"
  | "STRING"
  | "OP"
  | "PUNCT"
  | "EOF";

export interface Token {
  type: TokenType;
  value: string;
  line: number;
  col: number;
}

const KEYWORDS = new Set([
  "floor",
  "project",
  "seek",
  "not",
  "toward",
  "via",
  "until",
  "converge",
  "diverge",
  "decline",
  "otherwise",
  "yield",
  "goal",
  "catalyst",
  "namespace",
  "input",
  "output",
  "claims",
  "coherence",
  "let",
  "if",
  "then",
  "else",
  "import",
  "module",
  "export",
  "assert",
  "in",
  "as",
  "with",
]);

/** Multi-character operators, checked longest-match-first. */
const MULTI_CHAR_OPERATORS = [":=", ">>", "||", ">=", "<=", "==", "->"];
const SINGLE_CHAR_OPERATORS = new Set([">", "<", ".", ":"]);
const PUNCTUATION = new Set(["(", ")", "{", "}", "[", "]", ","]);

export class LexError extends Error {
  constructor(message: string, readonly line: number, readonly col: number) {
    super(`${message} (line ${line}, col ${col})`);
    this.name = "LexError";
  }
}

function isLetter(ch: string): boolean {
  return /[A-Za-z_]/.test(ch);
}

function isDigit(ch: string): boolean {
  return /[0-9]/.test(ch);
}

function isIdentChar(ch: string): boolean {
  return /[A-Za-z0-9_]/.test(ch);
}

/**
 * Tokenise a .grf source string. Comments (`--` to end of line) and
 * whitespace are discarded; Graffiti is not layout-sensitive.
 */
export function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;

  const advance = (n = 1) => {
    for (let k = 0; k < n; k++) {
      if (source[i] === "\n") {
        line++;
        col = 1;
      } else {
        col++;
      }
      i++;
    }
  };

  while (i < source.length) {
    const ch = source[i]!;

    if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
      advance();
      continue;
    }

    if (ch === "-" && source[i + 1] === "-") {
      while (i < source.length && source[i] !== "\n") advance();
      continue;
    }

    if (ch === '"') {
      const startLine = line;
      const startCol = col;
      advance();
      let value = "";
      while (i < source.length && source[i] !== '"') {
        if (source[i] === "\\" && i + 1 < source.length) {
          const next = source[i + 1];
          const escapes: Record<string, string> = { n: "\n", t: "\t", r: "\r", '"': '"', "\\": "\\" };
          value += escapes[next!] ?? next!;
          advance(2);
        } else {
          value += source[i];
          advance();
        }
      }
      if (i >= source.length) {
        throw new LexError("Unterminated string literal", startLine, startCol);
      }
      advance(); // closing quote
      tokens.push({ type: "STRING", value, line: startLine, col: startCol });
      continue;
    }

    if (isDigit(ch)) {
      const startLine = line;
      const startCol = col;
      let value = "";
      while (i < source.length && isDigit(source[i]!)) {
        value += source[i];
        advance();
      }
      if (source[i] === "." && isDigit(source[i + 1] ?? "")) {
        value += source[i];
        advance();
        while (i < source.length && isDigit(source[i]!)) {
          value += source[i];
          advance();
        }
      }
      if (source[i] === "e" || source[i] === "E") {
        let lookahead = i + 1;
        let expStr: string = source[i]!;
        if (source[lookahead] === "+" || source[lookahead] === "-") {
          expStr += source[lookahead]!;
          lookahead++;
        }
        if (isDigit(source[lookahead] ?? "")) {
          while (lookahead < source.length && isDigit(source[lookahead]!)) {
            expStr += source[lookahead]!;
            lookahead++;
          }
          advance(expStr.length);
          value += expStr;
        }
      }
      tokens.push({ type: "NUMBER", value, line: startLine, col: startCol });
      continue;
    }

    if (isLetter(ch)) {
      const startLine = line;
      const startCol = col;
      let value = "";
      while (i < source.length && isIdentChar(source[i]!)) {
        value += source[i];
        advance();
      }
      tokens.push({
        type: KEYWORDS.has(value) ? "KEYWORD" : "IDENT",
        value,
        line: startLine,
        col: startCol,
      });
      continue;
    }

    const multi = MULTI_CHAR_OPERATORS.find((op) => source.startsWith(op, i));
    if (multi) {
      tokens.push({ type: "OP", value: multi, line, col });
      advance(multi.length);
      continue;
    }

    if (SINGLE_CHAR_OPERATORS.has(ch)) {
      tokens.push({ type: "OP", value: ch, line, col });
      advance();
      continue;
    }

    if (PUNCTUATION.has(ch)) {
      tokens.push({ type: "PUNCT", value: ch, line, col });
      advance();
      continue;
    }

    throw new LexError(`Unexpected character '${ch}'`, line, col);
  }

  tokens.push({ type: "EOF", value: "", line, col });
  return tokens;
}

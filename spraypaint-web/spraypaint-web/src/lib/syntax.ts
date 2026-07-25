// ── GRF syntax highlighting for the code panel ──

const KEYWORDS = new Set([
  "floor", "project", "seek", "not", "toward", "via", "until",
  "converge", "diverge", "decline", "yield", "goal", "catalyst",
  "namespace", "input", "output", "claims", "coherence", "let",
  "if", "then", "else", "import", "module", "export", "assert",
  "in", "as", "with", "otherwise",
]);

const TYPES = new Set(["Claim", "Region", "Catalyst", "Chain", "Residue"]);

interface Token {
  text: string;
  type: "keyword" | "type" | "comment" | "string" | "number" | "operator" | "identifier" | "whitespace" | "punctuation";
}

export function tokenizeLine(line: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < line.length) {
    // comment
    if (line[i] === "-" && line[i + 1] === "-") {
      tokens.push({ text: line.slice(i), type: "comment" });
      break;
    }

    // whitespace
    if (/\s/.test(line[i])) {
      let j = i;
      while (j < line.length && /\s/.test(line[j])) j++;
      tokens.push({ text: line.slice(i, j), type: "whitespace" });
      i = j;
      continue;
    }

    // string
    if (line[i] === '"') {
      let j = i + 1;
      while (j < line.length && line[j] !== '"') {
        if (line[j] === "\\") j++;
        j++;
      }
      tokens.push({ text: line.slice(i, j + 1), type: "string" });
      i = j + 1;
      continue;
    }

    // number
    if (/\d/.test(line[i])) {
      let j = i;
      while (j < line.length && /[\d.eE+\-]/.test(line[j])) j++;
      tokens.push({ text: line.slice(i, j), type: "number" });
      i = j;
      continue;
    }

    // operators and punctuation
    if (line[i] === ">" && line[i + 1] === ">") {
      tokens.push({ text: ">>", type: "operator" });
      i += 2;
      continue;
    }
    if (line[i] === "|" && line[i + 1] === "|") {
      tokens.push({ text: "||", type: "operator" });
      i += 2;
      continue;
    }
    if (line[i] === ":" && line[i + 1] === "=") {
      tokens.push({ text: ":=", type: "operator" });
      i += 2;
      continue;
    }
    if (line[i] === ">" && line[i + 1] === "=") {
      tokens.push({ text: ">=", type: "operator" });
      i += 2;
      continue;
    }
    if (line[i] === "<" && line[i + 1] === "=") {
      tokens.push({ text: "<=", type: "operator" });
      i += 2;
      continue;
    }
    if (line[i] === "=" && line[i + 1] === "=") {
      tokens.push({ text: "==", type: "operator" });
      i += 2;
      continue;
    }
    if (/[{}()\[\],:.<>=\->]/.test(line[i])) {
      tokens.push({ text: line[i], type: "punctuation" });
      i++;
      continue;
    }

    // identifier or keyword
    if (/[A-Za-z_]/.test(line[i])) {
      let j = i;
      while (j < line.length && /[A-Za-z0-9_]/.test(line[j])) j++;
      const word = line.slice(i, j);
      if (KEYWORDS.has(word)) {
        tokens.push({ text: word, type: "keyword" });
      } else if (TYPES.has(word)) {
        tokens.push({ text: word, type: "type" });
      } else {
        tokens.push({ text: word, type: "identifier" });
      }
      i = j;
      continue;
    }

    // fallback
    tokens.push({ text: line[i], type: "whitespace" });
    i++;
  }

  return tokens;
}

export const TOKEN_COLORS: Record<Token["type"], string> = {
  keyword: "#569cd6",
  type: "#4ec9b0",
  comment: "#6a9955",
  string: "#ce9178",
  number: "#b5cea8",
  operator: "#d4d4d4",
  identifier: "#9cdcfe",
  whitespace: "",
  punctuation: "#cccccc",
};

import React, { useCallback, useRef, useState } from "react";
import * as graffiti from "@/graffiti";

let nextCellId = 1;

function createCell(source = "") {
  return { id: nextCellId++, source, output: null, error: null };
}

/**
 * Each cell runs in its own function scope but shares one persistent
 * `scope` object across the notebook, so a function defined in one cell
 * is callable from a later cell -- the one property a REPL needs that a
 * single <textarea> does not give you for free. `scope.graffiti` exposes
 * the whole library (CatalystRegistry, runSource, parseProgram, ...) so a
 * cell can compile and execute .grf source directly; the function is
 * async so a cell body may `await` a seek's result.
 */
function runCell(source, scope) {
  const fn = new Function(
    "scope",
    `with (scope) { return (async function() { ${source}\n})(); }`,
  );
  return fn(scope);
}

function Cell({ cell, index, onChange, onRun, onDelete }) {
  const textareaRef = useRef(null);

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      onRun(cell.id);
    }
  };

  return (
    <div className="mb-4 border border-white/10 rounded-md">
      <div className="flex items-center justify-between px-3 py-1 text-xs text-white/40 border-b border-white/10">
        <span>[{index + 1}]</span>
        <div className="flex gap-3">
          <button
            className="hover:text-white/80"
            onClick={() => onRun(cell.id)}
            title="Run (Ctrl/Cmd+Enter)"
          >
            run
          </button>
          <button className="hover:text-white/80" onClick={() => onDelete(cell.id)}>
            delete
          </button>
        </div>
      </div>
      <textarea
        ref={textareaRef}
        className="w-full bg-transparent text-white/90 font-mono text-sm p-3 outline-none resize-none"
        rows={Math.max(3, cell.source.split("\n").length)}
        value={cell.source}
        onChange={(e) => onChange(cell.id, e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="write a function..."
        spellCheck={false}
      />
      {(cell.output !== null || cell.error !== null) && (
        <div className="px-3 py-2 border-t border-white/10 font-mono text-sm whitespace-pre-wrap">
          {cell.error ? (
            <span className="text-red-400">{cell.error}</span>
          ) : (
            <span className="text-white/70">{cell.output}</span>
          )}
        </div>
      )}
    </div>
  );
}

function formatOutput(value) {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export default function Repl() {
  const [cells, setCells] = useState([createCell()]);
  const scopeRef = useRef({ graffiti });

  const handleChange = useCallback((id, source) => {
    setCells((prev) => prev.map((c) => (c.id === id ? { ...c, source } : c)));
  }, []);

  const handleRun = useCallback(async (id) => {
    const logs = [];
    const scope = scopeRef.current;
    scope.console = { ...console, log: (...args) => logs.push(args.map(formatOutput).join(" ")) };

    let source;
    setCells((prev) => {
      source = prev.find((c) => c.id === id)?.source ?? "";
      return prev;
    });

    try {
      const result = await runCell(source, scope);
      const output = [...logs, result !== undefined ? formatOutput(result) : null]
        .filter((x) => x !== null && x !== "")
        .join("\n");
      setCells((prev) => prev.map((c) => (c.id === id ? { ...c, output, error: null } : c)));
    } catch (err) {
      setCells((prev) =>
        prev.map((c) =>
          c.id === id ? { ...c, output: null, error: String(err && err.message ? err.message : err) } : c,
        ),
      );
    }
  }, []);

  const handleDelete = useCallback((id) => {
    setCells((prev) => (prev.length > 1 ? prev.filter((c) => c.id !== id) : prev));
  }, []);

  const handleAddCell = useCallback(() => {
    setCells((prev) => [...prev, createCell()]);
  }, []);

  return (
    <div className="min-h-screen w-full bg-black px-6 py-10 sm:px-3">
      <div className="mx-auto max-w-3xl">
        {cells.map((cell, i) => (
          <Cell
            key={cell.id}
            cell={cell}
            index={i}
            onChange={handleChange}
            onRun={handleRun}
            onDelete={handleDelete}
          />
        ))}
        <button
          className="text-sm text-white/40 hover:text-white/80 font-mono"
          onClick={handleAddCell}
        >
          + add cell
        </button>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import { FILE_TREE } from "@/data/samples";
import FileExplorer from "./FileExplorer";
import CodeEditor from "./CodeEditor";
import OutputPanel from "./OutputPanel";
import {
  ExecutionState,
  ScriptDiff,
  applyDiff,
  executeScript,
} from "@/lib/engine";
import { UndoStack } from "@/lib/undo";
import {
  VscPlay,
  VscDebugStop,
  VscDiscard,
  VscRedo,
  VscTerminal,
  VscSettingsGear,
} from "react-icons/vsc";

export default function IDE() {
  const [activeFile, setActiveFile] = useState<string | null>("prompt.grf");
  const [code, setCode] = useState(
    FILE_TREE[0]?.children?.[0]?.children?.[0]?.content ?? ""
  );
  const [execState, setExecState] = useState<ExecutionState | null>(null);
  const [pendingDiff, setPendingDiff] = useState<ScriptDiff | null>(null);
  const [running, setRunning] = useState(false);
  const [col1Width, setCol1Width] = useState(200);
  const [col2Ratio, setCol2Ratio] = useState(0.45); // fraction of remaining space
  const undoStackRef = useRef(new UndoStack());

  // file selection
  const handleFileSelect = useCallback(
    (name: string, content: string) => {
      setActiveFile(name);
      setCode(content);
      setPendingDiff(null);
      // mark results as stale since script changed
      setExecState((prev) =>
        prev ? { ...prev, stale: true } : null
      );
      undoStackRef.current.push({
        script: content,
        timestamp: Date.now(),
        source: "editor",
        description: `Opened ${name}`,
      });
    },
    []
  );

  // code changes from editor
  const handleCodeChange = useCallback(
    (newCode: string) => {
      setCode(newCode);
      setPendingDiff(null);
      setExecState((prev) =>
        prev ? { ...prev, stale: true } : null
      );
      // debounce undo push
      undoStackRef.current.push({
        script: newCode,
        timestamp: Date.now(),
        source: "editor",
        description: "Manual edit",
      });
    },
    []
  );

  // crossfilter: chart interaction → script diff → auto-apply
  const handleCrossfilter = useCallback(
    (diff: ScriptDiff) => {
      const newCode = applyDiff(code, diff);
      if (newCode === code) return; // diff didn't match anything

      setCode(newCode);
      setPendingDiff(diff);
      setExecState((prev) =>
        prev ? { ...prev, stale: true } : null
      );
      undoStackRef.current.push({
        script: newCode,
        timestamp: Date.now(),
        source: "crossfilter",
        description: diff.description,
      });

      // clear the diff highlight after a moment
      setTimeout(() => setPendingDiff(null), 3000);
    },
    [code]
  );

  // run script
  const handleRun = useCallback(() => {
    setRunning(true);
    // simulate execution delay
    setTimeout(() => {
      const result = executeScript(code, execState ?? undefined);
      setExecState(result);
      setPendingDiff(null);
      setRunning(false);
    }, 800);
  }, [code, execState]);

  // undo / redo
  const handleUndo = useCallback(() => {
    const entry = undoStackRef.current.undo();
    if (entry) {
      setCode(entry.script);
      setExecState((prev) =>
        prev ? { ...prev, stale: true } : null
      );
    }
  }, []);

  const handleRedo = useCallback(() => {
    const entry = undoStackRef.current.redo();
    if (entry) {
      setCode(entry.script);
      setExecState((prev) =>
        prev ? { ...prev, stale: true } : null
      );
    }
  }, []);

  // keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      }
      if (
        (e.metaKey || e.ctrlKey) &&
        (e.key === "y" || (e.key === "z" && e.shiftKey))
      ) {
        e.preventDefault();
        handleRedo();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleRun();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleUndo, handleRedo, handleRun]);

  // column resizing
  const draggingRef = useRef<"col1" | "col2" | null>(null);

  const handleMouseDown = useCallback(
    (col: "col1" | "col2") => (e: React.MouseEvent) => {
      e.preventDefault();
      draggingRef.current = col;
    },
    []
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      if (draggingRef.current === "col1") {
        setCol1Width(Math.max(140, Math.min(400, e.clientX)));
      } else {
        const remaining = window.innerWidth - col1Width;
        const editorX = e.clientX - col1Width;
        setCol2Ratio(Math.max(0.2, Math.min(0.7, editorX / remaining)));
      }
    };
    const handleMouseUp = () => {
      draggingRef.current = null;
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [col1Width]);

  const remaining = `calc(100vw - ${col1Width}px)`;
  const col2Width = `calc(${remaining} * ${col2Ratio})`;
  const col3Width = `calc(${remaining} * ${1 - col2Ratio})`;

  return (
    <div className="h-screen flex flex-col bg-[#1e1e1e] text-[#cccccc] overflow-hidden select-none">
      {/* ── Title Bar ── */}
      <div className="h-[30px] bg-[#323233] flex items-center px-3 gap-3 text-[12px] text-[#cccccc] shrink-0 border-b border-[#1e1e1e]">
        <span className="font-semibold tracking-wide text-[#4fc1ff]">
          spraypaint
        </span>
        <span className="text-[#858585]">—</span>
        <span className="text-[#858585]">
          semantic causal propagation
        </span>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={handleUndo}
            className="p-1 hover:bg-[#ffffff15] rounded"
            title="Undo (Ctrl+Z)"
          >
            <VscDiscard className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleRedo}
            className="p-1 hover:bg-[#ffffff15] rounded"
            title="Redo (Ctrl+Shift+Z)"
          >
            <VscRedo className="w-3.5 h-3.5" />
          </button>
          <div className="w-px h-3 bg-[#555]" />
          <button
            onClick={handleRun}
            disabled={running}
            className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[12px] font-medium transition-colors ${
              running
                ? "bg-[#555] text-[#999] cursor-wait"
                : "bg-[#89d185] text-[#1e1e1e] hover:bg-[#9bdb97]"
            }`}
            title="Run (Ctrl+Enter)"
          >
            {running ? (
              <>
                <VscDebugStop className="w-3.5 h-3.5" />
                executing…
              </>
            ) : (
              <>
                <VscPlay className="w-3.5 h-3.5" />
                Run
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Column 1: File Explorer */}
        <div style={{ width: col1Width }} className="shrink-0">
          <FileExplorer
            tree={FILE_TREE}
            activeFile={activeFile}
            onSelect={handleFileSelect}
          />
        </div>

        {/* Resize handle */}
        <div
          className="w-[3px] bg-[#1e1e1e] hover:bg-[#007acc] cursor-col-resize transition-colors shrink-0"
          onMouseDown={handleMouseDown("col1")}
        />

        {/* Column 2: Code Editor */}
        <div style={{ width: col2Width }} className="shrink-0">
          <CodeEditor
            code={code}
            onChange={handleCodeChange}
            fileName={activeFile}
            pendingDiff={pendingDiff}
            stale={execState?.stale ?? false}
          />
        </div>

        {/* Resize handle */}
        <div
          className="w-[3px] bg-[#1e1e1e] hover:bg-[#007acc] cursor-col-resize transition-colors shrink-0"
          onMouseDown={handleMouseDown("col2")}
        />

        {/* Column 3: Output Panel */}
        <div style={{ width: col3Width }} className="min-w-0">
          <OutputPanel state={execState} onCrossfilter={handleCrossfilter} />
        </div>
      </div>

      {/* ── Status Bar ── */}
      <div className="h-[22px] bg-[#007acc] flex items-center px-3 gap-4 text-[11px] text-white shrink-0">
        <span className="flex items-center gap-1">
          <VscTerminal className="w-3 h-3" />
          GRF
        </span>
        {execState && (
          <>
            <span>
              M = {execState.invariants.committedCount}
            </span>
            <span>
              {execState.seeks.length} seek{execState.seeks.length !== 1 ? "s" : ""}
            </span>
            <span>
              {execState.scenes.filter((s) => s.allocated > 0).length} active scenes
            </span>
          </>
        )}
        <span className="ml-auto">
          {execState?.stale
            ? "⚠ script modified — results stale"
            : execState
            ? "✓ results current"
            : "ready"}
        </span>
      </div>
    </div>
  );
}

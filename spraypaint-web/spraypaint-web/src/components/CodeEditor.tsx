"use client";

import React, { useRef, useCallback, useEffect } from "react";
import { tokenizeLine, TOKEN_COLORS } from "@/lib/syntax";
import { ScriptDiff } from "@/lib/engine";

interface Props {
  code: string;
  onChange: (code: string) => void;
  fileName: string | null;
  pendingDiff: ScriptDiff | null;
  stale: boolean;
}

export default function CodeEditor({
  code,
  onChange,
  fileName,
  pendingDiff,
  stale,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  const syncScroll = useCallback(() => {
    if (textareaRef.current && highlightRef.current) {
      highlightRef.current.scrollTop = textareaRef.current.scrollTop;
      highlightRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }, []);

  useEffect(() => {
    syncScroll();
  }, [code, syncScroll]);

  const lines = code.split("\n");

  const renderHighlightedLine = (line: string, lineIdx: number) => {
    const tokens = tokenizeLine(line);
    const isDiffLine =
      pendingDiff && line.includes(pendingDiff.newText) && !line.includes(pendingDiff.oldText);

    return (
      <div
        key={lineIdx}
        className={`flex ${isDiffLine ? "bg-[#2ea04333]" : ""}`}
        style={{ minHeight: "20px", lineHeight: "20px" }}
      >
        {/* line number */}
        <span
          className="shrink-0 text-right pr-4 select-none text-[#858585] w-[50px]"
          style={{ fontSize: "13px", fontFamily: "inherit" }}
        >
          {lineIdx + 1}
        </span>
        {/* token spans */}
        <span style={{ fontSize: "13px" }}>
          {tokens.map((t, i) => (
            <span key={i} style={{ color: TOKEN_COLORS[t.type] || "#d4d4d4" }}>
              {t.text}
            </span>
          ))}
          {tokens.length === 0 && "\u00a0"}
        </span>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col bg-[#1e1e1e]">
      {/* tab bar */}
      <div className="flex items-center bg-[#252526] border-b border-[#1e1e1e] h-[35px] shrink-0">
        {fileName && (
          <div className="flex items-center gap-2 px-3 py-1 bg-[#1e1e1e] border-r border-[#252526] text-[13px] text-[#cccccc] h-full">
            <span className={`w-2 h-2 rounded-full ${stale ? "bg-[#cca700]" : "bg-[#89d185]"}`} />
            <span>{fileName}</span>
          </div>
        )}
        {stale && (
          <span className="ml-auto mr-3 text-[11px] text-[#cca700] tracking-wide">
            MODIFIED — results from previous run
          </span>
        )}
      </div>

      {/* diff banner */}
      {pendingDiff && (
        <div className="bg-[#007acc22] border-b border-[#007acc55] px-4 py-1 text-[12px] text-[#4fc1ff] flex items-center gap-2">
          <span className="font-mono">↻</span>
          <span>{pendingDiff.description}</span>
        </div>
      )}

      {/* editor area: overlay of textarea + highlight layer */}
      <div className="flex-1 relative overflow-hidden">
        {/* highlighted layer (visual) */}
        <div
          ref={highlightRef}
          className="absolute inset-0 overflow-auto pointer-events-none font-mono whitespace-pre py-2"
          aria-hidden="true"
        >
          {lines.map((line, i) => renderHighlightedLine(line, i))}
        </div>

        {/* editable textarea (invisible text, captures input) */}
        <textarea
          ref={textareaRef}
          value={code}
          onChange={(e) => onChange(e.target.value)}
          onScroll={syncScroll}
          spellCheck={false}
          className="absolute inset-0 w-full h-full resize-none bg-transparent text-transparent caret-[#aeafad] font-mono outline-none py-2 pl-[50px] overflow-auto"
          style={{
            fontSize: "13px",
            lineHeight: "20px",
            tabSize: 2,
            caretColor: "#aeafad",
          }}
        />
      </div>
    </div>
  );
}

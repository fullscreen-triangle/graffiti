"use client";

import React, { useState } from "react";
import { FileNode } from "@/data/samples";
import {
  VscChevronRight,
  VscChevronDown,
  VscFile,
  VscFolder,
  VscFolderOpened,
} from "react-icons/vsc";

interface Props {
  tree: FileNode[];
  activeFile: string | null;
  onSelect: (name: string, content: string) => void;
}

function TreeNode({
  node,
  depth,
  activeFile,
  onSelect,
}: {
  node: FileNode;
  depth: number;
  activeFile: string | null;
  onSelect: (name: string, content: string) => void;
}) {
  const [open, setOpen] = useState(depth < 2);
  const isActive = node.type === "file" && node.name === activeFile;

  if (node.type === "folder") {
    return (
      <div>
        <div
          className="flex items-center gap-1 px-1 py-[2px] cursor-pointer hover:bg-[#2a2d2e] select-none text-[13px]"
          style={{ paddingLeft: depth * 12 + 4 }}
          onClick={() => setOpen(!open)}
        >
          {open ? (
            <VscChevronDown className="w-4 h-4 shrink-0 text-[#cccccc]" />
          ) : (
            <VscChevronRight className="w-4 h-4 shrink-0 text-[#cccccc]" />
          )}
          {open ? (
            <VscFolderOpened className="w-4 h-4 shrink-0 text-[#dcb67a]" />
          ) : (
            <VscFolder className="w-4 h-4 shrink-0 text-[#dcb67a]" />
          )}
          <span className="text-[#cccccc] truncate">{node.name}</span>
        </div>
        {open &&
          node.children?.map((child, i) => (
            <TreeNode
              key={child.name + i}
              node={child}
              depth={depth + 1}
              activeFile={activeFile}
              onSelect={onSelect}
            />
          ))}
      </div>
    );
  }

  const isGrf = node.name.endsWith(".grf");

  return (
    <div
      className={`flex items-center gap-1 px-1 py-[2px] cursor-pointer select-none text-[13px] ${
        isActive ? "bg-[#094771]" : "hover:bg-[#2a2d2e]"
      }`}
      style={{ paddingLeft: depth * 12 + 4 }}
      onClick={() => onSelect(node.name, node.content ?? "")}
    >
      <span className="w-4 h-4 shrink-0" />
      <VscFile
        className={`w-4 h-4 shrink-0 ${
          isGrf ? "text-[#519aba]" : "text-[#8c8c8c]"
        }`}
      />
      <span className="text-[#cccccc] truncate">{node.name}</span>
    </div>
  );
}

export default function FileExplorer({ tree, activeFile, onSelect }: Props) {
  return (
    <div className="h-full flex flex-col bg-[#252526] border-r border-[#1e1e1e]">
      {/* sidebar header */}
      <div className="flex items-center justify-between px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-[#bbbbbb] border-b border-[#1e1e1e]">
        Explorer
      </div>

      {/* tree */}
      <div className="flex-1 overflow-y-auto py-1 scrollbar-thin">
        {tree.map((node, i) => (
          <TreeNode
            key={node.name + i}
            node={node}
            depth={0}
            activeFile={activeFile}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}

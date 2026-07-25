// ── Unified undo stack: every semantic action produces a script snapshot ──

export interface UndoEntry {
  script: string;
  timestamp: number;
  source: "prompt" | "editor" | "crossfilter";
  description: string;
}

export class UndoStack {
  private stack: UndoEntry[] = [];
  private cursor = -1;
  private maxSize = 200;

  push(entry: UndoEntry) {
    // discard anything after cursor (we branched)
    this.stack = this.stack.slice(0, this.cursor + 1);
    this.stack.push(entry);
    if (this.stack.length > this.maxSize) {
      this.stack.shift();
    }
    this.cursor = this.stack.length - 1;
  }

  undo(): UndoEntry | null {
    if (this.cursor <= 0) return null;
    this.cursor--;
    return this.stack[this.cursor];
  }

  redo(): UndoEntry | null {
    if (this.cursor >= this.stack.length - 1) return null;
    this.cursor++;
    return this.stack[this.cursor];
  }

  current(): UndoEntry | null {
    return this.stack[this.cursor] ?? null;
  }

  history(): UndoEntry[] {
    return this.stack.slice(0, this.cursor + 1);
  }

  canUndo(): boolean {
    return this.cursor > 0;
  }

  canRedo(): boolean {
    return this.cursor < this.stack.length - 1;
  }
}

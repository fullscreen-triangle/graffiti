// ── Undo stack over whole values ──
//
// Generic over the value type; used as `UndoStack<AskQuery>`. Storing whole
// values rather than diffs is what makes undo exact: restoring is an assignment,
// not a reverse text-replace that has to find its own edit again.

export interface UndoEntry<T> {
  value: T;
  timestamp: number;
  source: "init" | "editor" | "gesture";
  description: string;
}

export class UndoStack<T> {
  private stack: UndoEntry<T>[] = [];
  private cursor = -1;
  private readonly maxSize: number;

  constructor(maxSize = 200) {
    this.maxSize = Math.max(2, maxSize);
  }

  push(entry: UndoEntry<T>) {
    // Discard the redo branch — we just took a different path.
    this.stack = this.stack.slice(0, this.cursor + 1);
    this.stack.push(entry);
    if (this.stack.length > this.maxSize) {
      // Dropping the oldest entry shifts every index down by one, so the cursor
      // must follow. Without this it points one entry too far right and undo
      // silently skips a step once the cap is reached.
      const overflow = this.stack.length - this.maxSize;
      this.stack = this.stack.slice(overflow);
      this.cursor = Math.max(0, this.cursor - overflow);
    }
    this.cursor = this.stack.length - 1;
  }

  undo(): UndoEntry<T> | null {
    if (this.cursor <= 0) return null;
    this.cursor--;
    return this.stack[this.cursor];
  }

  redo(): UndoEntry<T> | null {
    if (this.cursor >= this.stack.length - 1) return null;
    this.cursor++;
    return this.stack[this.cursor];
  }

  current(): UndoEntry<T> | null {
    return this.stack[this.cursor] ?? null;
  }

  /** Entries up to and including the cursor, oldest first. */
  history(): UndoEntry<T>[] {
    return this.stack.slice(0, this.cursor + 1);
  }

  canUndo(): boolean {
    return this.cursor > 0;
  }

  canRedo(): boolean {
    return this.cursor < this.stack.length - 1;
  }
}

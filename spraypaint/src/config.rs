//! Index configuration: which files enter the corpus, which directories are
//! skipped, and the chunking parameters. Defaults mirror `purpose`'s extension
//! and ignore lists, widened for full-text (prose + config) indexing.

/// Static configuration for a build. Constructed from defaults + CLI flags.
#[derive(Debug, Clone)]
pub struct SprayConfig {
    /// Code extensions (definition-dense source).
    pub source_exts: Vec<String>,
    /// Prose extensions (markdown, plain text).
    pub prose_exts: Vec<String>,
    /// LaTeX / bibliography extensions.
    pub latex_exts: Vec<String>,
    /// Directory names skipped anywhere in the tree, on top of `.gitignore`.
    pub skip_dirs: Vec<String>,
    /// Files larger than this are skipped (bytes).
    pub max_file_bytes: u64,
    /// Passage window length in lines.
    pub window: usize,
    /// Overlap between consecutive windows in lines.
    pub overlap: usize,
}

fn to_vec(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

impl Default for SprayConfig {
    fn default() -> Self {
        SprayConfig {
            source_exts: to_vec(&[
                "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp",
                "cs", "rb", "php", "swift", "kt", "scala", "toml", "yaml", "yml", "json",
                "sh", "sql",
            ]),
            prose_exts: to_vec(&["md", "txt", "rst", "org"]),
            latex_exts: to_vec(&["tex", "bib"]),
            skip_dirs: to_vec(&[
                ".git",
                "node_modules",
                "target",
                "build",
                "dist",
                "out",
                "vendor",
                ".spraypaint",
                ".purpose",
                "__pycache__",
                ".venv",
                "venv",
                ".next",
                ".nuxt",
                ".cache",
                "coverage",
                ".idea",
                ".vscode",
            ]),
            max_file_bytes: 2 * 1024 * 1024,
            window: 40,
            overlap: 10,
        }
    }
}

impl SprayConfig {
    /// Kind of a file by extension, or `None` if it should not be indexed.
    pub fn kind_for_ext(&self, ext: &str) -> Option<FileKind> {
        let ext = ext.to_lowercase();
        if self.source_exts.iter().any(|e| e == &ext) {
            Some(FileKind::Code)
        } else if self.prose_exts.iter().any(|e| e == &ext) {
            Some(FileKind::Prose)
        } else if self.latex_exts.iter().any(|e| e == &ext) {
            Some(FileKind::Latex)
        } else {
            None
        }
    }

    /// Stride between window starts.
    pub fn stride(&self) -> usize {
        self.window.saturating_sub(self.overlap).max(1)
    }
}

/// Broad content class of an indexed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Code,
    Prose,
    Latex,
}

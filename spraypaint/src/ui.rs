//! The embedded web UI.
//!
//! The statically-exported Next.js app is compiled into the executable by
//! `rust-embed`, so `spraypaint serve` needs no Node, no network, and no files
//! beside the binary.
//!
//! Serving only from the embed is also the path-traversal defence, and it is a
//! structural one rather than a filter: there is no filesystem read here to
//! escape from. A request for `../../../etc/passwd` is looked up as a key in a
//! fixed compile-time map, misses, and 404s. No amount of `%2e%2e%2f` encoding
//! changes that, because nothing in this module ever touches a path on disk.

use rust_embed::RustEmbed;

/// The exported UI. Built by `npm run export` into `ui/dist/`.
///
/// The folder is committed with a `.gitkeep` so this compiles in a fresh clone
/// with no UI built. When it is empty the binary still runs — see
/// [`is_empty`] and the plain-text notice `serve` falls back to. A server that
/// refused to start without a UI would be strictly worse: the JSON API is
/// useful on its own, and the user would get a mystery instead of a diagnosis.
#[derive(RustEmbed)]
#[folder = "ui/dist/"]
pub struct Ui;

/// Was a UI embedded in this build?
pub fn is_empty() -> bool {
    Ui::iter().next().is_none()
}

/// Content type for a path, by extension.
///
/// A hand-written match rather than `mime_guess` — which is in the tree anyway,
/// pulled in transitively by `rust-embed`, so this is not a dependency saving.
/// It is a correctness one: we control exactly which file types a Next.js static
/// export emits, so this list is complete for our input, and it lets us pin two
/// things `mime_guess` will not. `charset=utf-8` on every text type, because the
/// export is UTF-8 and a browser left to guess may decode it as Latin-1; and
/// `application/octet-stream` for anything unrecognised, which together with
/// `nosniff` is what stops an unexpected file being executed as script.
pub fn content_type(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("");
    match ext {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "txt" => "text/plain; charset=utf-8",
        // Unknown types are served as bytes, never sniffed. Combined with
        // `X-Content-Type-Options: nosniff` this means an unexpected file
        // downloads rather than executing as script.
        _ => "application/octet-stream",
    }
}

/// Resolve a request path to an embedded file.
///
/// Handles the three shapes a Next.js static export produces:
///
///   * `/`            -> `index.html`
///   * `/foo/`        -> `foo/index.html`   (`trailingSlash: true`)
///   * `/foo/bar.css` -> `foo/bar.css`      (verbatim)
///
/// Returns the resolved key and its bytes. `None` means no such asset — the
/// caller decides whether that is a 404 or an SPA fallback.
pub fn resolve(req_path: &str) -> Option<(String, std::borrow::Cow<'static, [u8]>)> {
    let p = req_path.trim_start_matches('/');

    // Directory-ish requests map to their index.html.
    let candidates: [String; 3] = if p.is_empty() {
        [
            "index.html".to_string(),
            "index.html".to_string(),
            "index.html".to_string(),
        ]
    } else if p.ends_with('/') {
        let base = p.trim_end_matches('/');
        [
            format!("{base}/index.html"),
            p.to_string(),
            base.to_string(),
        ]
    } else {
        [
            p.to_string(),
            format!("{p}/index.html"),
            format!("{p}.html"),
        ]
    };

    for c in candidates {
        if let Some(f) = Ui::get(&c) {
            return Some((c, f.data));
        }
    }
    None
}

/// Cache policy for an embedded asset.
///
/// Next.js emits hashed filenames under `_next/static/`, so those are immutable
/// and can be cached hard. Everything else — above all `index.html` — must be
/// revalidated, or a user who upgrades the binary keeps being served the old
/// shell out of their browser cache and sees an app that does not match the
/// API it is talking to.
pub fn cache_control(path: &str) -> &'static str {
    if path.starts_with("_next/static/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_covers_the_export_and_defaults_safely() {
        assert_eq!(content_type("a/b/index.html"), "text/html; charset=utf-8");
        assert_eq!(content_type("x.woff2"), "font/woff2");
        assert_eq!(content_type("noextension"), "application/octet-stream");
        // An unknown extension must not fall through to a text type — with
        // nosniff set, octet-stream is what stops a stray file executing.
        assert_eq!(content_type("payload.wasm"), "application/octet-stream");
    }

    #[test]
    fn hashed_assets_are_immutable_and_the_shell_is_not() {
        assert!(cache_control("_next/static/chunks/x.js").contains("immutable"));
        assert_eq!(cache_control("index.html"), "no-cache");
    }
}

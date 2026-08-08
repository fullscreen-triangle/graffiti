//! `spraypaint serve` — the local HTTP server.
//!
//! This inverts the usual topology. A hosted web page cannot spawn a process on
//! a visitor's machine, so the binary hosts the UI instead: same origin, no
//! CORS, no bridge, and the "server" is a program the user already trusts
//! enough to have run. The UI is embedded (see [`crate::ui`]), so this works
//! offline with no Node and no network.
//!
//! Everything that touches the index, the lock, or the count goes through
//! [`crate::actions`] — the same functions the CLI calls, in the same order.
//! Nothing invariant-critical is reimplemented here; this module is transport,
//! status codes, and security headers.

mod api;
mod index_job;

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tiny_http::{Header, Request, Response, Server};

use crate::ui;

pub use index_job::IndexJob;

/// Runtime configuration for the server.
pub struct ServeConfig {
    pub root: PathBuf,
    pub port: u16,
    pub host: String,
    pub allow_index: bool,
    pub no_cache: bool,
    pub open: bool,
}

/// Shared, immutable-after-start state handed to every worker thread.
pub struct ServerState {
    pub root: PathBuf,
    pub allow_index: bool,
    pub no_cache: bool,
    /// The set of `Host` header values this server will answer to.
    pub allowed_hosts: Vec<String>,
    pub job: IndexJob,
    /// Cached index, keyed by (mtime, len) of `index.json`. See `api::load_cached`.
    pub cache: std::sync::Mutex<Option<api::CachedIndex>>,
}

/// Maximum request body we will read, in bytes.
///
/// An `AskQuery` is a few hundred bytes; 64 KiB is generous. The cap exists so
/// a malformed or hostile `Content-Length` cannot make us allocate without
/// bound on a machine the user is also working on.
const MAX_BODY: usize = 64 * 1024;

/// Number of worker threads.
///
/// Fixed at 4 rather than `available_parallelism()`: the work is serialised by
/// file locks anyway, the UI is single-user, and a small pool bounds how many
/// threads can pile up blocked behind an exclusive index lock. Excess requests
/// wait in the accept backlog, which is correct backpressure rather than
/// unbounded thread growth.
const WORKERS: usize = 4;

/// Start the server and block forever.
pub fn run(cfg: ServeConfig) -> Result<()> {
    let ip: IpAddr = cfg
        .host
        .parse()
        .with_context(|| format!("--host is not a valid IP address: {}", cfg.host))?;
    let addr = SocketAddr::new(ip, cfg.port);

    let state = Arc::new(ServerState {
        root: cfg.root.clone(),
        allow_index: cfg.allow_index,
        no_cache: cfg.no_cache,
        allowed_hosts: allowed_hosts(&cfg.host, cfg.port),
        job: IndexJob::new(),
        cache: std::sync::Mutex::new(None),
    });

    let server = Server::http(addr).map_err(|e| {
        // Do NOT silently pick another port. `--open` and anything the user
        // copy-pasted would then point at the wrong process — and if that other
        // process is a different spraypaint, at the wrong repo.
        anyhow!(
            "cannot bind {addr}: {e}\n\n\
             If the port is already in use, pick another explicitly:\n    \
             spraypaint serve --port {}",
            cfg.port.wrapping_add(1)
        )
    })?;

    let url = format!("http://{}:{}/", display_host(&cfg.host), cfg.port);
    eprintln!("spraypaint serving {} at {url}", cfg.root.display());
    if ui::is_empty() {
        eprintln!(
            "note: no UI embedded in this build — the JSON API under /api works, \
             but / returns a plain-text notice."
        );
    }
    if !cfg.allow_index {
        eprintln!("note: POST /api/index is disabled; pass --allow-index to enable it.");
    }
    if cfg.open {
        open_browser(&url);
    }
    eprintln!("press Ctrl-C to stop.");

    let server = Arc::new(server);
    let mut handles = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        handles.push(std::thread::spawn(move || loop {
            match server.recv() {
                Ok(req) => handle(&state, req),
                Err(_) => break,
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// Which `Host` values this server answers to.
fn allowed_hosts(bind: &str, port: u16) -> Vec<String> {
    let mut names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "[::1]".to_string(),
        "::1".to_string(),
    ];
    if !names.iter().any(|n| n == bind) {
        names.push(bind.to_string());
    }
    // Both with and without the port, since browsers include a non-default port
    // in `Host` but omit it on 80/443.
    let mut out = Vec::with_capacity(names.len() * 2);
    for n in &names {
        out.push(format!("{n}:{port}"));
        out.push(n.clone());
    }
    out
}

fn display_host(bind: &str) -> String {
    match bind {
        "0.0.0.0" | "::" => "localhost".to_string(),
        b if b.contains(':') => format!("[{b}]"),
        b => b.to_string(),
    }
}

/// Is this request's `Host` header one we serve?
///
/// **This is the single most important control in the server.** Binding to
/// 127.0.0.1 keeps other machines out, but it does *not* keep out a web page
/// the user visits: any site can make their browser issue requests to
/// `127.0.0.1:7373`, and with DNS rebinding it can do so from an origin that
/// passes a same-origin check. What such a request cannot do is forge the
/// `Host` header — the browser sets it from the URL. So pinning `Host` to the
/// loopback names is what actually prevents a hostile page from reading the
/// user's source tree through this API.
///
/// Missing `Host` is rejected too: HTTP/1.1 requires it, and the only clients
/// that omit it are not browsers.
fn host_allowed(state: &ServerState, req: &Request) -> bool {
    let host = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Host"))
        .map(|h| h.value.as_str().to_ascii_lowercase());
    match host {
        Some(h) => state.allowed_hosts.iter().any(|a| a.eq_ignore_ascii_case(&h)),
        None => false,
    }
}

fn header(k: &str, v: &str) -> Header {
    // Both sides are compile-time-known ASCII at every call site.
    Header::from_bytes(k.as_bytes(), v.as_bytes()).expect("static header is valid")
}

/// Headers applied to every response.
///
/// Note what is *absent*: there is no `Access-Control-Allow-Origin`. The UI is
/// same-origin and does not need it, and adding `*` would hand back exactly the
/// cross-origin read access the `Host` check exists to deny.
fn security_headers() -> Vec<Header> {
    vec![
        header("X-Content-Type-Options", "nosniff"),
        header("Referrer-Policy", "no-referrer"),
        // Self-only, and no framing: a hostile page cannot embed the UI and
        // drive it via clickjacking.
        header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; \
             img-src 'self' data:; font-src 'self' data:; \
             connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        ),
    ]
}

fn respond(req: Request, mut resp: Response<std::io::Cursor<Vec<u8>>>) {
    for h in security_headers() {
        resp.add_header(h);
    }
    let _ = req.respond(resp);
}

fn text(req: Request, code: u16, body: &str) {
    let resp = Response::from_string(body)
        .with_status_code(code)
        .with_header(header("Content-Type", "text/plain; charset=utf-8"));
    respond(req, resp);
}

pub(crate) fn json(req: Request, code: u16, body: String) {
    let resp = Response::from_string(body)
        .with_status_code(code)
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        // Never let an intermediary or the browser replay a response that
        // committed an act — a cached /api/ask would double-count.
        .with_header(header("Cache-Control", "no-store"));
    respond(req, resp);
}

/// Route one request.
fn handle(state: &ServerState, req: Request) {
    if !host_allowed(state, &req) {
        // 421 Misdirected Request: "this server is not authoritative for that
        // host". Semantically exact, and distinguishable from 403 in a log.
        text(
            req,
            421,
            "Misdirected request: this server only answers to localhost.\n\
             If you reached this from a web page, that page was trying to read \
             your local index.\n",
        );
        return;
    }

    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();

    if path == "/api" || path.starts_with("/api/") {
        api::route(state, req, &path, &url);
        return;
    }
    serve_static(req, &path);
}

fn serve_static(req: Request, path: &str) {
    if ui::is_empty() {
        text(
            req,
            200,
            "spraypaint serve is running, but no web UI was embedded in this build.\n\n\
             The JSON API is available under /api — try /api/health.\n\n\
             To get the UI, install a release binary, or build it yourself:\n  \
             cd spraypaint-web/spraypaint-web && npm ci && npm run export\n  \
             cp -r out/. ../../spraypaint/ui/dist/ && cargo build --release\n",
        );
        return;
    }
    match ui::resolve(path) {
        Some((key, bytes)) => {
            let ct = ui::content_type(&key);
            let cc = ui::cache_control(&key);
            let resp = Response::from_data(bytes.into_owned())
                .with_header(header("Content-Type", ct))
                .with_header(header("Cache-Control", cc));
            respond(req, resp);
        }
        None => text(req, 404, "not found\n"),
    }
}

/// Read a request body, capped at [`MAX_BODY`].
pub(crate) fn read_body(req: &mut Request) -> Result<String> {
    use std::io::Read;
    let len = req.body_length().unwrap_or(0);
    if len > MAX_BODY {
        return Err(anyhow!("request body too large ({len} bytes, max {MAX_BODY})"));
    }
    let mut buf = String::new();
    req.as_reader()
        .take(MAX_BODY as u64)
        .read_to_string(&mut buf)
        .context("reading request body")?;
    Ok(buf)
}

/// Best-effort browser launch for `--open`. Failure is not an error: the URL is
/// already printed, and a headless or unusual environment is not a reason to
/// refuse to serve.
fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    let r = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if r.is_err() {
        eprintln!("(could not open a browser automatically — open {url} yourself)");
    }
}

/// Resolve the serving root, failing early if there is no index yet.
///
/// Called once at startup so the user is told at the prompt rather than by a
/// 404 in a browser tab. A missing index is a warning, not a fatal error: with
/// `--allow-index` the UI can build one.
pub fn preflight(root: &Path, allow_index: bool) {
    if !crate::root::index_path(root).exists() {
        eprintln!(
            "warning: no index at {} — run `spraypaint index` first{}",
            crate::root::index_path(root).display(),
            if allow_index {
                ", or build one from the UI."
            } else {
                "."
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_for(bind: &str, port: u16) -> ServerState {
        ServerState {
            root: PathBuf::from("."),
            allow_index: false,
            no_cache: false,
            allowed_hosts: allowed_hosts(bind, port),
            job: IndexJob::new(),
            cache: std::sync::Mutex::new(None),
        }
    }

    /// The DNS-rebinding guard. If this test ever goes green for `evil.com`,
    /// any website the user visits can read their indexed source tree.
    #[test]
    fn only_loopback_hosts_are_allowed() {
        let s = state_for("127.0.0.1", 7373);
        let ok = |h: &str| s.allowed_hosts.iter().any(|a| a.eq_ignore_ascii_case(h));

        assert!(ok("localhost:7373"));
        assert!(ok("127.0.0.1:7373"));
        assert!(ok("LOCALHOST:7373"), "Host is case-insensitive");
        assert!(ok("[::1]:7373"));

        assert!(!ok("evil.com"));
        assert!(!ok("evil.com:7373"));
        // The classic rebinding shape: an attacker hostname that resolves to
        // loopback. The address is fine; the *name* is what we reject.
        assert!(!ok("spraypaint.evil.com:7373"));
        // A different port is a different server.
        assert!(!ok("localhost:9999"));
        // Suffix/prefix confusion must not slip through.
        assert!(!ok("notlocalhost:7373"));
        assert!(!ok("localhost.evil.com:7373"));
    }

    #[test]
    fn a_custom_bind_address_is_allowed_but_does_not_widen_others() {
        let s = state_for("192.168.1.5", 7373);
        let ok = |h: &str| s.allowed_hosts.iter().any(|a| a.eq_ignore_ascii_case(h));
        assert!(ok("192.168.1.5:7373"));
        assert!(ok("localhost:7373"), "loopback still works when bound wide");
        assert!(!ok("evil.com:7373"));
    }
}

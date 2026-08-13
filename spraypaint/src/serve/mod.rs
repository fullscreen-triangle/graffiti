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
pub mod pairing;

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tiny_http::{Header, Request, Response, Server};

use crate::ui;

pub use index_job::IndexJob;
pub use pairing::Token;

/// Runtime configuration for the server.
pub struct ServeConfig {
    pub root: PathBuf,
    pub port: u16,
    pub host: String,
    pub allow_index: bool,
    pub no_cache: bool,
    pub open: bool,
    /// Origin to pair with, already normalised. `None` = same-origin only.
    pub pair_origin: Option<String>,
}

/// Shared, immutable-after-start state handed to every worker thread.
pub struct ServerState {
    pub root: PathBuf,
    pub allow_index: bool,
    pub no_cache: bool,
    /// The set of `Host` header values this server will answer to.
    pub allowed_hosts: Vec<String>,
    /// The single paired origin, if `--pair` was given.
    pub pair_origin: Option<String>,
    /// The bearer token required on `/api/*` when paired. `None` = not paired,
    /// in which case same-origin requests need no credential at all.
    pub token: Option<Token>,
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

    // Mint the token only when pairing. An unpaired server has no credential to
    // leak and no code path that would check one.
    let token = match cfg.pair_origin.as_deref() {
        Some(_) => Some(Token::mint().context("could not generate a pairing token")?),
        None => None,
    };

    let state = Arc::new(ServerState {
        root: cfg.root.clone(),
        allow_index: cfg.allow_index,
        no_cache: cfg.no_cache,
        allowed_hosts: allowed_hosts(&cfg.host, cfg.port),
        pair_origin: cfg.pair_origin.clone(),
        token,
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
    if let (Some(origin), Some(tok)) = (cfg.pair_origin.as_deref(), state.token.as_ref()) {
        print_pairing_banner(origin, tok, &cfg.host, cfg.port);
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

/// Print the pairing token and what it authorises.
///
/// Written to stderr, like every other status line here, so `spraypaint serve >
/// file` does not silently capture the secret into a file the user forgot about.
///
/// The wording states the grant in terms of consequences rather than mechanism.
/// A user pasting a token into a web page is making a security decision, and
/// "lets that page read any file in this repo" is the fact they need — not
/// "enables CORS for that origin".
fn print_pairing_banner(origin: &str, token: &Token, host: &str, port: u16) {
    eprintln!(
        "\n\
         ── pairing ─────────────────────────────────────────────────────────\n\
         Paired with: {origin}\n\
         \n\
         Token (paste into that page; it is not stored anywhere):\n\
         \n    \
         {}\n\
         \n\
         That page will be able to read the content of any indexed file in this\n\
         repo and to increment the committed count. Only paste it into {origin}.\n\
         The token lives in this process only — stopping the server revokes it,\n\
         and restarting mints a new one.\n\
         \n\
         Server URL for the pairing form:  http://{}:{}\n\
         ────────────────────────────────────────────────────────────────────",
        token.reveal(),
        display_host(host),
        port,
    );
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

/// Read one request header, case-insensitively.
///
/// `name` is `&'static str` because `tiny_http`'s `equiv` requires it; every
/// call site passes a literal anyway.
fn header_value(req: &Request, name: &'static str) -> Option<String> {
    req.headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str().to_string())
}

/// Is this request from the paired origin?
///
/// Compared against the single normalised origin from `--pair` — never a
/// wildcard, never a suffix match. A suffix test (`ends_with(".vercel.app")`)
/// would pair every deployment on a shared domain, including other people's.
fn is_paired_origin(state: &ServerState, req: &Request) -> bool {
    let (Some(paired), Some(origin)) = (state.pair_origin.as_deref(), header_value(req, "Origin"))
    else {
        return false;
    };
    origin.trim().eq_ignore_ascii_case(paired)
}

/// Does this request carry the pairing token?
///
/// Accepts `Authorization: Bearer <token>` only. A `?token=` query parameter is
/// deliberately *not* supported: URLs end up in browser history, in `Referer`
/// headers, and in server logs, and this token is the only thing standing
/// between a hostile page and the user's source tree.
fn token_ok(state: &ServerState, req: &Request) -> bool {
    let Some(tok) = state.token.as_ref() else {
        // Not paired: no token exists, so nothing can present one. Same-origin
        // requests are authorised by the Host check alone, exactly as before.
        return false;
    };
    let Some(auth) = header_value(req, "Authorization") else {
        return false;
    };
    let candidate = match auth.split_once(' ') {
        Some((scheme, rest)) if scheme.eq_ignore_ascii_case("Bearer") => rest.trim(),
        _ => return false,
    };
    tok.matches(candidate)
}

/// CORS headers for a paired request.
///
/// Echoes the one paired origin rather than sending `*`, and pairs it with
/// `Vary: Origin` so a cache can never serve an allow-header minted for the
/// paired origin to some other origin's request.
///
/// `Access-Control-Allow-Private-Network` is what makes this work at all from a
/// public HTTPS page in Chromium: since Chrome 142, Local Network Access gates
/// public → loopback requests behind a preflight carrying
/// `Access-Control-Request-Private-Network: true`, which must be answered with
/// this header or the fetch fails with a bare `TypeError`.
///
/// Note `Allow-Credentials` is absent. The token is an explicit header, so
/// cookies and TLS client certs are never needed — and omitting it means a
/// hostile page cannot ride the user's ambient credentials.
fn cors_headers(origin: &str) -> Vec<Header> {
    vec![
        header("Access-Control-Allow-Origin", origin),
        header("Vary", "Origin"),
        header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        header("Access-Control-Allow-Headers", "Authorization, Content-Type"),
        header("Access-Control-Allow-Private-Network", "true"),
        header(
            "Access-Control-Max-Age",
            &pairing::PREFLIGHT_MAX_AGE.as_secs().to_string(),
        ),
    ]
}

fn header(k: &str, v: &str) -> Header {
    // Both sides are compile-time-known ASCII at every call site.
    Header::from_bytes(k.as_bytes(), v.as_bytes()).expect("static header is valid")
}

/// Headers applied to every response.
///
/// There is deliberately no unconditional `Access-Control-Allow-Origin` here:
/// `*` would hand back exactly the cross-origin read access the `Host` check
/// exists to deny. When `--pair` is in effect, [`cors_headers`] adds a header
/// naming that one origin, and only on requests that actually came from it.
///
/// The CSP below governs the *embedded* UI, which is same-origin by
/// construction, so `connect-src 'self'` stays correct even when paired — the
/// paired page is served by Vercel and carries its own CSP, not this one.
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

/// Respond, adding CORS headers when the request came from the paired origin.
///
/// **Every** response leaving this server goes through here, including the ones
/// built by `api.rs`. That is deliberate and load-bearing: a browser discards a
/// cross-origin response whose `Access-Control-Allow-Origin` is missing *even
/// after a successful preflight*, so a single handler that bypassed this would
/// fail in a browser while looking perfectly fine to `curl`.
///
/// The echo is conditional on the request's own `Origin` matching the paired
/// one — computed from the request, never reflected from an arbitrary value the
/// client sent. Reflecting whatever `Origin` arrives is the classic way this
/// goes wrong, and is equivalent to `*`.
fn respond(state: &ServerState, req: Request, mut resp: Response<std::io::Cursor<Vec<u8>>>) {
    for h in security_headers() {
        resp.add_header(h);
    }
    if is_paired_origin(state, &req) {
        if let Some(o) = state.pair_origin.as_deref() {
            for h in cors_headers(o) {
                resp.add_header(h);
            }
        }
    }
    let _ = req.respond(resp);
}

fn text(state: &ServerState, req: Request, code: u16, body: &str) {
    let resp = Response::from_string(body)
        .with_status_code(code)
        .with_header(header("Content-Type", "text/plain; charset=utf-8"));
    respond(state, req, resp);
}

pub(crate) fn json(state: &ServerState, req: Request, code: u16, body: String) {
    let resp = Response::from_string(body)
        .with_status_code(code)
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        // Never let an intermediary or the browser replay a response that
        // committed an act — a cached /api/ask would double-count.
        .with_header(header("Cache-Control", "no-store"));
    respond(state, req, resp);
}

/// Route one request.
///
/// ## The authorisation ladder, in order
///
/// A request reaches the API by satisfying **exactly one** of two disjoint
/// paths, and the order below is deliberate:
///
/// 1. **Paired origin + valid token.** Cross-origin, so the `Host` check cannot
///    apply — the token is the credential. Checked first because a paired
///    request legitimately carries a `Host` of `127.0.0.1:7373` *and* an
///    `Origin` of the deployed site, and we must not reject it for the latter.
/// 2. **Same-origin loopback.** No `Origin` header (or a loopback one), `Host`
///    pinned to a loopback name. This is the embedded UI, and it needs no token.
///
/// Anything else is refused. Critically, a request from an *unpaired* origin is
/// refused even when its `Host` is `localhost` — that is precisely the DNS
/// rebinding shape, and the `Origin` header is what exposes it.
fn handle(state: &ServerState, req: Request) {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();
    let is_api = path == "/api" || path.starts_with("/api/");

    // CORS preflight. Must be answered before any auth check: a preflight
    // deliberately carries no `Authorization` header (that is the header it is
    // asking permission to send), so requiring a token here would make every
    // paired request fail at the first hop.
    if req.method() == &tiny_http::Method::Options {
        return cors_preflight(state, req);
    }

    let origin = header_value(&req, "Origin");
    let paired = is_paired_origin(state, &req);

    // A cross-origin request from anywhere we have not paired with is rejected
    // outright — including when it targets a static asset. Serving the UI to a
    // hostile page is harmless in itself, but answering at all tells that page
    // the server exists and which port it is on.
    if let Some(o) = origin.as_deref() {
        if !paired && !is_loopback_origin(o) {
            return text(
                state,
                req,
                403,
                "Forbidden: this origin is not paired with this server.\n\n\
                 A page you are visiting tried to read your local spraypaint index.\n\
                 If that was you, restart the server with:\n    \
                 spraypaint serve --pair <that-origin>\n",
            );
        }
    }

    if !host_allowed(state, &req) {
        // 421 Misdirected Request: "this server is not authoritative for that
        // host". Semantically exact, and distinguishable from 403 in a log.
        text(
            state,
            req,
            421,
            "Misdirected request: this server only answers to localhost.\n\
             If you reached this from a web page, that page was trying to read \
             your local index.\n",
        );
        return;
    }

    if is_api {
        // The token is required from the paired origin and *only* from it. A
        // same-origin request from the embedded UI has no way to know the token
        // (it is printed on a terminal, not injected into the bundle), so
        // demanding one there would break the offline path this tool exists for.
        if paired && !token_ok(state, &req) {
            return json(
                state,
                req,
                401,
                serde_json::json!({
                    "error": {
                        "code": "unauthorized",
                        "message": "missing or invalid pairing token — paste the token \
                                    printed by `spraypaint serve --pair`"
                    }
                })
                .to_string(),
            );
        }
        api::route(state, req, &path, &url);
        return;
    }
    serve_static(state, req, &path);
}

/// Is this `Origin` a loopback one? Those are the embedded UI talking to itself.
fn is_loopback_origin(origin: &str) -> bool {
    let o = origin.trim().to_ascii_lowercase();
    let rest = match o.split_once("://") {
        Some((s, r)) if s == "http" || s == "https" => r,
        _ => return false,
    };
    let hostname = rest.rsplit_once(':').map(|(h, _)| h).unwrap_or(rest);
    hostname == "localhost" || hostname == "127.0.0.1" || hostname == "[::1]"
}

/// Answer a CORS preflight.
///
/// Only the paired origin gets an allow. Everything else gets a bare 403 with no
/// CORS headers at all, which is what makes the browser block the real request.
///
/// Named `cors_preflight` to distinguish it from [`preflight`], the startup
/// index check — different meanings of the same word.
fn cors_preflight(state: &ServerState, req: Request) {
    if !is_paired_origin(state, &req) {
        return text(state, req, 403, "origin not paired\n");
    }
    // `respond` attaches the CORS headers itself, since the origin has already
    // been verified as the paired one.
    let resp = Response::from_string("").with_status_code(204);
    respond(state, req, resp);
}

fn serve_static(state: &ServerState, req: Request, path: &str) {
    if ui::is_empty() {
        text(
            state,
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
            respond(state, req, resp);
        }
        None => text(state, req, 404, "not found\n"),
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
            pair_origin: None,
            token: None,
            job: IndexJob::new(),
            cache: std::sync::Mutex::new(None),
        }
    }

    fn paired_state(origin: &str, token: &str) -> ServerState {
        ServerState {
            root: PathBuf::from("."),
            allow_index: false,
            no_cache: false,
            allowed_hosts: allowed_hosts("127.0.0.1", 7373),
            pair_origin: Some(origin.to_string()),
            token: Some(Token::from_raw(token)),
            job: IndexJob::new(),
            cache: std::sync::Mutex::new(None),
        }
    }

    /// Origin matching is exact. A suffix or prefix match would pair every
    /// deployment on a shared domain — on `*.vercel.app`, everyone's.
    #[test]
    fn a_paired_origin_does_not_widen_other_origins() {
        let s = paired_state("https://acrylic-spray-paint-inky.vercel.app", "tok");
        let paired = |o: &str| {
            s.pair_origin
                .as_deref()
                .is_some_and(|p| o.trim().eq_ignore_ascii_case(p))
        };

        assert!(paired("https://acrylic-spray-paint-inky.vercel.app"));
        assert!(paired("HTTPS://Acrylic-Spray-Paint-Inky.Vercel.App"));

        // A different app on the same shared domain.
        assert!(!paired("https://someone-elses-app.vercel.app"));
        // Suffix confusion: an attacker registering a lookalike.
        assert!(!paired("https://acrylic-spray-paint-inky.vercel.app.evil.com"));
        assert!(!paired("https://evil.com/acrylic-spray-paint-inky.vercel.app"));
        // Scheme downgrade must not match — a plaintext origin can be spoofed
        // by anyone on the network path.
        assert!(!paired("http://acrylic-spray-paint-inky.vercel.app"));
        assert!(!paired("null"));
    }

    /// The embedded UI is same-origin and must keep working with no token.
    #[test]
    fn loopback_origins_are_recognised_as_same_origin() {
        assert!(is_loopback_origin("http://localhost:7373"));
        assert!(is_loopback_origin("http://127.0.0.1:7373"));
        assert!(is_loopback_origin("http://[::1]:7373"));
        assert!(is_loopback_origin("HTTP://LOCALHOST:7373"));
        assert!(is_loopback_origin("http://localhost"));

        assert!(!is_loopback_origin("https://evil.com"));
        // The rebinding lookalike: a hostname that merely *contains* localhost.
        assert!(!is_loopback_origin("https://localhost.evil.com"));
        assert!(!is_loopback_origin("https://notlocalhost"));
    }

    /// Token comparison is exact and case-sensitive. This is the whole control
    /// once an origin is paired, so a prefix must never pass.
    #[test]
    fn only_the_exact_token_authorises() {
        let s = paired_state("https://example.com", "deadbeefcafe");
        let t = s.token.as_ref().expect("paired state has a token");

        assert!(t.matches("deadbeefcafe"));
        assert!(!t.matches("deadbeefcaf"), "prefix must not pass");
        assert!(!t.matches("deadbeefcafee"));
        assert!(!t.matches("DEADBEEFCAFE"));
        assert!(!t.matches(""));
    }

    /// An unpaired server has no token, so `token_ok` must be unable to succeed
    /// no matter what the client sends — including an empty bearer value.
    #[test]
    fn an_unpaired_server_accepts_no_token_at_all() {
        let s = state_for("127.0.0.1", 7373);
        assert!(s.token.is_none());
        assert!(s.pair_origin.is_none());
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

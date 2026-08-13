//! Pairing: letting a **hosted** page drive this local server.
//!
//! ## Why this module has to exist, and why it is dangerous
//!
//! The rest of [`crate::serve`] is built on one assumption: the UI is served by
//! this binary, so every request is same-origin, and *any* cross-origin request
//! is hostile by definition. That assumption is what lets `host_allowed` reject
//! everything but loopback and lets [`super::security_headers`] send no
//! `Access-Control-Allow-Origin` at all. Together they make a page on the open
//! web structurally unable to read your source tree.
//!
//! Pairing deliberately gives that up for exactly one origin, so a deployed copy
//! of the UI can talk to a binary running on the user's machine. Since the
//! origin pin can no longer be the control, a **bearer token** becomes the
//! control in its place:
//!
//!   * it is minted per process, lives only in memory, and dies with the server;
//!   * it never touches disk, so it cannot leak through a backup or a repo;
//!   * it is 160 bits from the OS CSPRNG, so it cannot be guessed online;
//!   * it is compared in constant time, so it cannot be recovered by timing.
//!
//! The token is printed on the terminal. That matters: the only way to obtain it
//! is to be the person who ran the command, which is precisely the authorisation
//! we want to encode. A hostile page can now *reach* the API, but every request
//! it makes without the token is a 401.
//!
//! ## What pairing does NOT weaken
//!
//! Pairing is opt-in (`--pair`), scoped to one origin given on the command line,
//! and additive to the existing checks rather than a replacement for them. With
//! `--pair` absent the server behaves exactly as before, and the DNS-rebinding
//! guard still applies to every unpaired origin. See `only_loopback_hosts_are_allowed`
//! and `a_paired_origin_does_not_widen_other_origins` in [`super`].

use std::time::Duration;

/// Token length in bytes before encoding. 20 bytes = 160 bits.
///
/// Sized against an *online* attacker — the only kind there is, since the token
/// is never stored. At a wildly generous million guesses per second against a
/// loopback server, 2^160 keeps the expected time to a first hit far past the
/// heat death of the sun. 16 bytes would also be fine; 20 costs nothing.
const TOKEN_BYTES: usize = 20;

/// A minted pairing token.
///
/// Deliberately not `Clone`, `Debug`, `Serialize`, or `Display`. The only way to
/// see the secret is [`Token::reveal`], whose name is meant to be conspicuous at
/// a call site — the value must appear on the user's terminal and nowhere else,
/// least of all in a log line or an error message someone pastes into an issue.
pub struct Token(String);

impl Token {
    /// Mint a fresh token from the OS CSPRNG.
    ///
    /// Uses `getrandom` (via `blake3`'s dependency-free path is not available,
    /// so we read from the OS directly). A PRNG seeded from the clock would be
    /// guessable by anyone who knows roughly when the server started, which for
    /// a tool a user just launched is *everyone*.
    pub fn mint() -> std::io::Result<Self> {
        let mut buf = [0u8; TOKEN_BYTES];
        getrandom(&mut buf)?;
        Ok(Token(hex(&buf)))
    }

    /// Construct from a known string. Test-only: real tokens come from [`mint`].
    #[cfg(test)]
    pub fn from_raw(s: &str) -> Self {
        Token(s.to_string())
    }

    /// The secret, for printing to the terminal that started this process.
    pub fn reveal(&self) -> &str {
        &self.0
    }

    /// Constant-time equality against a candidate from a request header.
    ///
    /// `==` on `String` short-circuits at the first differing byte, which leaks
    /// the length of the matching prefix through response timing. Over enough
    /// requests that recovers the token byte by byte, turning a 160-bit secret
    /// into ~20 × 256 guesses. The loopback path makes timing *easier* to
    /// measure, not harder — there is no network jitter to hide in.
    ///
    /// Comparing lengths first is safe and not a leak: the length is a public
    /// constant of the format, identical for every token we mint.
    pub fn matches(&self, candidate: &str) -> bool {
        let a = self.0.as_bytes();
        let b = candidate.as_bytes();
        if a.len() != b.len() {
            return false;
        }
        let mut diff = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            diff |= x ^ y;
        }
        diff == 0
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEXDIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEXDIGITS[(b >> 4) as usize] as char);
        s.push(HEXDIGITS[(b & 0x0f) as usize] as char);
    }
    s
}

/// Fill `buf` with cryptographically secure random bytes from the OS.
#[cfg(unix)]
fn getrandom(buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read;
    // /dev/urandom is the right choice on every modern unix: after the pool is
    // seeded (long before a user can type a command) it is as strong as
    // /dev/random and never blocks.
    let mut f = std::fs::File::open("/dev/urandom")?;
    f.read_exact(buf)
}

#[cfg(windows)]
fn getrandom(buf: &mut [u8]) -> std::io::Result<()> {
    // `RtlGenRandom`, exported as `SystemFunction036`. Available since XP with
    // no extra crate and no CryptoAPI context handling.
    #[link(name = "advapi32")]
    extern "system" {
        #[link_name = "SystemFunction036"]
        fn RtlGenRandom(buf: *mut u8, len: u32) -> u8;
    }
    // SAFETY: `buf` is a valid, uniquely-borrowed slice of exactly `len` bytes,
    // and RtlGenRandom only writes within that range.
    let ok = unsafe { RtlGenRandom(buf.as_mut_ptr(), buf.len() as u32) };
    if ok == 0 {
        return Err(std::io::Error::other("RtlGenRandom failed"));
    }
    Ok(())
}

/// How long a browser may cache the CORS preflight for a paired origin.
///
/// 10 minutes: long enough that a burst of queries does not double its request
/// count, short enough that stopping the server is felt promptly rather than
/// hidden behind a day-long cached allow.
pub const PREFLIGHT_MAX_AGE: Duration = Duration::from_secs(600);

/// Normalise an origin for comparison, and reject anything that is not one.
///
/// An HTTP `Origin` is scheme + host + optional port, with **no** trailing slash
/// and no path. Users naturally paste `https://example.com/`, so we accept that
/// and normalise it rather than failing on a trailing slash. What we do not
/// accept is anything with a path, a query, or credentials: those are signs the
/// value is a URL that was never an origin, and silently truncating one would
/// pair a wider scope than the user typed.
///
/// `null` is rejected outright. It is the origin of sandboxed iframes and
/// `file://` documents — matching it would pair *every* such context at once.
pub fn normalise_origin(raw: &str) -> Result<String, String> {
    let s = raw.trim().trim_end_matches('/');
    if s.is_empty() {
        return Err("origin is empty".to_string());
    }
    if s.eq_ignore_ascii_case("null") {
        return Err(
            "`null` is not a pairable origin — it is shared by every sandboxed frame \
             and file:// document"
                .to_string(),
        );
    }
    let (scheme, rest) = s
        .split_once("://")
        .ok_or_else(|| format!("not an origin (expected scheme://host): {raw}"))?;
    let scheme_lc = scheme.to_ascii_lowercase();
    if scheme_lc != "http" && scheme_lc != "https" {
        return Err(format!("unsupported scheme `{scheme}` — use http or https"));
    }
    if rest.is_empty() {
        return Err(format!("origin has no host: {raw}"));
    }
    if rest.contains('/') {
        return Err(format!(
            "an origin has no path — use just the scheme, host and port, e.g. {}://{}",
            scheme_lc,
            rest.split('/').next().unwrap_or("")
        ));
    }
    if rest.contains('@') {
        return Err(format!("an origin has no credentials: {raw}"));
    }
    if rest.contains('?') || rest.contains('#') {
        return Err(format!("an origin has no query or fragment: {raw}"));
    }
    // Host is case-insensitive; the scheme is too. Lowercasing both makes the
    // later comparison a plain `==` with no room for a case-confusion bypass.
    Ok(format!("{}://{}", scheme_lc, rest.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minted_tokens_are_long_and_distinct() {
        let a = Token::mint().expect("mint");
        let b = Token::mint().expect("mint");
        assert_eq!(a.reveal().len(), TOKEN_BYTES * 2);
        assert!(a.reveal().chars().all(|c| c.is_ascii_hexdigit()));
        // Two mints colliding means the RNG is broken; the odds otherwise are 2^-160.
        assert_ne!(a.reveal(), b.reveal(), "tokens must not repeat");
    }

    #[test]
    fn matches_accepts_only_the_exact_token() {
        let t = Token::from_raw("abc123");
        assert!(t.matches("abc123"));

        assert!(!t.matches("abc124"));
        assert!(!t.matches("abc12"), "a prefix is not a match");
        assert!(!t.matches("abc1234"), "an extension is not a match");
        assert!(!t.matches(""));
        assert!(!t.matches("ABC123"), "tokens are case-sensitive");
    }

    #[test]
    fn origins_are_normalised() {
        assert_eq!(
            normalise_origin("https://example.vercel.app/").unwrap(),
            "https://example.vercel.app"
        );
        assert_eq!(
            normalise_origin("  https://Example.Vercel.App  ").unwrap(),
            "https://example.vercel.app"
        );
        assert_eq!(
            normalise_origin("HTTPS://example.com:8443").unwrap(),
            "https://example.com:8443"
        );
        assert_eq!(
            normalise_origin("http://localhost:3000").unwrap(),
            "http://localhost:3000"
        );
    }

    /// Each rejection here is a way a user could accidentally pair something
    /// broader than they meant to.
    #[test]
    fn non_origins_are_rejected() {
        assert!(normalise_origin("").is_err());
        assert!(normalise_origin("example.com").is_err(), "no scheme");
        assert!(normalise_origin("ftp://example.com").is_err(), "wrong scheme");
        assert!(normalise_origin("https://").is_err(), "no host");
        assert!(
            normalise_origin("https://example.com/app").is_err(),
            "a path is not part of an origin and must not be silently dropped"
        );
        assert!(normalise_origin("https://user:pw@example.com").is_err());
        assert!(normalise_origin("https://example.com?a=1").is_err());
    }

    /// `null` matches sandboxed iframes and file:// documents. Pairing it would
    /// authorise a whole class of contexts rather than one site.
    #[test]
    fn null_origin_is_rejected() {
        assert!(normalise_origin("null").is_err());
        assert!(normalise_origin("NULL").is_err());
    }
}

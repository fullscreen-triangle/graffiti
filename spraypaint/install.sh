#!/bin/sh
# spraypaint installer — macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/fullscreen-triangle/graffiti/main/spraypaint/install.sh | sh
#
# POSIX sh, not bash: this runs on Alpine, where /bin/sh is busybox ash and bash
# is usually absent. No arrays, no [[ ]], no `local` outside functions.
#
# Fails closed. Every download is verified against the published SHA256SUMS
# before anything is written to your PATH — a piped-to-shell installer that
# skips that check offers no more integrity than an unverified binary.
set -eu

REPO="fullscreen-triangle/graffiti"
INSTALL_DIR="${SPRAYPAINT_INSTALL_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
err() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || err "required command not found: $1"
}

# --- platform detection -----------------------------------------------------

detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin)
            case "$arch" in
                arm64|aarch64) echo "aarch64-apple-darwin" ;;
                x86_64)        echo "x86_64-apple-darwin" ;;
                *) err "unsupported macOS architecture: $arch" ;;
            esac
            ;;
        Linux)
            # Only x86_64 Linux is published today. Rejecting other
            # architectures by name is better than downloading an archive that
            # cannot execute and failing with "cannot execute binary file".
            [ "$arch" = "x86_64" ] || err "unsupported Linux architecture: $arch (only x86_64 is published)"
            # musl vs glibc. `ldd --version` writes its banner to stderr on
            # glibc and to stdout on musl, so both streams are merged. When ldd
            # is missing entirely the system is almost certainly not glibc, and
            # the musl build is the safe answer either way: it is static, so it
            # runs on glibc systems too. Guessing gnu when wrong yields a
            # "No such file or directory" for the loader, which is far more
            # confusing than a slightly slower binary.
            if command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
                echo "x86_64-unknown-linux-musl"
            elif command -v ldd >/dev/null 2>&1; then
                echo "x86_64-unknown-linux-gnu"
            else
                echo "x86_64-unknown-linux-musl"
            fi
            ;;
        *) err "unsupported operating system: $os (use install.ps1 on Windows)" ;;
    esac
}

# --- download helpers -------------------------------------------------------

# curl or wget, whichever exists. `-f` / `--server-response` handling matters:
# without curl's `-f`, a 404 page is written to disk as if it were an archive
# and the failure surfaces later as a corrupt tarball.
fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$1" -O "$2"
    else
        err "need curl or wget to download"
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        err "need sha256sum or shasum to verify the download"
    fi
}

# --- main -------------------------------------------------------------------

need uname
need tar
need mkdir

TARGET="$(detect_target)"
say "spraypaint installer"
say "  platform : $TARGET"

# Resolve the version. A caller-supplied SPRAYPAINT_VERSION pins an exact
# release; otherwise follow the /releases/latest redirect, which avoids needing
# the GitHub API (rate-limited and unauthenticated here).
if [ -n "${SPRAYPAINT_VERSION:-}" ]; then
    VERSION="$SPRAYPAINT_VERSION"
else
    need curl
    latest_url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
        "https://github.com/$REPO/releases/latest" 2>/dev/null)" \
        || err "could not reach GitHub to resolve the latest release"
    # .../releases/tag/spraypaint-v0.2.0 -> 0.2.0
    VERSION="${latest_url##*/spraypaint-v}"
    case "$VERSION" in
        ""|*/*) err "could not parse a version from: $latest_url" ;;
    esac
fi
say "  version  : $VERSION"

TAG="spraypaint-v$VERSION"
NAME="spraypaint-$VERSION-$TARGET"
ARCHIVE="$NAME.tar.gz"
BASE="https://github.com/$REPO/releases/download/$TAG"

# mktemp -d is POSIX-ish and present on both busybox and coreutils. The trap
# runs on normal exit and on failure, so a failed verification leaves no
# half-downloaded archive behind.
TMP="$(mktemp -d 2>/dev/null || mktemp -d -t spraypaint)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "  fetching : $BASE/$ARCHIVE"
fetch "$BASE/$ARCHIVE" "$TMP/$ARCHIVE" || err "download failed — does a release exist for $TARGET at $TAG?"
fetch "$BASE/SHA256SUMS" "$TMP/SHA256SUMS" || err "could not download SHA256SUMS — refusing to install unverified"

# Verify by extracting this archive's expected hash from SHA256SUMS and
# comparing. Not `sha256sum -c`, which would fail on the other four archives
# named in the file that we deliberately did not download.
expected="$(grep " \*\{0,1\}$ARCHIVE\$" "$TMP/SHA256SUMS" | cut -d' ' -f1 | head -n1)"
[ -n "$expected" ] || err "$ARCHIVE is not listed in SHA256SUMS"
actual="$(sha256_of "$TMP/$ARCHIVE")"
if [ "$expected" != "$actual" ]; then
    err "checksum mismatch for $ARCHIVE
  expected $expected
  actual   $actual
The download was corrupted or tampered with. Nothing was installed."
fi
say "  checksum : ok"

tar xzf "$TMP/$ARCHIVE" -C "$TMP" || err "could not extract $ARCHIVE"
[ -f "$TMP/$NAME/spraypaint" ] || err "archive did not contain the expected binary"

mkdir -p "$INSTALL_DIR" || err "could not create $INSTALL_DIR"
# Install to a temporary name in the destination directory and rename, so an
# interrupted copy cannot leave a truncated executable on PATH. rename within
# one filesystem is atomic; a plain cp over a running binary is not.
cp "$TMP/$NAME/spraypaint" "$INSTALL_DIR/.spraypaint.new" || err "could not write to $INSTALL_DIR"
chmod 755 "$INSTALL_DIR/.spraypaint.new"
mv -f "$INSTALL_DIR/.spraypaint.new" "$INSTALL_DIR/spraypaint"

# The binary is unsigned. On macOS the quarantine attribute makes the first run
# fail with a Gatekeeper dialog; clearing it here is the same action the user
# would take manually, on a file whose hash we just verified.
if [ "$(uname -s)" = "Darwin" ] && command -v xattr >/dev/null 2>&1; then
    xattr -d com.apple.quarantine "$INSTALL_DIR/spraypaint" 2>/dev/null || true
fi

say "  installed: $INSTALL_DIR/spraypaint"

# PATH check. Matching with delimiters on both sides avoids a false positive
# where /home/me/.local/binaries would "contain" /home/me/.local/bin.
case ":$PATH:" in
    *":$INSTALL_DIR:"*)
        say ""
        say "Run it:  spraypaint serve --open"
        ;;
    *)
        say ""
        say "warning: $INSTALL_DIR is not on your PATH."
        say "Add it to your shell profile:"
        say ""
        say "    export PATH=\"$INSTALL_DIR:\$PATH\""
        say ""
        say "Or run it directly:  $INSTALL_DIR/spraypaint serve --open"
        ;;
esac

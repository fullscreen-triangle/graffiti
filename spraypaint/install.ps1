# spraypaint installer - Windows.
#
#   irm https://raw.githubusercontent.com/fullscreen-triangle/graffiti/main/spraypaint/install.ps1 | iex
#
# Targets Windows PowerShell 5.1, the version shipped with Windows 11, so it
# deliberately avoids PowerShell 7 syntax that would be a *parse* error there --
# parse errors fire before any of the script runs, so a version check could not
# even print a message. That rules out: ternary (?:), null-coalescing (??),
# pipeline chain operators (&& ||), and null-conditional (?.).
#
# User scope only. Installs under %LOCALAPPDATA% and edits the *User* PATH, so
# it never needs administrator rights and cannot affect other accounts.

$ErrorActionPreference = 'Stop'

$Repo = 'fullscreen-triangle/graffiti'

if ($env:SPRAYPAINT_INSTALL_DIR) {
    $InstallDir = $env:SPRAYPAINT_INSTALL_DIR
} else {
    $InstallDir = Join-Path $env:LOCALAPPDATA 'spraypaint\bin'
}

function Fail($msg) {
    Write-Host "error: $msg" -ForegroundColor Red
    exit 1
}

Write-Host 'spraypaint installer'

# Only x86_64 Windows is published. ARM64 Windows can run x64 binaries under
# emulation, so this is a warning rather than a hard stop -- but say so, because
# emulated performance is a plausible cause of a later "why is this slow".
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -eq 'ARM64') {
    Write-Host '  note     : ARM64 detected; installing the x64 build (runs under emulation)' -ForegroundColor Yellow
} elseif ($arch -ne 'AMD64') {
    Fail "unsupported architecture: $arch (only x64 is published)"
}

$Target = 'x86_64-pc-windows-msvc'
Write-Host "  platform : $Target"

# TLS 1.2. Windows PowerShell 5.1 on older builds still defaults to TLS 1.0,
# which github.com refuses -- the failure looks like a connection reset rather
# than a protocol problem, so set it explicitly.
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
    # Already-modern .NET may not expose the enum member; not fatal.
}

# Resolve the version: an explicit SPRAYPAINT_VERSION pins a release, otherwise
# follow the /releases/latest redirect. -MaximumRedirection 0 makes the 302
# itself the result; in 5.1 that surfaces as a terminating error whose response
# still carries the Location header, so both paths are handled.
if ($env:SPRAYPAINT_VERSION) {
    $Version = $env:SPRAYPAINT_VERSION
} else {
    $location = $null
    try {
        $resp = Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest" `
            -MaximumRedirection 0 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($resp) { $location = $resp.Headers.Location }
    } catch {
        if ($_.Exception.Response) {
            $location = $_.Exception.Response.Headers['Location']
        }
    }
    if (-not $location) { Fail 'could not reach GitHub to resolve the latest release' }
    $Version = ($location -split 'spraypaint-v')[-1].Trim('/')
    if (-not $Version) { Fail "could not parse a version from: $location" }
}
Write-Host "  version  : $Version"

$Tag     = "spraypaint-v$Version"
$Name    = "spraypaint-$Version-$Target"
$Archive = "$Name.zip"
$Base    = "https://github.com/$Repo/releases/download/$Tag"

$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("spraypaint-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Tmp -Force | Out-Null

try {
    $zipPath = Join-Path $Tmp $Archive
    $sumPath = Join-Path $Tmp 'SHA256SUMS'

    Write-Host "  fetching : $Base/$Archive"
    try {
        Invoke-WebRequest -Uri "$Base/$Archive" -OutFile $zipPath -UseBasicParsing
    } catch {
        Fail "download failed - does a release exist for $Target at $Tag?"
    }
    try {
        Invoke-WebRequest -Uri "$Base/SHA256SUMS" -OutFile $sumPath -UseBasicParsing
    } catch {
        Fail 'could not download SHA256SUMS - refusing to install unverified'
    }

    # Verify before extracting. An installer that pipes from the internet and
    # skips this check gives no more integrity than downloading blind.
    $expected = $null
    foreach ($line in Get-Content $sumPath) {
        # sha256sum writes "<hash>  <name>" or "<hash> *<name>" in binary mode.
        if ($line -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
            if ($matches[2].Trim() -eq $Archive) { $expected = $matches[1].ToLower() }
        }
    }
    if (-not $expected) { Fail "$Archive is not listed in SHA256SUMS" }

    $actual = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) {
        Fail "checksum mismatch for $Archive`n  expected $expected`n  actual   $actual`nThe download was corrupted or tampered with. Nothing was installed."
    }
    Write-Host '  checksum : ok'

    # Clear the Mark of the Web before extracting. Without this, files unpacked
    # from a downloaded zip inherit the zone identifier and SmartScreen blocks
    # the first run with a dialog that offers no obvious way forward.
    try { Unblock-File -Path $zipPath -ErrorAction SilentlyContinue } catch { }

    Expand-Archive -Path $zipPath -DestinationPath $Tmp -Force
    $binSrc = Join-Path $Tmp "$Name\spraypaint.exe"
    if (-not (Test-Path $binSrc)) { Fail 'archive did not contain the expected binary' }

    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    $binDst = Join-Path $InstallDir 'spraypaint.exe'

    # Windows locks a running executable, so overwriting an in-use spraypaint.exe
    # fails outright. Say which process to stop rather than emitting the raw
    # "being used by another process" error.
    try {
        Copy-Item -Path $binSrc -Destination $binDst -Force
    } catch {
        Fail "could not write $binDst - if 'spraypaint serve' is running, stop it and re-run"
    }
    try { Unblock-File -Path $binDst -ErrorAction SilentlyContinue } catch { }

    Write-Host "  installed: $binDst"

    # User PATH only -- never Machine, which would require elevation and change
    # the environment for every account on the box. Read the stored User value
    # rather than $env:PATH: the latter is the merged Machine+User+session copy,
    # so writing it back would duplicate the machine entries into user scope.
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    if (-not $userPath) { $userPath = '' }

    $already = $false
    foreach ($entry in ($userPath -split ';')) {
        if ($entry.TrimEnd('\') -ieq $InstallDir.TrimEnd('\')) { $already = $true }
    }

    if ($already) {
        Write-Host ''
        Write-Host 'Run it:  spraypaint serve --open'
    } else {
        if ($userPath -eq '') {
            $newPath = $InstallDir
        } else {
            $newPath = $userPath.TrimEnd(';') + ';' + $InstallDir
        }
        [Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')
        # Also update this session, so the very next command works without a
        # restart. SetEnvironmentVariable only affects processes started later.
        $env:PATH = $env:PATH + ';' + $InstallDir
        Write-Host "  PATH     : added $InstallDir (user scope)"
        Write-Host ''
        Write-Host 'Open a new terminal for PATH to apply everywhere, then run:'
        Write-Host '    spraypaint serve --open'
    }
} finally {
    Remove-Item -Path $Tmp -Recurse -Force -ErrorAction SilentlyContinue
}

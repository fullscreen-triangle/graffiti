# Graffiti Search - Revolutionary Proof-Based Search Engine
# PowerShell build script for Windows development

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [switch]$Release,
    [switch]$Open,
    [switch]$Watch
)

function Write-Header {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Invoke-CommandWithCheck {
    param([string]$Command, [string]$Description)
    
    Write-Host "Running: $Description..." -ForegroundColor Yellow
    Invoke-Expression $Command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "$Description completed successfully"
    } else {
        Write-Error "$Description failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

switch ($Command.ToLower()) {
    "help" {
        Write-Header "Graffiti Search - Revolutionary Proof-Based Search Engine"
        Write-Host ""
        Write-Host "Available commands:" -ForegroundColor White
        Write-Host "  help                  - Show this help message"
        Write-Host "  install              - Install dependencies and setup environment"
        Write-Host "  build                - Build the project (add -Release for optimized)"
        Write-Host "  test                 - Run all tests"
        Write-Host "  test-atmospheric     - Run atmospheric processing tests"
        Write-Host "  test-environmental   - Run environmental measurement tests"
        Write-Host "  test-perturbation    - Run perturbation validation tests"
        Write-Host "  clean               - Clean build artifacts"
        Write-Host "  dev                 - Start development server (add -Watch for auto-reload)"
        Write-Host "  wasm                - Build WebAssembly version"
        Write-Host "  wasm-release        - Build optimized WASM for production"
        Write-Host "  web                 - Build web interface"
        Write-Host "  web-dev             - Start web development server (add -Open to open browser)"
        Write-Host "  serve               - Serve the application locally"
        Write-Host "  bench               - Run performance benchmarks"
        Write-Host "  fmt                 - Format code"
        Write-Host "  lint                - Run clippy linting"
        Write-Host "  doc                 - Generate documentation (add -Open to open browser)"
        Write-Host "  check               - Run all quality checks"
        Write-Host "  docker              - Build Docker image"
        Write-Host "  init-env            - Initialize environmental measurement systems"
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor White
        Write-Host "  .\build.ps1 build -Release     # Build optimized release"
        Write-Host "  .\build.ps1 dev -Watch         # Development with auto-reload"
        Write-Host "  .\build.ps1 doc -Open          # Generate and open documentation"
        Write-Host "  .\build.ps1 web-dev -Open      # Start web dev server and open browser"
    }
    
    "install" {
        Write-Header "Installing Dependencies"
        
        # Check if Rust is installed
        try {
            cargo --version | Out-Null
            Write-Success "Rust is already installed"
        } catch {
            Write-Error "Rust is not installed. Please install from https://rustup.rs/"
            exit 1
        }
        
        # Install WebAssembly target
        Invoke-CommandWithCheck "rustup target add wasm32-unknown-unknown" "Installing WebAssembly target"
        
        # Install Rust components
        Invoke-CommandWithCheck "rustup component add clippy rustfmt" "Installing Rust components"
        
        # Install wasm-pack
        try {
            wasm-pack --version | Out-Null
            Write-Success "wasm-pack is already installed"
        } catch {
            Write-Host "Installing wasm-pack..."
            Invoke-WebRequest -Uri "https://rustwasm.github.io/wasm-pack/installer/init.ps1" -OutFile "init.ps1"
            .\init.ps1
            Remove-Item "init.ps1"
            Write-Success "wasm-pack installed"
        }
        
        # Install trunk for web development
        Invoke-CommandWithCheck "cargo install trunk" "Installing trunk for web development"
        
        # Install cargo-watch for development
        Invoke-CommandWithCheck "cargo install cargo-watch" "Installing cargo-watch for development"
        
        Write-Success "All dependencies installed successfully!"
    }
    
    "build" {
        Write-Header "Building Graffiti Search"
        
        if ($Release) {
            Invoke-CommandWithCheck "cargo build --release" "Building release version"
        } else {
            Invoke-CommandWithCheck "cargo build" "Building debug version"
        }
    }
    
    "test" {
        Write-Header "Running All Tests"
        Invoke-CommandWithCheck "cargo test --workspace" "Running all tests"
    }
    
    "test-atmospheric" {
        Write-Header "Running Atmospheric Processing Tests"
        Invoke-CommandWithCheck "cargo test -p graffiti-atmospheric" "Running atmospheric tests"
    }
    
    "test-environmental" {
        Write-Header "Running Environmental Measurement Tests"
        Invoke-CommandWithCheck "cargo test -p graffiti-environmental" "Running environmental tests"
    }
    
    "test-perturbation" {
        Write-Header "Running Perturbation Validation Tests"
        Invoke-CommandWithCheck "cargo test -p graffiti-perturbation" "Running perturbation tests"
    }
    
    "clean" {
        Write-Header "Cleaning Build Artifacts"
        Invoke-CommandWithCheck "cargo clean" "Cleaning Rust artifacts"
        
        if (Test-Path "pkg") {
            Remove-Item -Recurse -Force "pkg"
            Write-Success "Removed WASM pkg directory"
        }
        
        if (Test-Path "crates\web\dist") {
            Remove-Item -Recurse -Force "crates\web\dist"
            Write-Success "Removed web dist directory"
        }
    }
    
    "dev" {
        Write-Header "Starting Development Server"
        
        if ($Watch) {
            Invoke-CommandWithCheck "cargo watch -x run" "Starting development server with auto-reload"
        } else {
            Invoke-CommandWithCheck "cargo run" "Starting development server"
        }
    }
    
    "wasm" {
        Write-Header "Building WebAssembly Version"
        Invoke-CommandWithCheck "wasm-pack build --target web --out-dir pkg crates\web" "Building WASM"
    }
    
    "wasm-release" {
        Write-Header "Building Optimized WebAssembly"
        Invoke-CommandWithCheck "wasm-pack build --target web --release --out-dir pkg crates\web" "Building optimized WASM"
    }
    
    "web" {
        Write-Header "Building Web Interface"
        
        # First build WASM
        & $PSCommandPath wasm
        
        # Then build web interface
        Set-Location "crates\web"
        try {
            Invoke-CommandWithCheck "trunk build --release" "Building web interface"
        } finally {
            Set-Location "..\\.."
        }
    }
    
    "web-dev" {
        Write-Header "Starting Web Development Server"
        
        Set-Location "crates\web"
        try {
            if ($Open) {
                Invoke-CommandWithCheck "trunk serve --open" "Starting web dev server with browser"
            } else {
                Invoke-CommandWithCheck "trunk serve" "Starting web dev server"
            }
        } finally {
            Set-Location "..\\.."
        }
    }
    
    "serve" {
        Write-Header "Serving Application Locally"
        
        # Build WASM first
        & $PSCommandPath wasm
        
        # Check if Python is available
        try {
            python --version | Out-Null
            Invoke-CommandWithCheck "python -m http.server 8080" "Serving with Python HTTP server"
        } catch {
            # Try with Node.js if available
            try {
                node --version | Out-Null
                Invoke-CommandWithCheck "npx http-server -p 8080 crates\web\dist" "Serving with Node.js HTTP server"
            } catch {
                Write-Error "Neither Python nor Node.js found. Please install one to serve the application."
                exit 1
            }
        }
    }
    
    "bench" {
        Write-Header "Running Performance Benchmarks"
        Invoke-CommandWithCheck "cargo bench" "Running all benchmarks"
    }
    
    "fmt" {
        Write-Header "Formatting Code"
        Invoke-CommandWithCheck "cargo fmt --all" "Formatting code"
    }
    
    "lint" {
        Write-Header "Running Clippy Linting"
        Invoke-CommandWithCheck "cargo clippy --workspace --all-targets --all-features -- -D warnings" "Running clippy"
    }
    
    "doc" {
        Write-Header "Generating Documentation"
        
        if ($Open) {
            Invoke-CommandWithCheck "cargo doc --no-deps --open" "Generating and opening documentation"
        } else {
            Invoke-CommandWithCheck "cargo doc --no-deps" "Generating documentation"
        }
    }
    
    "check" {
        Write-Header "Running All Quality Checks"
        
        & $PSCommandPath fmt
        & $PSCommandPath lint
        & $PSCommandPath test
        
        Write-Success "All quality checks completed!"
    }
    
    "docker" {
        Write-Header "Building Docker Image"
        Invoke-CommandWithCheck "docker build -t graffiti-search:latest ." "Building Docker image"
    }
    
    "init-env" {
        Write-Header "Initializing Environmental Measurement Systems"
        
        # Create data directories
        $dirs = @(
            "data\atmospheric",
            "data\environmental", 
            "data\quantum",
            "data\temporal",
            "data\molecular",
            "data\s-entropy",
            "simulations\output",
            "perturbation\results"
        )
        
        foreach ($dir in $dirs) {
            if (!(Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-Success "Created directory: $dir"
            }
        }
        
        Write-Success "Environmental measurement systems initialized!"
    }
    
    "advanced-help" {
        Write-Header "Advanced Commands"
        Write-Host "  profile             - Run performance profiling (requires perf tools)"
        Write-Host "  coverage            - Generate test coverage report"
        Write-Host "  audit               - Run security audit"
        Write-Host "  update              - Update all dependencies"
        Write-Host "  deep-clean          - Remove all generated files including target/"
    }
    
    "profile" {
        Write-Header "Performance Profiling"
        Write-Host "Building release version for profiling..."
        & $PSCommandPath build -Release
        
        # For Windows, we'd need different profiling tools
        Write-Host "For Windows profiling, consider using:"
        Write-Host "- Intel VTune Profiler"
        Write-Host "- Visual Studio Diagnostic Tools"
        Write-Host "- perf for WSL"
    }
    
    "coverage" {
        Write-Header "Generating Test Coverage"
        
        # Install tarpaulin if not present
        try {
            cargo tarpaulin --version | Out-Null
        } catch {
            Invoke-CommandWithCheck "cargo install cargo-tarpaulin" "Installing cargo-tarpaulin"
        }
        
        Invoke-CommandWithCheck "cargo tarpaulin --out html --output-dir coverage" "Generating coverage report"
    }
    
    "audit" {
        Write-Header "Running Security Audit"
        
        try {
            cargo audit --version | Out-Null
        } catch {
            Invoke-CommandWithCheck "cargo install cargo-audit" "Installing cargo-audit"
        }
        
        Invoke-CommandWithCheck "cargo audit" "Running security audit"
    }
    
    "update" {
        Write-Header "Updating Dependencies"
        Invoke-CommandWithCheck "cargo update" "Updating Rust dependencies"
    }
    
    "deep-clean" {
        Write-Header "Deep Cleaning All Generated Files"
        
        & $PSCommandPath clean
        
        if (Test-Path "target") {
            Remove-Item -Recurse -Force "target"
            Write-Success "Removed target directory"
        }
        
        # Remove data directories
        $dataDirs = @("data", "simulations", "perturbation")
        foreach ($dir in $dataDirs) {
            if (Test-Path $dir) {
                Remove-Item -Recurse -Force $dir
                Write-Success "Removed $dir directory"
            }
        }
    }
    
    default {
        Write-Error "Unknown command: $Command"
        Write-Host "Use '.\build.ps1 help' to see available commands."
        exit 1
    }
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

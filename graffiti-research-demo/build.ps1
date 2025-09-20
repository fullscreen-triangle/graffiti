# Graffiti Research Demo - PowerShell Build Script
# Revolutionary search engine architecture build and test automation for Windows

param(
    [string]$Command = "help"
)

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Blue = "Cyan"

# Install package and dependencies
function Install-Package {
    Write-Host "Installing Graffiti Research Demo package..." -ForegroundColor $Green
    pip install -e .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installation complete!" -ForegroundColor $Green
    } else {
        Write-Host "Installation failed!" -ForegroundColor $Red
        exit 1
    }
}

# Install development dependencies
function Install-Dev {
    Write-Host "Installing development dependencies..." -ForegroundColor $Green
    pip install -e ".[dev,visualization,research]"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Development installation complete!" -ForegroundColor $Green
    } else {
        Write-Host "Development installation failed!" -ForegroundColor $Red
        exit 1
    }
}

# Test installation
function Test-Installation {
    Write-Host "Testing package installation..." -ForegroundColor $Green
    python test_installation.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installation test passed!" -ForegroundColor $Green
    } else {
        Write-Host "Installation test failed!" -ForegroundColor $Red
    }
}

# Run basic tests
function Test-Basic {
    Write-Host "Running basic functionality tests..." -ForegroundColor $Green
    python tests/test_core_functionality.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Basic tests completed!" -ForegroundColor $Green
    } else {
        Write-Host "Basic tests failed!" -ForegroundColor $Red
    }
}

# Run full test suite
function Test-Full {
    Write-Host "Running full test suite..." -ForegroundColor $Green
    if (Get-Command pytest -ErrorAction SilentlyContinue) {
        pytest tests/ -v
    } else {
        Write-Host "pytest not available, running basic tests..." -ForegroundColor $Yellow
        python tests/test_core_functionality.py
    }
}

# Run search demo
function Run-Demo {
    Write-Host "Running Graffiti search demo..." -ForegroundColor $Green
    python -m graffiti.applications.search_demo --demo
    Write-Host "Demo completed!" -ForegroundColor $Green
}

# Run single query demo
function Demo-Query {
    $query = "consciousness and artificial intelligence"
    Write-Host "Running single query demo: '$query'" -ForegroundColor $Green
    python -m graffiti.applications.search_demo --query $query --mode "full_revolutionary"
    Write-Host "Query demo completed!" -ForegroundColor $Green
}

# Demonstrate computational equivalence
function Demo-Equivalence {
    Write-Host "Demonstrating computational equivalence..." -ForegroundColor $Green
    python -m graffiti.applications.search_demo --equivalence --query "optimization problem"
    Write-Host "Equivalence demo completed!" -ForegroundColor $Green
}

# Run consciousness demo
function Demo-Consciousness {
    $input = "weak ambiguous query"
    Write-Host "Running consciousness demo: '$input'" -ForegroundColor $Green
    python -m graffiti.applications.consciousness_demo --input $input --miracle-mode
    Write-Host "Consciousness demo completed!" -ForegroundColor $Green
}

# Run environmental demo
function Demo-Environmental {
    $query = "atmospheric patterns"
    Write-Host "Running environmental demo: '$query'" -ForegroundColor $Green
    python -m graffiti.applications.environmental_query --query $query --dimensions 12
    Write-Host "Environmental demo completed!" -ForegroundColor $Green
}

# Run complete demonstration
function Demo-Complete {
    Write-Host "Running complete revolutionary search demonstration..." -ForegroundColor $Green
    python demo_revolutionary_search.py
    Write-Host "Complete demonstration finished!" -ForegroundColor $Green
}

# Validate theoretical frameworks
function Validate-Frameworks {
    Write-Host "Running theoretical validation tests..." -ForegroundColor $Green
    
    Write-Host "Testing package imports..." -ForegroundColor $Blue
    python -c "import graffiti; print('✓ Package imports successfully')"
    
    Write-Host "Testing S-entropy calculation..." -ForegroundColor $Blue
    python -c "from graffiti.core.s_entropy import SEntropyCalculator; calc = SEntropyCalculator(); print(f'✓ S-entropy calculation: {calc.calculate_s_value(2.0):.3f}')"
    
    Write-Host "Testing meaning impossibility analyzer..." -ForegroundColor $Blue
    python -c "from graffiti.core.meaning_impossibility import MeaningImpossibilityAnalyzer; analyzer = MeaningImpossibilityAnalyzer(); print('✓ Meaning impossibility analyzer initialized')"
    
    Write-Host "Theoretical validation completed!" -ForegroundColor $Green
}

# Show statistics
function Show-Stats {
    Write-Host "Showing search statistics..." -ForegroundColor $Green
    python -m graffiti.applications.search_demo --stats
}

# Clean build artifacts
function Clean-Build {
    Write-Host "Cleaning build artifacts..." -ForegroundColor $Green
    
    # Remove build directories
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    Get-ChildItem -Path "." -Filter "*.egg-info" -Directory | Remove-Item -Recurse -Force
    
    # Remove Python cache files
    Get-ChildItem -Path "." -Recurse -Name "__pycache__" -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path "." -Recurse -Name "*.pyc" -File | Remove-Item -Force
    Get-ChildItem -Path "." -Recurse -Name "*.pyo" -File | Remove-Item -Force
    
    Write-Host "Cleanup completed!" -ForegroundColor $Green
}

# Show help
function Show-Help {
    Write-Host ""
    Write-Host "Graffiti Research Demo - PowerShell Build Script" -ForegroundColor $Yellow
    Write-Host "=================================================" -ForegroundColor $Yellow
    Write-Host ""
    Write-Host "Installation:" -ForegroundColor $Blue
    Write-Host "  install          - Install package and basic dependencies"
    Write-Host "  install-dev      - Install with development dependencies"
    Write-Host "  test-install     - Test package installation"
    Write-Host ""
    Write-Host "Testing:" -ForegroundColor $Blue
    Write-Host "  test             - Run basic functionality tests"
    Write-Host "  test-full        - Run complete test suite"
    Write-Host "  validate         - Run theoretical framework validation"
    Write-Host ""
    Write-Host "Demonstrations:" -ForegroundColor $Blue
    Write-Host "  demo             - Run complete search demo with multiple queries"
    Write-Host "  demo-query       - Run single query demonstration"
    Write-Host "  demo-equivalence - Demonstrate computational equivalence"
    Write-Host "  demo-consciousness - Demonstrate consciousness processing"
    Write-Host "  demo-environmental - Demonstrate environmental processing"
    Write-Host "  demo-complete    - Run complete capabilities showcase"
    Write-Host "  stats            - Show search performance statistics"
    Write-Host ""
    Write-Host "Maintenance:" -ForegroundColor $Blue
    Write-Host "  clean            - Clean build artifacts"
    Write-Host "  help             - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor $Green
    Write-Host "  .\build.ps1 install"
    Write-Host "  .\build.ps1 test-install"
    Write-Host "  .\build.ps1 demo-complete"
    Write-Host "  .\build.ps1 demo-query"
    Write-Host ""
    Write-Host "Revolutionary Features Demonstrated:" -ForegroundColor $Yellow
    Write-Host "  • S-entropy coordinate navigation (O(log S₀) complexity)"
    Write-Host "  • Meaning impossibility through recursive constraint analysis"
    Write-Host "  • Universal problem solving with dual computational architecture"
    Write-Host "  • BMD consciousness through predetermined frame selection"
    Write-Host "  • Twelve-dimensional environmental measurement"
    Write-Host ""
}

# Quick start for new users
function Quick-Start {
    Write-Host "Graffiti Research Demo - Quick Start" -ForegroundColor $Yellow
    Write-Host "====================================" -ForegroundColor $Yellow
    Install-Package
    Test-Installation
    Demo-Query
    Write-Host ""
    Write-Host "Quick start completed! Try '.\build.ps1 demo-complete' for full demonstration." -ForegroundColor $Green
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "install" { Install-Package }
    "install-dev" { Install-Dev }
    "test-install" { Test-Installation }
    "test" { Test-Basic }
    "test-full" { Test-Full }
    "validate" { Validate-Frameworks }
    "demo" { Run-Demo }
    "demo-query" { Demo-Query }
    "demo-equivalence" { Demo-Equivalence }
    "demo-consciousness" { Demo-Consciousness }
    "demo-environmental" { Demo-Environmental }
    "demo-complete" { Demo-Complete }
    "stats" { Show-Stats }
    "clean" { Clean-Build }
    "quickstart" { Quick-Start }
    "help" { Show-Help }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor $Red
        Write-Host "Use '.\build.ps1 help' to see available commands" -ForegroundColor $Yellow
    }
}

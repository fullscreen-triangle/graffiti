# Windows Setup Guide

Quick setup guide for running the Graffiti Research Demo on Windows.

## Prerequisites

1. **Python 3.8 or higher** installed
2. **Virtual environment** activated (recommended)
3. **PowerShell** or Command Prompt

## Quick Setup

### Option 1: Using PowerShell (Recommended)

```powershell
# 1. Install the package
.\build.ps1 install

# 2. Test installation
.\build.ps1 test-install

# 3. Run demo
.\build.ps1 demo-complete
```

### Option 2: Using Python directly

```powershell
# 1. Install the package
pip install -e .

# 2. Test installation
python test_installation.py

# 3. Run demo
python demo_revolutionary_search.py
```

## Available Commands

### PowerShell Build Script
```powershell
.\build.ps1 install          # Install package
.\build.ps1 test-install     # Test installation  
.\build.ps1 demo-complete    # Complete demo
.\build.ps1 demo-query       # Single query demo
.\build.ps1 help             # Show all commands
```

### Direct Python Usage
```powershell
# Search demo
python -m graffiti.applications.search_demo --query "your query" --mode full_revolutionary

# Consciousness demo  
python -m graffiti.applications.consciousness_demo --input "test input" --miracle-mode

# Environmental demo
python -m graffiti.applications.environmental_query --query "atmospheric patterns" --dimensions 12
```

## Console Scripts (After Installation)

If installation is successful, these commands should work:

```powershell
graffiti-demo --query "revolutionary search patterns" --mode full_revolutionary
graffiti-consciousness --input "weak ambiguous query" --miracle-mode  
graffiti-environmental --query "atmospheric patterns" --dimensions 12
```

## Troubleshooting

### Import Errors
If you get import errors like `No module named 'graffiti.environmental.cross_modal_bmv'`:

1. **Reinstall the package:**
   ```powershell
   pip install -e . --force-reinstall
   ```

2. **Check installation:**
   ```powershell
   python test_installation.py
   ```

3. **Verify package structure:**
   ```powershell
   python -c "import graffiti; print(graffiti.__file__)"
   ```

### Console Script Errors  
If console scripts don't work:

1. **Use Python module syntax instead:**
   ```powershell
   python -m graffiti.applications.search_demo --help
   ```

2. **Check pip installation:**
   ```powershell
   pip show graffiti-research-demo
   ```

### Missing Dependencies
If you get dependency errors:

```powershell
pip install numpy scipy pandas sympy scikit-learn pydantic loguru
```

## Package Structure

```
graffiti-research-demo/
├── graffiti/
│   ├── core/                  # Core theoretical frameworks
│   ├── environmental/         # Environmental integration
│   ├── integration/           # Framework orchestration  
│   └── applications/          # Demo applications
├── tests/                     # Test suite
├── build.ps1                  # Windows build script
├── test_installation.py       # Installation test
└── demo_revolutionary_search.py # Complete demo
```

## Examples

### Basic Search
```powershell
python -m graffiti.applications.search_demo --query "consciousness patterns" --mode s_entropy_navigation
```

### Advanced Processing
```powershell
python -m graffiti.applications.search_demo --query "optimization algorithms" --mode full_revolutionary --urgency high
```

### Performance Testing
```powershell
python tests/test_core_functionality.py
```

## Expected Output

Successful installation should show:
- ✓ All imports working
- ✓ S-entropy calculations
- ✓ Environmental measurements  
- ✓ Search demo functionality

## Support

If issues persist:
1. Check Python version: `python --version` (should be 3.8+)
2. Check virtual environment is activated
3. Try: `pip install -e . --force-reinstall --no-deps`
4. Run: `python test_installation.py` to diagnose issues

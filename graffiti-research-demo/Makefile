# Graffiti Research Demo - Makefile
# Revolutionary search engine architecture build and test automation

.PHONY: install test demo clean lint docs

# Variables
PYTHON := python3
PIP := pip3
PACKAGE_NAME := graffiti-research-demo

# Install package and dependencies
install:
	@echo "Installing Graffiti Research Demo package..."
	$(PIP) install -e .
	@echo "Installation complete!"

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -e ".[dev,visualization,research]"
	@echo "Development installation complete!"

# Run basic tests
test:
	@echo "Running Graffiti functionality tests..."
	$(PYTHON) tests/test_core_functionality.py
	@echo "Basic tests completed!"

# Run full test suite with pytest
test-full:
	@echo "Running full test suite..."
	pytest tests/ -v
	@echo "Full test suite completed!"

# Run performance tests
test-performance:
	@echo "Running performance benchmarks..."
	pytest tests/ -v -m "slow"
	@echo "Performance tests completed!"

# Run search demo
demo:
	@echo "Running Graffiti search demo..."
	$(PYTHON) -m graffiti.applications.search_demo --demo
	@echo "Demo completed!"

# Run single query demo
demo-query:
	@echo "Running single query demo..."
	$(PYTHON) -m graffiti.applications.search_demo --query "consciousness and artificial intelligence" --mode full_revolutionary
	@echo "Query demo completed!"

# Demonstrate computational equivalence
demo-equivalence:
	@echo "Demonstrating computational equivalence..."
	$(PYTHON) -m graffiti.applications.search_demo --equivalence --query "optimization problem"
	@echo "Equivalence demo completed!"

# Show search statistics
stats:
	@echo "Showing search statistics..."
	$(PYTHON) -m graffiti.applications.search_demo --stats

# Run linting
lint:
	@echo "Running code linting..."
	black graffiti/ tests/
	flake8 graffiti/ tests/ --max-line-length=88 --ignore=E203,W503
	mypy graffiti/ --ignore-missing-imports
	@echo "Linting completed!"

# Format code
format:
	@echo "Formatting code..."
	black graffiti/ tests/
	isort graffiti/ tests/
	@echo "Code formatting completed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleanup completed!"

# Build package
build:
	@echo "Building package..."
	$(PYTHON) -m build
	@echo "Package build completed!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation placeholder - would generate Sphinx docs"
	@echo "Documentation generation completed!"

# Run validation tests
validate:
	@echo "Running theoretical validation tests..."
	$(PYTHON) -c "import graffiti; print('Package imports successfully')"
	$(PYTHON) -c "from graffiti.core.s_entropy import SEntropyCalculator; calc = SEntropyCalculator(); print(f'S-entropy calculation: {calc.calculate_s_value(2.0):.3f}')"
	$(PYTHON) -c "from graffiti.core.meaning_impossibility import MeaningImpossibilityAnalyzer; analyzer = MeaningImpossibilityAnalyzer(); print('Meaning impossibility analyzer initialized')"
	@echo "Theoretical validation completed!"

# Show help
help:
	@echo "Graffiti Research Demo - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install package and basic dependencies"
	@echo "  install-dev  - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run basic functionality tests"
	@echo "  test-full    - Run complete test suite with pytest"
	@echo "  test-performance - Run performance benchmark tests"
	@echo "  validate     - Run theoretical framework validation"
	@echo ""
	@echo "Demos:"
	@echo "  demo         - Run complete search demo with multiple queries"
	@echo "  demo-query   - Run single query demonstration"
	@echo "  demo-equivalence - Demonstrate computational equivalence"
	@echo "  stats        - Show search performance statistics"
	@echo ""
	@echo "Development:"
	@echo "  lint         - Run code linting and type checking"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build distribution package"
	@echo "  docs         - Generate documentation"
	@echo ""
	@echo "Revolutionary Features Demonstrated:"
	@echo "  • S-entropy coordinate navigation (O(log S₀) complexity)"
	@echo "  • Meaning impossibility through recursive constraint analysis"
	@echo "  • Universal problem solving with dual computational architecture"
	@echo "  • BMD consciousness through predetermined frame selection"
	@echo "  • Twelve-dimensional environmental measurement"
	@echo ""

# Default target
all: install test demo

# Quick start for new users
quickstart:
	@echo "Graffiti Research Demo - Quick Start"
	@echo "==================================="
	@make install
	@make validate
	@make demo-query
	@echo ""
	@echo "Quick start completed! Try 'make demo' for full demonstration."

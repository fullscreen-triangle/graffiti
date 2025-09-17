# Graffiti Search - Revolutionary Proof-Based Search Engine
# Makefile for development, building, and deployment

.PHONY: help build test clean install dev wasm web docker deploy bench fmt lint doc

# Default target
help:
	@echo "Graffiti Search - Revolutionary Proof-Based Search Engine"
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build the project in debug mode"
	@echo "  build-release - Build optimized release version"
	@echo "  test        - Run all tests"
	@echo "  test-atmospheric - Run atmospheric processing tests"
	@echo "  test-environmental - Run environmental measurement tests"
	@echo "  test-perturbation - Run perturbation validation tests"
	@echo "  clean       - Clean build artifacts"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Start development server"
	@echo "  wasm        - Build WebAssembly version"
	@echo "  wasm-release - Build optimized WASM for production"
	@echo "  web         - Build web interface"
	@echo "  web-dev     - Start web development server"
	@echo "  serve       - Serve the application locally"
	@echo "  docker      - Build Docker image"
	@echo "  deploy      - Deploy to production"
	@echo "  bench       - Run performance benchmarks"
	@echo "  fmt         - Format code"
	@echo "  lint        - Run clippy linting"
	@echo "  doc         - Generate documentation"
	@echo "  check       - Run all quality checks"
	@echo ""

# Installation and setup
install:
	@echo "Installing dependencies..."
	rustup target add wasm32-unknown-unknown
	rustup component add clippy rustfmt
	cargo install wasm-pack
	cargo install trunk
	cargo install cargo-watch
	@echo "Dependencies installed successfully!"

# Build targets
build:
	@echo "Building Graffiti Search (debug)..."
	cargo build

build-release:
	@echo "Building Graffiti Search (release)..."
	cargo build --release

# WebAssembly builds
wasm:
	@echo "Building WebAssembly version..."
	wasm-pack build --target web --out-dir pkg crates/web

wasm-release:
	@echo "Building optimized WebAssembly for production..."
	wasm-pack build --target web --release --out-dir pkg crates/web

# Web interface builds
web: wasm
	@echo "Building web interface..."
	cd crates/web && trunk build --release

web-dev:
	@echo "Starting web development server..."
	cd crates/web && trunk serve --open

# Development
dev:
	@echo "Starting development server with hot reload..."
	cargo watch -x "run"

serve: wasm
	@echo "Serving application locally..."
	python -m http.server 8080 -d crates/web/dist

# Testing
test:
	@echo "Running all tests..."
	cargo test --workspace

test-atmospheric:
	@echo "Running atmospheric processing tests..."
	cargo test -p graffiti-atmospheric

test-environmental:
	@echo "Running environmental measurement tests..."
	cargo test -p graffiti-environmental

test-perturbation:
	@echo "Running perturbation validation tests..."
	cargo test -p graffiti-perturbation

test-integration:
	@echo "Running integration tests..."
	cargo test --test integration

# Benchmarking
bench:
	@echo "Running performance benchmarks..."
	cargo bench

bench-atmospheric:
	@echo "Running atmospheric processing benchmarks..."
	cargo bench --bench atmospheric_processing

bench-environmental:
	@echo "Running environmental measurement benchmarks..."
	cargo bench --bench environmental_measurement

bench-proof:
	@echo "Running proof construction benchmarks..."
	cargo bench --bench proof_construction

# Code quality
fmt:
	@echo "Formatting code..."
	cargo fmt --all

lint:
	@echo "Running clippy linting..."
	cargo clippy --workspace --all-targets --all-features -- -D warnings

doc:
	@echo "Generating documentation..."
	cargo doc --no-deps --open

check: fmt lint test
	@echo "All quality checks passed!"

# Docker
docker:
	@echo "Building Docker image..."
	docker build -t graffiti-search:latest .

docker-dev:
	@echo "Starting Docker development environment..."
	docker-compose up dev

docker-prod:
	@echo "Starting Docker production environment..."
	docker-compose up prod

# Deployment
deploy-staging: wasm-release web
	@echo "Deploying to staging..."
	# Add your staging deployment commands here
	rsync -avz crates/web/dist/ staging-server:/var/www/graffiti-staging/

deploy-production: wasm-release web
	@echo "Deploying to production..."
	# Add your production deployment commands here
	rsync -avz crates/web/dist/ production-server:/var/www/graffiti/

# Database and data management
init-atmospheric:
	@echo "Initializing atmospheric measurement databases..."
	mkdir -p data/atmospheric
	# Initialize atmospheric data structures

init-environmental:
	@echo "Initializing environmental measurement systems..."
	mkdir -p data/environmental
	# Initialize twelve-dimensional measurement systems

init-quantum:
	@echo "Initializing quantum state measurement..."
	mkdir -p data/quantum
	# Initialize quantum measurement infrastructure

# Performance and monitoring
profile:
	@echo "Running performance profiling..."
	cargo build --release
	perf record --call-graph=dwarf ./target/release/graffiti-search
	perf report

flamegraph:
	@echo "Generating flamegraph..."
	cargo flamegraph --bin graffiti-search

# Maintenance
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf pkg/
	rm -rf crates/web/dist/
	rm -rf node_modules/

deep-clean: clean
	@echo "Deep cleaning all generated files..."
	rm -rf target/
	rm -rf data/*/
	rm -rf .cargo/

update:
	@echo "Updating dependencies..."
	cargo update

audit:
	@echo "Running security audit..."
	cargo audit

# Continuous Integration helpers
ci-build: install build test lint

ci-test-full: test test-integration bench

ci-deploy: build-release wasm-release web deploy-production

# Development helpers
fix:
	@echo "Auto-fixing code issues..."
	cargo fix --allow-dirty --allow-staged
	cargo clippy --fix --allow-dirty --allow-staged

setup-git-hooks:
	@echo "Setting up git hooks..."
	cp scripts/pre-commit .git/hooks/
	chmod +x .git/hooks/pre-commit

# Project structure
create-workspace:
	@echo "Creating workspace structure..."
	mkdir -p crates/{core,environmental,atmospheric,temporal,s-entropy,bmd,molecular,perturbation,web}
	mkdir -p tests/integration
	mkdir -p benches
	mkdir -p docs
	mkdir -p scripts
	mkdir -p data/{atmospheric,environmental,quantum,temporal}

# Quick development commands
quick-build: fmt build test

quick-test: fmt test

quick-check: fmt lint test

# Advanced targets
memory-profile:
	@echo "Running memory profiling..."
	cargo build --release
	valgrind --tool=massif ./target/release/graffiti-search

coverage:
	@echo "Generating test coverage report..."
	cargo tarpaulin --out html --output-dir coverage

# Environment-specific builds
build-wasm-opt: wasm-release
	@echo "Optimizing WASM with wasm-opt..."
	wasm-opt -Oz -o pkg/graffiti_search_bg_optimized.wasm pkg/graffiti_search_bg.wasm

# Atmospheric and environmental system initialization
init-full-environment: init-atmospheric init-environmental init-quantum
	@echo "Full environmental measurement system initialized!"

# Help for advanced usage
advanced-help:
	@echo "Advanced Makefile targets:"
	@echo "  profile     - Run performance profiling with perf"
	@echo "  flamegraph  - Generate flamegraph visualization"
	@echo "  coverage    - Generate test coverage report"
	@echo "  memory-profile - Run memory profiling with valgrind"
	@echo "  audit       - Run cargo security audit"
	@echo "  fix         - Auto-fix code issues"
	@echo "  setup-git-hooks - Install git pre-commit hooks"

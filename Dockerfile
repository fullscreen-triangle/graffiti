# Graffiti Search - Revolutionary Proof-Based Search Engine
# Multi-stage Docker build for production deployment

# Build stage
FROM rust:1.75-bullseye as builder

# Install WebAssembly target and tools
RUN rustup target add wasm32-unknown-unknown
RUN cargo install wasm-pack trunk

# Install Node.js for web tooling
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Create app directory
WORKDIR /app

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/*/

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Build WebAssembly components
RUN wasm-pack build --target web --release --out-dir pkg crates/web

# Build web interface
WORKDIR /app/crates/web
RUN trunk build --release

# Runtime stage - minimal image for production
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r graffiti && useradd -r -g graffiti graffiti

# Create app directory
WORKDIR /app

# Copy built application
COPY --from=builder /app/target/release/graffiti-search ./
COPY --from=builder /app/crates/web/dist ./web/

# Copy configuration files
COPY config/ ./config/

# Create data directories with proper permissions
RUN mkdir -p data/{atmospheric,environmental,quantum,temporal} && \
    chown -R graffiti:graffiti /app

# Switch to non-root user
USER graffiti

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Environment variables
ENV RUST_LOG=info
ENV GRAFFITI_PORT=8080
ENV GRAFFITI_HOST=0.0.0.0

# Run the application
CMD ["./graffiti-search"]

# Development stage - includes development tools
FROM builder as development

WORKDIR /app

# Install development tools
RUN cargo install cargo-watch cargo-audit cargo-tarpaulin

# Install debugging tools
RUN apt-get update && \
    apt-get install -y \
    gdb \
    valgrind \
    perf-tools-unstable \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Environment for development
ENV RUST_LOG=debug
ENV GRAFFITI_ENV=development

# Expose additional ports for development
EXPOSE 8080 8000 3000

# Development command
CMD ["cargo", "watch", "-x", "run"]

# Testing stage - for CI/CD
FROM builder as testing

WORKDIR /app

# Install testing tools
RUN cargo install cargo-tarpaulin

# Run tests
RUN cargo test --workspace --release

# Run benchmarks
RUN cargo bench --no-run

# Generate test coverage
RUN cargo tarpaulin --out xml --output-dir coverage

# Production stage with nginx for static file serving
FROM nginx:alpine as web-server

# Copy built web interface
COPY --from=builder /app/crates/web/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Health check for nginx
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/health || exit 1

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]

# Atmospheric processing worker stage
FROM builder as atmospheric-worker

WORKDIR /app

# Install scientific computing libraries
RUN apt-get update && \
    apt-get install -y \
    libopenmpi-dev \
    libfftw3-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy atmospheric processing binary
COPY --from=builder /app/target/release/graffiti-atmospheric ./

# Create atmospheric data directory
RUN mkdir -p data/atmospheric && \
    chown -R graffiti:graffiti /app/data

USER graffiti

# Environment for atmospheric processing
ENV RUST_LOG=info
ENV ATMOSPHERIC_WORKERS=auto
ENV MOLECULAR_PROCESSING_MODE=distributed

# Command for atmospheric worker
CMD ["./graffiti-atmospheric", "--worker"]

# Environmental measurement stage
FROM builder as environmental-sensor

WORKDIR /app

# Install sensor libraries and drivers
RUN apt-get update && \
    apt-get install -y \
    libudev-dev \
    libusb-1.0-0-dev \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy environmental measurement binary
COPY --from=builder /app/target/release/graffiti-environmental ./

# Create environmental data directory
RUN mkdir -p data/environmental && \
    chown -R graffiti:graffiti /app/data

USER graffiti

# Environment for environmental sensors
ENV RUST_LOG=info
ENV SENSOR_POLLING_INTERVAL=100ms
ENV TWELVE_DIMENSIONAL_MODE=true

# Command for environmental sensor
CMD ["./graffiti-environmental", "--sensor-daemon"]

# Quantum measurement stage (specialized for quantum processing)
FROM builder as quantum-processor

WORKDIR /app

# Install quantum computing libraries
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python quantum libraries
RUN pip3 install qiskit cirq pennylane

# Copy quantum processing binary
COPY --from=builder /app/target/release/graffiti-quantum ./

USER graffiti

# Environment for quantum processing
ENV RUST_LOG=info
ENV QUANTUM_BACKEND=simulator
ENV COHERENCE_TIME_THRESHOLD=100us

# Command for quantum processor
CMD ["./graffiti-quantum", "--quantum-daemon"]

# Complete system stage - orchestrates all components
FROM debian:bullseye-slim as system-orchestrator

# Install runtime dependencies and orchestration tools
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    libssl1.1 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all built components
COPY --from=builder /app/target/release/graffiti-search ./
COPY --from=builder /app/target/release/graffiti-atmospheric ./
COPY --from=builder /app/target/release/graffiti-environmental ./
COPY --from=builder /app/target/release/graffiti-quantum ./

# Copy orchestration scripts
COPY docker/orchestrator.sh ./
COPY docker/health-check.sh ./

# Create all necessary directories
RUN mkdir -p data/{atmospheric,environmental,quantum,temporal,molecular,s-entropy} && \
    mkdir -p logs/{atmospheric,environmental,quantum,temporal} && \
    groupadd -r graffiti && useradd -r -g graffiti graffiti && \
    chown -R graffiti:graffiti /app

USER graffiti

# Health check for complete system
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD ./health-check.sh

# Expose orchestrator port
EXPOSE 8080

# Environment for complete system
ENV RUST_LOG=info
ENV GRAFFITI_MODE=orchestrator
ENV SYSTEM_HEALTH_CHECK_INTERVAL=30s

# Start system orchestrator
CMD ["./orchestrator.sh"]

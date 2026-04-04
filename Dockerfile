# ── Stage 1: Build React dashboard ───────────────────────────────────────────
FROM node:20-alpine AS dashboard
WORKDIR /build/dashboard
COPY dashboard/package*.json ./
RUN npm ci --silent
COPY dashboard/ ./
RUN npm run build

# ── Stage 2: Build Rust binaries ─────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder

# Pre-fetch dependencies using a stub workspace (layer cache trick).
# Only re-runs when Cargo.toml / Cargo.lock changes.
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/tracerazor-core/Cargo.toml      crates/tracerazor-core/Cargo.toml
COPY crates/tracerazor-ingest/Cargo.toml    crates/tracerazor-ingest/Cargo.toml
COPY crates/tracerazor-semantic/Cargo.toml  crates/tracerazor-semantic/Cargo.toml
COPY crates/tracerazor-store/Cargo.toml     crates/tracerazor-store/Cargo.toml
COPY crates/tracerazor-server/Cargo.toml    crates/tracerazor-server/Cargo.toml
COPY crates/tracerazor-proxy/Cargo.toml     crates/tracerazor-proxy/Cargo.toml
COPY crates/tracerazor-cli/Cargo.toml       crates/tracerazor-cli/Cargo.toml

# Stub every lib/main so cargo can resolve the dependency graph.
RUN for crate in tracerazor-core tracerazor-ingest tracerazor-semantic \
        tracerazor-store tracerazor-proxy; do \
      mkdir -p crates/$crate/src && echo "pub fn _stub() {}" > crates/$crate/src/lib.rs; \
    done && \
    mkdir -p crates/tracerazor-server/src && echo "fn main() {}" > crates/tracerazor-server/src/main.rs && \
    mkdir -p crates/tracerazor-cli/src && echo "fn main() {}" > crates/tracerazor-cli/src/main.rs

RUN cargo fetch

# Now copy real source and build for release.
COPY crates/ crates/
RUN cargo build --release -p tracerazor-server -p tracerazor

# ── Stage 3: Minimal runtime image ───────────────────────────────────────────
FROM debian:bookworm-slim

# ca-certificates needed for HTTPS calls to OpenAI API.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/tracerazor-server ./
COPY --from=builder /build/target/release/tracerazor ./
COPY --from=dashboard /build/dashboard/dist ./dashboard/dist
COPY traces/ ./traces/

# Data directory for the persistent SurrealDB file.
RUN mkdir -p /app/data

ENV TRACERAZOR_DB_PATH=/app/data/tracerazor.db
ENV PORT=8080

EXPOSE 8080

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/api/ || exit 1

CMD ["./tracerazor-server"]

# ── Stage 1: Build React dashboard ───────────────────────────────────────────
FROM node:20-alpine AS dashboard
WORKDIR /build/dashboard
COPY dashboard/package*.json ./
RUN npm ci --silent
COPY dashboard/ ./
RUN npm run build

# ── Stage 2: Build Rust binaries ─────────────────────────────────────────────
FROM rust:1.88-bookworm@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0 AS builder

# Pre-fetch dependencies using a stub workspace (layer cache trick).
# Only re-runs when Cargo.toml / Cargo.lock changes.
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/tracerazor-core/Cargo.toml      crates/tracerazor-core/Cargo.toml
COPY crates/tracerazor-ingest/Cargo.toml    crates/tracerazor-ingest/Cargo.toml
COPY crates/tracerazor-semantic/Cargo.toml  crates/tracerazor-semantic/Cargo.toml
COPY crates/tracerazor-store/Cargo.toml     crates/tracerazor-store/Cargo.toml
COPY crates/tracerazor-server/Cargo.toml    crates/tracerazor-server/Cargo.toml
COPY crates/tracerazor-cli/Cargo.toml       crates/tracerazor-cli/Cargo.toml

# Stub every lib/main so cargo can resolve the dependency graph.
RUN for crate in tracerazor-core tracerazor-ingest tracerazor-semantic \
        tracerazor-store; do \
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

# Data directory for the persistent SQLite database file.
RUN mkdir -p /app/data

# Run as an unprivileged user so an app-level RCE can't trivially own the
# container as root. The chown must happen BEFORE the VOLUME instruction —
# changes to a volume path made afterwards are discarded by the builder — so
# the runtime user can write the SQLite database into the named volume.
RUN useradd --system --uid 10001 --user-group --no-create-home appuser \
 && chown -R appuser:appuser /app

ENV TRACERAZOR_DB_PATH=/app/data/tracerazor.db
ENV PORT=8080
# Keep the standalone image on the server's fail-closed loopback default.
# `docker run -p` therefore does not expose an unauthenticated backend. The
# default Compose topology explicitly provisions bearer auth and a Caddy TLS
# boundary before it opts the backend into a private non-loopback listener.
ENV TRACERAZOR_BIND_ADDR=127.0.0.1

EXPOSE 8080

VOLUME ["/app/data"]

USER appuser

# Use the server binary's own probe mode — no curl/wget needed in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/app/tracerazor-server", "--health-check"]

CMD ["./tracerazor-server"]

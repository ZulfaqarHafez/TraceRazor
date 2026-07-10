from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_standalone_dashboard_image_has_no_unsafe_listener_default():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV TRACERAZOR_BIND_ADDR=0.0.0.0" not in dockerfile
    assert "TRACERAZOR_TLS_TERMINATED" not in dockerfile
    assert "TRACERAZOR_API_TOKEN" not in dockerfile
    assert "server's fail-closed loopback default" in dockerfile
    assert "ENV TRACERAZOR_BIND_ADDR=127.0.0.1" in dockerfile
    assert 'CMD ["./tracerazor-server"]' in dockerfile
    assert (
        "rust:1.88-bookworm@sha256:"
        "af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0"
        in dockerfile
    )


def test_compose_publishes_only_the_tls_gateway_on_host_loopback():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    backend, gateway = compose.split("\n  gateway:\n", 1)

    assert "ports:" not in backend
    assert 'expose:\n      - "8080"' in backend
    assert "TRACERAZOR_BIND_ADDR: 0.0.0.0" in backend
    assert 'TRACERAZOR_TLS_TERMINATED: "true"' in backend
    assert "TRACERAZOR_API_TOKEN:?" in backend

    assert (
        "caddy:2.10.2-alpine@sha256:"
        "4c6e91c6ed0e2fa03efd5b44747b625fec79bc9cd06ac5235a779726618e530d"
        in gateway
    )
    assert '"127.0.0.1:${PORT:-8080}:8443"' in gateway
    assert "TRACERAZOR_API_TOKEN:?" in gateway
    assert "condition: service_healthy" in gateway
    assert "./Caddyfile.compose:/etc/caddy/Caddyfile:ro" in gateway
    assert '"${PORT:-8080}:8080"' not in compose
    assert '"0.0.0.0:${PORT:-8080}' not in compose


def test_compose_caddyfile_has_real_tls_and_injects_backend_auth():
    caddyfile = (ROOT / "Caddyfile.compose").read_text(encoding="utf-8")

    assert "admin off" in caddyfile
    assert "https://localhost:8443" in caddyfile
    assert "tls internal" in caddyfile
    assert "reverse_proxy tracerazor:8080" in caddyfile
    assert 'header_up Authorization "Bearer {$TRACERAZOR_API_TOKEN}"' in caddyfile
    assert "http://localhost:8443" not in caddyfile


def test_container_documentation_states_secret_ca_and_standalone_boundaries():
    docs = (ROOT / "docs" / "container.md").read_text(encoding="utf-8")

    assert "deliberately refuses to render" in docs
    assert "tracerazor-local-ca.crt" in docs
    assert "https://localhost:8080" in docs
    assert "does **not** expose the service" in docs
    assert "must not be set merely to bypass startup validation" in docs

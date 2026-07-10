# Secure local dashboard container

The default Compose topology is local-only and fail-closed. The TraceRazor
backend listens on its private Compose network with bearer authentication. A
Caddy sidecar terminates HTTPS, injects that bearer credential upstream, and is
the only service published to the host, on IPv4 `127.0.0.1`.

## Start it

Generate a random token, then add the printed line to the repository's
untracked `.env` file without replacing any existing entries:

```bash
python -c "import secrets; print('TRACERAZOR_API_TOKEN=' + secrets.token_hex(32))"
docker compose config                 # validates the required secret
docker compose up --build
```

Compose deliberately refuses to render when `TRACERAZOR_API_TOKEN` is absent.
The dashboard is served at `https://localhost:8080` by default. Set `PORT` in
`.env` to change only the host-loopback port.

Caddy uses a persisted local CA. After the first start, copy its root
certificate and trust it in the browser or operating-system trust store:

```bash
docker compose cp gateway:/data/caddy/pki/authorities/local/root.crt ./tracerazor-local-ca.crt
curl --cacert ./tracerazor-local-ca.crt https://localhost:8080/healthz
```

The gateway replaces the upstream `Authorization` header, so the token is not
sent to browser code. The backend port is only `expose`d to the Compose network;
it is never published to the host.

## Standalone image behavior

The dashboard `Dockerfile` intentionally leaves the server on its loopback
default. Consequently, this does **not** expose the service:

```bash
docker run --rm -p 8080:8080 tracerazor:1.1.0
```

That behavior prevents an ordinary port publication from silently turning an
unauthenticated local server into a network service. Use the default Compose
topology for local dashboard access.

For a different deployment, supply a real TLS reverse proxy, block direct
access to the backend port, set a bearer token, and only then set
`TRACERAZOR_BIND_ADDR=0.0.0.0` and `TRACERAZOR_TLS_TERMINATED=true` on the
backend. The TLS assertion describes an existing boundary; it never enables
native TLS and must not be set merely to bypass startup validation.

The bundled Caddy configuration is for loopback development. Public hosting
requires a real domain/certificate, explicit ingress policy, and separate
deployment review.

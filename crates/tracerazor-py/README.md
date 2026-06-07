# tracerazor-native (PyO3 bindings)

Native, in-process Python bindings for the TraceRazor auditor core. Lets the
Python `teacher` package call the Rust auditor directly instead of shelling out
to the CLI — no subprocess, no temp files.

## Status

This crate is **excluded from the root Cargo workspace** (see the `exclude`
entry in the top-level `Cargo.toml`) so that `cargo build --workspace` and the
CI gate never depend on `pyo3`. It is an **opt-in** acceleration layer.

> Note: this binding was authored against the existing core API
> (`tracerazor_core::analyse` / `tracerazor_ingest::parse` /
> `tracerazor_semantic::default_similarity_fn`) but has **not** been compiled in
> the environment it was written in (no crates.io access to fetch `pyo3`). Build
> it where crates.io is reachable with the command below; the Python layer works
> without it via the subprocess backend in the meantime.

## Build

```bash
pip install maturin
maturin develop -m crates/tracerazor-py/Cargo.toml --release
```

## Use

Once installed, `teacher.Diagnoser` auto-detects it:

```python
from teacher import Diagnoser
d = Diagnoser()           # backend == "native" if tracerazor_native importable
print(d.backend)
```

Or call it directly:

```python
import json, tracerazor_native
report = json.loads(tracerazor_native.audit_json(json.dumps(trace_dict)))
print(report["score"]["score"])      # TAS
```

## API

- `audit_json(trace_json: str) -> str` — audit auto-detected JSON, return report JSON.
- `audit_json_with_format(trace_json: str, fmt: str) -> str` — `fmt` ∈
  `{"auto","raw","langsmith","otel"}`.

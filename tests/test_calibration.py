"""Tests for the TAS weight-calibration math and dataset loading.

The fitting tests are hermetic (pure numpy/scipy, no binary). End-to-end
feature extraction is covered by the CI calibration smoke step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from calibration.calibrate import (  # noqa: E402
    METRICS,
    fit_weights,
    load_dataset,
    predict,
    r2,
    recoverable_fraction,
)


def test_weights_are_a_convex_combination():
    rng = np.random.default_rng(0)
    X = rng.random((40, len(METRICS)))
    y = rng.random(40)
    w = fit_weights(X, y)
    assert w.shape == (len(METRICS),)
    assert (w >= -1e-9).all(), "weights must be non-negative"
    assert abs(w.sum() - 1.0) < 1e-6, "weights must sum to 1"


def test_fit_recovers_a_known_linear_target():
    # y is exactly metric 0 (SRR). The fit should put ~all mass on metric 0.
    rng = np.random.default_rng(1)
    X = rng.random((80, len(METRICS)))
    y = X[:, 0]
    w = fit_weights(X, y)
    assert w[0] > 0.9, f"expected mass on metric 0, got {w}"
    assert r2(y, predict(X, w)) > 0.99


def test_l2_pulls_weights_toward_uniform():
    rng = np.random.default_rng(2)
    X = rng.random((30, len(METRICS)))
    y = X[:, 0]
    spread_no_reg = fit_weights(X, y, l2=0.0).std()
    spread_reg = fit_weights(X, y, l2=5.0).std()
    assert spread_reg < spread_no_reg, "L2 should spread weight toward uniform"


def test_recoverable_fraction_math():
    assert abs(recoverable_fraction(1000, 600) - 0.4) < 1e-9
    assert recoverable_fraction(1000, 1200) == 0.0   # after bigger -> clamp to 0
    assert recoverable_fraction(0, 0) == 0.0          # guard against div by zero
    assert recoverable_fraction(1000, 0) == 1.0


def test_load_dataset_before_after_defers_fraction(tmp_path):
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.json").write_text("{}")
    manifest = {"name": "t", "entries": [{"before": "b.json", "after": "a.json"}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))

    name, samples = load_dataset(tmp_path / "m.json")
    assert name == "t"
    assert len(samples) == 1
    # Fraction is computed later from audited token totals, not at load.
    assert samples[0].target_fraction is None
    assert samples[0].after_path is not None
    assert samples[0].after_path.name == "a.json"


def test_load_dataset_explicit_fraction(tmp_path):
    trace = {"trace_id": "x", "steps": []}
    (tmp_path / "x.json").write_text(json.dumps(trace))
    manifest = {"entries": [{"trace": "x.json", "recoverable_fraction": 0.25}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    _, samples = load_dataset(tmp_path / "m.json")
    assert abs(samples[0].target_fraction - 0.25) < 1e-9


# ── Adapter: harness exports -> manifest ──────────────────────────────────────
from calibration.adapt import main as adapt_main  # noqa: E402


def _touch(p):
    p.write_text("{}")
    return p


def test_adapt_csv_before_after(tmp_path):
    _touch(tmp_path / "b1.json"); _touch(tmp_path / "a1.json")
    (tmp_path / "runs.csv").write_text("before_path,after_path\nb1.json,a1.json\n")
    out = tmp_path / "manifest.json"
    rc = adapt_main(["--csv", str(tmp_path / "runs.csv"),
                     "--before-col", "before_path", "--after-col", "after_path",
                     "--out", str(out)])
    assert rc == 0
    m = json.loads(out.read_text())
    assert len(m["entries"]) == 1
    e = m["entries"][0]
    assert e["before"].endswith("b1.json") and e["after"].endswith("a1.json")
    assert Path(e["before"]).is_absolute()


def test_adapt_jsonl_labelled(tmp_path):
    _touch(tmp_path / "t1.json")
    (tmp_path / "runs.jsonl").write_text(json.dumps({"path": "t1.json", "frac": 0.3}) + "\n")
    out = tmp_path / "m.json"
    rc = adapt_main(["--jsonl", str(tmp_path / "runs.jsonl"),
                     "--trace-col", "path", "--label-col", "frac", "--out", str(out)])
    assert rc == 0
    e = json.loads(out.read_text())["entries"][0]
    assert e["recoverable_fraction"] == 0.3
    assert e["trace"].endswith("t1.json")


def test_adapt_label_out_of_range_errors(tmp_path):
    _touch(tmp_path / "t1.json")
    (tmp_path / "runs.csv").write_text("path,frac\nt1.json,1.5\n")
    with pytest.raises(SystemExit):
        adapt_main(["--csv", str(tmp_path / "runs.csv"),
                    "--trace-col", "path", "--label-col", "frac",
                    "--out", str(tmp_path / "m.json")])


def test_adapt_pair_dir(tmp_path):
    _touch(tmp_path / "task1_before.json"); _touch(tmp_path / "task1_after.json")
    _touch(tmp_path / "task2_before.json"); _touch(tmp_path / "task2_after.json")
    out = tmp_path / "m.json"
    rc = adapt_main(["--pair-dir", str(tmp_path),
                     "--before-suffix", "_before.json", "--after-suffix", "_after.json",
                     "--out", str(out)])
    assert rc == 0
    entries = json.loads(out.read_text())["entries"]
    assert len(entries) == 2
    assert all("before" in e and "after" in e for e in entries)


def test_adapt_missing_file_errors(tmp_path):
    (tmp_path / "runs.csv").write_text("before_path,after_path\nnope_b.json,nope_a.json\n")
    with pytest.raises(SystemExit):
        adapt_main(["--csv", str(tmp_path / "runs.csv"),
                    "--before-col", "before_path", "--after-col", "after_path",
                    "--out", str(tmp_path / "m.json")])


def test_adapt_allow_missing_skips_rows(tmp_path):
    _touch(tmp_path / "ok.json")
    (tmp_path / "runs.csv").write_text(
        "trace,frac\nok.json,0.2\ngone.json,0.3\n")
    out = tmp_path / "m.json"
    rc = adapt_main(["--csv", str(tmp_path / "runs.csv"),
                     "--trace-col", "trace", "--label-col", "frac",
                     "--allow-missing", "--out", str(out)])
    assert rc == 0
    entries = json.loads(out.read_text())["entries"]
    # the missing row is dropped, not written pointing at a nonexistent file
    assert len(entries) == 1
    assert entries[0]["trace"].endswith("ok.json")


def test_adapt_rejects_both_modes(tmp_path):
    _touch(tmp_path / "x.json")
    (tmp_path / "runs.csv").write_text("b,a,t,f\nx.json,x.json,x.json,0.1\n")
    with pytest.raises(SystemExit):
        adapt_main(["--csv", str(tmp_path / "runs.csv"),
                    "--before-col", "b", "--after-col", "a",
                    "--trace-col", "t", "--label-col", "f",
                    "--out", str(tmp_path / "m.json")])


def test_adapt_pair_dir_rejects_equal_suffixes(tmp_path):
    _touch(tmp_path / "x_before.json")
    with pytest.raises(SystemExit):
        adapt_main(["--pair-dir", str(tmp_path),
                    "--before-suffix", ".json", "--after-suffix", ".json",
                    "--out", str(tmp_path / "m.json")])

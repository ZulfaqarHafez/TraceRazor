"""Tests for the TAS weight-calibration math and dataset loading.

The fitting tests are hermetic (pure numpy/scipy, no binary). End-to-end
feature extraction is covered by the CI calibration smoke step.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from calibration.calibrate import (  # noqa: E402
    METRICS,
    fit_weights,
    load_dataset,
    predict,
    r2,
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


def test_load_dataset_before_after_fraction(tmp_path):
    before = {"trace_id": "b", "agent_name": "a", "framework": "raw",
              "total_tokens": 1000, "steps": []}
    after = {"trace_id": "c", "agent_name": "a", "framework": "raw",
             "total_tokens": 600, "steps": []}
    (tmp_path / "b.json").write_text(json.dumps(before))
    (tmp_path / "a.json").write_text(json.dumps(after))
    manifest = {"name": "t", "entries": [{"before": "b.json", "after": "a.json"}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))

    name, samples = load_dataset(tmp_path / "m.json")
    assert name == "t"
    assert len(samples) == 1
    # (1000 - 600) / 1000 = 0.4
    assert abs(samples[0].target_fraction - 0.4) < 1e-9


def test_load_dataset_explicit_fraction(tmp_path):
    trace = {"trace_id": "x", "steps": []}
    (tmp_path / "x.json").write_text(json.dumps(trace))
    manifest = {"entries": [{"trace": "x.json", "recoverable_fraction": 0.25}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    _, samples = load_dataset(tmp_path / "m.json")
    assert abs(samples[0].target_fraction - 0.25) < 1e-9

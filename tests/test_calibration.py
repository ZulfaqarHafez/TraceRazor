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


# ── Connector: messages-format trajectories -> traces + manifest ─────────────
from calibration.sources.from_messages import main as fm_main, messages_to_trace  # noqa: E402
import argparse as _argparse  # noqa: E402


def _convo(verbose: bool):
    """A short agent conversation in OpenAI messages format (>=5 agent steps)."""
    pad = " carefully reconsidering every detail to be thorough" if verbose else ""
    return [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "Inspect the failing test." + pad},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "open_file", "arguments": "test_x.py"}}]},
        {"role": "tool", "content": "file contents ok"},
        {"role": "assistant", "content": "Apply the fix to the handler." + pad},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "edit", "arguments": "patch"}}]},
        {"role": "tool", "content": "tests passed"},
    ]


def test_messages_to_trace_maps_steps():
    rec = {"instance_id": "i1", "model": "m", "resolved": True, "messages": _convo(False)}
    ns = _argparse.Namespace(messages_field="messages", id_field="instance_id",
                             model_field="model")
    trace = messages_to_trace(rec, ns)
    assert trace is not None
    types = {s["step_type"] for s in trace["steps"]}
    assert "reasoning" in types and "tool_call" in types
    assert all(s["tokens"] > 0 for s in trace["steps"])


def test_from_messages_builds_before_after_pairs(tmp_path):
    # 2 instances, each solved by a verbose and a lean model -> 2 before/after pairs
    records = []
    for inst in ("i1", "i2"):
        records.append({"instance_id": inst, "model": "verbose", "resolved": True,
                        "messages": _convo(True)})
        records.append({"instance_id": inst, "model": "lean", "resolved": True,
                        "messages": _convo(False)})
    # an unresolved run that must be ignored
    records.append({"instance_id": "i3", "model": "x", "resolved": False,
                    "messages": _convo(True)})
    jsonl = tmp_path / "traj.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records))

    manifest = tmp_path / "m.json"
    rc = fm_main(["--jsonl", str(jsonl), "--out-dir", str(tmp_path / "conv"),
                  "--manifest", str(manifest)])
    assert rc == 0
    entries = json.loads(manifest.read_text())["entries"]
    assert len(entries) == 2
    for e in entries:
        before = json.loads(Path(e["before"]).read_text())
        after = json.loads(Path(e["after"]).read_text())
        bt = sum(s["tokens"] for s in before["steps"])
        at = sum(s["tokens"] for s in after["steps"])
        assert bt > at, "verbose run should have more tokens than lean run"
    # the manifest must load in the calibrator
    _, samples = load_dataset(manifest)
    assert len(samples) == 2
    assert all(s.after_path is not None for s in samples)


from calibration.sources.from_taubench import main as tb_main  # noqa: E402


def test_from_taubench_emits_successful_episodes(tmp_path):
    # tau-bench shape: <model>-<domain>.json, list of episodes with reward/traj/task_id
    eps = [
        {"task_id": 0, "reward": 1.0, "trial": 0, "traj": _convo(True)},
        {"task_id": 0, "reward": 0.0, "trial": 1, "traj": _convo(False)},  # failed -> dropped
        {"task_id": 1, "reward": 1.0, "trial": 0, "traj": _convo(False)},
    ]
    d = tmp_path / "historical_trajectories"
    d.mkdir()
    (d / "gpt4o-airline.json").write_text(json.dumps(eps))
    out = tmp_path / "tb.jsonl"
    rc = tb_main(["--dir", str(d), "--out", str(out)])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2  # only the two reward=1 episodes
    assert all(r["resolved"] is True for r in rows)
    assert rows[0]["instance_id"] == "airline-0"
    assert rows[0]["model"].startswith("gpt4o")

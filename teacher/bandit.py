"""LinUCB contextual bandit for intervention arm selection.

Each unique intervention key is an "arm". The context vector encodes the
current waste pattern's kind (one-hot over WasteKind values), severity,
estimated token waste fraction, and log-trace-length.  Reward is the verified
token saving fraction on ACCEPT or -0.05 (small penalty) on REJECT.

Uses stdlib only — no numpy/scipy. Matrix dimension d=11 so Gaussian
elimination is O(d^3) ≈ 1331 ops per call; negligible latency.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

_WASTE_KINDS = ["srr", "ldi", "tca", "rda", "cce", "vdi", "shl"]
_DIM = 4 + len(_WASTE_KINDS)  # bias + severity + waste_pct + log_len + 7 = 11


def _ctx(severity: float, token_waste_pct: float, trace_len: int, kind_val: str) -> list[float]:
    onehot = [1.0 if kind_val == k else 0.0 for k in _WASTE_KINDS]
    return [
        1.0,
        min(float(severity), 1.0),
        min(float(token_waste_pct) / 100.0, 1.0),
        math.log1p(max(int(trace_len), 1)) / 10.0,
        *onehot,
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(M: list[list[float]], x: list[float]) -> list[float]:
    return [_dot(row, x) for row in M]


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A θ = b via Gaussian elimination (stdlib, no numpy)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        prow = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[prow] = M[prow], M[col]
        piv = M[col][col]
        if abs(piv) < 1e-12:
            continue
        for row in range(col + 1, n):
            f = M[row][col] / piv
            for c in range(col, n + 1):
                M[row][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        denom = M[i][i]
        if abs(denom) < 1e-12:
            continue
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / denom
    return x


def _inv(A: list[list[float]]) -> list[list[float]]:
    """Invert A via Gauss-Jordan elimination."""
    n = len(A)
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        prow = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[prow] = M[prow], M[col]
        piv = M[col][col]
        if abs(piv) < 1e-12:
            continue
        for c in range(2 * n):
            M[col][c] /= piv
        for row in range(n):
            if row != col and abs(M[row][col]) > 1e-12:
                f = M[row][col]
                for c in range(2 * n):
                    M[row][c] -= f * M[col][c]
    return [row[n:] for row in M]


class _Arm:
    """Per-arm ridge state: A = I + Σ xx^T, b = Σ r·x."""
    __slots__ = ("A", "b", "t")

    def __init__(self) -> None:
        d = _DIM
        self.A: list[list[float]] = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.b: list[float] = [0.0] * d
        self.t: int = 0

    def update(self, x: list[float], reward: float) -> None:
        for i in range(_DIM):
            self.b[i] += reward * x[i]
            for j in range(_DIM):
                self.A[i][j] += x[i] * x[j]
        self.t += 1

    def ucb_score(self, x: list[float], alpha: float) -> float:
        theta = _solve(self.A, self.b)
        Ainv = _inv(self.A)
        return _dot(theta, x) + alpha * math.sqrt(max(_dot(x, _mat_vec(Ainv, x)), 0.0))

    def to_dict(self) -> dict:
        return {"A": self.A, "b": self.b, "t": self.t}

    @classmethod
    def from_dict(cls, d: dict) -> "_Arm":
        arm = cls()
        arm.A, arm.b, arm.t = d["A"], d["b"], d.get("t", 0)
        return arm


class LinUCBBandit:
    """Disjoint LinUCB bandit — one ridge model per intervention key.

    Context: [bias, severity, waste_pct, log_trace_len, *waste_kind_onehot].
    Reward: token saving fraction on ACCEPT; -0.05 on REJECT.

    Optionally pass ``path`` to persist arm state as JSON across Teacher runs
    (same playbook-persistence pattern as ``Playbook``).
    """

    def __init__(self, alpha: float = 1.0, path: Optional[str] = None) -> None:
        self.alpha = alpha
        self.path = path
        self.arms: dict[str, _Arm] = {}
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()

    def _arm(self, key: str) -> _Arm:
        if key not in self.arms:
            self.arms[key] = _Arm()
        return self.arms[key]

    @staticmethod
    def _encode(iv, diagnosis) -> list[float]:
        kind_val = iv.waste_pattern.value if hasattr(iv.waste_pattern, "value") else str(iv.waste_pattern)
        severity, waste_pct = 0.5, 0.0
        total = getattr(diagnosis, "total_tokens", 1) or 1
        for p in getattr(diagnosis, "patterns", []):
            p_kind = p.kind.value if hasattr(p.kind, "value") else str(p.kind)
            if p_kind == kind_val:
                severity = getattr(p, "severity", 0.5)
                waste_pct = 100.0 * getattr(p, "est_token_waste", 0) / total
                break
        n_pats = max(len(getattr(diagnosis, "patterns", [])), 1)
        return _ctx(severity, waste_pct, n_pats, kind_val)

    def select(self, candidates: list, diagnosis) -> object:
        """Return the candidate intervention with the highest UCB score."""
        best, best_score = candidates[0], float("-inf")
        for iv in candidates:
            score = self._arm(iv.key).ucb_score(self._encode(iv, diagnosis), self.alpha)
            if score > best_score:
                best_score, best = score, iv
        return best

    def update(self, arm_key: str, iv, diagnosis, reward: float) -> None:
        """Record the outcome (reward) for arm_key in the current context."""
        self._arm(arm_key).update(self._encode(iv, diagnosis), reward)
        if self.path:
            self.save()

    def save(self) -> None:
        if not self.path:
            return
        with open(self.path, "w") as fh:
            json.dump({"alpha": self.alpha,
                       "arms": {k: v.to_dict() for k, v in self.arms.items()}}, fh)

    def _load(self) -> None:
        with open(self.path) as fh:
            d = json.load(fh)
        self.alpha = d.get("alpha", self.alpha)
        for k, v in d.get("arms", {}).items():
            self.arms[k] = _Arm.from_dict(v)

    def summary(self) -> str:
        if not self.arms:
            return "(bandit: no data)"
        lines = ["  arm                  trials  θ₀(bias)"]
        for k, arm in sorted(self.arms.items(), key=lambda kv: kv[1].t, reverse=True):
            theta0 = _solve(arm.A, arm.b)[0] if arm.t > 0 else 0.0
            lines.append(f"  {k:<22} {arm.t:4d}   {theta0:+.3f}")
        return "\n".join(lines)

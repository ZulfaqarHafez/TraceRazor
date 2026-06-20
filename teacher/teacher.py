"""The Teacher / Orchestrator.

Runs the teaching loop:  perceive -> diagnose -> plan (curriculum) -> act
-> verify (quality gate) -> remember.  Drives a target agent toward token
efficiency without ever regressing task success.

Three modes:
  * CURRICULUM  -- re-run a task set, greedily apply gate-passing interventions
                   tier-by-tier until TAS plateaus. Promotes a config.
  * AUTONOMOUS  -- same loop, single pass over the current diagnosis.
  * COACH       -- dry-run: emit a ranked report + config diff, promote nothing.

The Teacher is deliberately scaffolded with deterministic control flow; the
"policy head" decisions (which intervention to try first) are made by a simple
curriculum + playbook prior, standing in for the LLM ranker described in the
design spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .bandit import LinUCBBandit
from .diagnose import Diagnoser
from .gate import QualityGate
from .interventions import apply, propose
from .memory import Playbook
from .runner import Task
from .runners import OfflineRunner, Runner
from .schemas import (
    AgentConfig,
    Decision,
    Intervention,
    Outcome,
    Tier,
    VerifiedResult,
)


class Mode(str, Enum):
    CURRICULUM = "curriculum"
    AUTONOMOUS = "autonomous"
    COACH = "coach"


@dataclass
class TeacherResult:
    base_config: AgentConfig
    final_config: AgentConfig
    history: list[VerifiedResult]
    tas_trajectory: list[float]
    tokens_trajectory: list[float]
    mode: Mode
    promoted: bool
    runner: str = "offline"

    @property
    def total_token_saving_pct(self) -> float:
        if not self.tokens_trajectory or self.tokens_trajectory[0] == 0:
            return 0.0
        first, last = self.tokens_trajectory[0], self.tokens_trajectory[-1]
        return 100.0 * (first - last) / first

    @property
    def accepted(self) -> list[VerifiedResult]:
        return [h for h in self.history if h.accepted]


class Teacher:
    def __init__(self, base_config: AgentConfig, *, framework: str = "langgraph",
                 mode: Mode = Mode.CURRICULUM, gate: QualityGate | None = None,
                 playbook: Playbook | None = None, diagnoser: Diagnoser | None = None,
                 bandit: LinUCBBandit | None = None, patience: int = 2):
        self.base_config = base_config
        self.framework = framework
        self.mode = mode
        self.gate = gate or QualityGate()
        self.playbook = playbook or Playbook()
        self.diagnoser = diagnoser or Diagnoser()
        self.bandit = bandit
        self.patience = patience

    # -- main entry point --------------------------------------------------- #
    def improve(self, tasks: "list[Task] | None" = None, max_rounds: int = 8,
                runner: "Runner | None" = None) -> TeacherResult:
        """Run the curriculum.

        Pass ``runner`` to choose how candidates are exercised + measured:
          * default / ``OfflineRunner(tasks, ...)`` -- deterministic mock agent;
          * ``teacher.online.OnlineRunner(agent, holdout, ...)`` -- real HTTP
            agent loop (pair the Teacher with ``gate=StatGate()``).
        ``tasks`` is only needed when no ``runner`` is supplied.
        """
        if runner is None:
            if not tasks:
                raise ValueError("improve() needs `tasks` or a `runner`")
            runner = OfflineRunner(tasks, self.diagnoser)

        cfg = self.base_config.clone()
        baseline = runner.evaluate(cfg)
        history: list[VerifiedResult] = []
        tas_traj = [baseline.tas]
        tok_traj = [baseline.mean_tokens]
        rejects_in_a_row = 0
        tried_keys: set[str] = set()
        promote = self.mode is not Mode.COACH

        for _ in range(max_rounds):
            # PERCEIVE + DIAGNOSE on a fresh trace from the current config.
            sample_trace = runner.sample_trace(cfg)
            diagnosis = self.diagnoser.diagnose(sample_trace)
            candidates = [
                iv for iv in propose(diagnosis) if iv.id not in tried_keys
            ]
            if not candidates:
                break

            # PLAN: curriculum -- lowest unexhausted tier first, then by
            # playbook-primed expected payoff.
            iv = self._select(candidates, diagnosis)
            tried_keys.add(iv.id)

            # ACT: apply to a candidate config and RE-RUN on the holdout.
            trial_cfg = apply(iv, cfg)
            trial = runner.evaluate(trial_cfg)

            # VERIFY: quality-preservation gate.
            decision = self.gate.decide(baseline, trial)
            vr = VerifiedResult(
                intervention=iv, decision=decision,
                tokens_before=baseline.mean_tokens, tokens_after=trial.mean_tokens,
                success_before=baseline.success_rate, success_after=trial.success_rate,
                tas_before=baseline.tas, tas_after=trial.tas)
            history.append(vr)

            # FEEDBACK + REMEMBER.
            outcome = Outcome(
                pattern_signature=self._signature(iv, diagnosis),
                intervention_id=iv.id, accepted=vr.accepted,
                token_delta_pct=vr.token_delta_pct, tas_delta=vr.tas_after - vr.tas_before)
            self.playbook.record(outcome.pattern_signature, iv.waste_pattern,
                                 self.framework, iv.key, outcome)
            if self.bandit is not None:
                # Reward = fractional token saving on ACCEPT; small penalty on REJECT
                # so the bandit learns both what works and what wastes an iteration.
                reward = -vr.token_delta_pct / 100.0 if vr.accepted else -0.05
                self.bandit.update(iv.key, iv, diagnosis, reward)

            if decision is Decision.ACCEPT and promote:
                cfg = trial_cfg
                baseline = trial            # ratchet
                tas_traj.append(trial.tas)
                tok_traj.append(trial.mean_tokens)
                rejects_in_a_row = 0
            else:
                rejects_in_a_row += 1
                if rejects_in_a_row >= self.patience and self._tier_exhausted(
                        candidates, iv):
                    # plateau: cheap tiers exhausted, nothing landing.
                    pass
            if self.mode is Mode.AUTONOMOUS:
                break

        return TeacherResult(
            base_config=self.base_config, final_config=cfg, history=history,
            tas_trajectory=tas_traj, tokens_trajectory=tok_traj,
            mode=self.mode, promoted=promote, runner=getattr(runner, "name", "offline"))

    # -- COACH over real, non-rerunnable captured traces -------------------- #
    def coach(self, traces: list[dict]):
        """Diagnose captured traces, rank interventions by predicted savings +
        playbook prior, and return a ``CoachReport``. Promotes nothing.

        Used for traces from a live agent (e.g. via the LangGraph adapter) that
        cannot be re-run offline, so the closed-loop gate does not apply; we
        rank by the auditor's own ``estimated_token_savings`` instead.
        """
        from .report import CoachReport, Recommendation

        diagnoses = [self.diagnoser.diagnose(t) for t in traces]
        backend = diagnoses[0].backend if diagnoses else "builtin"

        # Aggregate interventions across traces (sum predicted savings, count).
        agg: dict[str, dict] = {}
        for d in diagnoses:
            for iv in propose(d):
                slot = agg.setdefault(iv.id, {"iv": iv, "save": 0, "n": 0, "sig": ""})
                slot["save"] += iv.predicted_savings
                slot["n"] += 1
                slot["sig"] = self._signature(iv, d)

        recs = []
        for slot in agg.values():
            iv = slot["iv"]
            prior = self.playbook.prior_winrate(slot["sig"], iv.key)
            recs.append(Recommendation(iv, slot["save"], slot["n"], prior))
        recs.sort(key=lambda r: (r.total_predicted_savings * r.prior_winrate),
                  reverse=True)

        # Build a proposed config: apply the applicable (non-STRUCT) edits.
        proposed = self.base_config.clone()
        for r in recs:
            if r.intervention.tier is not Tier.STRUCT:
                proposed = apply(r.intervention, proposed)

        mean_tas = sum(d.tas_score for d in diagnoses) / max(len(diagnoses), 1)
        total_tokens = sum(d.total_tokens for d in diagnoses)
        return CoachReport(
            n_traces=len(diagnoses), mean_tas=round(mean_tas, 1),
            total_tokens=total_tokens, recommendations=recs,
            base_config=self.base_config, proposed_config=proposed, backend=backend)

    # -- planning helpers --------------------------------------------------- #
    def _select(self, candidates: list[Intervention], diagnosis) -> Intervention:
        lowest = min(c.tier for c in candidates)
        tier_cands = [c for c in candidates if c.tier == lowest]
        # When a LinUCB bandit is wired in, let it choose (explore/exploit).
        # Without one, fall back to the playbook-primed greedy score.
        if self.bandit is not None:
            return self.bandit.select(tier_cands, diagnosis)  # type: ignore[return-value]
        def score(iv: Intervention) -> float:
            prior = self.playbook.prior_winrate(self._signature(iv, diagnosis), iv.key)
            return prior * iv.predicted_savings / (1.0 + iv.predicted_risk)
        return max(tier_cands, key=score)

    @staticmethod
    def _tier_exhausted(candidates: list[Intervention], iv: Intervention) -> bool:
        return all(c.tier >= iv.tier for c in candidates)

    @staticmethod
    def _signature(iv: Intervention, diagnosis) -> str:
        for p in diagnosis.patterns:
            if p.kind is iv.waste_pattern:
                return p.signature
        return f"{iv.waste_pattern.value}|lo"

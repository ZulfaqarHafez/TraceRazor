"""TraceRazor Teacher / Orchestrator (v2 prototype).

A guiding "teacher" agent that observes a target agent's traces, diagnoses
token waste with the TraceRazor auditor, and *alters the agent* -- behind a
quality-preservation gate -- so it stops wasting tokens, learning across runs
via a shared Playbook.

Quickstart (offline, no API keys):

    from teacher import Teacher, AgentConfig, Task

    cfg = AgentConfig()                       # a wasteful starting agent
    tasks = [Task("t1", "refund order ORD-1", ["get_order", "refund"])]
    result = Teacher(cfg).improve(tasks)
    print(result.total_token_saving_pct)      # tokens cut, success preserved

See ``examples/demo_teacher_offline.py`` for an end-to-end demo, and
``docs/v2_improvement_plan.md`` for the full production roadmap.
"""
from .adapters import FrameworkAdapter, LangGraphAdapter, RunRecorder
from .diagnose import Diagnoser
from .gate import QualityGate
from .interventions import apply, from_auditor_fixes, propose
from .memory import Playbook
from .report import CoachReport, Recommendation, config_diff, render
from .runner import Task, evaluate, run_task
from .schemas import (
    AgentConfig,
    Decision,
    Diagnosis,
    Intervention,
    Target,
    Tier,
    VerifiedResult,
    WasteKind,
    WastePattern,
)
from .teacher import Mode, Teacher, TeacherResult

__all__ = [
    "Teacher", "TeacherResult", "Mode",
    "AgentConfig", "Task", "QualityGate", "Playbook", "Diagnoser",
    "Intervention", "Diagnosis", "WastePattern", "WasteKind",
    "Target", "Tier", "Decision", "VerifiedResult",
    "propose", "from_auditor_fixes", "apply", "evaluate", "run_task",
    "render", "config_diff", "CoachReport", "Recommendation",
    "LangGraphAdapter", "RunRecorder", "FrameworkAdapter",
]

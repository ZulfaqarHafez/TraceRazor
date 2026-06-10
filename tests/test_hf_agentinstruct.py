"""Hermetic tests for the Hugging Face AgentInstruct loader + converter.

These never touch the network: they exercise the vendored real sample
(``benchmark/data/_agentinstruct_hf_sample.py``) and the ReAct→TraceRazor
converter (``tools/convert_agentinstruct.py``).
"""
import json
from pathlib import Path

import pytest

from benchmark.hf_loader import HFAgentInstructLoader
from tools.convert_agentinstruct import (
    convert_conversations,
    convert_row,
    _classify_gpt_turn,
    _looks_failed,
    _real_task_turns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "traces" / "external" / "huggingface" / "agentinstruct"


# ── Loader ───────────────────────────────────────────────────────────────────

class TestHFLoader:
    def test_bundled_loads_real_rows(self):
        rows = HFAgentInstructLoader(source="bundled").load()
        assert len(rows) >= 8
        for r in rows:
            assert "id" in r and "conversations" in r
            assert all("from" in t and "value" in t for t in r["conversations"])

    def test_split_filter(self):
        os_rows = HFAgentInstructLoader(source="bundled", split="os").load()
        assert len(os_rows) >= 5
        assert all(r["domain"] == "os" for r in os_rows)

    def test_max_rows_truncates(self):
        rows = HFAgentInstructLoader(source="bundled", max_rows=3).load()
        assert len(rows) == 3

    def test_disk_requires_existing_file(self):
        with pytest.raises(FileNotFoundError):
            HFAgentInstructLoader(source="disk", jsonl_path="/no/such.jsonl").load()

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            HFAgentInstructLoader(source="bogus")

    def test_live_requires_split(self):
        with pytest.raises(ValueError):
            HFAgentInstructLoader(source="live").load()


# ── Converter classification ─────────────────────────────────────────────────

class TestClassifyTurn:
    def test_bash_action(self):
        a = _classify_gpt_turn("Think: list files.\n\nAct: bash\n\n```bash\nls -l\n```")
        assert a["kind"] == "bash"
        assert a["code"] == "ls -l"

    def test_sql_action(self):
        a = _classify_gpt_turn("Join tables.\nAction: Operation\n```sql\nSELECT 1;\n```")
        assert a["kind"] == "sql"
        assert "SELECT 1;" in a["code"]

    def test_answer_action(self):
        assert _classify_gpt_turn("Think: done.\n\nAct: answer(220)")["kind"] == "answer"

    def test_sql_answer_action(self):
        assert _classify_gpt_turn("Action: Answer\nFinal Answer: [0]")["kind"] == "answer"

    def test_plain_ack_is_reasoning(self):
        assert _classify_gpt_turn("Ok.")["kind"] == "reasoning"


class TestFailureDetection:
    def test_real_failure_flagged(self):
        assert _looks_failed("bash: foo.txt: No such file or directory")

    def test_word_error_in_log_not_flagged(self):
        # The literal word "error" inside legitimate command output is NOT a
        # tool failure (regression guard against the naive substring check).
        assert not _looks_failed("Timeout error --- task:33 --- on:worker:1908")


# ── Few-shot scaffolding exclusion ───────────────────────────────────────────

class TestRealTaskTurns:
    def test_loss_flags_split_demo_from_real_task(self):
        # AgentInstruct os rows: gpt turns with loss=False are the dataset's
        # one-shot demo; the real trajectory carries loss=True.
        rows = {r["id"]: r for r in HFAgentInstructLoader(source="bundled").load()}
        turns = _real_task_turns(rows["os_0"]["conversations"])
        gpt = [t for t in turns if t["from"] == "gpt"]
        assert gpt and all(t["loss"] is True for t in gpt)
        # The kept human turn states the *real* problem, not the demo's.
        assert "new problem" in turns[0]["value"].lower()

    def test_db_ack_scaffolding_excluded(self):
        # The db split's "Ok." acknowledgement is scaffolding (loss=False).
        rows = {r["id"]: r for r in HFAgentInstructLoader(source="bundled").load()}
        trace = convert_row(rows["db_0"])
        assert all("ok." != s["content"].strip().lower() for s in trace["steps"])
        assert len(trace["steps"]) == 2

    def test_marker_fallback_without_loss_flags(self):
        conv = [
            {"from": "human", "value": "demo task"},
            {"from": "gpt", "value": "demo answer"},
            {"from": "human", "value": "Now, I will start a new problem in a new OS. My problem is: real task"},
            {"from": "gpt", "value": "real answer"},
        ]
        turns = _real_task_turns(conv)
        assert len(turns) == 2
        assert "real task" in turns[0]["value"]

    def test_passthrough_without_scaffolding(self):
        conv = [
            {"from": "human", "value": "the task"},
            {"from": "gpt", "value": "the answer", "loss": True},
        ]
        assert _real_task_turns(conv) == conv

    def test_first_steps_differ_across_rows(self):
        # Before scaffolding exclusion every converted trace began with the
        # identical demo step ("count the files in /etc"), pseudo-replicating
        # it into the statistics. Real first steps must differ across tasks.
        rows = HFAgentInstructLoader(source="bundled", split="os").load()
        firsts = {convert_row(r)["steps"][0]["content"] for r in rows}
        assert len(firsts) == len(rows), "first steps must be task-specific"


# ── Converter end-to-end ─────────────────────────────────────────────────────

class TestConvertConversations:
    def test_os_trajectory_structure(self):
        rows = HFAgentInstructLoader(source="bundled", split="os").load()
        trace = convert_row(rows[0])
        assert trace["trace_id"].startswith("agentinstruct-")
        assert trace["framework"] == "raw"
        assert trace["total_tokens"] > 0
        assert trace["metadata"]["source"] == "huggingface:zai-org/AgentInstruct"
        # The real task instruction anchors goal metrics.
        assert "task" in trace["metadata"]
        # ReAct bash turns become tool_call steps; outputs are attached.
        tool_steps = [s for s in trace["steps"] if s["type"] == "tool_call"]
        assert tool_steps, "expected at least one tool_call step"
        assert any(s.get("output") for s in tool_steps)
        assert all(s.get("tool_name") == "bash" for s in tool_steps)

    def test_retry_trajectory_marks_failures(self):
        # os_5 hits "No such file or directory" → at least one failed tool step.
        rows = {r["id"]: r for r in HFAgentInstructLoader(source="bundled").load()}
        trace = convert_row(rows["os_5"])
        assert any(s.get("tool_success") is False for s in trace["steps"])

    def test_step_ids_are_sequential(self):
        rows = HFAgentInstructLoader(source="bundled", split="os").load()
        trace = convert_row(rows[0])
        ids = [s["id"] for s in trace["steps"]]
        assert ids == list(range(1, len(ids) + 1))


# ── Generated corpus on disk ─────────────────────────────────────────────────

class TestGeneratedCorpus:
    def test_corpus_files_exist_and_parse(self):
        files = sorted(CORPUS_DIR.glob("agentinstruct-*.json"))
        assert len(files) >= 8, "run tools/convert_agentinstruct.py --bundled"
        for f in files:
            trace = json.loads(f.read_text())
            assert trace["steps"]
            assert trace["total_tokens"] == sum(s["tokens"] for s in trace["steps"])

    def test_corpus_shape_documents_floor_coverage(self):
        # With the dataset's few-shot scaffolding excluded, most real
        # AgentInstruct trajectories are 3-4 steps. The corpus keeps that
        # sub-floor majority on purpose: it measures the 5-step analysis
        # floor's coverage cost on real data (and exercises the skip path).
        files = sorted(CORPUS_DIR.glob("agentinstruct-*.json"))
        n_steps = [len(json.loads(f.read_text())["steps"]) for f in files]
        analysable = sum(1 for n in n_steps if n >= 5)
        sub_floor = sum(1 for n in n_steps if n < 5)
        assert analysable >= 4
        assert sub_floor >= analysable

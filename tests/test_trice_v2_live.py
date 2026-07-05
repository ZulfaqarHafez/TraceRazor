import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tracerazor.trice.adapters import CommandRepairAdapter, JsonPatchAdapter
from tracerazor.trice.bundle import export_evidence_bundle, verify_evidence_bundle
from tracerazor.trice.claim import build_claim_card, render_claim_card_markdown, render_claim_ladder_svg
from tracerazor.trice.artifact import build_artifact_card, render_artifact_markdown, render_artifact_svg
from tracerazor.trice.contract import build_contract_card, render_contract_markdown, render_contract_svg, verify_contract_card_file
from tracerazor.trice.crates import build_crates_card, render_crates_markdown, render_crates_svg, verify_crates_card_file
from tracerazor.trice.design import build_design_card, render_design_markdown, render_design_svg, verify_design_card_file
from tracerazor.trice.integrity import build_integrity_card, render_integrity_markdown, render_integrity_svg, verify_integrity_card_file, write_integrity_outputs
from tracerazor.trice.install import render_install_markdown, render_install_svg, verify_install_card_file
from tracerazor.trice.protocol import build_protocol_lock, render_protocol_markdown, render_protocol_svg, verify_protocol_lock_file
from tracerazor.trice.readiness import build_suite_readiness, render_readiness_markdown, render_readiness_svg, verify_readiness_file
from tracerazor.trice.release import build_release_card, render_release_markdown, render_release_svg, verify_release_card_file
from tracerazor.trice.release_evidence import build_release_evidence_card, render_release_evidence_markdown, render_release_evidence_svg, verify_release_evidence_file, write_release_evidence_outputs
from tracerazor.trice.reproduction import build_reproduction_card, render_reproduction_markdown, render_reproduction_svg, verify_reproduction_card_file
from tracerazor.trice.research import build_research_card, render_research_markdown, render_research_svg, verify_research_card_file
from tracerazor.trice.evidence import canonical_json, verify_manifest
from tracerazor.trice.live import LiveTask, run_live_learning_loop
from tracerazor.trice.recall import evidence_recall_from_policy
from tracerazor.trice.receipt import validate_run_receipt_file
from tracerazor.trice.schemas import load_schema, schema_path, validate_adapter_profile_file, validate_patch_spec_file, validate_suite_manifest_file
from tracerazor.trice.stats import bootstrap_mean_ci, claim_gate_from_rounds, clustered_bootstrap_mean_ci, wilson_ci
from tracerazor.trice.suite import run_suite_manifest, scaffold_suite_manifest, verify_suite_evidence
from tracerazor.trice.user import UserPreferenceProfile


REPO = Path(__file__).resolve().parents[1]


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": [3, {"z": 1}]}) == canonical_json({"a": [3, {"z": 1}], "b": 2})


def test_public_tracerazor_trice_import_surface():
    import tracerazor.trice as trice

    assert trice.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert callable(trice.run_live_learning_loop)
    assert callable(trice.scaffold_suite_manifest)
    assert callable(trice.evidence_recall_from_policy)
    assert callable(trice.build_claim_card)
    assert callable(trice.build_artifact_card)
    assert callable(trice.build_contract_card)
    assert callable(trice.build_crates_card)
    assert callable(trice.build_design_card)
    assert callable(trice.build_integrity_card)
    assert callable(trice.build_install_card)
    assert callable(trice.build_protocol_lock)
    assert callable(trice.build_release_card)
    assert callable(trice.build_release_evidence_card)
    assert callable(trice.build_reproduction_card)
    assert callable(trice.build_research_card)
    assert callable(trice.build_suite_readiness)
    assert callable(trice.verify_readiness_file)
    assert callable(trice.verify_release_card_file)
    assert callable(trice.verify_release_evidence_file)
    assert callable(trice.verify_claim_card_file)
    assert callable(trice.verify_artifact_card_file)
    assert callable(trice.verify_contract_card_file)
    assert callable(trice.verify_crates_card_file)
    assert callable(trice.verify_install_card_file)
    assert callable(trice.verify_protocol_lock_file)
    assert callable(trice.verify_design_card_file)
    assert callable(trice.verify_integrity_card_file)
    assert callable(trice.verify_reproduction_card_file)
    assert callable(trice.verify_research_card_file)
    assert callable(trice.claim_gate_from_rounds)
    assert callable(trice.CommandRepairAdapter.from_dict)
    assert callable(trice.JsonPatchAdapter.from_dict)
    assert callable(trice.export_evidence_bundle)
    assert callable(trice.verify_evidence_bundle)
    assert trice.load_schema("patch")["title"] == "TRICE deterministic patch spec"


def test_schema_helpers_validate_example_patch():
    path = schema_path("patch")
    assert path.name == "trice_patch_spec.schema.json"
    assert load_schema("manifest")["title"] == "TRICE evidence manifest"
    assert load_schema("suite")["title"] == "TRICE live suite manifest"
    assert load_schema("bundle")["title"] == "TRICE evidence bundle manifest"
    assert load_schema("adapter")["title"] == "TRICE adapter profile"
    assert load_schema("receipt")["title"] == "TRICE run receipt"
    assert load_schema("claim")["title"] == "TRICE deterministic claim card"
    assert load_schema("readiness")["title"] == "TRICE suite readiness preflight"
    assert load_schema("artifact")["title"] == "TRICE artifact review card"
    assert load_schema("protocol")["title"] == "TRICE protocol lock"
    assert load_schema("design")["title"] == "TRICE statistical design card"
    assert load_schema("reproduction")["title"] == "TRICE reproduction card"
    assert load_schema("release")["title"] == "TRICE release card"
    assert load_schema("contract")["title"] == "TRICE public contract card"
    assert load_schema("contract-card")["title"] == "TRICE public contract card"
    assert load_schema("release-evidence")["title"] == "TRICE release evidence"
    assert load_schema("integrity")["title"] == "TRICE integrity card"
    assert load_schema("crates")["title"] == "TRICE crates publish card"
    assert load_schema("install")["title"] == "TRICE installability card"
    assert load_schema("install-card")["title"] == "TRICE installability card"
    assert load_schema("research")["title"] == "TRICE research card"
    assert load_schema("research-card")["title"] == "TRICE research card"
    verdict = validate_patch_spec_file(REPO / "examples" / "trice_patch_fix_offbyone.json")
    assert verdict["ok"] is True
    assert verdict["edit_count"] == 1
    suite_verdict = validate_suite_manifest_file(REPO / "examples" / "trice_suite_fix_offbyone.json")
    assert suite_verdict["ok"] is True
    assert suite_verdict["task_count"] == 1
    assert suite_verdict["run_count"] == 3
    adapter_verdict = validate_adapter_profile_file(REPO / "examples" / "trice_adapter_profile_echo.json")
    assert adapter_verdict["ok"] is True
    assert adapter_verdict["type"] == "command"
    bundled_adapter = validate_adapter_profile_file(REPO / "examples" / "trice_adapter_profile_bundled_tasks.json")
    assert bundled_adapter["ok"] is True
    assert "trice_repair_bundled_tasks.py" in bundled_adapter["command"][1]
    bundled_suite = validate_suite_manifest_file(REPO / "examples" / "trice_suite_bundled_live.json")
    assert bundled_suite["ok"] is True
    assert bundled_suite["task_count"] == 6


def test_deterministic_stats_are_repeatable_and_gate_local_claim():
    values = [0.773, 0.802, 0.811, 0.749, 0.776, 0.808]
    ci1 = bootstrap_mean_ci(values)
    ci2 = bootstrap_mean_ci(list(reversed(values)))
    assert ci1.mean == ci2.mean
    assert ci1.low > 0.70
    assert ci1.high < 0.83
    pass_ci = wilson_ci(6, 6)
    assert pass_ci.mean == 1.0
    rounds = [
        {
            "measured_input_savings": v,
            "baseline": {"passed": True},
            "optimized": {"passed": True},
            "accepted": True,
        }
        for v in values
    ]
    gate = claim_gate_from_rounds(rounds, target_savings=0.60)
    assert gate.smoke_gate_passed is True
    assert gate.broad_claim_allowed is False
    assert gate.savings_ci.low >= 0.60
    failed_gate = claim_gate_from_rounds(
        [
            {
                "measured_input_savings": 0.9,
                "baseline": {"passed": False},
                "optimized": {"passed": False},
                "accepted": True,
            }
        ],
        target_savings=0.60,
    )
    assert failed_gate.smoke_gate_passed is False
    clustered = clustered_bootstrap_mean_ci({"repo-a": [0.7, 0.8], "repo-b": [0.6]})
    assert clustered.mean == 0.7
    assert 0.6 <= clustered.low <= clustered.high <= 0.8


def test_evidence_recall_from_policy_requires_recallable_essential_evidence():
    policy = {
        "constraints": {"evidence_recall_min": 0.95},
        "decisions": [
            {
                "segment_id": "s1",
                "step_id": 1,
                "state": "essential",
                "action": "keep",
                "original_tokens": 80,
                "locked": True,
                "receipt": "a" * 64,
                "rehydrate_pointer": "trace:t:step:1",
            },
            {
                "segment_id": "s2",
                "step_id": 2,
                "state": "essential",
                "action": "mask_with_receipt",
                "original_tokens": 20,
                "locked": True,
                "receipt": "",
                "rehydrate_pointer": None,
            },
            {
                "segment_id": "s3",
                "step_id": 3,
                "state": "redundant",
                "action": "mask_with_receipt",
                "original_tokens": 1000,
                "locked": False,
                "receipt": "",
                "rehydrate_pointer": None,
            },
        ],
    }

    report = evidence_recall_from_policy(policy)

    assert report.evidence_recall == 0.8
    assert report.passed is False
    assert report.obligation_count == 2
    assert report.missing[0]["segment_id"] == "s2"


def test_user_profile_learns_live_aggressive_target_from_feedback(tmp_path):
    profile = UserPreferenceProfile.load(tmp_path / "profile.json")
    profile.ingest_feedback("real runs, not replay runs, learn from the user, hit 60% savings")
    assert profile.require_live_rollout is True
    assert profile.target_savings == 0.60
    assert profile.budget_ratio <= 0.40
    assert any("live rollout" in lesson for lesson in profile.lessons)


def test_json_patch_adapter_refuses_test_edits_and_path_escape(tmp_path):
    workspace = tmp_path / "repo"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "tests" / "test_x.py").write_text("x = 1\n", encoding="utf-8")
    adapter = JsonPatchAdapter.from_dict(
        {"edits": [{"op": "replace", "path": "tests/test_x.py", "old": "1", "new": "2"}]}
    )
    with pytest.raises(ValueError, match="forbidden path"):
        adapter.apply_fix(object(), workspace)
    escape = JsonPatchAdapter.from_dict(
        {"edits": [{"op": "write", "path": "../escape.py", "content": "bad"}]}
    )
    with pytest.raises(ValueError, match="inside workspace"):
        escape.apply_fix(object(), workspace)


def test_command_repair_adapter_runs_real_command_and_blocks_test_edits(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    repair = tmp_path / "repair.py"
    repair.write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "p = Path('app.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('1', '2'), encoding='utf-8')\n"
        "Path('.trice').mkdir(exist_ok=True)\n"
        "Path('.trice/agent_receipt.json').write_text(json.dumps({\n"
        "  'model': 'unit-model',\n"
        "  'token_accounting': {\n"
        "    'input_tokens': int(os.environ['TRICE_INPUT_TOKENS']),\n"
        "    'baseline_input_tokens': int(os.environ['TRICE_BASELINE_INPUT_TOKENS']),\n"
        "  },\n"
        "  'context_mode': os.environ['TRICE_CONTEXT_MODE'],\n"
        "  'paths': {\n"
        "    'context': os.environ['TRICE_CONTEXT_PATH'],\n"
        "    'policy': os.environ['TRICE_POLICY_PATH'],\n"
        "    'trace': os.environ['TRICE_TRACE_PATH'],\n"
        "    'verify_cmd': json.loads(os.environ['TRICE_VERIFY_CMD_JSON']),\n"
        "  },\n"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    adapter = CommandRepairAdapter(command=(sys.executable, str(repair)), timeout_s=30)
    changed = adapter.apply_fix(
        type(
            "Task",
            (),
            {
                "task_id": "cmd",
                "prompt": "fix value",
                "trice_context": {
                    "schema_version": "trice-context-envelope/v1",
                    "condition": "trice-v2",
                    "context_mode": "trice_policy",
                    "input_tokens": 456,
                    "baseline_input_tokens": 1234,
                    "policy_tokens": 456,
                    "budget_tokens": 494,
                    "budget_ratio": 0.4,
                    "realized_budget_ratio": 0.36953,
                    "projected_input_savings_pct": 63.05,
                    "policy_path": str(tmp_path / "policy.json"),
                    "compressed_context_path": str(tmp_path / "context.txt"),
                    "trace_path": str(tmp_path / "trace.json"),
                    "verify_cmd": ["python", "-m", "pytest"],
                    "policy_action_counts": {"keep": 2, "extract": 1},
                },
            },
        )(),
        workspace,
    )
    assert changed == ["app.py"]
    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert adapter.last_receipt
    assert adapter.last_receipt["agent_reported"]["token_accounting"]["input_tokens"] == 456
    assert adapter.last_receipt["agent_reported"]["token_accounting"]["baseline_input_tokens"] == 1234
    assert adapter.last_receipt["agent_reported"]["paths"]["context"] == str(tmp_path / "context.txt")
    assert adapter.last_receipt["agent_reported"]["paths"]["policy"] == str(tmp_path / "policy.json")
    assert adapter.last_receipt["agent_reported"]["paths"]["trace"] == str(tmp_path / "trace.json")
    assert adapter.last_receipt["agent_reported"]["paths"]["verify_cmd"] == ["python", "-m", "pytest"]
    assert adapter.last_receipt["trice_context"]["input_tokens"] == 456
    assert "TRICE_INPUT_TOKENS" in adapter.last_receipt["reserved_env_keys"]
    assert "TRICE_CONTEXT_PATH" in adapter.last_receipt["reserved_env_keys"]
    assert "TRICE_POLICY_PATH" in adapter.last_receipt["reserved_env_keys"]
    assert "TRICE_TRACE_PATH" in adapter.last_receipt["reserved_env_keys"]
    assert "TRICE_VERIFY_CMD_JSON" in adapter.last_receipt["reserved_env_keys"]
    assert len(adapter.last_receipt["workspace_before_sha256"]) == 64
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(adapter.last_receipt, indent=2, sort_keys=True), encoding="utf-8")
    receipt_verdict = validate_run_receipt_file(receipt_path)
    assert receipt_verdict["agent_reported_input_tokens"] == 456
    assert receipt_verdict["trice_context_mode"] == "trice_policy"
    assert receipt_verdict["trice_input_tokens"] == 456

    bad_workspace = tmp_path / "bad-repo"
    bad_workspace.mkdir()
    bad = tmp_path / "bad_repair.py"
    bad.write_text(
        "from pathlib import Path\n"
        "Path('tests').mkdir(exist_ok=True)\n"
        "Path('tests/test_x.py').write_text('def test_x():\\n    assert True\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden path"):
        CommandRepairAdapter(command=(sys.executable, str(bad)), timeout_s=30).apply_fix(object(), bad_workspace)


def test_trice_v2_live_rollout_edits_real_workspace_and_updates_profile(tmp_path):
    task = LiveTask.from_dir(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one")
    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path,
        user_feedback="real runs, not replay runs, learn from the user, target 60% token savings",
        rounds=1,
    )

    assert result.rounds
    live_round = result.rounds[0]
    assert live_round.baseline.passed is True
    assert live_round.optimized.passed is True
    assert live_round.pass_noninferior is True
    assert live_round.measured_input_savings >= 0.60
    assert live_round.accepted is True

    workspace = Path(live_round.optimized.workspace)
    assert workspace.is_dir()
    assert "size - 1" not in (workspace / "chunker.py").read_text(encoding="utf-8")
    assert (workspace / "chunker.py").read_text(encoding="utf-8").count("size") >= 2
    assert live_round.optimized.policy_path
    assert Path(live_round.optimized.policy_path).is_file()
    assert live_round.optimized.decision_trace_path
    assert Path(live_round.optimized.decision_trace_path).is_file()
    assert Path(result.report_path or "").is_file()
    assert Path(result.result_path or "").is_file()
    assert Path(result.manifest_path or "").is_file()
    verdict = verify_manifest(result.manifest_path)
    assert verdict["ok"] is True
    receipt = Path(live_round.optimized.receipt_path or "")
    assert receipt.is_file()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["trice_context"]["trace_path"] == live_round.optimized.decision_trace_path
    assert receipt_payload["trice_context"]["verify_cmd"] == list(task.verify_cmd)
    receipt_verdict = validate_run_receipt_file(receipt)
    assert receipt_verdict["adapter_type"] == "managed_python"
    assert receipt_verdict["trice_context_mode"] == "trice_policy"
    assert receipt_verdict["trice_input_tokens"] == live_round.optimized.input_tokens
    assert receipt_verdict["evidence_recall"] == 1.0
    assert receipt_verdict["evidence_recall_passed"] is True
    assert live_round.optimized.evidence_recall == 1.0
    assert live_round.optimized.evidence_recall_passed is True
    baseline_receipt = validate_run_receipt_file(live_round.baseline.receipt_path)
    assert baseline_receipt["trice_context_mode"] == "full_context"
    assert baseline_receipt["trice_input_tokens"] == live_round.baseline.input_tokens
    assert any("accepted" in lesson for lesson in result.profile["lessons"])


def test_generic_repo_json_patch_rollout(tmp_path):
    seed = REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed"
    task = LiveTask.from_repo(
        seed,
        task_id="generic-fix-offby-one",
        prompt="Fix chunker.py without editing tests.",
        verify_cmd=["python", "-m", "pytest", "-q", "--tb=short"],
    )
    adapter = JsonPatchAdapter.from_dict(
        {
            "name": "generic-offby-one",
            "edits": [{"op": "replace", "path": "chunker.py", "old": "size - 1", "new": "size"}],
        }
    )
    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path / "generic",
        user_feedback="real runs, not replay; target 60% savings",
        rounds=1,
        adapter=adapter,
    )
    live_round = result.rounds[0]
    assert live_round.optimized.passed is True
    assert live_round.optimized.modified_files == ["chunker.py"]
    assert validate_run_receipt_file(live_round.optimized.receipt_path)["adapter_type"] == "json_patch"
    assert live_round.measured_input_savings >= 0.60
    assert result.claim_gate["smoke_gate_passed"] is True
    assert verify_manifest(result.manifest_path)["ok"] is True


def test_json_patch_rollout_clears_stale_python_bytecode_after_same_size_edit(tmp_path):
    repo = tmp_path / "same-size-repo"
    (repo / "src" / "sample").mkdir(parents=True)
    (repo / "src" / "sample" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "sample" / "simple.py").write_text(
        "def add_one(number):\n"
        "    return number + 1\n",
        encoding="utf-8",
    )
    verify_cmd = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, 'src'); from sample.simple import add_one; assert add_one(1) == 3",
    ]
    task = LiveTask.from_repo(
        repo,
        task_id="same-size-pyc",
        prompt="Change add_one so add_one(1) returns 3.",
        verify_cmd=verify_cmd,
    )
    adapter = JsonPatchAdapter.from_dict(
        {
            "name": "same-size-pyc-repair",
            "edits": [
                {
                    "op": "replace",
                    "path": "src/sample/simple.py",
                    "old": "return number + 1",
                    "new": "return number + 2",
                }
            ],
        }
    )

    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path / "same-size-out",
        user_feedback="real runs, not replay; target 60% savings",
        rounds=1,
        adapter=adapter,
    )

    live_round = result.rounds[0]
    assert live_round.baseline.passed is True
    assert live_round.optimized.passed is True
    assert live_round.accepted is True
    assert result.claim_gate["trice_pass_rate"] == 1.0


def test_manifest_driven_suite_runs_real_repo_and_deep_verifies(tmp_path):
    source_repo = tmp_path / "source-repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", source_repo)
    subprocess.run(["git", "init"], cwd=source_repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "trice@example.test"], cwd=source_repo, check=True)
    subprocess.run(["git", "config", "user.name", "TRICE Test"], cwd=source_repo, check=True)
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=source_repo, check=True, capture_output=True, text=True)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_repo, check=True, capture_output=True, text=True).stdout.strip()
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch_path = patch_dir / "fix.json"
    shutil.copy2(REPO / "examples" / "trice_patch_fix_offbyone.json", patch_path)
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-suite/v1",
                "name": "tmp-real-suite",
                "user_feedback": "real runs, not replay; target 60% savings",
                "target_savings": 0.60,
                "rounds": 1,
                "replicates": 2,
                "tasks": [
                    {
                        "task_id": "tmp-fix-offby-one",
                        "git": {"url": source_repo.resolve().as_uri(), "rev": git_rev},
                        "patch_spec": "patches/fix.json",
                        "prompt": "Fix chunker.py without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "suite-out"
    result = run_suite_manifest(manifest_path, out_dir=out_dir, rounds=1)

    assert len(result.tasks) == 2
    assert result.tasks[0].mean_savings >= 0.60
    assert result.tasks[0].accepted_rounds == 1
    assert result.tasks[0].source["source_type"] == "git"
    assert result.tasks[0].source["git"]["resolved_commit"] == git_rev
    assert result.tasks[0].source["repo_tree"]["algorithm"] == "trice-tree-sha256/v1"
    assert result.tasks[0].source["repo_tree"]["file_count"] >= 2
    assert len(result.tasks[0].source["repo_tree"]["digest"]) == 64
    assert len(result.tasks[0].source["patch_sha256"]) == 64
    assert result.tasks[1].replicate_index == 2
    assert result.claim_gate["replicate_count"] == 2
    assert result.claim_gate["task_cluster_count"] == 1
    assert result.claim_gate["clustered_savings_ci"]["low"] >= 0.60
    assert result.claim_gate["evidence_recall_minimum"] == 1.0
    assert result.claim_gate["evidence_recall_failures"] == 0
    assert result.claim_gate["smoke_gate_passed"] is True
    assert result.claim_gate["s_tier_gate"]["passed"] is False
    assert "task_clusters" in result.claim_gate["s_tier_gate"]["missing_requirements"]
    assert "adapter_profiles" in result.claim_gate["s_tier_gate"]["missing_requirements"]
    assert (out_dir / "trice_suite_sources.json").is_file()
    verdict = verify_suite_evidence(out_dir / str(result.manifest_path))
    assert verdict["ok"] is True
    assert len(verdict["children"]) == 2
    assert all(child["ok"] for child in verdict["children"])

    bundle = export_evidence_bundle(out_dir / str(result.manifest_path), out_dir / "suite.trice.zip")
    bundle_verdict = verify_evidence_bundle(bundle)
    assert bundle_verdict["ok"] is True
    assert bundle_verdict["entry_count"] >= 10

    tampered = out_dir / "tampered.trice.zip"
    shutil.copy2(bundle, tampered)
    with tampered.open("r+b") as fh:
        fh.seek(-24, 2)
        byte = fh.read(1)
        fh.seek(-24, 2)
        fh.write(bytes([byte[0] ^ 1]))
    assert verify_evidence_bundle(tampered)["ok"] is False

    card = build_claim_card(out_dir / str(result.result_path), manifest_path=out_dir / str(result.manifest_path))
    assert card["schema_version"] == "trice-claim-card/v1"
    assert card["claim_level"] == "smoke"
    assert card["claim_allowed"] is False
    assert card["verification"]["ok"] is True
    assert card["metrics"]["mean_input_token_savings"] >= 0.60
    assert any("Not an S-tier claim" in item for item in card["non_claims"])
    assert "TRICE Claim Card" in render_claim_card_markdown(card)
    assert "deterministic claim ladder" in render_claim_ladder_svg(card)


def test_suite_scaffold_generates_locked_remote_git_manifest(tmp_path):
    source = tmp_path / "remote-git-list.json"
    out = tmp_path / "suite.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "trice-remote-git-list/v1",
                "name": "held-out-pilot",
                "adapter_profile": "profiles/codex-adapter.json",
                "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                "replicates": 2,
                "tasks": [
                    {
                        "task_id": "sample-python-fix",
                        "url": "https://github.com/example/project.git",
                        "rev": "0123456789abcdef0123456789abcdef01234567",
                        "prompt": "Fix the failing parser test without editing tests.",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = scaffold_suite_manifest(source, out)

    assert out.is_file()
    assert manifest["schema_version"] == "trice-suite/v1"
    assert manifest["replicates"] == 2
    assert manifest["s_tier_gate"]["min_task_clusters"] == 50
    assert manifest["tasks"][0]["git"]["url"] == "https://github.com/example/project.git"
    assert manifest["tasks"][0]["git"]["rev"] == "0123456789abcdef0123456789abcdef01234567"
    assert manifest["tasks"][0]["adapter_profile"] == "profiles/codex-adapter.json"
    assert validate_suite_manifest_file(out)["ok"] is True

    readiness = build_suite_readiness(out)
    assert readiness["schema_version"] == "trice-suite-readiness/v1"
    assert readiness["readiness_level"] == "smoke_ready"
    assert readiness["pilot_execution_ready"] is False
    assert readiness["claim_execution_ready"] is False
    assert "pilot_task_clusters" in readiness["missing_for_pilot"]
    assert "claim_task_clusters" in readiness["missing_for_claim"]
    assert "TRICE Suite Readiness" in render_readiness_markdown(readiness)
    assert "TRICE suite readiness preflight" in render_readiness_svg(readiness)
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_readiness_file(readiness_path, manifest_path=out)["ok"] is True

    protocol = build_protocol_lock(out)
    assert protocol["schema_version"] == "trice-protocol-lock/v1"
    assert protocol["protocol_level"] == "smoke_protocol_locked"
    assert protocol["claim_allowed_by_protocol"] is False
    assert "TRICE Protocol Lock" in render_protocol_markdown(protocol)
    assert "TRICE protocol lock" in render_protocol_svg(protocol)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_protocol_lock_file(protocol_path, manifest_path=out)["ok"] is True

    design = build_design_card(protocol_path, suite_result_path=REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_results.json")
    assert design["schema_version"] == "trice-design-card/v1"
    assert design["design_level"] == "smoke_design_observed"
    assert design["claim_design_ready"] is False
    assert "TRICE Design Card" in render_design_markdown(design)
    assert "TRICE statistical design card" in render_design_svg(design)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_design_card_file(
        design_path,
        protocol_path=protocol_path,
        suite_result_path=REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_results.json",
    )["ok"] is True

    reproduction = build_reproduction_card()
    assert reproduction["schema_version"] == "trice-reproduction-card/v1"
    assert reproduction["reproduction_level"] == "reviewer_replay_ready_smoke"
    assert reproduction["claim_allowed"] is False
    assert "TRICE Reproduction Card" in render_reproduction_markdown(reproduction)
    assert "TRICE reproduction card" in render_reproduction_svg(reproduction)
    reproduction_path = tmp_path / "reproduction.json"
    reproduction_path.write_text(json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_reproduction_card_file(reproduction_path)["ok"] is True

    contract = build_contract_card()
    assert contract["schema_version"] == "trice-contract-card/v1"
    assert contract["contract_level"] == "library_contract_locked"
    assert contract["contract_score"] == 100
    assert "TRICE Contract Card" in render_contract_markdown(contract)
    assert "TRICE public contract card" in render_contract_svg(contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_contract_card_file(contract_path)["ok"] is True

    crates = build_crates_card(offline=True)
    assert crates["schema_version"] == "trice-crates-card/v1"
    assert crates["crates_card_level"] == "publish_plan_locked"
    assert crates["local_publish_plan_locked"] is True
    assert crates["cargo_install_claim_allowed"] is False
    assert "TRICE Crates Publish Card" in render_crates_markdown(crates)
    assert "TRICE crates publish card" in render_crates_svg(crates)
    crates_path = tmp_path / "crates.json"
    crates_path.write_text(json.dumps(crates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_crates_card_file(crates_path)["ok"] is True

    install_path = REPO / "docs" / "trice_install_card.json"
    install_verdict = verify_install_card_file(install_path)
    assert install_verdict["ok"] is True
    assert install_verdict["install_level"] in {"python_trice_install_ready", "full_cli_install_ready"}
    install_card = json.loads(install_path.read_text(encoding="utf-8"))
    assert "TRICE Installability Card" in render_install_markdown(install_card)
    assert "TRICE installability card" in render_install_svg(install_card)

    research = build_research_card()
    assert research["schema_version"] == "trice-research-card/v1"
    assert research["research_level"] == "research_basis_locked"
    assert research["research_score"] == 100
    assert research["source_count"] >= 150
    assert "TRICE Research Card" in render_research_markdown(research)
    assert "TRICE research basis card" in render_research_svg(research)
    research_path = tmp_path / "research.json"
    research_path.write_text(json.dumps(research, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_research_card_file(research_path)["ok"] is True

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "tracerazor-1.0.3-py3-none-any.whl").write_bytes(b"fake wheel for release evidence\n")
    (dist / "tracerazor-1.0.3.tar.gz").write_bytes(b"fake sdist for release evidence\n")
    cli_binary = tmp_path / ("tracerazor.exe" if sys.platform.startswith("win") else "tracerazor")
    cli_binary.write_bytes(b"fake cli binary for release evidence\n")
    release_evidence = build_release_evidence_card(dist_dir=dist, cli_binary_path=cli_binary, sidecar_stem="release-evidence")
    assert release_evidence["schema_version"] == "trice-release-evidence/v1"
    assert release_evidence["release_evidence_level"] == "release_evidence_ready"
    assert release_evidence["release_evidence_score"] == 100
    assert "TRICE Release Evidence" in render_release_evidence_markdown(release_evidence)
    assert "TRICE release evidence" in render_release_evidence_svg(release_evidence)
    evidence_path = tmp_path / "release-evidence.json"
    write_release_evidence_outputs(release_evidence, evidence_path)
    evidence_verdict = verify_release_evidence_file(evidence_path)
    assert evidence_verdict["ok"] is True
    assert "rust_cli" in evidence_verdict["checked_artifacts"]
    assert "release-evidence.checksums.txt" in evidence_verdict["checked_sidecars"]

    release = build_release_card(offline=True)
    assert release["schema_version"] == "trice-release-card/v1"
    assert release["release_level"] == "local_release_candidate"
    assert release["public_release_ready"] is False
    assert "TRICE Release Card" in render_release_markdown(release)
    assert "TRICE release card" in render_release_svg(release)
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_release_card_file(release_path)["ok"] is True

    integrity = build_integrity_card(release_path=release_path, release_evidence_path=evidence_path, crates_path=crates_path, install_path=install_path, research_path=research_path)
    assert integrity["schema_version"] == "trice-integrity-card/v1"
    assert integrity["integrity_level"] == "proof_graph_integrity_locked"
    assert integrity["integrity_score"] == 100
    assert "TRICE Integrity Card" in render_integrity_markdown(integrity)
    assert "TRICE proof graph integrity" in render_integrity_svg(integrity)
    integrity_path = tmp_path / "integrity.json"
    write_integrity_outputs(integrity, integrity_path)
    integrity_verdict = verify_integrity_card_file(integrity_path)
    assert integrity_verdict["ok"] is True
    assert "release_evidence" in integrity_verdict["checked_inputs"]
    assert "crates_card" in integrity_verdict["checked_inputs"]
    assert "install_card" in integrity_verdict["checked_inputs"]
    assert "research_card" in integrity_verdict["checked_inputs"]


def test_suite_readiness_identifies_pilot_ready_manifest(tmp_path):
    source = tmp_path / "remote-git-list.json"
    out = tmp_path / "pilot-suite.json"
    tasks = [
        {
            "task_id": f"pilot-task-{idx}",
            "url": "https://github.com/example/project.git",
            "rev": f"{idx:040x}",
            "prompt": "Fix the failing parser test without editing tests.",
            "verify_cmd": ["python", "-m", "pytest", "-q"],
        }
        for idx in range(1, 11)
    ]
    source.write_text(
        json.dumps(
            {
                "schema_version": "trice-remote-git-list/v1",
                "name": "pilot-ready-suite",
                "adapter_profile": "profiles/codex-adapter.json",
                "replicates": 2,
                "target_savings": 0.60,
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = scaffold_suite_manifest(source, out)
    assert manifest["replicates"] == 2

    readiness = build_suite_readiness(out)

    assert readiness["readiness_level"] == "pilot_ready"
    assert readiness["pilot_execution_ready"] is True
    assert readiness["claim_execution_ready"] is False
    assert readiness["planned_execution"]["planned_runs"] == 20
    assert readiness["planned_execution"]["verify_command_invocations_min"] == 40
    assert readiness["missing_for_claim"] == ["claim_task_clusters", "claim_replicates_per_task"]
    assert any(row["name"] == "evidence_recall_gate" and row["passed"] for row in readiness["checks"])

    protocol = build_protocol_lock(out)
    assert protocol["protocol_level"] == "pilot_protocol_ready"
    assert protocol["suite_shape"]["planned_runs"] == 20
    assert protocol["claim_allowed_by_protocol"] is False
    assert protocol["evaluation_contract"]["evidence_recall_min"] == 0.95


def test_manifest_suite_can_use_command_repair_adapter(tmp_path):
    source_repo = tmp_path / "source-repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", source_repo)
    repair = tmp_path / "repair_offbyone.py"
    repair.write_text(
        "from pathlib import Path\n"
        "p = Path('chunker.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('size - 1', 'size'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "suite-command.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-suite/v1",
                "name": "tmp-command-suite",
                "user_feedback": "real runs, not replay; target 60% savings",
                "target_savings": 0.60,
                "rounds": 1,
                "replicates": 1,
                "tasks": [
                    {
                        "task_id": "tmp-command-fix",
                        "repo": str(source_repo),
                        "repair_cmd": [sys.executable, str(repair)],
                        "repair_timeout_s": 30,
                        "prompt": "Fix chunker.py without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_suite_manifest_file(manifest_path)["ok"] is True
    out_dir = tmp_path / "suite-command-out"
    result = run_suite_manifest(manifest_path, out_dir=out_dir, rounds=1)

    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.patch_spec is None
    assert task.repair_cmd == [sys.executable, str(repair)]
    assert task.adapter_profile is None
    assert task.source["adapter_type"] == "command"
    assert "patch_sha256" not in task.source
    assert task.source["repair_timeout_s"] == 30
    assert task.mean_savings >= 0.60
    assert task.accepted_rounds == 1
    assert verify_suite_evidence(out_dir / str(result.manifest_path))["ok"] is True


def test_manifest_suite_can_use_adapter_profile(tmp_path):
    source_repo = tmp_path / "source-repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", source_repo)
    repair = tmp_path / "repair_profile.py"
    repair.write_text(
        "from pathlib import Path\n"
        "p = Path('chunker.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('size - 1', 'size'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    profile = tmp_path / "adapter-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "trice-adapter-profile/v1",
                "name": "tmp-profile-command",
                "type": "command",
                "command": [sys.executable, str(repair)],
                "timeout_s": 30,
                "allow_test_edits": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "suite-profile.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-suite/v1",
                "name": "tmp-profile-suite",
                "user_feedback": "real runs, not replay; target 60% savings",
                "rounds": 1,
                "replicates": 1,
                "s_tier_gate": {
                    "min_task_clusters": 1,
                    "min_replicates_per_task": 1,
                    "require_locked_git_sources": False,
                    "require_remote_git_sources": False,
                    "require_adapter_profiles": True,
                    "min_mean_savings": 0.60,
                    "min_clustered_savings_ci_low": 0.60,
                    "min_evidence_recall": 0.95,
                },
                "tasks": [
                    {
                        "task_id": "tmp-profile-fix",
                        "repo": str(source_repo),
                        "adapter_profile": str(profile),
                        "prompt": "Fix chunker.py without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_suite_manifest(manifest_path, out_dir=tmp_path / "suite-profile-out", rounds=1)

    task = result.tasks[0]
    assert task.adapter_profile == str(profile)
    assert task.source["adapter_type"] == "command_profile"
    assert len(task.source["adapter_profile_sha256"]) == 64
    assert task.mean_savings >= 0.60
    assert result.claim_gate["adapter_breakdown"]["command_profile"]["runs"] == 1
    assert result.claim_gate["failure_breakdown"]["pass_regression_runs"] == 0
    assert result.claim_gate["evidence_recall_minimum"] == 1.0
    assert result.claim_gate["s_tier_gate"]["requirements"]["evidence_recall"]["passed"] is True
    assert result.claim_gate["s_tier_gate"]["passed"] is True
    assert result.claim_gate["s_tier_gate"]["claim_level"] == "s_tier"


def test_live_rollout_manifest_repeats_for_identical_real_run(tmp_path):
    task = LiveTask.from_dir(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one")
    out_dir = tmp_path / "repeatable"

    def run_once() -> dict:
        result = run_live_learning_loop(
            [task],
            out_dir=out_dir,
            user_feedback="real runs, not replay; target 60% savings",
            rounds=1,
        )
        manifest = json.loads(Path(result.manifest_path or "").read_text(encoding="utf-8"))
        trace = json.loads(Path(result.rounds[0].optimized.trace_path).read_text(encoding="utf-8"))
        assert "wall_s" not in trace["metadata"]
        assert "in <duration>" in result.rounds[0].optimized.verify_output_excerpt
        assert verify_manifest(result.manifest_path)["ok"] is True
        return manifest

    first = run_once()
    shutil.rmtree(out_dir)
    second = run_once()

    assert second["result_sha256"] == first["result_sha256"]
    assert second["canonical_result_sha256"] == first["canonical_result_sha256"]
    assert second["artifacts"] == first["artifacts"]


def test_manifest_verification_rejects_invalid_hashed_run_receipt(tmp_path):
    task = LiveTask.from_dir(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one")
    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path / "bad-receipt",
        user_feedback="real runs, not replay; target 60% savings",
        rounds=1,
    )
    receipt_path = Path(result.rounds[0].optimized.receipt_path or "")
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["changed_file_count"] = 999
    receipt_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = Path(result.manifest_path or "")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"].endswith("trice-v2/run_receipt.json"):
            artifact["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            artifact["bytes"] = receipt_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verdict = verify_manifest(manifest_path)
    assert verdict["ok"] is False
    assert any("invalid run receipt" in err for err in verdict["errors"])

import json

import core
import cli
import report


def _write(tmp_path, rows):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return str(p)

def test_new_name_exists_old_gone():
    assert hasattr(core, "load_records")
    assert not hasattr(core, "fetch_data")

def test_cli_uses_new_name(tmp_path):
    assert cli.main(_write(tmp_path, [{"a": 1}, {"a": 2}])) == 2

def test_report_uses_new_name(tmp_path):
    s = report.summarize(_write(tmp_path, [{"a": 1, "b": 2}]))
    assert s == {"count": 1, "keys": ["a", "b"]}

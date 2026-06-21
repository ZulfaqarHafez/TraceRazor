from filt import filter_rows


def _write(tmp_path, text):
    p = tmp_path / "rows.csv"
    p.write_text(text, encoding="utf-8")
    return str(p)

def test_basic_filter(tmp_path):
    p = _write(tmp_path, "name,score\nann,10\nbob,3\ncid,7\n")
    assert filter_rows(p, 7) == [
        {"name": "ann", "score": 10},
        {"name": "cid", "score": 7},
    ]

def test_scores_are_ints(tmp_path):
    p = _write(tmp_path, "name,score\nann,10\n")
    [row] = filter_rows(p, 0)
    assert isinstance(row["score"], int)

def test_header_only(tmp_path):
    assert filter_rows(_write(tmp_path, "name,score\n"), 0) == []

def test_none_qualify(tmp_path):
    assert filter_rows(_write(tmp_path, "name,score\nann,1\n"), 5) == []

def test_order_preserved(tmp_path):
    p = _write(tmp_path, "name,score\nz,9\na,8\nm,9\n")
    assert [r["name"] for r in filter_rows(p, 8)] == ["z", "a", "m"]

from tracerazor._trice.evidence import write_text_lf


def test_write_text_lf_is_platform_independent(tmp_path):
    target = tmp_path / "evidence.json"

    write_text_lf(target, "first\r\nsecond\rthird\n")

    assert target.read_bytes() == b"first\nsecond\nthird\n"

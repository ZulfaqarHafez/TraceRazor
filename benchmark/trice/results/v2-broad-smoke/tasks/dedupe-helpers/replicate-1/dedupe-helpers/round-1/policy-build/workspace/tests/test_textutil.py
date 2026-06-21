import textutil
import utils_a
import utils_b


def test_single_shared_implementation():
    assert utils_a.normalize_name is textutil.normalize_name
    assert utils_b.normalize_name is textutil.normalize_name

def test_lowercases_and_strips():
    assert textutil.normalize_name("  Ada LOVELACE  ") == "ada lovelace"

def test_collapses_all_whitespace():
    assert textutil.normalize_name("Ada\t  Lovelace\n") == "ada lovelace"

def test_callers_still_work():
    assert utils_a.label_for(" Grace  Hopper ") == "user:grace hopper"
    assert utils_b.greeting("ALAN\tTuring") == "hello alan turing"

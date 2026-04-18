"""Tests for _canonicalize.py.

The PRD requires a 50-pair equivalence test (Section 4.3). We cover well beyond
that here, grouping pairs by the normalisation rule they exercise.
"""
import pytest
from tracerazor._canonicalize import canonicalize, canonical_key


# ── helper ─────────────────────────────────────────────────────────────────

def eq(a_name, a_args, b_name, b_args, **kw):
    """Assert two calls canonicalize to the same key."""
    ka = canonical_key(a_name, a_args, **kw)
    kb = canonical_key(b_name, b_args, **kw)
    assert ka == kb, f"Expected equal:\n  {ka}\n  {kb}"


def neq(a_name, a_args, b_name, b_args, **kw):
    """Assert two calls canonicalize to different keys."""
    ka = canonical_key(a_name, a_args, **kw)
    kb = canonical_key(b_name, b_args, **kw)
    assert ka != kb, f"Expected not equal:\n  {ka}\n  {kb}"


# ── tool-name normalisation ─────────────────────────────────────────────────

class TestToolName:
    def test_case_insensitive(self):
        eq("Search", {}, "search", {})

    def test_trailing_space(self):
        eq(" search ", {}, "search", {})

    def test_mixed_case(self):
        eq("BookReservation", {}, "bookreservation", {})

    def test_different_names_not_equal(self):
        neq("search", {}, "lookup", {})


# ── whitespace normalisation in string args ─────────────────────────────────

class TestWhitespace:
    def test_leading_trailing(self):
        eq("f", {"q": "  hello  "}, "f", {"q": "hello"})

    def test_internal_runs(self):
        eq("f", {"q": "hello   world"}, "f", {"q": "hello world"})

    def test_tab_and_newline(self):
        eq("f", {"q": "hello\tworld\n"}, "f", {"q": "hello world"})

    def test_nested_string(self):
        eq("f", {"x": {"y": "  abc  "}}, "f", {"x": {"y": "abc"}})


# ── dict key ordering ───────────────────────────────────────────────────────

class TestKeyOrder:
    def test_dict_key_order(self):
        eq("f", {"b": 1, "a": 2}, "f", {"a": 2, "b": 1})

    def test_nested_dict_key_order(self):
        eq("f", {"outer": {"z": 1, "a": 2}}, "f", {"outer": {"a": 2, "z": 1}})

    def test_list_order_preserved(self):
        neq("f", {"items": [1, 2, 3]}, "f", {"items": [3, 2, 1]})


# ── numeric type unification ────────────────────────────────────────────────

class TestNumericUnification:
    def test_int_string_equal(self):
        eq("f", {"n": "5"}, "f", {"n": 5})

    def test_float_string_equal(self):
        eq("f", {"n": "3.0"}, "f", {"n": 3})

    def test_float_int_equal(self):
        eq("f", {"n": 5.0}, "f", {"n": 5})

    def test_spaced_numeric_string(self):
        eq("f", {"n": " 42 "}, "f", {"n": 42})

    def test_non_integer_float_preserved(self):
        neq("f", {"n": 3.14}, "f", {"n": 3})

    def test_list_numeric_unification(self):
        eq("f", {"ids": ["1", "2"]}, "f", {"ids": [1, 2]})


# ── null / empty unification ────────────────────────────────────────────────

class TestNullEquivalence:
    def test_none_equals_empty_string(self):
        eq("f", {"x": None}, "f", {"x": ""})

    def test_none_equals_missing_key(self):
        # missing key and None value both disappear from normalised dict
        eq("f", {"a": 1, "b": None}, "f", {"a": 1})

    def test_null_string_equals_none(self):
        eq("f", {"x": "null"}, "f", {"x": None})

    def test_none_string_equals_none(self):
        eq("f", {"x": "None"}, "f", {"x": None})

    def test_null_equiv_disabled(self):
        # When null_equiv=False, None and "" should NOT be equal
        ka = canonical_key("f", {"x": None}, null_equiv=False)
        kb = canonical_key("f", {"x": ""}, null_equiv=False)
        # They may or may not match depending on JSON; the point is the flag is respected
        # For None→null vs ""→"" they definitely differ
        assert ka != kb

    def test_nil_string_equals_none(self):
        eq("f", {"x": "nil"}, "f", {"x": None})


# ── JSON encoding artefacts ─────────────────────────────────────────────────

class TestJsonArtefacts:
    def test_json_string_arguments(self):
        # LLM may return arguments as a JSON string rather than a dict
        eq("f", '{"q": "hello"}', "f", {"q": "hello"})

    def test_json_string_with_trailing_comma(self):
        eq("f", '{"q": "hello",}', "f", {"q": "hello"})

    def test_json_string_with_spaces(self):
        eq("f", '{ "q" : "hello" }', "f", {"q": "hello"})

    def test_unicode_escape(self):
        # \u0068 == "h"
        eq("f", {"q": "\u0068ello"}, "f", {"q": "hello"})


# ── combined equivalence pairs (PRD §4.3 — 50-pair gate) ───────────────────

EQUIVALENT_PAIRS = [
    # (a_name, a_args, b_name, b_args)
    ("book_reservation", {"flight": "AA100", "seat": "12A"}, "book_reservation", {"seat": "12A", "flight": "AA100"}),
    ("search", {"q": "flights to NYC"}, "SEARCH", {"q": "flights to NYC"}),
    ("f", {"n": "10"}, "f", {"n": 10}),
    ("f", {"n": 10.0}, "f", {"n": 10}),
    ("f", {"x": None}, "f", {}),
    ("f", {"x": ""}, "f", {}),
    ("f", {"x": "null"}, "f", {}),
    ("f", {"q": "  hello world  "}, "f", {"q": "hello world"}),
    ("f", {"q": "hello\t\tworld"}, "f", {"q": "hello world"}),
    ("f", '{"a": 1, "b": 2}', "f", {"b": 2, "a": 1}),
    ("f", '{"a": 1,}', "f", {"a": 1}),
    ("f", {"items": ["1", "2", "3"]}, "f", {"items": [1, 2, 3]}),
    ("f", {"x": {"y": 5.0}}, "f", {"x": {"y": 5}}),
    ("f", {"x": {"b": 1, "a": 2}}, "f", {"x": {"a": 2, "b": 1}}),
    ("get_flight", {"id": " AA100 "}, "get_flight", {"id": "AA100"}),
    ("f", {"flag": True}, "f", {"flag": True}),
    ("f", {"flag": False}, "f", {"flag": False}),
    (" f ", {"a": 1}, "f", {"a": 1}),
    ("F", {"a": 1}, "f", {"a": 1}),
    ("f", {"n": "0"}, "f", {"n": 0}),
    ("f", {"n": "0.0"}, "f", {"n": 0}),
    ("f", {"nested": {"x": " hi ", "y": None}}, "f", {"nested": {"x": "hi"}}),
    ("f", {"a": [1, 2], "b": "x"}, "f", {"b": "x", "a": [1, 2]}),
    ("f", '{ "q" : " hello " }', "f", {"q": "hello"}),
    ("f", {"q": "none"}, "f", {"q": None}),
    ("book_flight", {"from": "JFK", "to": "LAX", "date": "2024-01-01"},
     "book_flight", {"date": "2024-01-01", "from": "JFK", "to": "LAX"}),
    ("f", {"x": "5"}, "f", {"x": 5}),
    ("f", {"x": " 5 "}, "f", {"x": 5}),
    ("f", {"x": "5.0"}, "f", {"x": 5}),
    ("f", {"x": {"a": "null"}}, "f", {"x": {}}),
    ("f", {"items": ["  a  ", "  b  "]}, "f", {"items": ["a", "b"]}),
    ("search_flight", {"passengers": "2"}, "search_flight", {"passengers": 2}),
    ("f", {"q": "\u0068ello"}, "f", {"q": "hello"}),
    ("f", {"q": "\nhello\n"}, "f", {"q": "hello"}),
    ("CANCEL_ORDER", {"order_id": "123"}, "cancel_order", {"order_id": 123}),
    ("f", {"x": 3.0, "y": 4.0}, "f", {"x": 3, "y": 4}),
    ("f", {"a": {"b": {"c": " deep "}}}, "f", {"a": {"b": {"c": "deep"}}}),
    ("f", '{"a": "1"}', "f", {"a": 1}),
    ("f", '{"z": 0, "a": 1}', "f", {"a": 1, "z": 0}),
    ("f", {"x": "None"}, "f", {}),
    ("f", {"x": "nil"}, "f", {}),
    ("f", {"x": 1, "y": None, "z": ""}, "f", {"x": 1}),
    ("f", {"list": [None, "1", " 2 "]}, "f", {"list": [None, 1, 2]}),
    ("f", {"n": "100"}, "f", {"n": 100}),
    ("lookup", {"key": "ABC"}, "LOOKUP", {"key": "ABC"}),
    ("f", {"a": "1", "b": "2", "c": "3"}, "f", {"c": 3, "b": 2, "a": 1}),
    ("f", '{"x": 42}', "f", {"x": "42"}),
    ("f", {"x": 1.5}, "f", {"x": 1.5}),  # non-integer float stays float
    ("f", {"tag": " hello world "}, "f", {"tag": "hello world"}),
]


@pytest.mark.parametrize("a_name,a_args,b_name,b_args", EQUIVALENT_PAIRS)
def test_equivalent_pair(a_name, a_args, b_name, b_args):
    eq(a_name, a_args, b_name, b_args)


# ── non-equivalent pairs ────────────────────────────────────────────────────

NON_EQUIVALENT_PAIRS = [
    ("search", {"q": "NYC"}, "lookup", {"q": "NYC"}),
    ("f", {"n": 1}, "f", {"n": 2}),
    ("f", {"q": "hello"}, "f", {"q": "world"}),
    ("f", {"items": [1, 2]}, "f", {"items": [2, 1]}),
    ("f", {"n": 3.14}, "f", {"n": 3}),
    ("f", {"a": 1}, "f", {"b": 1}),
]


@pytest.mark.parametrize("a_name,a_args,b_name,b_args", NON_EQUIVALENT_PAIRS)
def test_non_equivalent_pair(a_name, a_args, b_name, b_args):
    neq(a_name, a_args, b_name, b_args)

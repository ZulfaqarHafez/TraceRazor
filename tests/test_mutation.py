"""Tests for MutationMetadata tool classification."""
import pytest
from tracerazor._mutation import MutationMetadata


M = MutationMetadata()


class TestBuiltInClassifications:
    def test_book_reservation_is_mutating(self):
        assert M.is_mutating("book_reservation") is True

    def test_cancel_order_is_mutating(self):
        assert M.is_mutating("cancel_order") is True

    def test_search_direct_flight_is_read_only(self):
        assert M.is_mutating("search_direct_flight") is False

    def test_get_order_details_is_read_only(self):
        assert M.is_mutating("get_order_details") is False

    def test_web_search_is_read_only(self):
        assert M.is_mutating("web_search") is False

    def test_send_email_is_mutating(self):
        assert M.is_mutating("send_email") is True

    def test_write_file_is_mutating(self):
        assert M.is_mutating("write_file") is True

    def test_read_file_is_read_only(self):
        assert M.is_mutating("read_file") is False


class TestCaseInsensitivity:
    def test_uppercase_name(self):
        assert M.is_mutating("BOOK_RESERVATION") is True

    def test_mixed_case(self):
        assert M.is_mutating("Search_Direct_Flight") is False

    def test_trailing_space(self):
        assert M.is_mutating(" cancel_order ") is True


class TestUnknownToolDefaultsMutating:
    def test_completely_unknown_tool(self):
        assert M.is_mutating("my_custom_tool") is True

    def test_novel_tool_name(self):
        assert M.is_mutating("process_payment_v2") is True


class TestOverrides:
    def test_override_read_only_to_mutating(self):
        m = MutationMetadata(overrides={"search": True})
        assert m.is_mutating("search") is True

    def test_override_unknown_to_read_only(self):
        m = MutationMetadata(overrides={"my_tool": False})
        assert m.is_mutating("my_tool") is False

    def test_override_case_insensitive(self):
        m = MutationMetadata(overrides={"MyTool": False})
        assert m.is_mutating("mytool") is False
        assert m.is_mutating("MYTOOL") is False

    def test_override_does_not_affect_other_tools(self):
        m = MutationMetadata(overrides={"my_tool": False})
        assert m.is_mutating("book_reservation") is True

    def test_empty_overrides(self):
        m = MutationMetadata(overrides={})
        assert m.is_mutating("book_reservation") is True


class TestClassifyAll:
    def test_classify_all_returns_dict(self):
        result = M.classify_all(["search_direct_flight", "book_reservation"])
        assert result == {"search_direct_flight": False, "book_reservation": True}

    def test_classify_all_empty(self):
        assert M.classify_all([]) == {}

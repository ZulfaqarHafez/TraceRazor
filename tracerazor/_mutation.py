"""MutationMetadata: classify tools as mutating (side-effecting) or read-only.

Unknown tools default to mutating. This is the safe default: treating a read-only
tool as mutating wastes one re-fan, treating a mutating tool as read-only would
execute the side effect K times.
"""
from __future__ import annotations

from typing import Dict, Optional

# ── built-in classifications ───────────────────────────────────────────────

_MUTATING: frozenset[str] = frozenset({
    # tau2-bench airline domain
    "book_reservation",
    "cancel_reservation",
    "update_reservation",
    "send_certificate",
    "send_email",
    # tau2-bench retail domain
    "cancel_order",
    "modify_order",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    # common LangChain / LangGraph tools
    "write_file",
    "delete_file",
    "append_to_file",
    "execute_code",
    "run_shell",
    "sql_query_write",
    "db_insert",
    "db_update",
    "db_delete",
    # generic HTTP mutations
    "post_request",
    "put_request",
    "delete_request",
    "patch_request",
})

_READ_ONLY: frozenset[str] = frozenset({
    # tau2-bench airline domain
    "search_direct_flight",
    "search_onestop_flight",
    "search_direct_flight_simple",
    "get_flight_status",
    "get_airport_info",
    "calculate_fare",
    "list_all_airports",
    # tau2-bench retail domain
    "get_order_details",
    "get_product_details",
    "list_orders",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_product_details_by_name",
    # common tools
    "search",
    "web_search",
    "tavily_search",
    "get_request",
    "read_file",
    "calculator",
    "lookup",
    "get_weather",
    "get_current_time",
})


class MutationMetadata:
    """Tool mutation classifier.

    Parameters
    ----------
    overrides:
        Optional dict mapping tool_name → is_mutating (bool).
        Overrides take precedence over built-in classifications.
    """

    def __init__(self, overrides: Optional[Dict[str, bool]] = None) -> None:
        self._overrides: Dict[str, bool] = {
            k.strip().lower(): v for k, v in (overrides or {}).items()
        }

    def is_mutating(self, tool_name: str) -> bool:
        name = tool_name.strip().lower()
        if name in self._overrides:
            return self._overrides[name]
        if name in _MUTATING:
            return True
        if name in _READ_ONLY:
            return False
        return True  # unknown → safe default

    def classify_all(self, tool_names: list[str]) -> Dict[str, bool]:
        """Return {tool_name: is_mutating} for every name in the list."""
        return {n: self.is_mutating(n) for n in tool_names}

"""LLM adapter factories for OpenAI and Anthropic clients.

Usage
-----
from tracerazor._adapters import openai_llm, anthropic_llm
from openai import AsyncOpenAI

llm = openai_llm(AsyncOpenAI(), model="gpt-4.1")
node = AdaptiveKNode(llm=llm, tools=my_tools)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


# ── OpenAI ─────────────────────────────────────────────────────────────────


def openai_llm(
    client: Any,
    model: str = "gpt-4.1",
    temperature: float = 1.0,
    **kwargs: Any,
) -> Callable:
    """Return an async LLM callable backed by an AsyncOpenAI client.

    The callable signature is ``(messages, tool_schema) -> dict`` as expected
    by AdaptiveKNode.

    Parameters
    ----------
    client:
        An ``openai.AsyncOpenAI`` instance.
    model:
        Model ID.  Default is gpt-4.1 (tau2-bench reference model).
    temperature:
        Sampling temperature.  Use > 0 for branch diversity.
    """

    async def _call(messages: List[dict], tool_schema: List[dict]) -> dict:
        tools_param = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                for t in tool_schema
            ]
            if tool_schema
            else None
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_param,
            temperature=temperature,
            **kwargs,
        )
        msg = response.choices[0].message
        usage = response.usage

        result: dict = {}
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            import json as _json
            result["tool_name"] = tc.function.name
            try:
                result["arguments"] = _json.loads(tc.function.arguments)
            except Exception:
                result["arguments"] = {"raw": tc.function.arguments}
        else:
            result["final_answer"] = msg.content or ""

        result["input_tokens"] = getattr(usage, "prompt_tokens", 0)
        result["output_tokens"] = getattr(usage, "completion_tokens", 0)
        # OpenAI prompt_tokens_details contains cached_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        result["cached_tokens"] = getattr(details, "cached_tokens", 0) if details else 0

        return result

    return _call


# ── Anthropic ──────────────────────────────────────────────────────────────


def anthropic_llm(
    client: Any,
    model: str = "claude-sonnet-4-5",
    temperature: float = 1.0,
    **kwargs: Any,
) -> Callable:
    """Return an async LLM callable backed by an AsyncAnthropic client.

    Parameters
    ----------
    client:
        An ``anthropic.AsyncAnthropic`` instance.
    model:
        Model ID.  Default is claude-sonnet-4-5 (secondary benchmark model).
    temperature:
        Sampling temperature.
    """

    async def _call(messages: List[dict], tool_schema: List[dict]) -> dict:
        # Separate system messages (Anthropic takes system as a top-level param)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        system_text = " ".join(m.get("content", "") for m in system_msgs) or None

        tools_param = (
            [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tool_schema
            ]
            if tool_schema
            else []
        )

        create_kwargs: dict = dict(
            model=model,
            messages=other_msgs,
            max_tokens=4096,
            temperature=temperature,
            **kwargs,
        )
        if system_text:
            create_kwargs["system"] = system_text
        if tools_param:
            create_kwargs["tools"] = tools_param

        response = await client.messages.create(**create_kwargs)
        usage = response.usage

        result: dict = {}
        for block in response.content:
            if block.type == "tool_use":
                result["tool_name"] = block.name
                result["arguments"] = block.input or {}
                break
        else:
            texts = [b.text for b in response.content if hasattr(b, "text")]
            result["final_answer"] = " ".join(texts)

        result["input_tokens"] = getattr(usage, "input_tokens", 0)
        result["output_tokens"] = getattr(usage, "output_tokens", 0)
        result["cached_tokens"] = getattr(usage, "cache_read_input_tokens", 0)

        return result

    return _call


# ── Generic / test helper ──────────────────────────────────────────────────


def mock_llm(responses: list) -> Callable:
    """Deterministic mock LLM for tests and offline demos.

    Returns the *same* response to all K branches at a given step (branches
    share identical message histories), then advances to the next response when
    the message history grows (indicating a new step has started).

    Each item in ``responses`` is a dict as returned by the real adapters::

        llm = mock_llm([
            {"tool_name": "search", "arguments": {"q": "hello"},
             "input_tokens": 100, "output_tokens": 20},
            {"final_answer": "The answer is 42",
             "input_tokens": 120, "output_tokens": 10},
        ])
    """
    _state = {"idx": 0, "last_msg_len": -1}

    async def _call(messages: List[dict], tool_schema: List[dict]) -> dict:
        msg_len = len(messages)
        # New step: message history grew → advance response pointer
        if msg_len != _state["last_msg_len"]:
            if _state["last_msg_len"] >= 0:
                _state["idx"] = (_state["idx"] + 1) % len(responses)
            _state["last_msg_len"] = msg_len
        return dict(responses[_state["idx"]])

    return _call

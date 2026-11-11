#!/usr/bin/env python3
"""context_tree.py - read chat/agent history as trees (EXPERIMENTAL research prototype).

One chat (session) is one tree; one repo/project is a grove; everything is the forest.

    soil      standing context before anything grows (system prompt, tool definitions,
              memory files): the context size of the first model call
    water     what the user pours in: user prompts
    roots     where nutrients come from: one root per tool/source, sized by content drawn in
    sun       the model's own work: output tokens
    trunk     the main conversation; every model call adds one growth ring
    branches  subagents
    leaves    blocks of content that entered context (tool results, prompts, replies)
    cells     the lines of a leaf; a cell is LIVE if something later reuses it, else DEAD
    fruit     file edits/writes
    scars     compactions;  winters: idle gaps longer than the cache lifetime

Honesty rules for every number this prints:
  * "live"/"dead" is an exact-text proxy: a line counts as reused only when a later
    message repeats a distinctive token that this line introduced. Content can be used
    without being quoted, so DEAD IS AN UPPER BOUND ON WASTE, never a measurement of it.
  * "carry" = characters x later model calls the block sat in context. It assumes blocks
    stay until the session ends; hosts that clear old results make it an overestimate.
  * Token figures come from the transcript's usage fields when present, otherwise they are
    estimated as characters/4 and flagged "est".
  * Nothing here is a saving. A saving exists only after a measured before/after rerun.

Privacy: runs fully offline, standard library only. Outputs contain counts, tool names and
sizes - never message text - unless you pass --labels.

Usage:
    python tools/context_tree.py ~/.claude/projects --out tree_out
    python tools/context_tree.py conversations.json --out tree_out      # claude.ai export
    python tools/context_tree.py ~/.claude/projects --out tree_out --render 12
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_./\-]{5,}|\d{3,}")
MIN_LEAF_CHARS = 200
WINTER_SECONDS = 3600  # idle gap after which a prompt cache has expired (1h lifetime)
FRUIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch", "str_replace_editor", "create_file"}
AGES = (1, 2, 3, 5, 8, 12, 20, 30, 50)
QUIET = (1, 2, 3, 5, 10)


# ----------------------------------------------------------------------------- loading
def _flat(content) -> str:
    """Text of a message content field (string, or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        if isinstance(b, str):
            out.append(b)
        elif isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                out.append(b.get("text") or "")
            elif t == "tool_result":
                out.append(_flat(b.get("content")))
    return "\n".join(out)


def _ts(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_claude_code(path):
    """One Claude Code transcript (.jsonl) -> list of events. Usage is de-duplicated per
    message id because the transcript repeats it on every content block of one message."""
    events, seen_usage = [], set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            if e.get("subtype") == "compact_boundary" or e.get("isCompactSummary"):
                events.append({"role": "scar", "ts": _ts(e.get("timestamp"))})
            m = e.get("message")
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
                continue
            content = m.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [b for b in (content or []) if isinstance(b, dict)]
            usage = None
            if m.get("role") == "assistant":
                mid = m.get("id") or e.get("uuid")
                if m.get("usage") and mid not in seen_usage:
                    seen_usage.add(mid)
                    usage = m["usage"]
                events.append({"role": "assistant", "id": mid, "blocks": blocks, "usage": usage, "ts": _ts(e.get("timestamp"))})
            else:
                events.append({"role": "user", "blocks": blocks, "ts": _ts(e.get("timestamp"))})
    return events


def load_claude_ai_export(path):
    """claude.ai data export (conversations.json) -> {chat name/uuid: events}.
    The export carries no token usage, so every token figure for these trees is estimated.
    NOTE: written against the documented shape (chat_messages[].sender/text/content);
    it has not been validated against a large real export."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("conversations") or [data]
    out = {}
    for i, conv in enumerate(data):
        if not isinstance(conv, dict):
            continue
        events = []
        for msg in conv.get("chat_messages") or conv.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            sender = msg.get("sender") or msg.get("role")
            role = "assistant" if sender == "assistant" else "user"
            blocks = [b for b in (msg.get("content") or []) if isinstance(b, dict)]
            if not blocks and msg.get("text"):
                blocks = [{"type": "text", "text": msg["text"]}]
            for att in msg.get("attachments") or []:
                if isinstance(att, dict) and att.get("extracted_content"):
                    blocks.append({"type": "tool_result", "tool_use_id": "attachment", "content": att["extracted_content"]})
            # tool results in exports sit inside the assistant message; move them to a user-side event
            results = [b for b in blocks if b.get("type") == "tool_result"]
            rest = [b for b in blocks if b.get("type") != "tool_result"]
            ts = _ts(msg.get("created_at"))
            if role == "assistant":
                pre, post, seen_result = [], [], False
                for b in blocks:
                    if b.get("type") == "tool_result":
                        seen_result = True
                    (post if seen_result else pre).append(b)
                if not results:
                    events.append({"role": "assistant", "id": msg.get("uuid"), "blocks": rest, "usage": None, "ts": ts})
                else:  # split around tool results so later text counts as a later call
                    chunk, n = [], 0
                    for b in blocks:
                        if b.get("type") == "tool_result":
                            if chunk:
                                events.append({"role": "assistant", "id": f"{msg.get('uuid')}:{n}", "blocks": chunk, "usage": None, "ts": ts})
                                chunk, n = [], n + 1
                            events.append({"role": "user", "blocks": [b], "ts": ts})
                        else:
                            chunk.append(b)
                    if chunk:
                        events.append({"role": "assistant", "id": f"{msg.get('uuid')}:{n}", "blocks": chunk, "usage": None, "ts": ts})
            else:
                events.append({"role": "user", "blocks": blocks, "ts": ts})
        key = conv.get("uuid") or f"chat-{i}"
        out[key] = events
    return out


def discover(root):
    """Yield (grove, tree_id, branch_id or None, loader) for every transcript under root."""
    if os.path.isfile(root):
        if root.endswith(".jsonl"):
            yield ("-", os.path.splitext(os.path.basename(root))[0], None, lambda p=root: load_claude_code(p))
        else:
            for key, events in load_claude_ai_export(root).items():
                yield ("claude.ai", key, None, lambda ev=events: ev)
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            if name == "conversations.json":
                for key, events in load_claude_ai_export(full).items():
                    yield ("claude.ai", key, None, lambda ev=events: ev)
            elif name.endswith(".jsonl") and name != "journal.jsonl":
                rel = os.path.relpath(full, root).split(os.sep)
                stem = os.path.splitext(name)[0]
                if "subagents" in rel:
                    i = rel.index("subagents")
                    tree = rel[i - 1] if i >= 1 else stem
                    grove = rel[i - 2] if i >= 2 else "-"
                    yield (grove, tree, stem, lambda p=full: load_claude_code(p))
                else:
                    grove = rel[-2] if len(rel) >= 2 else "-"
                    yield (grove, stem, None, lambda p=full: load_claude_code(p))


# ---------------------------------------------------------------------------- analysis
def _context_size(usage):
    if not usage:
        return None
    return sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def analyse(events):
    """Analyse one transcript (a trunk or one branch). Returns a dict of counts only."""
    tool_names, seen, owner = {}, set(), {}
    leaves, cells = [], []  # leaf: dict ; cell: [chars, created_call, ages:set, leaf_index]
    call = 0
    ctx, growth_est, out_tokens, usage_seen = [], [], 0, False
    call_ts, prompts, water_chars, fruit, scars = [], 0, 0, 0, 0
    pending_chars = 0  # characters that entered context since the previous model call
    roots = Counter()
    last_id = object()

    def add_leaf(kind, text):
        nonlocal pending_chars
        pending_chars += len(text)
        if len(text) < MIN_LEAF_CHARS:
            seen.update(TOK.findall(text))
            return
        idx = len(leaves)
        leaves.append({"kind": kind, "chars": len(text), "created": call, "cells": 0})
        local = set()
        for line in text.split("\n"):
            toks = set(TOK.findall(line))
            novel = toks - seen - local
            if novel:
                for t in novel:
                    owner[t] = len(cells)
                cells.append([len(line), call, set(), idx])
                leaves[idx]["cells"] += 1
            local |= toks
        seen.update(local)

    def reference(text, at_call):
        for t in set(TOK.findall(text)):
            c = owner.get(t)
            if c is not None and at_call > cells[c][1]:
                cells[c][2].add(at_call - cells[c][1])
        return text

    for e in events:
        if e["role"] == "scar":
            scars += 1
            continue
        if e["role"] == "assistant":
            if e.get("id") != last_id or e.get("id") is None:
                call += 1
                last_id = e.get("id")
                call_ts.append(e.get("ts"))
                growth_est.append(pending_chars // 4)
                pending_chars = 0
                ctx.append(None)
            if e.get("usage"):
                usage_seen = True
                ctx[-1] = _context_size(e["usage"])
                out_tokens += int(e["usage"].get("output_tokens") or 0)
            produced = []
            for b in e["blocks"]:
                if b.get("type") == "text":
                    produced.append(b.get("text") or "")
                elif b.get("type") == "tool_use":
                    name = b.get("name") or "tool"
                    tool_names[b.get("id")] = name
                    if name in FRUIT_TOOLS:
                        fruit += 1
                    produced.append(json.dumps(b.get("input"), default=str))
            text = "\n".join(produced)
            reference(text, call)
            if not usage_seen:
                out_tokens += len(text) // 4
            reply = "\n".join(b.get("text") or "" for b in e["blocks"] if b.get("type") == "text")
            pending_chars += len(text)
            if len(reply) >= MIN_LEAF_CHARS:  # a reply is itself a leaf carried on later turns
                pending_chars -= len(reply)
                add_leaf("reply", reply)
            else:
                seen.update(TOK.findall(text))
        else:
            texts = []
            for b in e["blocks"]:
                if b.get("type") == "tool_result":
                    name = tool_names.get(b.get("tool_use_id"), "attachment" if b.get("tool_use_id") == "attachment" else "tool")
                    body = _flat(b.get("content"))
                    roots[name] += len(body)
                    add_leaf("tool:" + name, body)
                elif b.get("type") == "text":
                    texts.append(b.get("text") or "")
            prompt = "\n".join(texts)
            if prompt.strip():
                prompts += 1
                water_chars += len(prompt)
                reference(prompt, call + 1)  # a user quoting earlier content keeps it alive
                add_leaf("prompt", prompt)

    total_calls = call
    for leaf in leaves:
        leaf.update(window=total_calls - leaf["created"], live_chars=0, cell_chars=0, last_use=0, carry=0, dead_carry=0, stale_carry=0)
    for chars, created, ages, idx in cells:
        leaf = leaves[idx]
        window = total_calls - created
        leaf["cell_chars"] += chars
        if ages:
            leaf["live_chars"] += chars
            leaf["last_use"] = max(leaf["last_use"], max(ages))
            leaf["stale_carry"] += chars * max(0, window - max(ages))
        else:
            leaf["dead_carry"] += chars * window
        leaf["carry"] += chars * window
    # growth ring per model call: tokens the context grew by before that call
    rings = []
    prev = None
    for i in range(total_calls):
        if ctx[i] is not None:
            rings.append(max(0, ctx[i] - prev) if prev is not None else 0)
            prev = ctx[i]
        else:
            rings.append(growth_est[i])
    known = [c for c in ctx if c is not None]
    winters = sum(1 for a, b in zip(call_ts, call_ts[1:]) if a and b and b - a > WINTER_SECONDS)
    return {
        "calls": total_calls, "prompts": prompts, "water_chars": water_chars,
        "soil_tokens": known[0] if known else None, "peak_context_tokens": max(known) if known else None,
        "sun_tokens": out_tokens, "tokens_estimated": not usage_seen,
        "roots": dict(roots), "fruit": fruit, "scars": scars, "winters": winters,
        "rings": rings, "leaves": leaves, "cells": [(c, cr, sorted(a), i) for c, cr, a, i in cells],
        "start": next((t for t in call_ts if t), None),
    }


def tree_summary(tree_id, grove, trunk, branches):
    parts = [trunk] + branches if trunk else branches
    leaves = [l for p in parts for l in p["leaves"]]
    carry = sum(l["carry"] for l in leaves)
    dead = sum(l["dead_carry"] for l in leaves)
    stale = sum(l["stale_carry"] for l in leaves)
    cell_chars = sum(l["cell_chars"] for l in leaves)
    live_chars = sum(l["live_chars"] for l in leaves)
    return {
        "tree": tree_id, "grove": grove,
        "calls": sum(p["calls"] for p in parts), "trunk_calls": trunk["calls"] if trunk else 0,
        "branches": len(branches),
        # water is what the *user* pours in: trunk prompts only (a branch's prompt comes from the agent)
        "prompts": (trunk or parts[0])["prompts"],
        "soil_tokens": trunk["soil_tokens"] if trunk else None,
        "peak_context_tokens": trunk["peak_context_tokens"] if trunk else None,
        "sun_tokens": sum(p["sun_tokens"] for p in parts),
        "water_chars": (trunk or parts[0])["water_chars"],
        "tokens_estimated": any(p["tokens_estimated"] for p in parts),
        "fruit": sum(p["fruit"] for p in parts), "scars": sum(p["scars"] for p in parts),
        "winters": sum(p["winters"] for p in parts),
        "leaves": len(leaves), "leaf_chars": sum(l["chars"] for l in leaves),
        "live_cell_share": (live_chars / cell_chars) if cell_chars else None,
        "carry": carry,
        "dead_on_arrival_share": (dead / carry) if carry else None,
        "stale_share": (stale / carry) if carry else None,
        "roots": dict(sum((Counter(p["roots"]) for p in parts), Counter())),
    }


def _quantiles(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return {"n": len(v), "p25": round(q(0.25), 4), "median": round(q(0.5), 4), "p75": round(q(0.75), 4)}


def forest_summary(trees, parts_by_tree):
    """Pooled and per-tree statistics. Per-tree medians are the honest headline: pooled
    figures are dominated by the few largest trees."""
    leaves = [(l, t) for t, parts in parts_by_tree.items() for p in parts for l in p["leaves"]]
    cells = [c for parts in parts_by_tree.values() for p in parts for c in [(ch, p["calls"] - cr, ages) for ch, cr, ages, _ in p["cells"]]]
    by_kind = defaultdict(list)
    for l, _ in leaves:
        kind = l["kind"] if not l["kind"].startswith("tool:") else l["kind"]
        by_kind[kind].append(l)

    def survival(ls):
        row = {}
        for a in AGES:
            elig = [l for l in ls if l["window"] >= a]
            if len(elig) >= 30:
                row[str(a)] = {"n": len(elig), "still_referenced": round(sum(1 for l in elig if l["last_use"] >= a) / len(elig), 3)}
        return row

    quiet = {}
    for k in QUIET:
        elig = [c for c in cells if c[1] > k and not any(a <= k for a in c[2])]
        if len(elig) >= 100:
            quiet[str(k)] = {"n_cells": len(elig), "reused_later": round(sum(1 for c in elig if c[2]) / len(elig), 3)}
    carry = sum(l["carry"] for l, _ in leaves)
    dead_sorted = sorted((l["dead_carry"] for l, _ in leaves), reverse=True)
    top = max(1, len(dead_sorted) // 20)
    kinds = {}
    for kind, ls in sorted(by_kind.items(), key=lambda kv: -sum(l["carry"] for l in kv[1]))[:15]:
        c = sum(l["carry"] for l in ls)
        kinds[kind] = {
            "leaves": len(ls), "chars": sum(l["chars"] for l in ls),
            "share_of_all_carry": round(c / carry, 3) if carry else None,
            "dead_on_arrival_share": round(sum(l["dead_carry"] for l in ls) / c, 3) if c else None,
            "stale_share": round(sum(l["stale_carry"] for l in ls) / c, 3) if c else None,
            "survival_by_age_in_calls": survival(ls),
        }
    multi = [t for t in trees if t["calls"] >= 5 and t["carry"]]
    return {
        "caveats": [
            "live/dead is an exact-text proxy; dead is an upper bound on waste, not a measurement",
            "carry assumes content stays in context until the session ends",
            "nothing here is a saving; savings need a measured before/after rerun",
        ],
        "trees": len(trees), "groves": len({t["grove"] for t in trees}),
        "trees_with_5plus_calls": len(multi),
        "trees_with_estimated_tokens": sum(1 for t in trees if t["tokens_estimated"]),
        "model_calls": sum(t["calls"] for t in trees), "leaves": len(leaves),
        "per_tree": {
            "calls": _quantiles([t["calls"] for t in trees]),
            "soil_tokens": _quantiles([t["soil_tokens"] for t in trees]),
            "live_cell_share": _quantiles([t["live_cell_share"] for t in multi]),
            "dead_on_arrival_share_of_carry": _quantiles([t["dead_on_arrival_share"] for t in multi]),
            "stale_share_of_carry": _quantiles([t["stale_share"] for t in multi]),
        },
        "pooled": {
            "dead_on_arrival_share_of_carry": round(sum(l["dead_carry"] for l, _ in leaves) / carry, 3) if carry else None,
            "stale_share_of_carry": round(sum(l["stale_carry"] for l, _ in leaves) / carry, 3) if carry else None,
            "top_5pct_of_leaves_share_of_dead_carry": round(sum(dead_sorted[:top]) / sum(dead_sorted), 3) if sum(dead_sorted) else None,
        },
        "survival_all_leaves_by_age_in_calls": survival([l for l, _ in leaves]),
        "cell_not_reused_within_k_calls_then_reused_later": quiet,
        "by_leaf_kind": kinds,
    }


# --------------------------------------------------------------------------- rendering
PALETTE = {"live": "#3f8f4f", "dead": "#a8763e", "trunk": "#6b4a2f", "soil": "#4a3a2c", "water": "#3b82c4",
           "sun": "#e6a817", "fruit": "#c2413b", "ink": "#22262b", "muted": "#6b7280", "paper": "#fbfaf6", "scar": "#111"}


def _leaf_svg(x, y, r, live_share, side, title):
    """A leaf: an ellipse whose green part is the live-cell share (by length, not area)."""
    w, h = r * 2.2, r * 1.1
    x0 = x if side > 0 else x - w
    gid = f"g{abs(hash((round(x, 2), round(y, 2), title))) % 10**9}"
    cut = max(0.0, min(1.0, live_share))
    stops = (f'<stop offset="{cut:.3f}" stop-color="{PALETTE["live"]}"/><stop offset="{cut:.3f}" stop-color="{PALETTE["dead"]}"/>')
    if side < 0:
        stops = (f'<stop offset="{1 - cut:.3f}" stop-color="{PALETTE["dead"]}"/><stop offset="{1 - cut:.3f}" stop-color="{PALETTE["live"]}"/>')
    return (f'<defs><linearGradient id="{gid}">{stops}</linearGradient></defs>'
            f'<ellipse cx="{x0 + w / 2:.1f}" cy="{y:.1f}" rx="{w / 2:.1f}" ry="{h / 2:.1f}" fill="url(#{gid})" stroke="#2c4a30" stroke-width="0.6">'
            f'<title>{html.escape(title)}</title></ellipse>')


def render_tree(summary, trunk, branches, labels=False, top_n=24):
    W, H, ground, cx = 980, 720, 560, 330
    calls = max(1, trunk["calls"])
    px = (ground - 60) / calls
    leaves = sorted(trunk["leaves"], key=lambda l: -l["carry"])
    big, small = leaves[:top_n], leaves[top_n:]
    max_chars = max([l["chars"] for l in leaves] or [1])
    # trunk width at a height = carry of everything created at that call (oldest = thickest)
    per_call = Counter()
    for l in trunk["leaves"]:
        per_call[l["created"]] += l["chars"] * l["window"]
    cum = [sum(v for c, v in per_call.items() if c >= i) for i in range(calls + 1)]
    wmax = max(cum[0], 1)
    half = lambda i: 4 + 34 * math.sqrt(cum[min(i, calls)] / wmax)
    y = lambda i: ground - i * px
    left = " ".join(f"{cx - half(i):.1f},{y(i):.1f}" for i in range(calls + 1))
    right = " ".join(f"{cx + half(i):.1f},{y(i):.1f}" for i in range(calls, -1, -1))
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="{PALETTE["paper"]}"/>']
    # soil band: thickness = standing context before the first call
    soil = summary.get("soil_tokens") or 0
    soil_h = 18 + min(90, 9 * math.log10(1 + soil))
    s.append(f'<rect x="20" y="{ground}" width="620" height="{soil_h:.0f}" fill="{PALETTE["soil"]}" opacity="0.85"><title>soil: {soil:,} tokens of standing context at the first call</title></rect>')
    s.append(f'<text x="28" y="{ground + 14}" fill="#e9dfd2">soil {soil:,} tok standing context</text>')
    # roots: one per source, thickness by characters drawn in
    roots = sorted(summary["roots"].items(), key=lambda kv: -kv[1])[:7]
    rmax = max([v for _, v in roots] or [1])
    for i, (name, chars) in enumerate(roots):
        ang = math.radians(200 + i * (140 / max(1, len(roots) - 1)) if len(roots) > 1 else 270)
        ex, ey = cx + 230 * math.cos(ang), ground + 20 + 95 * abs(math.sin(ang)) + 22
        s.append(f'<path d="M{cx},{ground} Q{cx},{ey:.0f} {ex:.0f},{ey:.0f}" fill="none" stroke="#8a6a4a" stroke-width="{1.5 + 9 * math.sqrt(chars / rmax):.1f}" stroke-linecap="round"><title>root {html.escape(name)}: {chars:,} chars drawn in</title></path>')
        short = name.split("__")[-1][:18]
        s.append(f'<text x="{ex:.0f}" y="{ey + 14 + (i % 2) * 12:.0f}" text-anchor="middle" fill="{PALETTE["ink"]}">{html.escape(short)} {chars / 1000:.0f}k</text>')
    s.append(f'<polygon points="{left} {right}" fill="{PALETTE["trunk"]}"><title>trunk: {calls} model calls; width = content carried from that point on</title></polygon>')
    # leaves
    for n, l in enumerate(big):
        side = 1 if n % 2 == 0 else -1
        ly = y(l["created"]) - 2
        r = 5 + 26 * math.sqrt(l["chars"] / max_chars)
        bx = cx + side * (half(l["created"]) + 14 + (n % 5) * 26)
        live = (l["live_chars"] / l["cell_chars"]) if l["cell_chars"] else 0
        s.append(f'<line x1="{cx + side * half(l["created"]):.1f}" y1="{ly:.1f}" x2="{bx:.1f}" y2="{ly:.1f}" stroke="{PALETTE["trunk"]}" stroke-width="1.2"/>')
        s.append(_leaf_svg(bx, ly, r, live, side, f'{l["kind"]} | {l["chars"]:,} chars | born at call {l["created"]} | carried {l["window"]} calls | live cells {live:.0%} | last reused {l["last_use"]} calls after birth'))
    for l in small:
        side = 1 if l["created"] % 2 else -1
        live = (l["live_chars"] / l["cell_chars"]) if l["cell_chars"] else 0
        s.append(f'<circle cx="{cx + side * (half(l["created"]) + 5):.1f}" cy="{y(l["created"]):.1f}" r="2" fill="{PALETTE["live"] if live >= 0.5 else PALETTE["dead"]}" opacity="0.7"/>')
    # branches (subagents): a stub with one leaf cluster each
    for i, b in enumerate(branches[:14]):
        by = y(calls * (i + 1) / (min(len(branches), 14) + 1))
        side = -1 if i % 2 == 0 else 1
        cc = sum(l["cell_chars"] for l in b["leaves"]) or 1
        live = sum(l["live_chars"] for l in b["leaves"]) / cc
        chars = sum(l["chars"] for l in b["leaves"])
        ex = cx + side * (200 + (i % 3) * 30)
        s.append(f'<path d="M{cx},{by:.0f} Q{(cx + ex) / 2:.0f},{by - 26:.0f} {ex:.0f},{by - 34:.0f}" fill="none" stroke="{PALETTE["trunk"]}" stroke-width="{2 + min(6, b["calls"] / 8):.1f}"/>')
        s.append(_leaf_svg(ex, by - 34, 8 + 16 * math.sqrt(chars / max(max_chars, chars, 1)), live, side, f'branch (subagent): {b["calls"]} calls | {chars:,} chars in leaves | live cells {live:.0%}'))
    # water, sun, fruit, scars
    s.append(f'<circle cx="590" cy="70" r="{14 + min(40, 4 * math.log10(1 + summary["sun_tokens"])):.0f}" fill="{PALETTE["sun"]}"><title>sun: {summary["sun_tokens"]:,} output tokens{" (est)" if summary["tokens_estimated"] else ""}</title></circle>')
    s.append(f'<text x="590" y="74" text-anchor="middle" fill="#3a2a00">{summary["sun_tokens"] / 1000:.0f}k out</text>')
    s.append(f'<text x="24" y="40" fill="{PALETTE["water"]}">water: {summary["prompts"]} prompts, {summary["water_chars"] / 1000:.1f}k chars</text>')
    s.append(f'<text x="24" y="56" fill="{PALETTE["fruit"]}">fruit: {summary["fruit"]} file edits</text>')
    s.append(f'<text x="24" y="72" fill="{PALETTE["muted"]}">scars (compactions): {summary["scars"]}   winters (idle &gt; 1h): {summary["winters"]}   branches: {summary["branches"]}</text>')
    # cross-section: growth rings, one per call (binned to <= 36), width = context growth
    rings = trunk["rings"] or [0]
    bins = min(36, len(rings))
    step = len(rings) / bins
    widths = [sum(rings[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(bins)]
    dead_by_call, all_by_call = Counter(), Counter()
    for ch, cr, ages, _ in trunk["cells"]:
        all_by_call[cr] += ch
        if not ages:
            dead_by_call[cr] += ch
    rx, ry, R = 810, 150, 110
    total = sum(widths) or 1
    radius = R
    s.append(f'<text x="{rx}" y="24" text-anchor="middle" fill="{PALETTE["ink"]}" font-size="12">growth rings (centre = first call)</text>')
    for i in range(bins - 1, -1, -1):  # outermost = newest
        lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
        a = sum(all_by_call[c] for c in range(lo, hi + 1)) or 1
        dead = sum(dead_by_call[c] for c in range(lo, hi + 1)) / a
        mix = lambda p, q: int(p + (q - p) * dead)
        col = f"rgb({mix(0x3f, 0xa8)},{mix(0x8f, 0x76)},{mix(0x4f, 0x3e)})"
        s.append(f'<circle cx="{rx}" cy="{ry}" r="{max(radius, 1):.1f}" fill="{col}" stroke="{PALETTE["paper"]}" stroke-width="0.5"><title>calls {lo + 1}-{hi}: context grew {widths[i]:,} tokens; {dead:.0%} of new cells never reused</title></circle>')
        radius -= R * widths[i] / total
    # ranked table: the precise view (length encodes carry; the tree is the overview)
    s.append(f'<text x="670" y="300" fill="{PALETTE["ink"]}" font-size="12">heaviest leaves by carry (chars x calls carried)</text>')
    cmax = max([l["carry"] for l in big] or [1])
    for i, l in enumerate(big[:14]):
        yy = 318 + i * 24
        live = (l["live_chars"] / l["cell_chars"]) if l["cell_chars"] else 0
        wbar = 280 * l["carry"] / cmax
        s.append(f'<rect x="670" y="{yy}" width="{wbar * live:.1f}" height="11" fill="{PALETTE["live"]}"/><rect x="{670 + wbar * live:.1f}" y="{yy}" width="{wbar * (1 - live):.1f}" height="11" fill="{PALETTE["dead"]}"/>')
        s.append(f'<text x="670" y="{yy + 21}" fill="{PALETTE["muted"]}" font-size="10">{html.escape(l["kind"])} · {l["chars"] / 1000:.1f}k chars · call {l["created"]} · carried {l["window"]} · live {live:.0%}</text>')
    s.append(f'<rect x="670" y="676" width="12" height="10" fill="{PALETTE["live"]}"/><text x="688" y="685" fill="{PALETTE["ink"]}">live cells (reused later)</text>'
             f'<rect x="820" y="676" width="12" height="10" fill="{PALETTE["dead"]}"/><text x="838" y="685" fill="{PALETTE["ink"]}">dead (never reused)</text>')
    s.append(f'<text x="670" y="704" fill="{PALETTE["muted"]}" font-size="10">dead = upper bound on waste (exact-text proxy), not a saving</text>')
    title = summary["tree"][:10] if not labels else summary["tree"]
    s.append(f'<text x="24" y="22" fill="{PALETTE["ink"]}" font-size="14" font-weight="600">tree {html.escape(title)} · {summary["calls"]} calls · live cells {(summary["live_cell_share"] or 0):.0%} · dead-on-arrival {(summary["dead_on_arrival_share"] or 0):.0%} of carry</text>')
    s.append("</svg>")
    return "".join(s)


def render_forest(trees):
    """Small multiples: one glyph per tree, grouped by grove. Height = model calls (log),
    canopy size = leaf content (sqrt), canopy split = live vs dead cells."""
    groves = defaultdict(list)
    for t in trees:
        groves[t["grove"]].append(t)
    cell, cols = 46, 24
    out, y0 = [], 30
    for gi, (grove, ts) in enumerate(sorted(groves.items(), key=lambda kv: -len(kv[1]))):
        out.append(f'<text x="10" y="{y0 - 8}" fill="{PALETTE["ink"]}" font-size="12">grove {gi + 1} · {len(ts)} trees</text>')
        ts = sorted(ts, key=lambda t: -t["calls"])
        for i, t in enumerate(ts):
            x = 14 + (i % cols) * cell + cell / 2
            base = y0 + (i // cols) * 112 + 104
            h = 6 + 9 * math.log2(1 + t["calls"])
            r = 2 + min(18, math.sqrt(t["leaf_chars"]) / 28)
            live = t["live_cell_share"] or 0
            out.append(f'<g><title>{t["calls"]} calls · {t["leaves"]} leaves · live cells {live:.0%} · dead-on-arrival {(t["dead_on_arrival_share"] or 0):.0%} of carry · {t["branches"]} branches</title>'
                       f'<line x1="{x}" y1="{base}" x2="{x}" y2="{base - h:.1f}" stroke="{PALETTE["trunk"]}" stroke-width="{1.5 + min(4, t["branches"] / 3):.1f}"/>'
                       f'<circle cx="{x}" cy="{base - h:.1f}" r="{r:.1f}" fill="{PALETTE["dead"]}"/>'
                       f'<path d="M{x},{base - h:.1f} m{-r:.1f},0 a{r:.1f},{r:.1f} 0 0 1 {2 * r:.1f},0 z" fill="{PALETTE["live"]}" transform="rotate({-90 + 180 * (1 - live):.0f} {x} {base - h:.1f})" opacity="{0.25 + 0.75 * live:.2f}"/></g>')
        y0 += (math.ceil(len(ts) / cols)) * 112 + 34
    return (f'<svg viewBox="0 0 {14 + cols * cell + 14} {y0}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">'
            f'<rect width="100%" height="100%" fill="{PALETTE["paper"]}"/>{"".join(out)}</svg>')


def page(title, body):
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f'<body style="margin:0;background:{PALETTE["paper"]};color:{PALETTE["ink"]};font-family:system-ui,sans-serif">'
            f'<div style="max-width:1100px;margin:0 auto;padding:16px">{body}</div></body>')


# -------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="Claude Code projects folder, a .jsonl transcript, or a claude.ai conversations.json")
    ap.add_argument("--out", default="tree_out", help="output folder (default: tree_out)")
    ap.add_argument("--render", type=int, default=6, help="render this many of the largest trees as HTML (default 6)")
    ap.add_argument("--labels", action="store_true", help="show full session ids in renders (off by default)")
    args = ap.parse_args(argv)

    trunks, branches, groves = {}, defaultdict(list), {}
    skipped = 0
    for root in args.paths:
        for grove, tree, branch, load in discover(os.path.expanduser(root)):
            try:
                result = analyse(load())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                skipped += 1
                print(f"skip {tree}: {type(exc).__name__}", file=sys.stderr)
                continue
            if result["calls"] == 0:
                continue
            groves.setdefault(tree, grove)
            if branch is None:
                trunks[tree] = result
            else:
                branches[tree].append(result)
    ids = sorted(set(trunks) | set(branches))
    if not ids:
        print("no transcripts with model calls found", file=sys.stderr)
        return 2
    grove_alias = {g: f"grove-{i + 1}" for i, g in enumerate(sorted(set(groves.values())))}
    trees, parts = [], {}
    for n, tid in enumerate(ids):
        alias = tid if args.labels else f"tree-{n + 1:04d}"
        summary = tree_summary(alias, grove_alias[groves[tid]], trunks.get(tid), branches.get(tid, []))
        trees.append(summary)
        parts[alias] = ([trunks[tid]] if tid in trunks else []) + branches.get(tid, [])
    forest = forest_summary(trees, parts)
    forest["skipped_files"] = skipped
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forest_summary.json"), "w") as fh:
        json.dump(forest, fh, indent=2)
    with open(os.path.join(args.out, "trees.json"), "w") as fh:
        json.dump(trees, fh, indent=1)
    body = [f"<h2>Forest: {len(trees)} trees in {forest['groves']} groves</h2>",
            "<p>Each glyph is one chat. Height = model calls (log scale). Canopy size = content in leaves. Green share = live cells. Hover for numbers.</p>",
            render_forest(trees)]
    ranked = sorted(range(len(ids)), key=lambda i: -(trees[i]["carry"]))[: max(0, args.render)]
    for i in ranked:
        tid = ids[i]
        if tid in trunks:
            body.append(render_tree(trees[i], trunks[tid], branches.get(tid, []), labels=args.labels))
    with open(os.path.join(args.out, "forest.html"), "w") as fh:
        fh.write(page("Context forest", "".join(body)))
    pt, pooled = forest["per_tree"], forest["pooled"]
    print(f"trees {forest['trees']} ({forest['trees_with_5plus_calls']} with 5+ calls) | groves {forest['groves']} | model calls {forest['model_calls']} | leaves {forest['leaves']}")
    print(f"per-tree median: live cells {pt['live_cell_share'] and pt['live_cell_share']['median']} | dead-on-arrival share of carry {pt['dead_on_arrival_share_of_carry'] and pt['dead_on_arrival_share_of_carry']['median']} | stale share {pt['stale_share_of_carry'] and pt['stale_share_of_carry']['median']}")
    print(f"pooled: dead-on-arrival {pooled['dead_on_arrival_share_of_carry']} | stale {pooled['stale_share_of_carry']} | top 5% of leaves hold {pooled['top_5pct_of_leaves_share_of_dead_carry']} of dead carry")
    print("dead = upper bound on waste (exact-text proxy). Not a saving.")
    print(f"wrote {args.out}/forest_summary.json, trees.json, forest.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

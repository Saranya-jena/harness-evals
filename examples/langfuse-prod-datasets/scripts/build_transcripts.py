#!/usr/bin/env python3
"""Build clean, readable conversation transcripts from hydrated Langfuse traces.

Unlike ``build_conversation_sessions.py`` (which summarizes), this reconstructs the
*actual* flow of each conversation — user prompts, assistant replies, and every tool
call (request + response) — from the richest source in a v3/autogen trace: the
``provider_call_*`` GENERATION observation's ``input.messages`` array (full Anthropic
message list) plus the final answer in ``process_chat_response.output_data``.

Outputs (into ``transcripts/`` by default):
  - ``<NN-env-module-shortid>.md``         one readable Markdown transcript per conversation
  - ``<NN-env-module-shortid>.tools.json`` full (untruncated) tool responses sidecar
  - ``labels.csv``                          one row per conversation for good/bad/useless labeling

Sources of conversations (no network needed): the existing session manifest files written
by ``build_conversation_sessions.py`` under ``sessions/`` and ``random-sample-15/sessions/``.
Each manifest lists interactions with ``trace_id`` + ``hydrated_trace_file``.

Usage:
  # POC — single conversation from cache
  python scripts/build_transcripts.py --conversation-id 74d532e9-582c-49a1-bbce-7a4abdd8580b

  # All cached sessions (pilot-30 + random-15)
  python scripts/build_transcripts.py

  # Fresh end-to-end: fetch a conversation straight from Langfuse, then build
  python scripts/build_transcripts.py --fetch --conversation-id <session_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_conversation_sessions import (  # noqa: E402
    LangfuseClient,
    load_langfuse_config,
)

DEFAULT_SESSION_DIRS = [ROOT / "sessions", ROOT / "random-sample-15" / "sessions"]
TRANSCRIPTS_DIR = ROOT / "transcripts"
TOOL_RESP_LIMIT = 500
SYSTEM_PROMPT_MARKER = "You are Harness AI"
CONTEXT_MARKERS = ("## Current Harness Context", "## Additional Context", "## User Context")
FEEDBACK_PREFIXES = ('{"reasons"',)


# --------------------------------------------------------------------------- parsing


def richest_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the message array from the provider_call with the highest message_count."""
    best: Any = None
    best_count = -1
    for obs in trace.get("observations") or []:
        name = obs.get("name") or ""
        if not name.startswith("provider_call"):
            continue
        inp = obs.get("input") or {}
        if not isinstance(inp, dict):
            continue
        msgs = inp.get("messages")
        count = inp.get("message_count") or 0
        if msgs and count >= best_count:
            best_count = count
            best = msgs
    if best is None:
        return []
    if isinstance(best, str):
        try:
            best = json.loads(best)
        except json.JSONDecodeError:
            return []
    return best if isinstance(best, list) else []


def final_answer(trace: dict[str, Any]) -> str:
    """Last non-empty process_chat_response.output_data (the turn's final answer)."""
    candidates: list[tuple[str, str]] = []
    for obs in trace.get("observations") or []:
        if (obs.get("name") or "") != "process_chat_response":
            continue
        out = obs.get("output") or {}
        data = out.get("output_data") if isinstance(out, dict) else None
        if isinstance(data, str) and data.strip():
            ts = obs.get("start_time") or obs.get("startTime") or ""
            candidates.append((ts, data.strip()))
    candidates.sort(key=lambda c: c[0])
    return candidates[-1][1] if candidates else ""


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") for b in _content_blocks(content) if b.get("type") == "text"]
    return "\n".join(p for p in parts if p and p.strip()).strip()


def _is_tool_result_msg(content: Any) -> bool:
    return any(b.get("type") == "tool_result" for b in _content_blocks(content))


def _is_plain_user_text(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    blocks = _content_blocks(content)
    has_text = any(b.get("type") == "text" for b in blocks)
    return has_text and not _is_tool_result_msg(content)


def _is_system(text: str) -> bool:
    return SYSTEM_PROMPT_MARKER in text[:400]


def clean_user_text(text: str) -> str:
    text = (text or "").strip()
    cut = len(text)
    for marker in CONTEXT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    # Also handle a trailing "\n\n{...currentUrl...}" JSON blob
    brace = text.find("\n\n{")
    if brace != -1 and ("currentUrl" in text[brace:] or "current_url" in text[brace:]):
        cut = min(cut, brace)
    return text[:cut].strip()


def build_turn(trace: dict[str, Any], seq: int) -> dict[str, Any]:
    """Reconstruct one interaction's flow from its richest message array + final answer."""
    msgs = richest_messages(trace)

    # Find this turn's real user prompt = last plain-text user message that is not the system prompt.
    last_user_idx = -1
    for i, m in enumerate(msgs):
        if _is_plain_user_text(m):
            if _is_system(_text_of(m.get("content"))):
                continue
            last_user_idx = i
    tail = msgs[last_user_idx:] if last_user_idx >= 0 else msgs

    # Map tool_use_id -> result content across the tail.
    results: dict[str, str] = {}
    for m in tail:
        if m.get("role") != "user":
            continue
        for b in _content_blocks(m.get("content")):
            if b.get("type") == "tool_result":
                content = b.get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                    )
                results[b.get("tool_use_id", "")] = content if isinstance(content, str) else json.dumps(content)

    user_text = ""
    if tail:
        user_text = clean_user_text(_text_of(tail[0].get("content")))

    events: list[dict[str, Any]] = []
    for m in tail[1:] if tail else []:
        if m.get("role") != "assistant":
            continue
        for b in _content_blocks(m.get("content")):
            btype = b.get("type")
            if btype == "text":
                txt = (b.get("text") or "").strip()
                if not txt or txt == "{}" or txt.startswith(FEEDBACK_PREFIXES):
                    continue
                events.append({"kind": "assistant_text", "text": txt})
            elif btype == "tool_use":
                tid = b.get("id", "")
                events.append(
                    {
                        "kind": "tool",
                        "id": tid,
                        "name": b.get("name", "?"),
                        "request": b.get("input", {}),
                        "response": results.get(tid, ""),
                    }
                )

    final = final_answer(trace)
    # Avoid duplicating the final answer if it is already the last narration block.
    if events and events[-1]["kind"] == "assistant_text" and events[-1]["text"] == final:
        final = ""

    return {
        "sequence": seq,
        "trace_id": trace.get("id"),
        "user_text": user_text,
        "events": events,
        "final": final,
        "cost_usd": trace.get("total_cost") or trace.get("totalCost") or 0.0,
    }


# --------------------------------------------------------------------------- rendering


def _truncate(text: str, limit: int = TOOL_RESP_LIMIT) -> tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    return text[:limit] + " …", True


def _compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
    except (TypeError, ValueError):
        return str(obj)


def render_markdown(meta: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    total_cost = sum(t["cost_usd"] for t in turns)
    tool_count = sum(1 for t in turns for e in t["events"] if e["kind"] == "tool")
    scope = "/".join(x for x in (meta.get("org_id"), meta.get("project_id")) if x)

    lines.append(f"# Conversation `{meta['conversation_id']}`")
    lines.append("")
    lines.append(
        f"- **env:** {meta.get('env')}  ·  **module:** {meta.get('module')}  ·  "
        f"**scope:** {scope or 'account'}"
    )
    lines.append(
        f"- **turns:** {len(turns)}  ·  **tool calls:** {tool_count}  ·  "
        f"**total cost:** ${total_cost:.4f}"
    )
    lines.append(f"- **first seen:** {meta.get('first_timestamp', '')}")
    lines.append("")
    lines.append("> System prompt on every turn: _[constant system prompt]_ (omitted for readability)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for turn in turns:
        n = turn["sequence"]
        lines.append(f"## Turn {n} · user")
        lines.append("")
        lines.append(turn["user_text"] or "_[no user text]_")
        lines.append("")
        for e in turn["events"]:
            if e["kind"] == "assistant_text":
                lines.append(f"**Turn {n} · assistant**")
                lines.append("")
                lines.append(e["text"])
                lines.append("")
            else:  # tool
                lines.append(f"**Turn {n} · assistant → tool `{e['name']}`**")
                lines.append("")
                lines.append("_request_")
                lines.append("```json")
                lines.append(_compact_json(e["request"]))
                lines.append("```")
                resp, truncated = _truncate(e["response"])
                note = "  _(full response in `.tools.json`)_" if truncated else ""
                lines.append(f"_response_{note}")
                lines.append("```json")
                lines.append(resp or "_[empty]_")
                lines.append("```")
                lines.append("")
        if turn["final"]:
            lines.append(f"## Turn {n} · assistant (final)")
            lines.append("")
            lines.append(turn["final"])
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_tool_sidecar(conversation_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in turns:
        for e in turn["events"]:
            if e["kind"] != "tool":
                continue
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "turn": turn["sequence"],
                    "tool_use_id": e["id"],
                    "name": e["name"],
                    "request": e["request"],
                    "response": e["response"],
                }
            )
    return rows


# --------------------------------------------------------------------------- I/O


def load_trace_from_cache(hydrated_file: str) -> dict[str, Any] | None:
    path = ROOT / hydrated_file if not Path(hydrated_file).is_absolute() else Path(hydrated_file)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def iter_manifests(session_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in session_dirs:
        if not d.is_dir():
            continue
        files.extend(sorted(p for p in d.glob("*.json") if p.name != "index.json"))
    return files


def conversation_meta(manifest: dict[str, Any]) -> dict[str, Any]:
    pm = manifest.get("pilot_metadata") or {}
    ss = manifest.get("session_summary") or {}
    return {
        "conversation_id": manifest.get("conversation_id"),
        "env": pm.get("env"),
        "module": pm.get("module"),
        "org_id": pm.get("org_id"),
        "project_id": pm.get("project_id"),
        "first_timestamp": ss.get("first_timestamp"),
    }


def process_manifest(
    manifest_path: Path,
    *,
    client: LangfuseClient | None,
    fetch: bool,
) -> dict[str, Any] | None:
    manifest = json.loads(manifest_path.read_text())
    meta = conversation_meta(manifest)
    interactions = manifest.get("interactions") or []
    if not interactions:
        return None

    turns: list[dict[str, Any]] = []
    for idx, inter in enumerate(interactions, start=1):
        trace: dict[str, Any] | None = None
        if fetch and client is not None:
            trace = client.fetch_trace_with_observations(inter["trace_id"])
        if trace is None:
            trace = load_trace_from_cache(inter.get("hydrated_trace_file", ""))
        if trace is None and client is not None:
            trace = client.fetch_trace_with_observations(inter["trace_id"])
        if trace is None:
            print(f"  ! missing trace {inter.get('trace_id')}", flush=True)
            continue
        turns.append(build_turn(trace, idx))

    if not turns:
        return None

    stem = manifest_path.stem
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = TRANSCRIPTS_DIR / f"{stem}.md"
    md_path.write_text(render_markdown(meta, turns))

    sidecar = build_tool_sidecar(meta["conversation_id"], turns)
    tools_path = TRANSCRIPTS_DIR / f"{stem}.tools.json"
    tools_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")

    tool_count = sum(1 for t in turns for e in t["events"] if e["kind"] == "tool")
    return {
        "conversation_id": meta["conversation_id"],
        "env": meta.get("env"),
        "module": meta.get("module"),
        "org_id": meta.get("org_id"),
        "project_id": meta.get("project_id"),
        "num_turns": len(turns),
        "num_tool_calls": tool_count,
        "total_cost_usd": round(sum(t["cost_usd"] for t in turns), 6),
        "first_timestamp": meta.get("first_timestamp"),
        "trace_ids": ";".join(str(t["trace_id"]) for t in turns),
        "transcript_file": md_path.name,
        "label": "",
        "notes": "",
    }


def write_labels_csv(rows: list[dict[str, Any]]) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TRANSCRIPTS_DIR / "labels.csv"
    fields = [
        "conversation_id",
        "env",
        "module",
        "org_id",
        "project_id",
        "num_turns",
        "num_tool_calls",
        "total_cost_usd",
        "first_timestamp",
        "trace_ids",
        "transcript_file",
        "label",
        "notes",
    ]
    rows_sorted = sorted(rows, key=lambda r: r["transcript_file"])
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_sorted)
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        action="append",
        type=Path,
        default=None,
        help="Session manifest directory (repeatable). Defaults to sessions/ + random-sample-15/sessions/",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Only build this conversation_id (session_id).",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch traces fresh from Langfuse instead of using the hydrated cache.",
    )
    args = parser.parse_args()

    session_dirs = args.session_dir or DEFAULT_SESSION_DIRS

    client: LangfuseClient | None = None
    if args.fetch:
        host, public, secret = load_langfuse_config()
        client = LangfuseClient(host, public, secret)

    manifests = iter_manifests([Path(d) for d in session_dirs])
    rows: list[dict[str, Any]] = []
    for mpath in manifests:
        try:
            manifest = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            continue
        if args.conversation_id and manifest.get("conversation_id") != args.conversation_id:
            continue
        print(f"Building {mpath.name} …", flush=True)
        row = process_manifest(mpath, client=client, fetch=args.fetch)
        if row:
            rows.append(row)

    if not rows:
        print("No conversations built. Check --conversation-id / --session-dir.", flush=True)
        return 1

    csv_path = write_labels_csv(rows)
    print(f"\nBuilt {len(rows)} transcript(s) in {TRANSCRIPTS_DIR}")
    print(f"Labeling CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

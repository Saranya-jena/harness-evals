#!/usr/bin/env python3
"""Build one review file per sampled conversation (full Langfuse session).

Reads ``pilot-sample-30.json``, fetches *all* traces for each ``conversation_id``
(Langfuse ``session_id``), hydrates missing traces to ``hydrated/traces/``, and
writes a readable combined session file under ``sessions/``.

Environment (or pass via CLI):
  LANGFUSE_HOST          default: https://langfuse-prod.harness.io
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY

Usage:
  python scripts/build_conversation_sessions.py
  python scripts/build_conversation_sessions.py --limit 3
  python scripts/build_conversation_sessions.py --skip-hydrate
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "pilot-sample-30.json"
HYDRATED_DIR = ROOT / "hydrated" / "traces"
SESSIONS_DIR = ROOT / "sessions"
INDEX_PATH = SESSIONS_DIR / "index.json"


def configure_dataset_paths(
    *,
    sample_file: Path | None = None,
    sessions_dir: Path | None = None,
    hydrated_dir: Path | None = None,
) -> None:
    """Override global output paths (used by main and external callers)."""
    global PILOT_PATH, HYDRATED_DIR, SESSIONS_DIR, INDEX_PATH
    if sample_file is not None:
        PILOT_PATH = sample_file
    if sessions_dir is not None:
        SESSIONS_DIR = sessions_dir
        INDEX_PATH = sessions_dir / "index.json"
    if hydrated_dir is not None:
        HYDRATED_DIR = hydrated_dir

HITL_PREFIXES = (
    "The user approved the entity",
    "The user approved use of",
    "The user declined",
    "The user denied",
    "The user cancelled",
    "The user clicked",
)

PREVIEW_CHARS = 2000


def load_langfuse_config() -> tuple[str, str, str]:
    host = os.environ.get("LANGFUSE_HOST", "https://langfuse-prod.harness.io").rstrip("/")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if public and secret:
        return host, public, secret

    mcp_json = ROOT.parents[2] / "ai-evals" / ".mcp.json"
    if mcp_json.is_file():
        cfg = json.loads(mcp_json.read_text())
        env = cfg.get("mcpServers", {}).get("langfuse", {}).get("env", {})
        return (
            env.get("LANGFUSE_HOST", host).rstrip("/"),
            env.get("LANGFUSE_PUBLIC_KEY", ""),
            env.get("LANGFUSE_SECRET_KEY", ""),
        )
    raise SystemExit(
        "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or ensure ai-evals/.mcp.json exists."
    )


class LangfuseClient:
    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self.host = host.rstrip("/")
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.host + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    def list_session_traces(self, session_id: str) -> list[dict[str, Any]]:
        """List all traces in a Langfuse session (= Harness conversation_id)."""
        payload = self._get(f"/api/public/sessions/{session_id}")
        traces = payload.get("traces") or []
        traces.sort(key=lambda t: t.get("timestamp") or "")
        return traces

    def fetch_trace_with_observations(self, trace_id: str) -> dict[str, Any]:
        """Fetch trace; observations are embedded in GET /traces/{id} on this host."""
        trace = self._get(f"/api/public/traces/{trace_id}")
        observations = list(trace.get("observations") or [])
        if not observations:
            observations = self._fetch_observations_paginated(trace_id)
        observations.sort(
            key=lambda o: o.get("startTime") or o.get("start_time") or ""
        )
        trace["observations"] = observations
        return normalize_trace(trace)

    def _fetch_observations_paginated(self, trace_id: str) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        page = 1
        while True:
            last_err: urllib.error.HTTPError | None = None
            payload: dict[str, Any] | None = None
            for attempt in range(3):
                try:
                    payload = self._get(
                        "/api/public/observations",
                        {"traceId": trace_id, "page": page, "limit": 100},
                    )
                    break
                except urllib.error.HTTPError as err:
                    last_err = err
                    if err.code not in (429, 422, 503) or attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            if payload is None:
                raise last_err  # type: ignore[misc]
            observations.extend(payload.get("data") or [])
            meta = payload.get("meta") or {}
            total_pages = meta.get("totalPages") or 1
            if page >= total_pages:
                break
            page += 1
        return observations


def normalize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Normalize Langfuse REST (camelCase) and MCP (snake_case) trace shapes."""
    if trace.get("session_id") or not trace.get("sessionId"):
        return trace
    meta = trace.get("metadata") or {}
    attrs = meta.get("attributes") or {}
    return {
        "id": trace.get("id"),
        "timestamp": trace.get("timestamp"),
        "name": trace.get("name"),
        "input": trace.get("input"),
        "output": trace.get("output"),
        "session_id": trace.get("sessionId"),
        "user_id": trace.get("userId"),
        "metadata": meta,
        "tags": trace.get("tags") or [],
        "environment": trace.get("environment"),
        "html_path": trace.get("htmlPath") or trace.get("html_path"),
        "latency": trace.get("latency"),
        "total_cost": trace.get("totalCost") or trace.get("total_cost"),
        "observations": trace.get("observations") or [],
        "is_hitl_resume": attrs.get("agent.is_hitl_resume") == "true",
    }


def is_hitl_message(text: str) -> bool:
    stripped = (text or "").strip()
    return any(stripped.startswith(prefix) for prefix in HITL_PREFIXES)


def strip_ui_context_suffix(text: str) -> str:
    """Remove Harness UI context JSON appended to user_message / llm_turn copies."""
    text = (text or "").strip()
    marker = "\n\n{"
    if marker not in text:
        return text
    head, tail = text.split(marker, 1)
    if "currentUrl" in tail or "current_url" in tail:
        return head.strip()
    return text


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"


def metadata_attrs(trace: dict[str, Any]) -> dict[str, Any]:
    meta = trace.get("metadata") or {}
    attrs = meta.get("attributes") or {}
    if isinstance(attrs, dict):
        return attrs
    return {}


def interaction_id_from_trace(trace: dict[str, Any]) -> str | None:
    meta = trace.get("metadata") or {}
    for key in ("harness.interaction.id", "interaction_id"):
        val = meta.get(key)
        if val:
            return str(val)
    attrs = metadata_attrs(trace)
    for key in ("harness.interaction.id", "harness.interaction.id"):
        val = attrs.get(key)
        if val:
            return str(val)
    inp = trace.get("input") or {}
    if inp.get("interaction_id"):
        return str(inp["interaction_id"])
    return None


def is_hitl_trace(trace: dict[str, Any]) -> bool:
    if trace.get("is_hitl_resume"):
        return True
    attrs = metadata_attrs(trace)
    if attrs.get("agent.is_hitl_resume") == "true":
        return True
    inp = trace.get("input") or {}
    msg = inp.get("user_message") or ""
    return is_hitl_message(msg)


def extract_user_messages_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered user messages for one trace/interaction."""
    messages: list[dict[str, Any]] = []
    ts = trace.get("timestamp") or ""

    inp = trace.get("input") or {}
    for key in ("prompt", "user_message"):
        text = (inp.get(key) or "").strip()
        if not text:
            continue
        messages.append(
            {
                "timestamp": ts,
                "source": f"trace.input.{key}",
                "text": text,
                "is_hitl": is_hitl_message(text),
            }
        )

    for obs in trace.get("observations") or []:
        ots = obs.get("startTime") or obs.get("start_time") or ts
        name = obs.get("name") or ""
        oinp = obs.get("input")

        if name == "chat_unified_agent" and isinstance(oinp, dict):
            text = (oinp.get("prompt") or "").strip()
            if text:
                messages.append(
                    {
                        "timestamp": ots,
                        "source": "chat_unified_agent.prompt",
                        "text": text,
                        "is_hitl": is_hitl_message(text),
                    }
                )

        if name.startswith("llm_turn_") and isinstance(oinp, dict):
            for msg in oinp.get("messages") or []:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                    if text.startswith("<system-reminder>"):
                        continue
                    messages.append(
                        {
                            "timestamp": ots,
                            "source": name,
                            "text": text,
                            "is_hitl": is_hitl_message(text),
                        }
                    )

    return messages


def pick_primary_user_message(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Pick one canonical user message per trace (not one per observability layer)."""
    messages = extract_user_messages_from_trace(trace)
    real = [m for m in messages if not m["is_hitl"]]
    pool = real or [m for m in messages if m["is_hitl"]]
    if not pool:
        return None

    source_priority = (
        "trace.input.prompt",
        "chat_unified_agent.prompt",
        "trace.input.user_message",
    )
    for source in source_priority:
        for msg in pool:
            if msg["source"] == source:
                chosen = dict(msg)
                chosen["text"] = strip_ui_context_suffix(msg["text"])
                return chosen

    chosen = dict(pool[0])
    chosen["text"] = strip_ui_context_suffix(chosen["text"])
    return chosen


def extract_tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for obs in trace.get("observations") or []:
        name = obs.get("name") or ""
        if name.startswith("harness_") or name.startswith("mcp__harness"):
            tools.append({"name": name, "type": obs.get("type")})
        if name == "mcp.tools/call" and isinstance(obs.get("input"), dict):
            tool_name = obs["input"].get("name") or obs["input"].get("toolName")
            if tool_name:
                tools.append({"name": str(tool_name), "type": "mcp.tools/call"})
    # dedupe preserving order
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in tools:
        key = item["name"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def extract_assistant_preview(trace: dict[str, Any]) -> str:
    out = trace.get("output") or {}
    if isinstance(out, dict):
        text = out.get("text") or out.get("content") or ""
        if text.strip():
            return preview(text)

    for obs in reversed(trace.get("observations") or []):
        if obs.get("name", "").startswith("llm_turn_") and isinstance(obs.get("output"), dict):
            content = obs["output"].get("content")
            if isinstance(content, list):
                chunks: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chunks.append(block.get("text") or "")
                joined = "\n".join(c for c in chunks if c.strip()).strip()
                if joined:
                    return preview(joined)
            if isinstance(content, str) and content.strip():
                return preview(content)
    return ""


def session_filename(index: int, conv: dict[str, Any]) -> str:
    env = conv.get("env") or "unknown"
    module = conv.get("module") or "none"
    short = conv["conversation_id"].split("-")[0]
    return f"{index:02d}-{env}-{module}-{short}.json"


def build_interaction(trace: dict[str, Any], sequence: int) -> dict[str, Any]:
    primary = pick_primary_user_message(trace)
    primary_user_message = primary["text"] if primary else ""
    message_kind = "unknown"
    if primary:
        message_kind = "hitl" if primary["is_hitl"] else "user"
    all_messages = extract_user_messages_from_trace(trace)
    real_messages = [m for m in all_messages if not m["is_hitl"]]
    hitl_messages = [m for m in all_messages if m["is_hitl"]]

    return {
        "sequence": sequence,
        "trace_id": trace.get("id"),
        "interaction_id": interaction_id_from_trace(trace),
        "timestamp": trace.get("timestamp"),
        "is_hitl_resume": is_hitl_trace(trace),
        "message_kind": message_kind,
        "user_message": preview(primary_user_message),
        "user_message_full_length": len(primary_user_message),
        "real_user_message_count": len(real_messages),
        "hitl_message_count": len(hitl_messages),
        "tool_calls": extract_tool_calls(trace),
        "assistant_preview": extract_assistant_preview(trace),
        "cost_usd": trace.get("totalCost") or trace.get("total_cost"),
        "latency_sec": trace.get("latency"),
        "hydrated_trace_file": f"hydrated/traces/{trace.get('id')}.json",
        "langfuse_url_path": trace.get("htmlPath") or trace.get("html_path"),
    }


def build_session(
    conv: dict[str, Any],
    traces: list[dict[str, Any]],
    *,
    index: int,
    sampled_trace_ids: set[str],
) -> dict[str, Any]:
    interactions = [build_interaction(trace, i + 1) for i, trace in enumerate(traces)]

    timeline: list[dict[str, Any]] = []
    for trace in traces:
        msg = pick_primary_user_message(trace)
        if not msg:
            continue
        timeline.append(
            {
                "timestamp": trace.get("timestamp") or msg["timestamp"],
                "trace_id": trace.get("id"),
                "source": msg["source"],
                "is_hitl": msg["is_hitl"],
                "text": preview(msg["text"], 500),
            }
        )

    real_timeline = [t for t in timeline if not t["is_hitl"]]
    original_prompt = real_timeline[0]["text"] if real_timeline else None
    original_trace_id = real_timeline[0]["trace_id"] if real_timeline else None

    timestamps = [t.get("timestamp") for t in traces if t.get("timestamp")]
    return {
        "session_file": session_filename(index, conv),
        "conversation_id": conv["conversation_id"],
        "pilot_metadata": {
            "env": conv.get("env"),
            "module": conv.get("module"),
            "account_id": conv.get("account_id"),
            "org_id": conv.get("org_id"),
            "project_id": conv.get("project_id"),
            "sampled_trace_ids": sorted(sampled_trace_ids),
        },
        "session_summary": {
            "interaction_count": len(traces),
            "sampled_trace_count": len(sampled_trace_ids),
            "additional_traces_fetched": max(0, len(traces) - len(sampled_trace_ids)),
            "first_timestamp": min(timestamps) if timestamps else None,
            "last_timestamp": max(timestamps) if timestamps else None,
            "total_cost_usd": sum((t.get("totalCost") or t.get("total_cost") or 0) for t in traces),
            "original_user_prompt": original_prompt,
            "original_user_prompt_trace_id": original_trace_id,
            "original_prompt_found": original_prompt is not None,
        },
        "user_message_timeline": timeline,
        "interactions": interactions,
    }


def hydrate_trace(client: LangfuseClient, trace_id: str, *, force: bool = False) -> Path:
    HYDRATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HYDRATED_DIR / f"{trace_id}.json"
    if out_path.exists() and not force:
        return out_path
    trace = client.fetch_trace_with_observations(trace_id)
    out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root (e.g. random-sample-15/). Sets sample + sessions paths.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=None,
        help="Sample manifest JSON (default: pilot-sample-30.json or dataset-dir/sample-15.json)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="Output sessions directory (default: sessions/ or dataset-dir/sessions/)",
    )
    parser.add_argument(
        "--hydrated-dir",
        type=Path,
        default=None,
        help="Hydrated traces cache (default: hydrated/traces/, shared across datasets)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N conversations")
    parser.add_argument("--skip-hydrate", action="store_true", help="Use existing hydrated traces only")
    parser.add_argument("--force-hydrate", action="store_true", help="Re-fetch traces even if cached")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir is not None:
        dataset_dir = dataset_dir if dataset_dir.is_absolute() else ROOT / dataset_dir
    sample_file = args.sample_file
    if sample_file is None and dataset_dir is not None:
        for candidate in ("sample-15.json", "sample.json", "pilot-sample.json"):
            if (dataset_dir / candidate).is_file():
                sample_file = dataset_dir / candidate
                break
    if sample_file is None:
        sample_file = PILOT_PATH
    elif not sample_file.is_absolute():
        sample_file = ROOT / sample_file

    sessions_dir = args.sessions_dir
    if sessions_dir is None and dataset_dir is not None:
        sessions_dir = dataset_dir / "sessions"
    elif sessions_dir is not None and not sessions_dir.is_absolute():
        sessions_dir = ROOT / sessions_dir

    hydrated_dir = args.hydrated_dir
    if hydrated_dir is not None and not hydrated_dir.is_absolute():
        hydrated_dir = ROOT / hydrated_dir

    configure_dataset_paths(
        sample_file=sample_file,
        sessions_dir=sessions_dir,
        hydrated_dir=hydrated_dir,
    )

    host, public, secret = load_langfuse_config()
    client = LangfuseClient(host, public, secret)

    pilot = json.loads(PILOT_PATH.read_text())
    conversations = pilot["conversations"]
    if args.limit:
        conversations = conversations[: args.limit]

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    stats = {
        "conversations": 0,
        "total_traces": 0,
        "newly_hydrated_traces": 0,
        "missing_original_prompt": 0,
    }

    for i, conv in enumerate(conversations, start=1):
        cid = conv["conversation_id"]
        sampled_ids = set(conv.get("trace_ids") or [])
        print(f"[{i}/{len(conversations)}] session {cid} …", flush=True)

        trace_summaries = [normalize_trace(t) for t in client.list_session_traces(cid)]
        trace_ids = [t["id"] for t in trace_summaries]
        stats["total_traces"] += len(trace_ids)

        summary_by_id = {t["id"]: t for t in trace_summaries}
        traces: list[dict[str, Any]] = []
        for j, trace_id in enumerate(trace_ids, start=1):
            cached = HYDRATED_DIR / f"{trace_id}.json"
            if cached.exists() and not args.force_hydrate:
                trace = normalize_trace(json.loads(cached.read_text()))
            elif args.skip_hydrate:
                trace = summary_by_id[trace_id]
            else:
                existed = cached.exists()
                hydrate_trace(client, trace_id, force=args.force_hydrate)
                if not existed:
                    stats["newly_hydrated_traces"] += 1
                trace = normalize_trace(json.loads(cached.read_text()))
                time.sleep(0.05)
                if j % 5 == 0:
                    print(f"  hydrated {j}/{len(trace_ids)} traces", flush=True)
            traces.append(trace)

        traces.sort(key=lambda t: t.get("timestamp") or "")
        session = build_session(conv, traces, index=i, sampled_trace_ids=sampled_ids)
        if not session["session_summary"]["original_prompt_found"]:
            stats["missing_original_prompt"] += 1

        out_path = SESSIONS_DIR / session["session_file"]
        out_path.write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n")

        index_rows.append(
            {
                "index": i,
                "file": session["session_file"],
                "conversation_id": cid,
                "env": conv.get("env"),
                "module": conv.get("module"),
                "interaction_count": session["session_summary"]["interaction_count"],
                "original_prompt_found": session["session_summary"]["original_prompt_found"],
                "original_user_prompt_preview": session["session_summary"]["original_user_prompt"],
            }
        )
        stats["conversations"] += 1
        print(
            f"  -> {len(traces)} interactions, original_prompt={'yes' if session['session_summary']['original_prompt_found'] else 'NO'}",
            flush=True,
        )

    index_doc = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_pilot": str(PILOT_PATH.relative_to(ROOT)),
        "sessions_dir": str(SESSIONS_DIR.relative_to(ROOT)) + "/",
        "hydrated_dir": str(HYDRATED_DIR.relative_to(ROOT)) + "/",
        "stats": stats,
        "sessions": index_rows,
    }
    INDEX_PATH.write_text(json.dumps(index_doc, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {stats['conversations']} session files to {SESSIONS_DIR}")
    print(f"Index: {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

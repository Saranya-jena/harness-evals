#!/usr/bin/env python3
"""Sample unified-agent conversations from Langfuse for manual review datasets.

Fetches recent prod traces, groups by ``session_id`` (= Harness ``conversation_id``),
and writes a sample manifest JSON (same shape as ``pilot-sample-30.json``).

Usage:
  python scripts/sample_conversations.py --count 15 --output-dir random-sample-15
  python scripts/sample_conversations.py --count 15 --strategy random --seed 99 \\
      --exclude pilot-sample-30.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_conversation_sessions import LangfuseClient, load_langfuse_config  # noqa: E402

PROD_ENVS = ("prod0", "prod1", "prod2", "prod3", "eu1", "prod")
TRACE_NAME = "unified_agent_chat"
DEFAULT_WINDOW_DAYS = 30


def get_env(trace: dict[str, Any]) -> str:
    env = (trace.get("environment") or trace.get("environment") or "").strip()
    if env in PROD_ENVS:
        return env
    for tag in trace.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("environment:"):
            val = tag.split(":", 1)[1]
            if val in PROD_ENVS:
                return val
    return env or "unknown"


def get_module(trace: dict[str, Any]) -> str:
    md = trace.get("metadata") or {}
    if isinstance(md, dict):
        for key in ("product", "agent.module", "harness.module"):
            val = md.get(key)
            if val:
                return str(val).lower()
    for tag in trace.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("product:"):
            return tag.split(":", 1)[1].lower()
    return "none"


def scope_fields(trace: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    md = trace.get("metadata") or {}
    if not isinstance(md, dict):
        return None, None, None
    account = md.get("account_id") or md.get("harness.account.id")
    org = md.get("org_id") or md.get("harness.org.id")
    project = md.get("project_id") or md.get("harness.project.id")
    return (
        str(account) if account else None,
        str(org) if org else None,
        str(project) if project else None,
    )


def fetch_traces_for_env(
    client: LangfuseClient,
    env: str,
    *,
    since: str,
    max_pages: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    traces: list[dict[str, Any]] = []
    pages_fetched = 0
    for page in range(1, max_pages + 1):
        payload = client._get(
            "/api/public/traces",
            {
                "environment": env,
                "name": TRACE_NAME,
                "fromTimestamp": since,
                "page": page,
                "limit": page_size,
            },
        )
        batch = payload.get("data") or []
        if not batch:
            break
        traces.extend(batch)
        pages_fetched = page
        meta = payload.get("meta") or {}
        if page >= (meta.get("totalPages") or page):
            break
        time.sleep(0.05)
    return traces, pages_fetched


def group_conversations(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        session_id = trace.get("sessionId") or trace.get("session_id")
        if not session_id:
            continue
        grouped[str(session_id)].append(trace)
    for session_id in grouped:
        grouped[session_id].sort(key=lambda t: t.get("timestamp") or "")
    return grouped


def build_conversation_record(session_id: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    first = traces[0]
    last = traces[-1]
    account_id, org_id, project_id = scope_fields(first)
    timestamps = [t.get("timestamp") for t in traces if t.get("timestamp")]
    return {
        "conversation_id": session_id,
        "env": get_env(first),
        "module": get_module(first),
        "trace_count": len(traces),
        "interaction_count": len(traces),
        "trace_ids": [t["id"] for t in traces],
        "account_id": account_id,
        "org_id": org_id,
        "project_id": project_id,
        "first_timestamp": min(timestamps) if timestamps else first.get("timestamp"),
        "last_timestamp": max(timestamps) if timestamps else last.get("timestamp"),
        "total_cost_usd": sum((t.get("totalCost") or t.get("total_cost") or 0) for t in traces),
        "total_latency_sec": sum((t.get("latency") or 0) for t in traces),
        "observation_count": sum(len(t.get("observations") or []) for t in traces),
    }


def load_exclude_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        for conv in doc.get("conversations") or []:
            cid = conv.get("conversation_id")
            if cid:
                excluded.add(str(cid))
    return excluded


def sample_random(
    conversations: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    exclude: set[str],
) -> list[dict[str, Any]]:
    pool = [c for c in conversations if c["conversation_id"] not in exclude]
    if len(pool) < count:
        raise SystemExit(
            f"Only {len(pool)} conversations available after exclusions; need {count}."
        )
    rng = random.Random(seed)
    picked = rng.sample(pool, count)
    picked.sort(key=lambda c: (c.get("env") or "", c.get("module") or "", c["conversation_id"]))
    return picked


def bucket_counts(conversations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for conv in conversations:
        key = f"{conv.get('env')}|{conv.get('module')}"
        counts[key] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=15, help="Number of conversations to sample")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "random-sample-15",
        help="Directory for sample manifest (default: random-sample-15/)",
    )
    parser.add_argument(
        "--sample-file",
        default="sample-15.json",
        help="Manifest filename inside output-dir",
    )
    parser.add_argument("--strategy", choices=("random",), default="random")
    parser.add_argument("--seed", type=int, default=99, help="Random seed")
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Look back window in days",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=12,
        help="Max API pages to fetch per prod environment",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Traces per API page (max 100)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["pilot-sample-30.json"],
        help="Sample manifest(s) whose conversation_ids to exclude (repeatable)",
    )
    args = parser.parse_args()

    host, public, secret = load_langfuse_config()
    client = LangfuseClient(host, public, secret)

    since = (
        datetime.now(timezone.utc) - timedelta(days=args.window_days)
    ).isoformat().replace("+00:00", "Z")

    exclude_paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.exclude]
    exclude_ids = load_exclude_ids(exclude_paths)
    print(f"Excluding {len(exclude_ids)} conversation_ids from prior samples", flush=True)

    all_traces: list[dict[str, Any]] = []
    pages_by_env: dict[str, int] = {}
    for env in ("prod0", "prod1", "prod2", "prod3"):
        print(f"Fetching {env} traces …", flush=True)
        traces, pages = fetch_traces_for_env(
            client,
            env,
            since=since,
            max_pages=args.max_pages,
            page_size=min(args.page_size, 100),
        )
        pages_by_env[env] = pages
        all_traces.extend(traces)
        print(f"  {len(traces)} traces ({pages} pages)", flush=True)

    grouped = group_conversations(all_traces)
    conversations = [build_conversation_record(sid, traces) for sid, traces in grouped.items()]
    print(
        f"Scanned {len(all_traces)} traces -> {len(conversations)} unique conversations",
        flush=True,
    )

    selected = sample_random(
        conversations,
        args.count,
        seed=args.seed,
        exclude=exclude_ids,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.sample_file

    doc = {
        "sample_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "scope": {
                "agent": "unified-agent",
                "environments": list(pages_by_env.keys()),
                "window_days": args.window_days,
                "trace_name": TRACE_NAME,
            },
            "sampling": {
                "strategy": args.strategy,
                "target_count": args.count,
                "random_seed": args.seed,
                "excluded_conversation_ids": len(exclude_ids),
            },
            "source": {
                "tool": "langfuse /api/public/traces",
                "pages_by_env": pages_by_env,
                "traces_scanned": len(all_traces),
                "unique_conversations_found": len(conversations),
                "bucket_counts_all": bucket_counts(conversations),
                "bucket_counts_selected": bucket_counts(selected),
            },
        },
        "conversations": selected,
    }
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(doc["sample_metadata"]["source"]["bucket_counts_selected"], indent=2))
    print(f"Wrote {len(selected)} conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

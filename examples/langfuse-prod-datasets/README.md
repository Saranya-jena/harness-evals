# Production Conversation Quality Dataset

This directory contains a reproducible pipeline for sampling Harness Agent v3
conversations from production Langfuse traces and categorizing them without
re-running the agent.

## Team workflow skill

Use the project skill `build-conversation-eval-dataset` when selecting
production conversations, writing portable overrides and elicitation hints,
tuning expected outcomes/SSE checks, or investigating low live-eval scores. Its
source is `.cursor/skills/build-conversation-eval-dataset/SKILL.md`.

## Artifacts

Each selected session has three files:

- `*.md` — human-readable transcript
- `*.tools.json` — complete tool requests and responses
- `*.conversation.json` — canonical cleaned conversation used by dataset export

Each `eval-datasets/*.jsonl` file contains multiple validated, pre-captured
`EvalCase` rows for one judge batch.

## 1. Fetch a fresh sample

```bash
python scripts/build_agent_transcripts.py \
  --random-count 15 \
  --module-count 30 \
  --use-cache
```

Future runs write Markdown, tool sidecars, and canonical conversation files in
the same pass.

## 2. Backfill existing samples without refetching

```bash
python scripts/build_agent_transcripts.py --backfill-conversations
```

This reads `random/labels.csv` and `module-coverage/labels.csv`, then rebuilds
canonical conversations from `cache/traces/*.json`. If a cached trace is
missing, the command fails and lists the missing traces. Fetch only those traces
with:

```bash
python scripts/build_agent_transcripts.py \
  --backfill-conversations \
  --fetch-missing
```

## 3. Build EvalCase batches

```bash
python scripts/build_eval_dataset.py --source module-coverage --limit 5
python scripts/build_eval_dataset.py --source module-coverage --limit 10
python scripts/build_eval_dataset.py --source module-coverage --limit 30
```

The exporter writes eligible EvalCases and a sibling `*.ineligible.jsonl` file
for structurally unusable conversations. It validates every eligible row with
`EvalCase.from_dict()`.

## 4. Validate without an LLM call

```bash
python scripts/run_conversation_quality_eval.py \
  --input eval-datasets/module-coverage-005.jsonl \
  --validate-only
```

## 5. Run the initial judge calibration

```bash
export OPENAI_API_KEY=...
python scripts/run_conversation_quality_eval.py \
  --input eval-datasets/module-coverage-005.jsonl \
  --provider openai \
  --model gpt-4o
```

The runner writes:

- `results.jsonl` — detailed category, component scores, confidence, reasoning,
  and cited evidence
- `review.csv` — blank human category/notes columns for calibration
- `summary.json` — category/module distribution and run configuration

The judge separates dataset usefulness (`useful | useless`) from agent quality
(`good | bad | unclear | not_applicable`). Human notes are not included in the
judge prompt.

## 6. Build live conversation goldens

Turn the categorized conversations into environment-portable
`ConversationGolden` rows that the live Harness SSE conversation runner can
replay against any Harness project.

```bash
python scripts/build_conversation_goldens.py \
  --review results/module-coverage-030/review.csv \
  --conversations module-coverage
```

Outputs (under the repo `examples/` directory):

- `prod-conversation.goldens.jsonl` — validated `ConversationGolden` rows
- `prod-conversation.goldens.manifest.jsonl` — one record per source
  conversation with the decision (`emitted` / `excluded`), the portability
  `action`, and the `reason`

### What the converter does

- **Filters** on the judge's `final_category`: keeps `good` / `bad` / `unclear`,
  drops `useless`. Pipeline error-analysis conversations are dropped (they need a
  specific failed execution that will not exist in the eval environment).
- **Ignores review-gate injections.** Synthetic platform messages ("The user
  approved the entity ...", "The user provided the following values ...") are
  elicitation continuations, not real user turns, so they never become scripted
  turns and never inflate `max_turns`.
- **Makes rows portable.** Production org/project identifiers are replaced with
  `${HARNESS_ORG}` / `${HARNESS_PROJECT}` placeholders. Read-only or write/mutation
  rows that depend on a production-specific named resource **fail closed** — they
  are excluded unless a curated entry exists in `conversation-golden-overrides.json`.
- **Scans for secrets/PII.** Emails, Harness/Bearer tokens, `api_key=` pairs, and
  non-documentation URLs in any emitted field cause the row to be excluded.
- **Validates** every emitted row with `ConversationGolden.from_dict()`.

There is no fixed yield. Inspect the manifest to see the included/excluded counts
by reason; the run prints the same breakdown.

### Curated overrides

`conversation-golden-overrides.json` holds explicit, reviewable portability
rewrites keyed by full `conversation_id`. Each entry either sets
`"exclude": true` with a `reason`, or supplies a portable `scenario`,
`expected_outcome`, and `turns`/`initial_prompt`. Never put concrete identifiers
in `elicitation_hints`.

### 7. Run the goldens against a live agent

Use a **disposable eval project** — write rows create real entities.

```bash
export SSE_ENDPOINT_URL=http://localhost:8000/stream
export HARNESS_ACCOUNT=...
export HARNESS_ORG=...
export HARNESS_PROJECT=<disposable-eval-project>
export TOKEN=...
export OPENAI_API_KEY=...
export EVAL_RUN_SUFFIX="$(date +%s)"   # unique-ify created entity names
PYTHONPATH=. poetry run harness-evals run examples/prod-conversation.eval.yaml
```

`examples/prod-conversation.eval.yaml` omits `conversation.mode` so each golden's
own mode (`scripted`) is respected, grades outcomes with the
`outcome_goal_accuracy` plugin metric (which reads the golden's curated
`expected_outcome`), and requires every per-row `sse_checks` assertion to pass
(`sse_events_match` threshold `1.0`).

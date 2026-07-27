---
name: build-conversation-eval-dataset
description: Converts real Harness agent conversations into portable, outcome-aware live conversation eval goldens. Use when selecting production conversations, curating ConversationGolden rows, writing overrides, adding elicitation hints and SSE checks, tuning expected outcomes or judge prompts, or diagnosing unexpectedly low conversation-eval scores.
---

# Build Conversation Eval Dataset

## Instructions

Create evals that measure whether the agent achieved the user's goal, without
overfitting to one production account, one elicitation wording, or one valid
workflow path.

### 1. Inspect the complete source conversation

Read all three artifacts when available:

- `*.conversation.json`: canonical user/assistant history
- `*.tools.json`: exact tool requests and results
- `*.md`: human-readable transcript

Identify:

- User goal and whether it spans multiple turns
- Required outcome versus incidental implementation details
- Tool operations and resource types needed for success
- Elicitations, options offered, and user answers
- Production-specific names, IDs, URLs, secrets, and dependencies
- Whether the source agent truly succeeded, partially succeeded, or failed

Do not infer success from the final assistant message alone. Verify tool results.
A claimed success after a failed tool call is a failure.

### 2. Decide whether the conversation is eval-worthy

Include rows categorized `good`, `bad`, or `unclear` when they test reusable
agent behavior. Exclude:

- `useless` conversations
- Error analysis requiring a production-only failed execution
- Flows that cannot be made self-contained and environment-portable
- Rows containing secrets, PII, or inaccessible production dependencies
- Duplicates that exercise the same goal, path, and failure mode

Keep distinct rows when they intentionally test different valid paths, such as
AWS-account buckets versus environment buckets.

### 3. Make the scenario portable

- Replace scope with `${HARNESS_ORG}` and `${HARNESS_PROJECT}`.
- Replace generated entity names with `${EVAL_RUN_SUFFIX}`.
- Rewrite named-resource dependencies into self-contained setup when possible.
- Require a curated override for production-specific read or write flows.
- Use a disposable eval project for mutation rows.
- Never place concrete production identifiers in elicitation hints.

The override file is the source of truth. Update
`examples/langfuse-prod-datasets/conversation-golden-overrides.json`; keep
`examples/prod-conversation.goldens.jsonl` synchronized through the converter.

### 4. Write a behavioral `expected_outcome`

State observable success:

1. What resource/action/result is required
2. Which essential fields or invariants must be present
3. Which tool operation/resource type should be used
4. What counts as an acceptable blocking error
5. Which alternate paths are valid

Do not write `"The assistant completes the request: ..."` or require one
incidental route. Judge core goal completion, not exact wording or tool count.

For multi-turn rows, label each turn's expected result. For read-only discovery,
an honest empty result or unavailable-module explanation with evidence can be
successful. For writes, an attempted tool call is not automatically success:
inspect the result and require confirmation or an honest blocking error.

### 5. Configure elicitation

Every golden must set:

```json
"elicitation_hints": {"llm_on_miss": true}
```

Deterministic intent matchers stabilize known questions. The simulator LLM must
answer when wording or options differ.

For each matcher:

- Match the semantic question using several realistic `question_contains` forms.
- Set an answer that is valid among the offered options.
- Put specific matchers before broad matchers.
- Keep answers coherent across rounds.
- Add `yaml.default_action: accept` for YAML review gates when appropriate.

Do not try to enumerate every wording. Add matchers for stable, important
branches; rely on `llm_on_miss` for the long tail.

### 6. Add trajectory checks

Use per-row `sse_checks` for required observable behavior:

- Read: expected read/search tool request and result
- Write: write tool request and result, including `resource_type` when important
- Multi-turn: checks for each essential operation
- Final response: `assistant_message` when user-visible closure is required

Keep `sse_events_match` at threshold `1.0`. It verifies trajectory, while
`outcome_goal_accuracy` verifies correctness. A passing SSE score does not prove
the API operation succeeded.

### 7. Review judge behavior

The outcome judge must:

- Read the full chronological `EvalCase.messages`, including tool calls/results
- Score against curated `expected_outcome`
- Accept coherent alternate workflows
- Treat elicitation and pending questions as visible progress
- Penalize absurd simulated answers, abandoned goals, unresolved tool errors,
  and false success claims

If a score seems wrong, compare the judge reason against actual tool results and
the expected outcome. Fix a weak golden or judge instruction; do not inflate the
score merely because the source row was labeled `good`.

### 8. Generate and validate

From `examples/langfuse-prod-datasets`:

```bash
python scripts/build_conversation_goldens.py \
  --review results/<batch>/review.csv \
  --conversations module-coverage
```

Inspect the manifest and generated JSONL. Confirm every row:

- Parses as `ConversationGolden`
- Has `elicitation_hints.llm_on_miss: true`
- Contains no concrete production scope or secret
- Has a specific behavioral expected outcome
- Uses suitable SSE checks
- Is distinct from nearby rows

Run focused converter/adapter tests, then the live eval:

```bash
PYTHONPATH=. poetry run harness-evals run examples/prod-conversation.eval.yaml
```

Review each score separately. Diagnose `outcome_goal_accuracy` from its reason
and `sse_events_match` from failed checks.

## Examples

Bad expected outcome:

```text
The assistant completes the request: create a feature flag.
```

Good expected outcome:

```text
The assistant creates a feature flag with treatments on, off, and current via
harness_create, after collecting required identifiers or using sensible
defaults, then confirms creation or clearly reports a blocking API error.
```

See [REFERENCE.md](REFERENCE.md) for complete override templates, matcher
patterns, and the review rubric.

## Performance Notes

- Curate the override, then regenerate; do not hand-maintain divergent copies.
- Use deterministic hints only for stable branch choices.
- Prefer a small set of behaviorally distinct rows over duplicated traces.
- Run focused tests before a paid live-agent/judge run.

## Troubleshooting

- **Unresolved elicitation:** verify `llm_on_miss: true`, simulator LLM config,
  matcher order, and that deterministic answers match offered options.
- **Low outcome score with successful tools:** strengthen the behavioral
  expected outcome and check whether chronological tool results reach the judge.
- **High outcome score after failed write:** require successful tool result or
  honest blocking-error reporting; reject false success claims.
- **SSE passes but outcome fails:** trajectory occurred, but goal completion did
  not. Inspect tool-result payloads and final response.
- **Regeneration removes edits:** update the override/source builder rather than
  only the generated JSONL.

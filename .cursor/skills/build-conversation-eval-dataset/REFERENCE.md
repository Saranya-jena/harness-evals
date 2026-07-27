# Conversation Eval Curation Reference

## Golden quality rubric

Approve a row only when all answers are yes:

### Goal

- Is the user goal clear and reusable outside the source account?
- Does `expected_outcome` define observable success?
- Are required fields/invariants separated from optional path choices?
- Does a multi-turn row specify the expected result of each turn?

### Correctness

- Were tool results inspected rather than trusting the final claim?
- Is an API/schema failure treated as a failure unless honestly reported as a
  blocking environment error?
- Are empty read results allowed when accurate and explained?
- Are alternate valid workflows accepted?

### Portability and safety

- Are org/project/name values parameterized?
- Is the flow self-contained in a disposable project?
- Are production IDs, URLs, emails, tokens, and secrets absent?
- Does every production-specific dependency have a curated rewrite or exclusion?

### Simulation

- Is `elicitation_hints.llm_on_miss` true?
- Are deterministic answers coherent and valid for likely options?
- Are specific matchers ordered before broad ones?
- Do distinct rows intentionally exercise distinct branches?

### Measurement

- Do SSE checks require the essential tool request and result?
- Do write checks include `resource_type` where useful?
- Does the expected outcome judge actual completion, not exact wording?
- Is the row behaviorally distinct from existing goldens?

## Override template

```json
{
  "<conversation-id>": {
    "scenario_type": "write",
    "scenario": "Create <portable resource> in ${HARNESS_PROJECT}",
    "expected_outcome": "The assistant creates ... via harness_create for resource_type ..., verifies the required fields ..., then confirms success or clearly reports a blocking API/environment error. Any valid ... path is acceptable.",
    "turns": [
      "First user turn",
      "Second user turn"
    ],
    "max_elicitation_rounds": 8,
    "user_persona": "Harness user who answers concisely using offered options",
    "context": [
      "Org: ${HARNESS_ORG}, Project: ${HARNESS_PROJECT}",
      "Use entity name eval_entity_${EVAL_RUN_SUFFIX}"
    ],
    "elicitation_hints": {
      "llm_on_miss": true,
      "intents": {
        "entity_name": "eval_entity_${EVAL_RUN_SUFFIX}",
        "branch_choice": "Valid option"
      },
      "matchers": [
        {
          "intent": "entity_name",
          "question_contains": [
            "name this entity",
            "what would you like to name"
          ]
        },
        {
          "intent": "branch_choice",
          "question_contains": [
            "which approach",
            "how should"
          ]
        }
      ],
      "yaml": {
        "default_action": "accept"
      }
    },
    "sse_checks": [
      {
        "event": "assistant_tool_request",
        "path": "$.v[*]",
        "match": [
          {"path": "$.name", "contains": "harness_create"},
          {"path": "$.arguments.resource_type", "equals": "<resource_type>"}
        ]
      },
      {
        "event": "assistant_tool_result",
        "path": "$.v[*]",
        "match": [
          {"path": "$.name", "contains": "harness_create"}
        ]
      },
      {
        "event": "assistant_message",
        "exists": true
      }
    ],
    "reason": "Why the production trace required this portable rewrite"
  }
}
```

Use `initial_prompt` instead of `turns` for a single-turn row.

## Matcher design

Prefer semantic fragments, not exact full questions:

```json
{
  "intent": "category_name",
  "question_contains": [
    "name this cost category",
    "what would you like to name",
    "name your cost category"
  ]
}
```

Avoid an overly broad matcher such as `"name"` because it can capture bucket,
pipeline, template, and category questions.

When two matchers overlap, order the narrower branch first:

1. `aws_grouping`: "group your AWS costs"
2. `bucket_count`: "how many cost buckets"
3. `bucket_type`: "what buckets do you want"

## Expected-outcome patterns

### Read/discovery

```text
The assistant lists <resources> accessible in the current scope using
harness_list. It reports the returned items accurately; an honest empty result
or unavailable-module explanation supported by tool errors is acceptable.
```

### Write

```text
The assistant creates <resource> with <required invariants> using harness_create
for resource_type <type>, then confirms the successful result or clearly reports
a blocking API/environment error. An attempted call followed by a failed result
does not count as creation.
```

### Validate

```text
The assistant calls <validation tool>, reports whether the input is valid, and
explains actionable schema/semantic errors. Echoing the input without validation
is a failure.
```

### Multi-turn mutation

```text
Turn 1: creates <resource A> with <invariants>.
Turn 2: retrieves/updates that same resource to add <change B>.
Both tool results must succeed, or the assistant must accurately report the
blocking failure instead of claiming success.
```

## Score triage

### `sse_events_match = 1`, outcome low

The expected tools ran, but completion may have failed. Inspect:

- Tool result error/schema validation
- Missing required field
- Final assistant false success claim
- Weak or overly strict expected outcome
- Missing chronological tool messages in `EvalCase.messages`

### Outcome high, `sse_events_match < 1`

The judge liked the narrative, but required observable behavior is absent.
Check tool naming, event paths, resource-type assertions, and whether the agent
answered without using the required API.

### Both low

Check scope/configuration first, then elicitation:

- `HARNESS_ORG`, `HARNESS_PROJECT`, disposable project
- `simulator_llm` configured
- `llm_on_miss: true`
- Option-valid deterministic hints
- Enough `max_elicitation_rounds`

## Key repository files

- Converter:
  `examples/langfuse-prod-datasets/scripts/build_conversation_goldens.py`
- Source overrides:
  `examples/langfuse-prod-datasets/conversation-golden-overrides.json`
- Generated goldens: `examples/prod-conversation.goldens.jsonl`
- Eval config: `examples/prod-conversation.eval.yaml`
- Elicitation adapter: `examples/harness_sse_elicitation_adapter.py`
- Outcome judge: `examples/outcome_goal_metric.py`
- SSE metric: `examples/sse_events_match_metric.py`
- Results: `examples/output/prod-conversation-results.jsonl`

"""LLM judge for usefulness and quality of captured Harness conversations."""

from __future__ import annotations

import json

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import LLMConversationMetric
from harness_evals.plugins import register_metric

PROMPT_VERSION = "conversation-quality-v1"

_SYSTEM_PROMPT = """You are a strict evaluator of production Harness AI agent conversations.
Use only evidence visible in the supplied conversation. Do not assume that missing explicit
user feedback means failure. Distinguish an agent failure from a platform capability that is
genuinely unavailable. Cite concise evidence for your decision."""

_PROMPT_TEMPLATE = """Evaluate this complete Harness AI conversation for dataset usefulness and agent quality.

Definitions:
- usefulness=useful: the conversation contains a meaningful user task and enough evidence to evaluate the agent.
- usefulness=useless: empty, corrupted, non-task, or insufficient evidence to evaluate.
- quality=good: the user's valid requests were substantially satisfied with a correct, actionable outcome.
- quality=bad: the agent materially failed, gave an unsupported/wrong result, abandoned the task, or used tools
  in a way that prevented satisfying the request.
- quality=unclear: the task is meaningful, but the visible evidence is insufficient to decide good versus bad.
- quality=not_applicable: use only when usefulness=useless.

Evaluation procedure:
1. Identify the user's original goal and every meaningful follow-up or correction.
2. Trace the assistant's responses and tool evidence in chronological order.
3. Decide whether the final state satisfies the latest valid user request.
4. Assess whether tool calls support the assistant's claims and whether failures were handled honestly.
5. Score goal achievement, resolution, and tool-use quality from 0.0 to 1.0.
6. Return confidence based on evidence quality, not on how strongly worded the answer is.

Complete conversation:
{conversation_text}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "usefulness",
        "quality",
        "goal_achievement",
        "resolution",
        "tool_use_quality",
        "confidence",
        "reasoning",
        "evidence",
    ],
    "properties": {
        "usefulness": {"type": "string", "enum": ["useful", "useless"]},
        "quality": {"type": "string", "enum": ["good", "bad", "unclear", "not_applicable"]},
        "goal_achievement": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "resolution": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "tool_use_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}


def _clamp(value: object) -> float:
    return max(0.0, min(1.0, float(value)))


def format_conversation(eval_case: EvalCase) -> str:
    """Render ordered messages and structured tool calls for the judge."""
    lines: list[str] = []
    for index, message in enumerate(eval_case.messages or [], start=1):
        role = message.role.upper()
        if message.content:
            lines.append(f"[{index}][{role}]\n{message.content}")
        for tool_call in message.tool_calls or []:
            lines.append(
                f"[{index}][ASSISTANT_TOOL_CALL:{tool_call.name}]\n"
                f"request={json.dumps(tool_call.input, ensure_ascii=False)}\n"
                f"response={tool_call.output or ''}"
            )
    return "\n\n".join(lines)


@register_metric("harness_conversation_quality")
class HarnessConversationQualityMetric(LLMConversationMetric):
    """Categorize a complete captured conversation without invoking the agent."""

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        *,
        max_conversation_chars: int = 120_000,
        **kwargs: object,
    ) -> None:
        super().__init__(
            llm=llm,
            threshold=threshold,
            name="harness_conversation_quality",
            dimension=Dimension.CORRECTNESS,
            **kwargs,
        )
        self.max_conversation_chars = max_conversation_chars

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if not eval_case.messages or len(eval_case.messages) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Conversation has fewer than two messages",
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "usefulness": "useless",
                    "quality": "not_applicable",
                    "final_category": "useless",
                    "confidence": 1.0,
                },
            )

        conversation_text = format_conversation(eval_case)
        if len(conversation_text) > self.max_conversation_chars:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=(
                    f"Conversation has {len(conversation_text)} characters, exceeding the "
                    f"{self.max_conversation_chars}-character judge limit"
                ),
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "usefulness": "useful",
                    "quality": "unclear",
                    "final_category": "unclear",
                    "confidence": 0.0,
                    "requires_chunked_evaluation": True,
                },
            )

        result = await self.llm.generate_json(
            _PROMPT_TEMPLATE.format(conversation_text=conversation_text),
            _RESPONSE_SCHEMA,
            system_prompt=_SYSTEM_PROMPT,
        )
        usefulness = str(result["usefulness"])
        quality = str(result["quality"])
        if usefulness == "useless":
            quality = "not_applicable"
            final_category = "useless"
        elif quality in {"good", "bad", "unclear"}:
            final_category = quality
        else:
            quality = "unclear"
            final_category = "unclear"

        goal_achievement = _clamp(result["goal_achievement"])
        resolution = _clamp(result["resolution"])
        tool_use_quality = _clamp(result["tool_use_quality"])
        confidence = _clamp(result["confidence"])
        value = (
            0.45 * goal_achievement
            + 0.35 * resolution
            + 0.20 * tool_use_quality
            if usefulness == "useful"
            else 0.0
        )
        reasoning = str(result["reasoning"])
        evidence = [str(item) for item in result.get("evidence") or []]

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={
                "prompt_version": PROMPT_VERSION,
                "usefulness": usefulness,
                "quality": quality,
                "final_category": final_category,
                "goal_achievement": goal_achievement,
                "resolution": resolution,
                "tool_use_quality": tool_use_quality,
                "confidence": confidence,
                "evidence": evidence,
            },
        )

"""Deterministic fake chat models for exercising deepagents' dynamic-subagent runtime.

`ScriptedParent` plays the coordinator: on its first turn it emits one `eval`
tool call containing a JavaScript orchestration script we supply; once it sees
the tool result it answers with that result as plain text.

`EchoWorker` plays every subagent: it sleeps for a fixed time (so we can see
whether dispatches overlap), records start/finish timestamps in a shared log,
and replies with a deterministic string derived from its prompt. Prompts
containing the token FAIL make it raise, so we can test error propagation.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

EVENTS: list[dict[str, Any]] = []  # shared timeline of worker activity


def reset_events() -> None:
    EVENTS.clear()


class _Base(BaseChatModel):
    def bind_tools(self, tools, **kwargs):  # tools are irrelevant to scripted fakes
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("use the async API")


class ScriptedParent(_Base):
    code: str
    tool_name: str = "eval"

    @property
    def _llm_type(self) -> str:
        return "scripted-parent"

    async def _agenerate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": {"code": self.code}, "id": "call_eval_1", "type": "tool_call"}],
            )
        else:
            msg = AIMessage(content=str(tool_msgs[-1].content))
        return ChatResult(generations=[ChatGeneration(message=msg)])


class EchoWorker(_Base):
    name_tag: str = "worker"
    delay: float = 1.0

    @property
    def _llm_type(self) -> str:
        return "echo-worker"

    async def _agenerate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        t0 = time.monotonic()
        EVENTS.append({"ev": "start", "who": self.name_tag, "prompt": prompt, "t": t0})
        await asyncio.sleep(self.delay)
        if "FAIL" in prompt:
            EVENTS.append({"ev": "error", "who": self.name_tag, "prompt": prompt, "t": time.monotonic()})
            raise RuntimeError(f"worker failed on: {prompt}")
        EVENTS.append({"ev": "end", "who": self.name_tag, "prompt": prompt, "t": time.monotonic()})
        reply = f"[{self.name_tag}] handled: {prompt}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


class ClassifierWorker(_Base):
    """Returns a deterministic JSON label based on keywords in the prompt."""

    delay: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "classifier-worker"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        EVENTS.append({"ev": "start", "who": "classifier", "prompt": prompt, "t": time.monotonic()})
        await asyncio.sleep(self.delay)
        low = prompt.lower()
        label = "bug" if ("crash" in low or "error" in low) else "feature" if "please add" in low else "question"
        EVENTS.append({"ev": "end", "who": "classifier", "prompt": prompt, "t": time.monotonic()})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=label))])


def summarize_timeline() -> dict[str, Any]:
    if not EVENTS:
        return {"dispatches": 0}
    t0 = min(e["t"] for e in EVENTS)
    starts = [e for e in EVENTS if e["ev"] == "start"]
    # max number of workers running at the same instant
    edges = sorted([(e["t"], 1) for e in starts] + [(e["t"], -1) for e in EVENTS if e["ev"] in ("end", "error")])
    cur = peak = 0
    for _, d in edges:
        cur += d
        peak = max(peak, cur)
    return {
        "dispatches": len(starts),
        "peak_concurrency": peak,
        "span_s": round(max(e["t"] for e in EVENTS) - t0, 2),
        "start_order": [e["prompt"][:40] for e in starts],
    }


def dumps(x: Any) -> str:
    return json.dumps(x, indent=2, default=str)

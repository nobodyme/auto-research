"""State/durability experiments for deepagents dynamic subagents.

1. long_sequential  - 8 x 1 s sequential task() calls: does the 5 s eval timeout
                      apply to time spent awaiting subagents?
2. persistence      - does a JS variable computed by a workflow in turn 1 survive
                      into turn 2 (mode="thread")?
3. interrupt_resume - a fan-out of 4 where one subagent hits a human-in-the-loop
                      interrupt. After resuming, which subagents run AGAIN?
                      (The analogue of Claude Workflow's resumeFromRunId cache.)
"""

from __future__ import annotations

import asyncio
import sys
import time

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from fakes import EVENTS, EchoWorker, ScriptedParent, _Base, dumps, reset_events, summarize_timeline


class TwoTurnParent(_Base):
    """Turn 1 runs code1; turn 2 (a new human message) runs code2."""

    code1: str
    code2: str

    @property
    def _llm_type(self) -> str:
        return "two-turn-parent"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        humans = sum(isinstance(m, HumanMessage) for m in messages)
        last = messages[-1]
        if isinstance(last, ToolMessage):
            msg = AIMessage(content=str(last.content))
        else:
            code = self.code1 if humans == 1 else self.code2
            msg = AIMessage(content="", tool_calls=[{"name": "eval", "args": {"code": code}, "id": f"call_eval_{humans}", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def ask_human(question: str) -> str:
    """Ask the human a question and wait for the answer."""
    return interrupt({"question": question})


class AskingWorker(_Base):
    """If the prompt contains ASK, calls ask_human first; otherwise answers directly."""

    @property
    def _llm_type(self) -> str:
        return "asking-worker"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = next(str(m.content) for m in messages if isinstance(m, HumanMessage))
        last = messages[-1]
        if "ASK" in prompt and not isinstance(last, ToolMessage):
            EVENTS.append({"ev": "start", "who": "asker", "prompt": prompt, "t": time.monotonic()})
            await asyncio.sleep(1.0)  # let the siblings FINISH before we interrupt
            msg = AIMessage(content="", tool_calls=[{"name": "ask_human", "args": {"question": "approve?"}, "id": "call_ask", "type": "tool_call"}])
        elif isinstance(last, ToolMessage):
            EVENTS.append({"ev": "end", "who": "asker", "prompt": prompt, "t": time.monotonic()})
            msg = AIMessage(content=f"{prompt} -> human said {last.content}")
        else:
            EVENTS.append({"ev": "start", "who": "asker", "prompt": prompt, "t": time.monotonic()})
            await asyncio.sleep(0.2)
            EVENTS.append({"ev": "end", "who": "asker", "prompt": prompt, "t": time.monotonic()})
            msg = AIMessage(content=f"{prompt} done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


LONG_SEQ = """
const out = [];
for (let i = 0; i < 8; i++) out.push(await task({ description: "step " + i, subagentType: "worker" }));
out.length;
"""

T1 = """
const report = await Promise.all(["x", "y"].map(s => task({ description: "analyse " + s, subagentType: "worker" })));
report.length;
"""
T2 = """
typeof report === "undefined" ? "report LOST" : "report still here: " + JSON.stringify(report);
"""

FANOUT_WITH_ASK = """
const r = await Promise.all(["job-1", "job-2 ASK", "job-3", "job-4"].map(d =>
  task({ description: d, subagentType: "asker" })));
r;
"""


async def long_sequential():
    agent = create_deep_agent(
        model=ScriptedParent(code=LONG_SEQ),
        subagents=[{"name": "worker", "description": "w", "system_prompt": "w", "model": EchoWorker(delay=1.0)}],
        middleware=[CodeInterpreterMiddleware()],
    )
    reset_events()
    t0 = time.monotonic()
    out = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    return {"wall_s": round(time.monotonic() - t0, 2), "eval_result": [m.content for m in out["messages"] if m.type == "tool"][-1]}


async def persistence():
    out = {}
    for mode in ("thread", "turn"):
        agent = create_deep_agent(
            model=TwoTurnParent(code1=T1, code2=T2),
            subagents=[{"name": "worker", "description": "w", "system_prompt": "w", "model": EchoWorker(delay=0.1)}],
            middleware=[CodeInterpreterMiddleware(mode=mode)],
            checkpointer=MemorySaver(),
        )
        cfg = {"configurable": {"thread_id": f"t-{mode}"}}
        await agent.ainvoke({"messages": [{"role": "user", "content": "turn one"}]}, config=cfg)
        r2 = await agent.ainvoke({"messages": [{"role": "user", "content": "turn two"}]}, config=cfg)
        out[mode] = [m.content for m in r2["messages"] if m.type == "tool"][-1]
    return out


async def interrupt_resume():
    agent = create_deep_agent(
        model=ScriptedParent(code=FANOUT_WITH_ASK),
        subagents=[{"name": "asker", "description": "a", "system_prompt": "a", "model": AskingWorker(), "tools": [ask_human]}],
        middleware=[CodeInterpreterMiddleware()],
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "resume-test"}}
    reset_events()
    first = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}, config=cfg)
    phase1 = [(e["ev"], e["prompt"]) for e in EVENTS]
    interrupts = first.get("__interrupt__")
    reset_events()
    second = await agent.ainvoke(Command(resume="yes"), config=cfg)
    phase2 = [(e["ev"], e["prompt"]) for e in EVENTS]
    return {
        "phase1_events": phase1,
        "interrupt_payload": [getattr(i, "value", i) for i in (interrupts or [])],
        "phase2_events_after_resume": phase2,
        "final_eval_result": [m.content for m in second["messages"] if m.type == "tool"][-1:],
    }


if __name__ == "__main__":
    for name in sys.argv[1:] or ["long_sequential", "persistence", "interrupt_resume"]:
        print(f"\n===== {name} =====")
        try:
            print(dumps(asyncio.run(globals()[name]())))
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"EXCEPTION: {type(e).__name__}: {e}")

"""Mechanics experiments for deepagents dynamic subagents (no real LLM needed).

Each experiment builds a deep agent whose coordinator is a ScriptedParent that
submits a fixed JavaScript orchestration script through the `eval` tool, and
whose subagents are deterministic fake workers. This isolates the runtime
behaviour (parallelism, chaining, failure handling, determinism, persistence)
from model behaviour.

Run:  python exp_mechanics.py [experiment-name ...]
"""

from __future__ import annotations

import asyncio
import sys
import time

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.memory import MemorySaver

from fakes import ClassifierWorker, EchoWorker, ScriptedParent, dumps, reset_events, summarize_timeline, EVENTS


def build(code: str, *, worker_delay: float = 1.0, mw_kwargs: dict | None = None, checkpointer=None, extra_subagents=()):
    return create_deep_agent(
        model=ScriptedParent(code=code),
        subagents=[
            {"name": "worker", "description": "Does one unit of work", "system_prompt": "work", "model": EchoWorker(delay=worker_delay)},
            {"name": "classifier", "description": "Labels a ticket bug/feature/question", "system_prompt": "classify", "model": ClassifierWorker()},
            {"name": "bug-fixer", "description": "Handles bugs", "system_prompt": "fix", "model": EchoWorker(name_tag="bug-fixer", delay=0.5)},
            {"name": "feature-analyst", "description": "Handles features", "system_prompt": "analyse", "model": EchoWorker(name_tag="feature-analyst", delay=0.5)},
            {"name": "support-agent", "description": "Handles questions", "system_prompt": "support", "model": EchoWorker(name_tag="support-agent", delay=0.5)},
            *extra_subagents,
        ],
        middleware=[CodeInterpreterMiddleware(**(mw_kwargs or {}))],
        checkpointer=checkpointer,
    )


async def run(agent, prompt="go", config=None):
    reset_events()
    t0 = time.monotonic()
    out = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    wall = round(time.monotonic() - t0, 2)
    tool_results = [m.content for m in out["messages"] if m.type == "tool"]
    return {"wall_s": wall, "eval_result": tool_results[-1] if tool_results else None, "timeline": summarize_timeline()}


# --- 1. Fan-out: is Promise.all over task() really parallel? Does the 5 s eval timeout bite? ---
FANOUT = """
const items = ["a", "b", "c", "d", "e", "f", "g", "h"];
const results = await Promise.all(items.map((it) =>
  task({ description: "item " + it, subagentType: "worker" })));
results;
"""

# --- 2. Sequential loop: same work, awaited one at a time ---
SEQUENTIAL = """
const out = [];
for (const it of ["a", "b", "c"]) {
  out.push(await task({ description: "item " + it, subagentType: "worker" }));
}
out;
"""

# --- 3. Chaining: classify -> route to a specialist by label (data-dependent control flow in code) ---
CHAIN = """
const tickets = [
  "App crashes when I upload a PNG",
  "Please add dark mode",
  "How do I reset my password?",
  "Error 500 on checkout",
];
const routed = await Promise.all(tickets.map(async (t) => {
  const label = (await task({ description: t, subagentType: "classifier" })).trim();
  const SPECIALIST = { bug: "bug-fixer", feature: "feature-analyst", question: "support-agent" };
  const handled = await task({ description: "Handle " + label + ": " + t, subagentType: SPECIALIST[label] });
  return { t, label, handled };
}));
routed.map(r => r.label + " -> " + r.handled).join("\\n");
"""

# --- 4. Failure semantics: one task() throws inside Promise.all ---
FAIL_ALL = """
const r = await Promise.all(["ok-1", "FAIL-2", "ok-3"].map((it) =>
  task({ description: it, subagentType: "worker" })));
r;
"""
FAIL_SETTLED = """
const r = await Promise.allSettled(["ok-1", "FAIL-2", "ok-3"].map((it) =>
  task({ description: it, subagentType: "worker" })));
r.map(x => x.status === "fulfilled" ? x.value : "ERR: " + String(x.reason).slice(0, 80));
"""

# --- 5. Unknown subagent type ---
UNKNOWN = """
await task({ description: "hello", subagentType: "does-not-exist" });
"""

# --- 6. Pure-CPU code still subject to the 5 s VM timeout ---
CPU_LOOP = """
let x = 0; while (true) { x++; }
"""

# --- 7. Large fan-out: is there a concurrency cap? ---
BIG_FANOUT = """
const n = 40;
const res = await Promise.all(Array.from({length: n}, (_, i) =>
  task({ description: "unit " + i, subagentType: "worker" })));
res.length;
"""

# --- 8. Loop-until-done with a data-dependent stop condition ---
LOOP_UNTIL = """
const found = [];
let round = 0;
while (round < 10) {
  round++;
  const r = await task({ description: "round " + round + (round >= 3 ? " nothing-new" : ""), subagentType: "worker" });
  if (r.includes("nothing-new")) break;
  found.push(r);
}
({ rounds: round, found: found.length });
"""


async def exp(name: str):
    if name == "fanout":
        return await run(build(FANOUT))
    if name == "sequential":
        return await run(build(SEQUENTIAL))
    if name == "chain":
        return await run(build(CHAIN))
    if name == "fail_all":
        r = await run(build(FAIL_ALL))
        r["events"] = [(e["ev"], e["prompt"]) for e in EVENTS]
        return r
    if name == "fail_settled":
        return await run(build(FAIL_SETTLED))
    if name == "unknown":
        return await run(build(UNKNOWN))
    if name == "cpu_loop":
        return await run(build(CPU_LOOP))
    if name == "big_fanout":
        return await run(build(BIG_FANOUT, worker_delay=1.0))
    if name == "loop_until":
        return await run(build(LOOP_UNTIL, worker_delay=0.1))
    if name == "determinism":
        # Same script 3x: do we get identical dispatch sets and results?
        runs = []
        for _ in range(3):
            r = await run(build(CHAIN))
            runs.append((r["eval_result"], sorted(e["prompt"] for e in EVENTS if e["ev"] == "start")))
        return {"identical_results": len({r[0] for r in runs}) == 1, "identical_dispatch_sets": len({tuple(r[1]) for r in runs}) == 1,
                "start_orders": [summarize_timeline()["start_order"]]}
    raise SystemExit(f"unknown experiment {name}")


ALL = ["fanout", "sequential", "chain", "fail_all", "fail_settled", "unknown", "cpu_loop", "big_fanout", "loop_until", "determinism"]

if __name__ == "__main__":
    names = [n for n in (sys.argv[1:] or ALL) if n != "schema_ignored"]
    for n in names:
        print(f"\n===== {n} =====")
        try:
            print(dumps(asyncio.run(exp(n))))
        except Exception as e:  # report, keep going
            print(f"EXCEPTION: {type(e).__name__}: {e}")


# --- 9. responseSchema is not enforced: a worker that ignores it ---
SCHEMA_IGNORED = """
const r = await task({ description: "rate this", subagentType: "worker",
  responseSchema: { type: "object", properties: { score: { type: "number" } }, required: ["score"] } });
({ typeofResult: typeof r, score: r.score, raw: r });
"""

if __name__ == "__main__" and sys.argv[1:] == ["schema_ignored"]:
    print(dumps(asyncio.run(run(build(SCHEMA_IGNORED, worker_delay=0.1)))))

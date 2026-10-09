"""Real-LLM experiment: does a deepagents coordinator actually write a workflow?

Coordinator: Claude Sonnet (via the `claude` CLI adapter).
Subagents:   Claude Haiku  (via the same adapter).

Scenario: triage 6 support tickets with 3 specialist subagents.
We run the same task with and without the trigger word "workflow", and the
"workflow" prompt twice, then record:
  - whether the coordinator used `eval` + `task()` or the plain `task` tool
  - the orchestration code it wrote (to compare across runs)
  - how many subagent dispatches happened, wall time, cost

Run:  python exp_real_llm.py [workflow|plain|workflow2 ...]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from claude_cli_model import CALL_LOG, ClaudeCLIChatModel

SCRATCH = "/tmp/claude-0/-home-user-auto-research/efc8a2ec-6ca5-5113-8ee3-c7f683d70030/scratchpad"
OUT = Path(__file__).parent / "real_llm_runs"

TICKETS = """\
T1: The app crashes every time I upload a PNG larger than 5MB.
T2: Please add a dark mode to the dashboard.
T3: How do I export my invoices as CSV?
T4: Checkout returns HTTP 500 when the cart has more than 20 items.
T5: It would be great to have SSO with Okta.
T6: Where can I change my notification email?"""

PROMPTS = {
    "workflow": f"Run a workflow that triages these support tickets: classify each as bug, feature or question, send each to the matching specialist, and give me a one-line outcome per ticket.\n\n{TICKETS}",
    "plain": f"Triage these support tickets: classify each as bug, feature or question, send each to the matching specialist, and give me a one-line outcome per ticket.\n\n{TICKETS}",
}
PROMPTS["workflow2"] = PROMPTS["workflow"]


def worker(tag: str) -> ClaudeCLIChatModel:
    return ClaudeCLIChatModel(model="haiku", tag=tag, workdir=SCRATCH)


def build():
    return create_deep_agent(
        model=ClaudeCLIChatModel(model="sonnet", tag="coordinator", workdir=SCRATCH),
        subagents=[
            {"name": "bug-fixer", "description": "Investigates bug reports and proposes likely cause and next step",
             "system_prompt": "You are a bug triage specialist. In 1-2 sentences give the likely cause and the next step.", "model": worker("bug-fixer")},
            {"name": "feature-analyst", "description": "Evaluates feature requests for effort and impact",
             "system_prompt": "You are a product analyst. In 1-2 sentences rate effort (S/M/L) and impact.", "model": worker("feature-analyst")},
            {"name": "support-agent", "description": "Answers user how-to questions",
             "system_prompt": "You are a support specialist. Answer in 1-2 sentences; say where in the product to look.", "model": worker("support-agent")},
        ],
        middleware=[CodeInterpreterMiddleware(timeout=10.0)],
    )


async def run(name: str) -> dict:
    CALL_LOG.clear()
    agent = build()
    events = []
    t0 = time.monotonic()
    final = None
    async for mode, chunk in agent.astream({"messages": [{"role": "user", "content": PROMPTS[name]}]},
                                           stream_mode=["custom", "values"], subgraphs=False):
        if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "subagent":
            events.append(chunk)
        elif mode == "values":
            final = chunk
    wall = round(time.monotonic() - t0, 1)
    msgs = final["messages"]
    coord_calls = [tc for m in msgs if m.type == "ai" for tc in m.tool_calls]
    eval_codes = [tc["args"].get("code") for tc in coord_calls if tc["name"] == "eval"]
    plain_task_calls = [tc["args"] for tc in coord_calls if tc["name"] == "task"]
    result = {
        "prompt": name,
        "wall_s": wall,
        "coordinator_tool_calls": [tc["name"] for tc in coord_calls],
        "eval_calls": len(eval_codes),
        "dispatches_from_code(task())": sum(1 for e in events if e.get("phase") == "start"),
        "dispatches_via_plain_task_tool": len(plain_task_calls),
        "subagent_errors": [e for e in events if e.get("phase") == "error"],
        "eval_code": eval_codes,
        "final_answer": msgs[-1].content if msgs[-1].type == "ai" else None,
        "cli_calls": len(CALL_LOG),
        "cost_usd": round(sum(c["cost_usd"] or 0 for c in CALL_LOG), 4),
        "call_log": list(CALL_LOG),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    for n in sys.argv[1:] or ["workflow", "plain", "workflow2"]:
        r = asyncio.run(run(n))
        print(f"\n===== {n} =====")
        print(json.dumps({k: v for k, v in r.items() if k not in ("call_log",)}, indent=2, default=str)[:6000])

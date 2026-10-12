"""Minimal deepagents dynamic-subagents example: triage support tickets.

    pip install "deepagents[quickjs]"
    export ANTHROPIC_API_KEY=...
    python sample_dynamic_subagents.py            # uses the Anthropic API
    python sample_dynamic_subagents.py --cli      # uses the local `claude` CLI (see claude_cli_model.py)

What makes it "dynamic": the coordinator gets an `eval` tool (QuickJS). Because
the agent has subagents, `eval` exposes a `task()` global, and the model writes
JavaScript that loops / Promise.all's over task() calls. The script prints the
JS the model wrote and a live line per subagent dispatch.
"""

import asyncio
import sys

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

TICKETS = """\
T1: The app crashes every time I upload a PNG larger than 5MB.
T2: Please add a dark mode to the dashboard.
T3: How do I export my invoices as CSV?
T4: Checkout returns HTTP 500 when the cart has more than 20 items."""

if "--cli" in sys.argv:  # no API key: drive Claude through the CLI adapter from this repo
    from claude_cli_model import ClaudeCLIChatModel

    coordinator, worker = ClaudeCLIChatModel(model="sonnet"), ClaudeCLIChatModel(model="haiku")
else:
    coordinator, worker = "anthropic:claude-sonnet-5-5", "anthropic:claude-haiku-5-5"

agent = create_deep_agent(
    model=coordinator,
    # Optional: make dispatch-from-code the rule instead of relying on the word "workflow".
    system_prompt="For any request covering several items, orchestrate subagents from code with task() inside eval.",
    subagents=[
        {"name": "bug-fixer", "description": "Investigates bug reports: likely cause and next step",
         "system_prompt": "You are a bug triage specialist. Answer in one sentence.", "model": worker},
        {"name": "feature-analyst", "description": "Rates feature requests for effort (S/M/L) and impact",
         "system_prompt": "You are a product analyst. Answer in one sentence.", "model": worker},
        {"name": "support-agent", "description": "Answers how-to questions",
         "system_prompt": "You are a support specialist. Answer in one sentence.", "model": worker},
    ],
    middleware=[CodeInterpreterMiddleware(timeout=10.0)],  # adds `eval`; task() is on because subagents exist
)


async def main() -> None:
    prompt = (
        "Run a workflow that triages these tickets: classify each as bug, feature or question, "
        f"send each to the matching specialist, and give a one-line outcome per ticket.\n\n{TICKETS}"
    )
    final = None
    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]}, stream_mode=["custom", "values"]
    ):
        if mode == "custom" and chunk.get("type") == "subagent":  # live fan-out events from task()
            if chunk["phase"] == "start":
                print(f"  -> {chunk['subagent_type']:<16} {chunk['label']}")
            elif chunk["phase"] == "error":
                print(f"  !! {chunk['id']} failed: {chunk['error']}")
        elif mode == "values":
            final = chunk

    for m in final["messages"]:
        for tc in getattr(m, "tool_calls", []) or []:
            if tc["name"] == "eval":
                print("\n----- JS the coordinator wrote -----\n" + tc["args"]["code"])
    print("\n----- Final answer -----\n" + final["messages"][-1].content)


asyncio.run(main())

"""A LangChain chat model backed by the `claude` CLI (`claude -p`).

The sandbox this research ran in had no Anthropic API key, but it did have an
authenticated Claude Code CLI. This adapter lets deepagents drive a real Claude
model anyway: each model call renders the system prompt, the bound tool
schemas and the conversation transcript into one prompt, and asks the CLI for
a structured reply `{text, tool_calls:[{name, args_json}]}` via `--json-schema`.

It is a test harness, not a production integration: no streaming, no images,
one subprocess per model call.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "args_json": {"type": "string"}},
                "required": ["name", "args_json"],
            },
        },
    },
    "required": ["text", "tool_calls"],
}

ADAPTER_RULES = """\
You are the language model behind an external agent framework. You are shown
the agent's system prompt, the ACTIONS (framework tools) it may request, and
the conversation so far. Produce the assistant's NEXT turn as data.

IMPORTANT: you have no native tools in this session except StructuredOutput.
Never attempt a native tool call named after an action (eval, task, ls, ...).
Requesting an action means writing it into the `tool_calls` field of your
StructuredOutput reply; the framework then executes it and shows you the
result in a later turn. Every listed action really is available that way.

- To request actions: fill `tool_calls` with {name, args_json} (args_json is
  the JSON-encoded arguments matching that action's schema). Keep `text`
  short or empty.
- To give the final answer: put it in `text` and return `tool_calls: []`.
"""

CALL_LOG: list[dict[str, Any]] = []  # every CLI call: who, duration, cost


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


class ClaudeCLIChatModel(BaseChatModel):
    model: str = "haiku"
    tag: str = "agent"
    timeout_s: float = 300.0
    tools_spec: list[dict] = []
    workdir: str = tempfile.gettempdir()

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        spec = [convert_to_openai_tool(t)["function"] for t in tools]
        return self.model_copy(update={"tools_spec": spec})

    def _render(self, messages: list[BaseMessage]) -> tuple[str, str]:
        system = "\n\n".join(_render_content(m.content) for m in messages if isinstance(m, SystemMessage))
        lines = []
        for m in messages:
            if isinstance(m, SystemMessage):
                continue
            if isinstance(m, HumanMessage):
                lines.append(f"### USER\n{_render_content(m.content)}")
            elif isinstance(m, AIMessage):
                body = _render_content(m.content)
                for tc in m.tool_calls:
                    body += f"\n<tool_call id={tc['id']} name={tc['name']}>{json.dumps(tc['args'])}</tool_call>"
                lines.append(f"### ASSISTANT\n{body}")
            elif isinstance(m, ToolMessage):
                lines.append(f"### TOOL RESULT (id={m.tool_call_id}, name={m.name})\n{_render_content(m.content)}")
        tools = json.dumps(self.tools_spec, indent=1) if self.tools_spec else "(none)"
        prompt = (
            f"===== AGENT SYSTEM PROMPT =====\n{system}\n\n===== ACTIONS (request via tool_calls) =====\n{tools}\n\n"
            f"===== CONVERSATION =====\n" + "\n\n".join(lines) + "\n\n===== Produce the next ASSISTANT turn now. ====="
        )
        return ADAPTER_RULES, prompt

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        sys_prompt, prompt = self._render(messages)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(sys_prompt)
            sys_file = f.name
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", "--model", self.model, "--tools", "", "--output-format", "json",
            "--system-prompt-file", sys_file, "--json-schema", json.dumps(REPLY_SCHEMA),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=self.workdir,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), self.timeout_s)
        finally:
            os.unlink(sys_file)
        data = json.loads(out.decode() or "{}")
        reply = data.get("structured_output") or {}
        if not reply and data.get("result"):
            try:
                reply = json.loads(data["result"])
            except json.JSONDecodeError:
                reply = {"text": data["result"], "tool_calls": []}
        CALL_LOG.append({"tag": self.tag, "model": self.model, "secs": round(loop.time() - t0, 1),
                         "cost_usd": data.get("total_cost_usd"), "n_tool_calls": len(reply.get("tool_calls", []))})
        tool_calls = []
        for tc in reply.get("tool_calls", []):
            try:
                args = json.loads(tc.get("args_json") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.get("args_json")}
            tool_calls.append({"name": tc["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"})
        msg = AIMessage(content=reply.get("text", ""), tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

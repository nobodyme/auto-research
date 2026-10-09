# Notes: deepagents "dynamic subagents" vs Claude Code Workflows

Claim to check: *"LangChain deepagents dynamic subagents is similar to Claude
workflow, which creates deterministic workflows chained together."*

## Setup
- deepagents 0.7.23, langchain-quickjs 0.3.8 (quickjs-rs 0.2.5), langgraph 1.2.14, langchain 1.4.4, Python 3.13.
- No Anthropic API key in the sandbox, but an authenticated `claude` CLI was available.
  So I wrote `claude_cli_model.py`, a LangChain `BaseChatModel` that shells out to
  `claude -p --json-schema ...` and maps the reply to tool calls.
- Claude Code's `Workflow` tool was available in the session, so I ran it directly.

## Reading (docs and source)
- Docs: https://docs.langchain.com/oss/python/deepagents/dynamic-subagents and .../interpreters
- "Dynamic subagents" = `CodeInterpreterMiddleware` (QuickJS) adds an `eval` tool. When
  the agent has subagents, the JS realm gets a `task({description, subagentType, responseSchema?, label?})`
  global. The **model writes JS at runtime** that loops / branches / `Promise.all`s over `task()`.
- The docs list the same patterns as Claude's workflow-authoring reference: classify-and-act,
  fan-out/synthesize, adversarial verification, generate-and-filter, tournament, loop-until-done.
- The interpreter system prompt (`langchain_quickjs/_prompt.py`) says that if the user says
  "workflow", the agent should fan out via `task()` in `eval`. The word is a trigger in both products.
- Source details:
  - `_repl.py`: `_MAX_TASK_CALLS_PER_THREAD = 32`. An asyncio.Semaphore caps concurrent `task()` calls.
  - The `timeout` (default 5 s) counts QuickJS VM time only. Awaiting host calls (subagents) is excluded.
  - `_subagent.py`: dispatch ids are a sha256 over (eval tool_call_id, ordinal, payload). The comment
    says this keeps them stable across an interrupt replay.
  - `GraphInterrupt` from a subagent propagates out through `eval` (HITL inside subagents works).
  - `task()` bypasses the parent's `interrupt_on` approvals (documented warning).
  - `deepagents/middleware/subagents.py`: if the subagent produces no `structured_response`, the task
    tool returns the last AI text. In JS, `JSON.parse` fails silently, so you get a string back.
- Claude Workflow (from the workflow-authoring reference, then verified by running it):
  - The main Claude writes a JS script with `agent()`, `parallel()`, `pipeline()`, `phase()`, `log()`,
    `workflow()`, `args`, `budget`. It is submitted to the `Workflow` tool and **runs in the background**.
    The tool returns a task id immediately and a notification arrives on completion.
  - `agent(prompt, {schema, model, effort, isolation:'worktree', agentType})`. With a schema, the output
    is validated at the tool layer and the model retries on mismatch.
  - `parallel()` maps a throwing thunk to `null` instead of rejecting. In `pipeline()` an item's
    failure drops only that item.
  - Concurrency cap is min(16, CPUs-2). There is a 1000-agent lifetime cap.
  - Determinism contract: `Date.now()`, `Math.random()` and `new Date()` are banned so that
    `resumeFromRunId` can replay the longest unchanged prefix of `agent()` calls from a journal.
  - Scripts can be saved and named (`.claude/workflows/`) and called with `args`, or nested via `workflow()`.
  - It is opt-in only: the user must ask for a workflow ("ultracode", "use a workflow", ...).

## Experiments: deepagents mechanics with deterministic fakes (`exp_mechanics.py`, `exp_state.py`)
`fakes.py`: the ScriptedParent emits a fixed `eval` script. EchoWorker sleeps a fixed time and
echoes, and raises if the prompt contains FAIL. This isolates the runtime from model behaviour.

Results (`results_mechanics.txt`, `results_state*.txt`):
| test | result |
|---|---|
| fan-out 8 x 1 s via Promise.all | peak concurrency 8, span 1.01 s. Results come back in input order. Start order is nondeterministic. |
| sequential 3 x 1 s | 3.03 s, concurrency 1 |
| classify -> route chain (4 tickets, 8 dispatches) | correct routing; per-item chains overlap (peak 4) |
| one task() throws inside Promise.all | the whole eval returns `<error type="RuntimeError">worker failed on: FAIL-2</error>`. Siblings still ran to completion, but their results were discarded. |
| Promise.allSettled | partial results survive, but the rejection reason is only `HostError: Host function failed` (the message is lost) |
| unknown subagentType | **does not throw**. task() resolves to the string "We cannot invoke subagent does-not-exist because it does not exist..." |
| CPU-bound infinite loop | `<error type="Timeout">interrupted</error>` after 5 s |
| 40 x 1 s fan-out | peak concurrency 32, 2.1 s. This confirms the hard 32 cap. |
| loop-until-done | stops on a data-dependent condition |
| same script 3x | identical results and identical dispatch sets; start order varies |
| 8 x 1 s sequential (> 5 s timeout) | finished in 9 s. The timeout ignores time spent awaiting subagents. |
| persistence, mode="thread" | a variable from turn 1's workflow is still in scope in turn 2. With mode="turn" it is lost. |
| interrupt in 1 of 4 subagents, siblings still running | after resume, the interrupted subagent continues from its checkpoint. The 3 siblings that were cancelled mid-flight **re-run from scratch**. |
| interrupt in 1 of 4, siblings already finished | after resume, **only** the interrupted subagent runs. The finished siblings were NOT re-executed (stable dispatch ids + LangGraph checkpoints act as a result cache). |
| responseSchema ignored by the worker | task() returns a string; `r.score` is `undefined`. No error. |

## Experiments: deepagents with a real LLM (`exp_real_llm.py`, `real_llm_runs/`)
Coordinator = Sonnet, subagents = Haiku, all via the CLI adapter. Task: triage 6 tickets.
- Adapter v1: the model said eval was "unavailable", so it fell back to 6 parallel *plain* `task`
  tool calls (a model-chosen batch, no code). v2: the model attempted native tool calls named
  eval/task inside the CLI. Fixed by telling it to express actions only as data (v3). Lesson: this is
  harness noise. The runs below all use v3.
- "Run a workflow that triages..." (run 1): 2 eval calls, 18 task() dispatches. Script 1 was
  6 x classify (general-purpose + responseSchema) -> 6 x specialist (responseSchema). The specialist
  outcomes came back empty: the schema was not honoured and the text fallback means `.outcome` is
  undefined. **The coordinator noticed and wrote a second script that reused the `cls` variable from
  REPL state** and re-dispatched without a schema. 104 s, $0.37.
- Same prompt (run 2): the same overall shape (classify -> route, then a repair eval), but different
  code text (arrays of tuples vs objects, different prompts). 18 dispatches, 51 s, $0.29.
- Prompt without "workflow": still used eval + task(), but the **coordinator classified the tickets
  itself** and wrote a 1-stage fan-out (6 dispatches), then a probe (1), then a re-run (6) = 13 dispatches.
  So the orchestration *structure* is chosen by the model per run.
- The schema failure is partly my adapter (Haiku via the CLI sometimes answered in text instead of
  calling the structured-output tool). The silent-`undefined` behaviour is deepagents' own, and
  `results_schema_ignored.txt` reproduces it with a fake.

## Experiments: Claude Code Workflow (`claude_workflow_probe.js`)
- First submission included `try { Date.now() } catch {}`. It was **rejected statically before running**:
  "Workflow scripts must be deterministic: Date.now()/Math.random()/new Date() are unavailable (breaks resume)".
- Second submission ran in the background (the tool returned immediately with a task id and run id).
- Run 1 (`claude_workflow_runs/run1_result.txt`): 8 Haiku agents, 16.4 s, 376k subagent tokens. All 4 tickets were
  classified and handled correctly, and every schema-validated `outcome` was present. A throwing thunk inside
  `parallel()` became `null` and was reported under `<failures>` ("parallel[1] failed: boom"). The run kept going.
- The journal shows `pipeline()` has no barrier: "T1 -> bug" started before "classify T4" had finished.
- Run 2: I edited only the stage-2 prompt and relaunched with `resumeFromRunId`. Only the 4 stage-2 agents started
  (journal lines 18-25). The 4 classifiers replayed from cache: 188k tokens (half), 9.9 s, and the stage-2
  outcomes changed (shorter).

## Verdict reasoning
Same core idea: an LLM writes a small JS program, and the *program* (loops, Promise.all, branching on
results) drives subagents. The pattern catalogues are almost identical, and so is the trigger word "workflow".
"Deterministic" only applies to the control flow once the script exists, in both systems. In deepagents
the script is re-authored by the model on every request (my two identical prompts gave different code, and
the "plain" prompt gave a different structure).

Material differences found by testing:
1. Where it runs: deepagents runs inside an `eval` tool call in the coordinator's turn (it blocks, the coordinator
   sees the result, then can write another eval that reuses variables). Claude runs as a background job launched by
   the main agent, with a notification on completion.
2. What can be dispatched: deepagents only dispatches developer-registered subagent types. Claude accepts any prompt,
   with per-call model, effort, worktree isolation and agentType.
3. Failure handling: deepagents uses raw JS promise semantics (one failure kills Promise.all and the finished
   siblings' results are thrown away; allSettled loses the message; an unknown type resolves to an error *string*).
   Claude turns a failure into `null` per item and reports it.
4. Schemas: deepagents is best-effort (silent text fallback). Claude validates and retries.
5. Durability: deepagents gets durability from LangGraph checkpoints plus stable dispatch ids (HITL interrupt -> resume
   does not redo finished siblings; REPL vars persist across turns in thread mode). Claude has a journal plus
   prefix-cached resume even after editing the script, with a static ban on Date.now()/Math.random().
6. Reuse: Claude scripts can be saved, named, parameterised (`args`) and nested. A deepagents script lives inside one
   tool call (functions do not survive the cross-turn snapshot).
7. Who decides to use it: deepagents leaves it to the model, which used it even without the word "workflow".
   Claude requires explicit user opt-in.
8. Limits: deepagents allows 32 concurrent task() calls, a 5 s CPU-only timeout and a 64 MB heap. Claude allows
   min(16, CPUs-2) concurrent agents, 1000 agents in total, and an optional token budget.

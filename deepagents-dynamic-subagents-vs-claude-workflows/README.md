# deepagents dynamic subagents vs Claude Code Workflows

**Claim tested:** "LangChain deepagents *dynamic subagents* is similar to Claude's
workflow feature, which creates deterministic workflows chained together."

**Verdict: partly true.** Both features share one core idea. A model writes a small
JavaScript program, and the program (loops, `Promise.all`, branching on results)
dispatches and chains subagents. The model does not issue one tool call per step.
Both publish nearly the same pattern catalogue (fan-out/synthesize, classify-and-route,
adversarial verify, tournament, loop-until-done), and both treat the word "workflow"
as the trigger.

The two are built differently, though, and "deterministic workflows" overstates both:

* In both systems only the **control flow** is deterministic once the script exists.
  Subagent outputs are not.
* In deepagents the script is **re-authored by the model on every request**. Two
  identical prompts produced different code. The same request phrased without
  "workflow" produced a structurally different plan (1 stage instead of 2).
* Under test, the runtimes differed in where the script runs, failure handling,
  schema enforcement, resume/durability, reuse, and who decides to use it (table below).

A good one-line description: **deepagents dynamic subagents is "programmatic tool calling
for subagents" inside an agent's loop. Claude Workflows is a background orchestration job
with a journaled, resumable script runtime.**

An explainer page for people who know deepagents but not this feature is published as an
artifact: https://claude.ai/artifact/3ykUe14cFjwTGSmkgDGtwx (private until shared; source: [`artifact.html`](artifact.html)).

## What was built and run

| File | What it does |
|---|---|
| `fakes.py` | Deterministic fake chat models. A *scripted coordinator* submits a fixed `eval` script. *Echo workers* sleep, log timestamps, echo, and fail on demand. |
| `exp_mechanics.py` | 11 runtime experiments (parallelism, chaining, failures, limits, determinism, schema). Output: `results_mechanics.txt`, `results_schema_ignored.txt` |
| `exp_state.py` | Timeout scope, cross-turn persistence, and human-in-the-loop interrupt then resume. Output: `results_state*.txt` |
| `claude_cli_model.py` | A LangChain `BaseChatModel` that drives the authenticated `claude -p` CLI with a JSON schema. Needed because the sandbox had no Anthropic API key. |
| `exp_real_llm.py` | Real deepagents run: Sonnet coordinator plus 3 Haiku specialist subagents triaging 6 tickets. Output: `real_llm_runs/*.json` |
| `claude_workflow_runs/` | The equivalent Claude Code Workflow (4 tickets, classify then route, 8 Haiku agents): script, an edited script for the resume test, the journal, and both results. |

Versions: deepagents 0.7.23, langchain-quickjs 0.3.8, langgraph 1.2.14, Python 3.13.

## Key results

### deepagents mechanics (deterministic fakes)
* **Parallel fan-out is real.** 8 one-second subagents finished in 1.01 s (peak concurrency 8).
  Results come back in input order. Start order varies from run to run.
* **Hard cap of 32 concurrent `task()` calls per REPL.** 40 dispatches peaked at 32 and took two waves.
  The cap is `_MAX_TASK_CALLS_PER_THREAD = 32` in `langchain_quickjs/_repl.py`.
* **The 5 s `eval` timeout counts only JS CPU time.** A busy loop died at 5 s. Eight sequential
  1 s subagents ran fine (9 s), because time spent awaiting subagents doesn't count.
* **Chaining works.** Classify → route-by-label with per-item chains overlapping. The same
  script run 3 times gave identical results and dispatch sets.
* **Failures follow raw JS promise rules.** One failing `task()` inside `Promise.all` turns the
  whole `eval` into an error, even though the siblings ran to completion and their results are
  thrown away. `Promise.allSettled` keeps partial results, but the error text is reduced to
  `HostError: Host function failed`.
* **A typo'd `subagentType` doesn't throw.** `task()` resolves to the string
  "We cannot invoke subagent … does not exist …", which flows on as if it were a result.
* **`responseSchema` is best-effort.** If a subagent ends without structured output, `task()`
  returns its plain text, so `r.score` is silently `undefined`.
* **Durability comes from LangGraph.**
  - In `mode="thread"`, variables computed by a workflow in turn 1 are still there in turn 2.
    In `mode="turn"` they are gone.
  - A subagent that hits a human-in-the-loop `interrupt()` stops the whole `eval`. After
    `Command(resume=…)`, the interrupted subagent continues from its checkpoint, and **siblings
    that had already finished are not re-run**. This relies on stable sha256 dispatch ids plus the
    checkpointer. Siblings that were still in flight when the interrupt landed **are re-run** from
    scratch.

### deepagents with a real model
* Prompt with "Run a workflow that…" (two runs): the coordinator wrote a 2-stage `eval`
  (6 × classify with a schema, then 6 × route to a specialist), 18 dispatches. The specialists
  ignored the schema, so outcomes came back `undefined`. **The coordinator noticed and wrote a
  second `eval` that reused the `cls` variable still in the REPL**, re-running only stage 2
  without a schema. Same shape both runs, different code text. Wall time was 104 s and 51 s;
  cost $0.37 and $0.29.
* Prompt without "workflow": the coordinator still used `eval` + `task()`, but classified the
  tickets itself and fanned out once. 13 dispatches in total, including a probe and a retry.
* Harness caveat: the schema failures are partly an artifact of my CLI adapter. The silent
  `undefined` behaviour is deepagents' own (reproduced with a fake).

### Claude Code Workflow (same task)
* `Date.now()` inside a `try` block was **rejected before the run started**, because the tool
  statically enforces determinism so that resume works.
* The run went to the background immediately and finished in 16.4 s with 8 agents. Every
  schema-validated `outcome` was present. A throwing thunk inside `parallel()` became `null`
  and was reported, and the run continued.
* The journal shows `pipeline()` has no barrier: ticket 1's stage 2 started before ticket 4's
  stage 1 finished.
* **Resume after editing the script:** I changed only the stage-2 prompt and relaunched with
  `resumeFromRunId`. The 4 classifiers replayed from cache and only the 4 stage-2 agents ran.
  Tokens halved, time fell from 16.4 s to 9.9 s, and the outcomes changed.

## Side-by-side

| | deepagents dynamic subagents | Claude Code Workflow |
|---|---|---|
| Script author | Coordinator model, inside an `eval` tool call | Main Claude agent, via the `Workflow` tool (or a saved, named script) |
| Runs | Synchronously inside the agent's turn. The model sees the result and can write another `eval` reusing variables. | As a background job. The main agent is notified on completion. |
| Dispatch | `task({description, subagentType, responseSchema?})`. Only **pre-registered** subagent types. | `agent(prompt, {schema, model, effort, isolation:'worktree', agentType})`. Any prompt. |
| Primitives | Plain JS: `Promise.all`, loops | `parallel()` (barrier, null-on-failure), `pipeline()` (no barrier), `phase`, `log`, `workflow()`, `args`, `budget` |
| One subagent fails | `Promise.all` rejects and the eval errors. Use `allSettled` (message lost). | That item becomes `null` and is reported. The rest continue. |
| Structured output | Best-effort, with a silent text fallback | Validated, and the model retries on mismatch |
| Concurrency / limits | 32 concurrent `task()`, 5 s CPU timeout, 64 MB heap | min(16, CPUs-2) concurrent, 1000 agents in total, optional token budget |
| Resume | LangGraph checkpointer. Interrupt/resume skips finished subagents. REPL state persists across turns. | Journal plus `resumeFromRunId`. A prefix cache survives script edits. `Date.now`/`Math.random` are banned. |
| Reuse | None built in. The script lives in one tool call, and functions don't survive turn snapshots. | Save as a named workflow, pass `args`, nest with `workflow()` |
| Who decides to use it | The model (it used `eval` even without the word "workflow") | Explicit user opt-in |
| Approvals | `task()` bypasses the parent's `interrupt_on` (documented) | Subagents keep the harness's normal tool permissions |
| Where it lives | A library feature for developers building agents | A feature of the Claude Code harness |

## When dynamic subagents are the right tool
* Many independent units (files, tickets, records) where model-chosen tool calls would only
  sample a few. Code covers every item.
* Multi-stage pipelines where stage 2 depends on stage 1's output, such as cheap classification
  then deep review of only the risky items.
* Verification and voting patterns, where results are combined in code and never pasted back
  into the model's context.
* Iterative work across turns, using `mode="thread"` to keep the working set in REPL variables.

Guardrails I'd add after testing:
* Use `Promise.allSettled` or per-task `.catch`.
* Validate `responseSchema` fields in JS rather than trusting them.
* Check for the "does not exist" string after each `task()` call.
* Gate `eval` itself if you need human approval.
* Use a checkpointer if subagents can interrupt.

## Sources
* deepagents dynamic subagents: https://docs.langchain.com/oss/python/deepagents/dynamic-subagents
* Interpreters (QuickJS, PTC, persistence, limits): https://docs.langchain.com/oss/python/deepagents/interpreters
* Launch post: https://langchain.com/blog/introducing-dynamic-subagents-in-deep-agents
* Source code read: `langchain_quickjs/{_repl,_subagent,_prompt,middleware}.py`, `deepagents/middleware/subagents.py`
* Claude Code Workflow tool reference (workflow-authoring skill), checked against the runs above.

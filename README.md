# Research projects carried out by AI tools

Each directory in this repo is a separate research project carried out by an LLM tool - usually [Claude Code](https://www.claude.com/product/claude-code). *Inspired by the blog from [Simon Willison](https://simonwillison.net/2025/Nov/6/async-code-research/)*

<!--[[[cog
import os
import re
import subprocess
import pathlib
from datetime import datetime, timezone

# Model to use for generating summaries: Anthropic if ANTHROPIC_API_KEY is set,
# otherwise OpenAI via OPENAI_API_KEY (GitHub Models has been retired)
if os.environ.get("ANTHROPIC_API_KEY"):
    MODEL = "anthropic/claude-opus-5-5"
elif os.environ.get("OPENAI_API_KEY"):
    MODEL = "gpt-4.1"
else:
    MODEL = None

# Get all subdirectories with their first commit dates
research_dir = pathlib.Path.cwd()
subdirs_with_dates = []

for d in research_dir.iterdir():
    if d.is_dir() and not d.name.startswith('.'):
        # Get the date of the first commit that touched this directory
        try:
            result = subprocess.run(
                ['git', 'log', '--diff-filter=A', '--follow', '--format=%aI', '--reverse', '--', d.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse first line (oldest commit)
                date_str = result.stdout.strip().split('\n')[0]
                commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                subdirs_with_dates.append((d.name, commit_date))
            else:
                # No git history, use directory modification time
                subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))
        except Exception:
            # Fallback to directory modification time
            subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))

# Print the heading with count
print(f"## {len(subdirs_with_dates)} research projects\n")

# Sort by date, most recent first
subdirs_with_dates.sort(key=lambda x: x[1], reverse=True)

for dirname, commit_date in subdirs_with_dates:
    folder_path = research_dir / dirname
    readme_path = folder_path / "README.md"
    summary_path = folder_path / "_summary.md"

    date_formatted = commit_date.strftime('%Y-%m-%d')

    # Get GitHub repo URL
    github_url = None
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            origin = result.stdout.strip()
            # Convert SSH URL to HTTPS URL for GitHub
            if origin.startswith('git@github.com:'):
                origin = origin.replace('git@github.com:', 'https://github.com/')
            if origin.endswith('.git'):
                origin = origin[:-4]
            github_url = f"{origin}/tree/main/{dirname}"
    except Exception:
        pass

    if github_url:
        print(f"### [{dirname}]({github_url}) ({date_formatted})\n")
    else:
        print(f"### {dirname} ({date_formatted})\n")

    # Check if summary already exists
    if summary_path.exists():
        # Use cached summary
        with open(summary_path, 'r') as f:
            description = f.read().strip()
            if description:
                print(description)
            else:
                print("*No description available.*")
    elif readme_path.exists():
        # Generate new summary using llm command
        if MODEL is None:
            raise RuntimeError(f"No summary for {dirname}: set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        prompt = """Summarize this research project concisely. Write just 1 paragraph (3-5 sentences) followed by an optional short bullet list if there are key findings. Vary your opening - don't start with "This report" or "This research". Include 1-2 links to key tools/projects. Be specific but brief. No emoji."""
        result = subprocess.run(
            ['llm', '-m', MODEL, '-s', prompt],
            stdin=open(readme_path),
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            error_msg = f"LLM command failed for {dirname} with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            raise RuntimeError(error_msg)
        if result.stdout.strip():
            description = result.stdout.strip()
            print(description)
            # Save to cache file
            with open(summary_path, 'w') as f:
                f.write(description + '\n')
        else:
            raise RuntimeError(f"LLM command returned no output for {dirname}")
    else:
        print("*No description available.*")

    print()  # Add blank line between entries

# Add AI-generated note to all project README.md files
# Note: we construct these marker strings via concatenation to avoid the HTML comment close sequence
AI_NOTE_START = "<!-- AI-GENERATED-NOTE --" + ">"
AI_NOTE_END = "<!-- /AI-GENERATED-NOTE --" + ">"
AI_NOTE_CONTENT = """> [!NOTE]
> This is an AI-generated research report. All text and code in this report was created by an LLM (Large Language Model). For more information on how these reports are created, see the [main research repository](https://github.com/simonw/research)."""

for dirname, _ in subdirs_with_dates:
    folder_path = research_dir / dirname
    readme_path = folder_path / "README.md"

    if not readme_path.exists():
        continue

    content = readme_path.read_text()

    # Check if note already exists
    if AI_NOTE_START in content:
        # Replace existing note
        pattern = re.escape(AI_NOTE_START) + r'.*?' + re.escape(AI_NOTE_END)
        new_note = f"{AI_NOTE_START}\n{AI_NOTE_CONTENT}\n{AI_NOTE_END}"
        new_content = re.sub(pattern, new_note, content, flags=re.DOTALL)
        if new_content != content:
            readme_path.write_text(new_content)
    else:
        # Add note after first heading (# ...)
        lines = content.split('\n')
        new_lines = []
        note_added = False
        for i, line in enumerate(lines):
            new_lines.append(line)
            if not note_added and line.startswith('# '):
                # Add blank line, then note, then blank line
                new_lines.append('')
                new_lines.append(AI_NOTE_START)
                new_lines.append(AI_NOTE_CONTENT)
                new_lines.append(AI_NOTE_END)
                note_added = True

        if note_added:
            readme_path.write_text('\n'.join(new_lines))

]]]-->
## 9 research projects

### [deepagents-dynamic-subagents-vs-claude-workflows](https://github.com/nobodyme/auto-research/tree/main/deepagents-dynamic-subagents-vs-claude-workflows) (2026-10-09)

Testing the claim that LangChain deepagents' *dynamic subagents* is the same thing as Claude Code Workflows, this project found it only partly true. In both systems the model writes a small JavaScript program that fans out, chains and routes subagents, and the two publish nearly identical pattern catalogues. They differ in how they run: [deepagents dynamic subagents](https://docs.langchain.com/oss/python/deepagents/dynamic-subagents) re-authors the script on every request and runs it synchronously inside the agent's turn via a QuickJS `eval`, while Claude Code Workflows runs a journaled, resumable script as a background job. The comparison rests on 11 deterministic-fake runtime experiments, a real Sonnet + Haiku ticket-triage run, and the equivalent Claude Workflow including an edited-script resume.

- deepagents caps concurrency at 32 `task()` calls per REPL and its 5 s `eval` timeout counts only JS CPU time, not time spent awaiting subagents.
- One failing `task()` inside `Promise.all` errors the whole `eval`; a typo'd subagent type and an ignored `responseSchema` fail silently instead of throwing.
- Claude Workflows validated every structured output, turned a throwing agent into a reported `null`, and on resume replayed unchanged stages from cache, halving tokens.
- Only control flow is deterministic in either system; subagent outputs are not.

### [excel-extraction](https://github.com/nobodyme/auto-research/tree/main/excel-extraction) (2026-07-19)

Five Python Excel libraries were benchmarked against a deliberately complex `.xlsx` workbook (5 sheets, 188 formulas, 6 charts, a structured table, merges, hyperlinks, a comment and a named range), scoring per-feature recall and load time against generated ground truth. [openpyxl](https://openpyxl.readthedocs.io/) is the only library that recovers every structural feature, including formula text, charts and defined names. Use `data_only=True` to get computed values instead of formula strings. [python-calamine](https://github.com/dimastbk/python-calamine), a Rust-backed reader, matched value accuracy at 12-15x the speed but exposes nothing beyond cell values. pandas and pyexcel also return values only, and `xlrd` 2.x no longer reads `.xlsx` at all.

- openpyxl: 100% recall on formulas, charts, tables, merges, hyperlinks, comments and defined names (~24 ms).
- python-calamine: 1.8 ms load, full value accuracy, zero structural features.
- The benchmark includes 38 pytest tests and chart outputs for speed versus coverage.

### [explaining-cog](https://github.com/nobodyme/auto-research/tree/main/explaining-cog) (2026-07-18)

A short explainer of [Cog](https://github.com/nedbat/cog), Ned Batchelder's tool for embedding Python inside Markdown or other files so a document's visible content is computed from its source. The note covers how code sits between HTML-comment markers and its printed output is spliced back into the file, the three flags that cover most workflows (`-r` to re-render in place, `-d` to delete output, `--check` to fail CI when output is stale), and why a single source of truth beats hand-maintained docs. It also walks through how this repository's root README is itself a Cog block. That block lists each project folder by first-commit date, reuses a cached `_summary.md` or generates one with the [`llm`](https://llm.datasette.io/) CLI, and stamps an AI-generated disclaimer into each project README. It closes by noting when a fuller doc system like MkDocs or Sphinx is the better choice.

### [pdf-extraction-research](https://github.com/nobodyme/auto-research/tree/main/pdf-extraction-research) (2026-02-27)

Four Python text extractors (PyMuPDF, pdfplumber, pypdf, pdfminer.six) and four table extractors (including camelot lattice and stream) were benchmarked test-first across 5 synthetic PDFs with ground truth and 4 real arXiv papers, backed by 62 pytest checks. On digital PDFs all four text extractors scored identically (0.967 overall, 95% keyword recall, 97.2% number preservation), so speed is the deciding factor. [PyMuPDF](https://pymupdf.readthedocs.io/) was 23x faster than pdfplumber. For tables, PyMuPDF and pdfplumber tied on accuracy, while camelot was slower, over-split tables, and in lattice mode skipped colour-shaded header rows, a defect the tests caught. For scanned or layout-heavy documents the report points to AI-based tools such as [Docling](https://github.com/docling-project/docling) and MinerU, and summarises public benchmarks like OmniDocBench and DP-Bench.

- Text: PyMuPDF 0.11 s average versus pdfplumber 2.48 s, with identical accuracy.
- Tables: PyMuPDF 0.919 score and 100% detection at 1.48 s; camelot-lattice 0.882 at 7.58 s.
- Recommendation: PyMuPDF (or `pymupdf4llm`) for LLM preprocessing pipelines.

### [claude-code-research](https://github.com/nobodyme/auto-research/tree/main/claude-code-research) (2026-01-14)

Leveraging LangChain and LangGraph, this project investigates Claude Code’s dynamic agent architecture, which enables modular, extensible AI assistants using a dual-layer approach: “skills” as reusable, metadata-driven instruction sets, and “subagents” as isolated agent processes for parallel task execution. The implementation demonstrates progressive skill loading (minimizing token usage), strict tool restrictions for security, and orchestrated parallel subagent workflows for complex tasks—all discoverable via filesystem. By integrating these features, developers can build AI agents with specialized knowledge injection, context isolation, and efficient routing, as detailed in the [Claude Code Skills documentation](https://code.claude.com/docs/en/skills) and exemplified through [official skill examples](https://github.com/anthropics/skills).

**Key Findings:**
- Skills rely on YAML metadata for model-driven invocation and can restrict tool access.
- Subagents operate in parallel, each having fresh context and tool configuration, but cannot recursively spawn.
- Progressive loading of skills conserves tokens and resources, supporting efficient scalability.
- The orchestrator pattern centralizes state management and result aggregation for dynamic agent workflows.

### [langgraph-aws-mcp](https://github.com/nobodyme/auto-research/tree/main/langgraph-aws-mcp) (2026-01-07)

Leveraging LangGraph and the official AWS Model Context Protocol (MCP) servers, this agent automates the diagnosis and troubleshooting of AWS application issues by integrating with CloudWatch, ECS, and DynamoDB. It parses issue descriptions, investigates logs and resource states, and uses Claude Sonnet 4.5 to synthesize root cause analysis and actionable remediation steps. The entire workflow operates in read-only mode for safety and transparency, requiring no changes to AWS resources. Installation is user-friendly and includes robust tests for MCP client interaction, service integration, and end-to-end diagnostics. For more details on the underlying components, see [LangGraph](https://github.com/langchain-ai/langgraph) and [AWS MCP Servers](https://github.com/awslabs/mcp).

**Key findings:**
- The agent reliably identifies issues such as ECS resource misconfigurations and DynamoDB timeouts, providing targeted recommendations.
- Read-only architecture supports safe operation even in production AWS environments.
- Comprehensive automated tests ensure stability across log analysis, compute, and database layers.

### [fine-tune-rf-detr-using-sam3](https://github.com/nobodyme/auto-research/tree/main/fine-tune-rf-detr-using-sam3) (2026-01-05)

A command-line tool that runs [RF-DETR](https://github.com/roboflow/rf-detr) object detection over a folder of images and can fine-tune the detector from a plain-text prompt. In detection mode it loads the Nano, Small or Medium variant, filters by confidence, and writes annotated images plus a JSON file of boxes. Pass `--prompt` (for example "detect yellow school buses") and it switches to fine-tuning mode. [SAM3](https://github.com/facebookresearch/sam3) labels the matching objects, the detections become COCO-format annotations, and RF-DETR is trained on them with configurable epochs, batch size and learning rate, after which the fine-tuned model runs detection and can be saved. The CLI, detector, SAM3 integration and fine-tuning pipeline each have their own pytest module.

### [comparison-of-object-detection-models](https://github.com/nobodyme/auto-research/tree/main/comparison-of-object-detection-models) (2026-01-05)

Delivering a thorough head-to-head analysis of modern object detection models, this project benchmarks YOLOv8, YOLOv10, YOLOv11, RT-DETR, and RF-DETR across both COCO and RF100-VL datasets, targeting low-latency, near real-time deployment. Performance metrics such as mean Average Precision (mAP), frames per second (FPS), and inference latency are compared graphically, helping pinpoint optimal models for balanced speed and accuracy. Notably, transformer-based RF-DETR models lead in COCO mAP, while YOLOv10 sets a new standard in low and stable latency. All evaluated models support fine-tuning for custom domains, with results and code accessible for reproducible experimentation.

Key findings:
- [RF-DETR](https://github.com/roboflow/rf-detr) achieves the highest accuracy (54.7% mAP) at real-time speeds; RF-DETR-N reaches 2.32ms latency.
- [YOLOv10](https://github.com/THU-MIG/yolov10) is 1.8× faster than RT-DETR-R18 with similar accuracy and offers stable latency.
- YOLOv8 combines proven stability and documentation, while RT-DETR excels in complex scene understanding.
- All models support batch detection, automated annotation, and robust visualization for efficient workflow integration.

### [exploring-cookies](https://github.com/nobodyme/auto-research/tree/main/exploring-cookies) (2026-01-05)

Modern HTTP cookies are an HTTP state mechanism (via `Set-Cookie`/`Cookie`) whose real power comes from scoping and security attributes—`Domain`, `Path`, `Expires/Max-Age`, plus `Secure`, `HttpOnly`, and `SameSite`—and from stricter naming rules like `__Host-` and `__Secure-` that reduce common attacks. Beyond session management and personalization, cookies historically enabled cross-site tracking through third-party embeds and tactics like cookie syncing, link decoration, CNAME cloaking, and bounce flows, increasingly supplemented by fingerprinting when cookies are constrained. Today’s practical reality is a patchwork: strong privacy regimes (GDPR/CCPA/CPRA and state laws) demand consent and transparency, while browsers have curtailed third-party tracking via blocking and partitioning, pushing developers toward first-party data, server-side measurement, and partitioned approaches like CHIPS. For developers, the takeaway is to keep cookies small and short-lived, treat them as untrusted input, and apply defense-in-depth against CSRF/XSS/session hijacking with `SameSite`, CSRF tokens, and hardened session cookies. Key references: [RFC 6265](https://datatracker.ietf.org/doc/html/rfc6265), [CHIPS (Partitioned cookies)](https://developer.chrome.com/docs/privacy-sandbox/chips/).

* **Best-practice session cookie:** `__Host-` + `Secure` + `HttpOnly` + `SameSite=Strict|Lax` + `Path=/`, short `Max-Age`, server-side sessions, rotate/regenerate on login.
* **Tracking trendline:** third-party cookies are increasingly blocked/partitioned; plan around first-party identity, consented measurement, and server-side integrations.
* **Security reminder:** `Path` is not a security boundary; protect against cookie tossing/session fixation by avoiding broad `Domain` and using `__Host-` where possible.

<!--[[[end]]]-->

---
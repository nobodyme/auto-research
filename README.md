# Research projects carried out by AI tools

Each directory in this repo is a separate research project carried out by an LLM tool - usually [Claude Code](https://www.claude.com/product/claude-code). *Inspired by the blog from [Simon Willison](https://simonwillison.net/2025/Nov/6/async-code-research/)*

<!--[[[cog
import os
import re
import subprocess
import pathlib
from datetime import datetime, timezone

# Model to use for generating summaries
MODEL = "github/gpt-4.1"

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
## 3 research projects

### [langgraph-aws-mcp](https://github.com/nobodyme/auto-research/tree/main/langgraph-aws-mcp) (2026-01-07)

Leveraging LangGraph and the official AWS Model Context Protocol (MCP) servers, this agent automates the diagnosis and troubleshooting of AWS application issues by integrating with CloudWatch, ECS, and DynamoDB. It parses issue descriptions, investigates logs and resource states, and uses Claude Sonnet 4.5 to synthesize root cause analysis and actionable remediation steps. The entire workflow operates in read-only mode for safety and transparency, requiring no changes to AWS resources. Installation is user-friendly and includes robust tests for MCP client interaction, service integration, and end-to-end diagnostics. For more details on the underlying components, see [LangGraph](https://github.com/langchain-ai/langgraph) and [AWS MCP Servers](https://github.com/awslabs/mcp).

**Key findings:**
- The agent reliably identifies issues such as ECS resource misconfigurations and DynamoDB timeouts, providing targeted recommendations.
- Read-only architecture supports safe operation even in production AWS environments.
- Comprehensive automated tests ensure stability across log analysis, compute, and database layers.

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
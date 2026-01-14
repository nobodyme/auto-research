# Claude Code Dynamic Skills and Subagents - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Skills System](#skills-system)
3. [Subagents System](#subagents-system)
4. [How to Add Custom Skills](#how-to-add-custom-skills)
5. [How to Add Custom Subagents](#how-to-add-custom-subagents)
6. [Building a Dynamic Agent System](#building-a-dynamic-agent-system)
7. [Architecture Patterns](#architecture-patterns)
8. [Best Practices](#best-practices)

---

## Overview

Claude Code implements two powerful mechanisms for extending agent capabilities dynamically:

1. **Skills** - Model-invoked instruction sets that teach Claude specialized workflows
2. **Subagents** - Isolated agent instances spawned via the Task tool for parallel task execution

Both systems enable dynamic, extensible behavior without modifying core agent code.

---

## Skills System

### What Are Skills?

Skills are specialized reusable instruction sets packaged as folders containing:
- A `SKILL.md` file (core instructions with YAML frontmatter)
- Optional supporting resources (scripts, documentation, templates)

**Key characteristic**: Skills are **model-invoked** - Claude automatically decides when to use them based on your request, unlike slash commands which require manual invocation.

### Skill Structure

```
my-skill/
├── SKILL.md                    # Required: Core instructions
│   ├── YAML frontmatter       # Metadata (name, description, tools)
│   └── Markdown instructions  # What Claude should do
├── scripts/                   # Optional: Executable helpers
│   ├── helper.py
│   └── deploy.sh
├── references/                # Optional: Detailed documentation
│   └── detailed-guide.md
└── assets/                    # Optional: Templates, binaries
    └── template.docx
```

### SKILL.md Format

```yaml
---
name: my-skill-name
description: Clear description of what this skill does and when Claude should use it
allowed-tools: Read, Grep, Glob    # Optional: restrict tool access
context: fork                       # Optional: run in isolated sub-agent
user-invocable: false              # Optional: hide from /skills/ menu
hooks:                             # Optional: lifecycle hooks
  Stop: [...]
---

# My Skill Name

[Your detailed instructions here that Claude follows when this skill is activated]

## When to Use
- Scenario 1
- Scenario 2

## Steps
1. First step
2. Second step

## Examples
Example usage patterns...
```

### Storage Locations

| Scope | Location | Sharing | Purpose |
|-------|----------|---------|---------|
| **Project** | `.claude/skills/*/SKILL.md` | Via git | Team-wide |
| **Personal** | `~/.claude/skills/*/SKILL.md` | Local only | All projects |
| **Built-in** | Anthropic-provided | Built-in | All users (PDF, DOCX, XLSX, PPTX) |

### Invocation Flow

1. **Startup**: Claude loads skill metadata (~100 tokens/skill) into system prompt
2. **Request Analysis**: Claude matches user intent against skill descriptions
3. **Skill Trigger**: Claude calls Skill tool with skill name
4. **Content Loading**: Full SKILL.md instructions load into context
5. **Resource Loading**: Referenced files load on-demand

### Skills vs Slash Commands

| Feature | Skills | Slash Commands |
|---------|--------|----------------|
| **Triggered by** | Claude (automatic) | User (manual `/cmd`) |
| **Format** | Directory with resources | Single `.md` file |
| **Best for** | Complex, context-dependent workflows | Repetitive, atomic actions |
| **Tool restriction** | Supported | Not supported |
| **Context loading** | Progressive (~100 tokens metadata) | Full file on invoke |

---

## Subagents System

### What Are Subagents?

Subagents are specialized AI assistants that run in isolated contexts:
- Each gets a fresh 200k token context window
- Custom system prompt and tool access
- Independent permissions (inherit parent, auto-deny unpre-approved)
- Cannot spawn their own subagents (no nested delegation)

### The Task Tool

The Task tool is Claude's delegation mechanism for spawning subagents:

```json
{
  "name": "Task",
  "parameters": {
    "subagent_type": "string",    // Required: type of agent
    "description": "string",       // Short task description
    "prompt": "string",            // Detailed instructions
    "model": "string",             // Optional: sonnet, opus, haiku
    "resume": "string"             // Optional: resume existing agent
  }
}
```

### Built-in Subagent Types

| Type | Purpose | Tools | Use Case |
|------|---------|-------|----------|
| **Explore** | Codebase search | Glob, Grep, Read | Finding files, understanding code |
| **Plan** | Architecture planning | All tools | Designing implementation strategy |
| **general-purpose** | Multi-step tasks | All tools | Complex research, code search |
| **claude-code-guide** | Documentation lookup | Glob, Grep, Read, WebFetch | Questions about Claude Code |

### Main Agent vs Subagent

| Aspect | Main Agent | Subagent |
|--------|-----------|----------|
| Context | Single shared | Isolated 200k |
| Spawning | Can spawn subagents | Cannot spawn subagents |
| Tool Access | Full suite | Configurable |
| MCP Tools | Available | NOT in background |
| State | Global | Isolated |

### State Management

- Subagents maintain separate contexts from main agent
- Information flows back via TaskOutput transcript
- File-based coordination for complex state sharing
- Subagent resumption available (retains full history)

---

## How to Add Custom Skills

### Step 1: Create Skill Directory

```bash
# Project skill (shared with team)
mkdir -p .claude/skills/my-skill

# Personal skill (available everywhere)
mkdir -p ~/.claude/skills/my-skill
```

### Step 2: Create SKILL.md

```yaml
---
name: my-skill
description: |
  When to use: Describe situations where Claude should activate this skill.
  What it does: Clear explanation of the skill's purpose.
allowed-tools: Read, Write, Bash  # Optional: tool restrictions
---

# My Custom Skill

## Purpose
Explain what this skill accomplishes.

## When to Activate
- Trigger condition 1
- Trigger condition 2

## Instructions
1. First, do this
2. Then, do that
3. Finally, complete with this

## Examples
### Example 1
User asks: "..."
Response approach: ...
```

### Step 3: Add Supporting Resources (Optional)

```bash
# Add helper scripts
mkdir -p .claude/skills/my-skill/scripts
echo '#!/bin/bash\n# Helper script' > .claude/skills/my-skill/scripts/helper.sh

# Add reference documentation
mkdir -p .claude/skills/my-skill/references
echo '# Detailed Guide' > .claude/skills/my-skill/references/guide.md
```

### Step 4: Enable in Settings (if needed)

```json
// settings.json
{
  "permissions": {
    "allow": ["Skill"]
  }
}
```

### Best Practices for Skills

1. **Keep SKILL.md concise** (<500 lines, <5000 words)
2. **Write clear descriptions** - Claude uses these to match intent
3. **Use progressive disclosure** - Put details in reference files
4. **Restrict tools when appropriate** - Use `allowed-tools` for safety
5. **Test activation triggers** - Ensure Claude activates when expected

---

## How to Add Custom Subagents

### Method 1: Filesystem (Recommended for Teams)

```bash
# Project subagent (shared)
mkdir -p .claude/agents

# Create subagent definition
cat > .claude/agents/code-reviewer.md << 'EOF'
---
name: code-reviewer
description: Expert code reviewer for quality, security, and best practices analysis
tools: Read, Grep, Glob, Bash
model: sonnet
disallowedTools: Task
---

You are a senior code reviewer with expertise in:
- Security vulnerability detection
- Performance optimization
- Code quality and maintainability
- Best practices enforcement

## Review Process
1. Run `git diff` to see changes
2. Analyze each changed file
3. Check for security issues
4. Identify performance concerns
5. Verify coding standards
6. Provide actionable feedback

## Output Format
Provide a structured review with:
- Summary of changes
- Issues found (Critical/Warning/Info)
- Specific recommendations
- Approval status
EOF
```

### Method 2: Programmatic (SDK)

```python
from anthropic import Anthropic

agents = {
    "code-reviewer": {
        "description": "Reviews code for quality, security, and best practices",
        "prompt": """You are a senior code reviewer...""",
        "tools": ["Read", "Grep", "Glob", "Bash"]
    },
    "test-runner": {
        "description": "Executes tests and analyzes results",
        "prompt": """You are a test execution specialist...""",
        "tools": ["Bash", "Read", "Grep"]
    }
}

# Use in Claude Agent SDK
client = Anthropic()
# ... configure agents parameter
```

### Method 3: CLI Flag

```bash
claude --agents '[
  {
    "name": "security-scanner",
    "description": "Scans code for security vulnerabilities",
    "prompt": "You are a security expert...",
    "tools": ["Read", "Grep", "Glob"]
  }
]'
```

### Subagent Definition Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier |
| `description` | Yes | When to use (Claude reads this) |
| `prompt` | No | System prompt (body of .md file) |
| `tools` | No | Allowed tools (omit = inherit all) |
| `model` | No | Model selection (sonnet, opus, haiku) |
| `disallowedTools` | No | Tools to block |

### Tool Access Patterns

```yaml
# Read-only analysis
tools: Read, Grep, Glob

# Full development
tools: Read, Write, Edit, Bash, Glob, Grep

# Research only
tools: Read, Grep, Glob, WebFetch, WebSearch

# Inherit all (default when omitted)
# tools: (not specified)
```

---

## Building a Dynamic Agent System

### Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                    │
│  - Receives user requests                               │
│  - Routes to appropriate skills/subagents               │
│  - Manages global state                                 │
│  - Consolidates results                                 │
└─────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                 ┌─────────────────────┐
│  Skill Registry │                 │  Subagent Registry  │
│  - Load skills  │                 │  - Load definitions │
│  - Match intent │                 │  - Spawn agents     │
│  - Inject code  │                 │  - Manage lifecycle │
└─────────────────┘                 └─────────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                 ┌─────────────────────┐
│ Skill Execution │                 │  Subagent Execution │
│ (Same context)  │                 │ (Isolated context)  │
└─────────────────┘                 └─────────────────────┘
```

### Key Components

1. **Skill Registry**
   - Scans filesystem for skills
   - Parses YAML frontmatter
   - Provides descriptions for intent matching
   - Loads content on activation

2. **Subagent Registry**
   - Loads subagent definitions
   - Validates tool configurations
   - Manages spawning lifecycle
   - Enforces isolation

3. **Router/Orchestrator**
   - Matches requests to skills/subagents
   - Decides execution strategy (skill vs subagent)
   - Handles parallel execution
   - Aggregates results

4. **State Manager**
   - Maintains global context
   - Coordinates file-based state sharing
   - Supports subagent resumption

### Implementation with LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    skills_registry: dict
    subagents_registry: dict
    active_skill: str | None
    pending_subagents: list

# Define nodes
def router(state: AgentState) -> str:
    """Route to skill, subagent, or direct response"""
    # Analyze request and match to registered skills/subagents
    ...

def skill_executor(state: AgentState) -> AgentState:
    """Execute skill in current context"""
    ...

def subagent_spawner(state: AgentState) -> AgentState:
    """Spawn subagent with isolated context"""
    ...

def result_aggregator(state: AgentState) -> AgentState:
    """Combine results from skills/subagents"""
    ...

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router)
workflow.add_node("skill_executor", skill_executor)
workflow.add_node("subagent_spawner", subagent_spawner)
workflow.add_node("aggregator", result_aggregator)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_decision)
workflow.add_edge("skill_executor", "aggregator")
workflow.add_edge("subagent_spawner", "aggregator")
workflow.add_edge("aggregator", END)

app = workflow.compile()
```

---

## Architecture Patterns

### Pattern 1: Progressive Skill Loading

```
Startup:     Load metadata only (~100 tokens/skill)
On Request:  Match intent to skill descriptions
On Match:    Load full SKILL.md (<5K tokens)
On Demand:   Load referenced resources as needed
```

### Pattern 2: Isolated Subagent Execution

```
Main Agent:  200k context - orchestration
Subagent 1:  200k context - specific task
Subagent 2:  200k context - parallel task
...
Result:      Summaries flow back to main agent
```

### Pattern 3: Parallel Subagent Execution

```python
# Spawn multiple subagents in single message
subagent_tasks = [
    {"type": "code-reviewer", "prompt": "Review changes"},
    {"type": "security-scanner", "prompt": "Scan for vulnerabilities"},
    {"type": "test-runner", "prompt": "Run test suite"}
]

# All execute concurrently
results = await asyncio.gather(*[spawn_subagent(t) for t in subagent_tasks])
```

### Pattern 4: Tool Restriction Enforcement

```python
def filter_tools(base_tools: list, allowed_tools: list) -> list:
    """Enforce tool restrictions from skill/subagent definition"""
    if allowed_tools:
        return [t for t in base_tools if t.name in allowed_tools]
    return base_tools
```

---

## Best Practices

### For Skills

1. **Clear descriptions** - Quality determines activation accuracy
2. **Focused scope** - One skill, one purpose
3. **Progressive disclosure** - Keep core instructions lean
4. **Tool restrictions** - Minimize attack surface
5. **Hot-reload friendly** - Test changes quickly

### For Subagents

1. **Single responsibility** - One agent, one job
2. **Explicit tool access** - Use least-privilege
3. **No nested spawning** - Don't include Task in subagent tools
4. **File-based coordination** - For complex state sharing
5. **Clear descriptions** - Claude uses these for delegation decisions

### For Orchestration

1. **Keep orchestrator lean** - Mostly read and route
2. **Explicit delegation** - Tell Claude when to use subagents
3. **Shared conventions** - Document in CLAUDE.md
4. **Parallel when possible** - Maximize throughput
5. **Monitor context usage** - Prune when needed

---

## References

### Official Documentation
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk)

### Community Resources
- [GitHub - anthropics/skills](https://github.com/anthropics/skills)
- [GitHub - awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
- [Claude Agent Skills Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)
- [Inside Claude Code Skills](https://mikhail.io/2025/10/claude-code-skills/)

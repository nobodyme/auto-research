# Claude Code Dynamic Skills and Subagents Research

Research and implementation of Claude Code's dynamic agent architecture using LangChain/LangGraph.

## Requirements

- Python 3.11+
- `langchain>=1.2.3`
- `langgraph>=1.0.5`
- `langchain-anthropic>=0.3.0`

## Overview

This research explores how Claude Code integrates **dynamic skills** and **subagents** to create extensible, powerful AI assistants. The key insight is that Claude Code uses a two-tier extension system:

1. **Skills** - Model-invoked instruction sets that inject specialized knowledge
2. **Subagents** - Isolated agent instances for parallel task execution

## Key Findings

### Skills System

| Aspect | Description |
|--------|-------------|
| **What** | Reusable instruction sets in `SKILL.md` files with YAML frontmatter |
| **Where** | `.claude/skills/` (project) or `~/.claude/skills/` (user) |
| **Trigger** | Model-invoked (Claude decides when to activate based on description) |
| **Loading** | Progressive: metadata (~100 tokens) → full content → resources |
| **Tools** | Can restrict via `allowed-tools` frontmatter |

#### Skill Structure
```
my-skill/
├── SKILL.md           # Required: Instructions + YAML metadata
├── scripts/           # Optional: Helper scripts
├── references/        # Optional: Detailed docs
└── assets/            # Optional: Templates
```

### Subagents System

| Aspect | Description |
|--------|-------------|
| **What** | Isolated agent instances with custom tools and prompts |
| **Spawn** | Via Task tool with `subagent_type` parameter |
| **Context** | Each gets fresh 200k token window |
| **Nesting** | Cannot spawn other subagents (prevents infinite recursion) |
| **Execution** | Supports parallel execution for speed |

#### Built-in Subagent Types
- **Explore** - Fast codebase search (Haiku-powered, read-only)
- **Plan** - Architecture and implementation planning
- **general-purpose** - Multi-step complex tasks

### Skills vs Subagents

| Feature | Skills | Subagents |
|---------|--------|-----------|
| Context | Same as main | Isolated 200k tokens |
| Invocation | Inject instructions | Spawn separate agent |
| Use case | Specialized knowledge | Parallel/heavy tasks |
| Tool access | Restrict in frontmatter | Configure per agent |

## Implementation

### Files

| File | Description |
|------|-------------|
| `dynamic_agent.py` | Main implementation with LangGraph workflow |
| `parallel_subagents.py` | Parallel subagent execution patterns |
| `test_dynamic_agent.py` | Comprehensive test suite (31 tests) |
| `docs.md` | Comprehensive documentation |
| `notes.md` | Research notes and process |
| `examples/` | Sample skills and agent definitions |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                    │
│  - Routes requests to skills/subagents                  │
│  - Manages global state                                 │
│  - Aggregates results                                   │
└─────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                 ┌─────────────────────┐
│  Skill Registry │                 │  Subagent Registry  │
│  - Discover     │                 │  - Define agents    │
│  - Match intent │                 │  - Spawn isolated   │
│  - Load content │                 │  - Filter tools     │
└─────────────────┘                 └─────────────────────┘
```

### Key Components

1. **SkillRegistry** - Discovers and loads skills from filesystem with progressive loading
2. **SubagentRegistry** - Manages subagent definitions and spawning with tool filtering
3. **LangGraph Workflow** - Orchestrates routing, execution, and result aggregation

## Usage

### Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

### Running Tests

```bash
# Run all tests
pytest test_dynamic_agent.py -v

# Run specific test class
pytest test_dynamic_agent.py::TestSkillRegistry -v
```

### Basic Usage

```python
from dynamic_agent import create_dynamic_agent

# Create agent with skill/subagent support
app, skills, agents = create_dynamic_agent(api_key="...")

# Execute with state
result = app.invoke({
    "messages": [HumanMessage(content="Review this code for security issues")],
    "skills_registry": {},
    "subagents_registry": {},
    "active_skill": None,
    "active_subagent": None,
    "subagent_results": [],
    "pending_tasks": [],
    "route_decision": None
})
```

### Adding Custom Skills

Create `.claude/skills/my-skill/SKILL.md`:

```yaml
---
name: my-skill
description: When to activate this skill and what it does
allowed-tools: read_file, search_files
---

# My Skill Instructions

[Detailed instructions for Claude to follow]
```

### Adding Custom Subagents

Create `.claude/agents/my-agent.md`:

```yaml
---
name: my-agent
description: What this agent specializes in
tools: read_file, write_file
model: claude-sonnet-4-20250514
---

You are a specialized agent for...

[System prompt instructions]
```

## Key Insights for Building Dynamic Agents

### 1. Progressive Loading
- Load only metadata at startup (~100 tokens/skill)
- Load full content only when activated
- Load resources on-demand

### 2. Tool Restrictions
- Skills can restrict tools via `allowed-tools`
- Subagents can use `tools` (allowlist) or `disallowedTools` (denylist)
- Always remove Task tool from subagents (prevents nesting)

### 3. Parallel Execution
- Spawn multiple subagents in single message
- Use thread pool for concurrent execution
- Aggregate results in orchestrator

### 4. State Isolation
- Each subagent gets fresh context
- Use file-based coordination for complex state
- Orchestrator maintains global plan

## References

### Official Documentation
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk)

### Community Resources
- [anthropics/skills](https://github.com/anthropics/skills) - Official skill examples
- [awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents) - 99+ subagent collection
- [Claude Agent Skills Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)

## Conclusion

Claude Code's architecture provides a flexible, extensible system for building powerful AI agents:

1. **Skills** enable specialized knowledge injection without code changes
2. **Subagents** enable parallel execution with context isolation
3. **Progressive loading** minimizes token usage
4. **Tool restrictions** provide security boundaries

The LangChain/LangGraph implementation in this repository demonstrates how to build a similar system with:
- Filesystem-based skill/agent discovery
- Automatic intent matching
- Parallel subagent execution
- Result aggregation

This architecture pattern is applicable to any AI agent system requiring extensibility and specialization.

# Research Notes: Claude Code Dynamic Skills and Subagents

## Research Session - January 14, 2026

### Objective
Understand how Claude Code integrates dynamic skills and subagents, with the goal of implementing a similar system using LangChain/LangGraph.

---

## Research Process

### Phase 1: Skills System Research

**What I investigated:**
- Skills definition and structure (SKILL.md files with YAML frontmatter)
- Storage locations (project `.claude/skills/`, user `~/.claude/skills/`)
- Invocation mechanism (model-invoked via Skill tool)
- Discovery and loading process (progressive disclosure)

**Key findings:**
1. Skills are model-invoked, not user-invoked like slash commands
2. Three-level loading: metadata (~100 tokens) → instructions (<5K tokens) → resources (on-demand)
3. Skills can restrict tool access via `allowed-tools` frontmatter
4. Hot-reload supported in Claude Code 2.1.0+

**Sources consulted:**
- Claude Code official docs (code.claude.com)
- Anthropic engineering blog
- Community deep-dive articles (leehanchung.github.io, mikhail.io)

### Phase 2: Subagents/Task Tool Research

**What I investigated:**
- Task tool architecture and spawning mechanism
- Available subagent types (Explore, Plan, General-purpose, Custom)
- Tool access restrictions per subagent type
- State management between main agent and subagents
- Custom subagent definition methods

**Key findings:**
1. Subagents run in isolated 200k token context windows
2. Critical constraint: subagents cannot spawn other subagents (no nested delegation)
3. Three definition methods: filesystem (Markdown+YAML), programmatic, CLI flag
4. MCP tools NOT available in background subagents
5. Parallel execution possible (multiple Task calls in single message)

**Sources consulted:**
- Claude Code docs (code.claude.com/docs/en/sub-agents)
- SDK documentation (platform.claude.com)
- Community best practices articles

---

## Architecture Insights

### Skills Architecture
```
┌─────────────────────────────────────────────────┐
│                  System Prompt                   │
│  (Contains skill metadata: ~100 tokens/skill)   │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│              User Request Analysis              │
│    (Claude matches intent → skill description)  │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│               Skill Tool Invocation             │
│        (Load full SKILL.md into context)        │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│              Resource Loading (On-demand)        │
│   (Scripts, references, assets as needed)       │
└─────────────────────────────────────────────────┘
```

### Subagent Architecture
```
┌─────────────────────────────────────────────────┐
│                Main Agent (Orchestrator)         │
│          - Global planning and coordination     │
│          - Task delegation via Task tool        │
│          - State management                     │
└─────────────────────────────────────────────────┘
            ↓ Task Tool Invocation ↓
┌──────────────┬──────────────┬──────────────────┐
│  Subagent 1  │  Subagent 2  │   Subagent N     │
│  (200k ctx)  │  (200k ctx)  │   (200k ctx)     │
│  [Tools: A]  │  [Tools: B]  │   [Tools: C]     │
└──────────────┴──────────────┴──────────────────┘
            ↓ Results Return ↓
┌─────────────────────────────────────────────────┐
│         Main Agent Consolidates Results         │
└─────────────────────────────────────────────────┘
```

---

## Implementation Considerations for LangChain/LangGraph

### Core Components Needed:

1. **Skill Registry**
   - Load skill metadata from filesystem
   - Support project and user-level skills
   - Provide skill descriptions to LLM for matching

2. **Skill Loader**
   - Progressive loading (metadata → full content → resources)
   - YAML frontmatter parsing
   - Tool restriction enforcement

3. **Subagent Orchestrator**
   - Spawn subagents with isolated state
   - Configure tool access per subagent
   - Support parallel execution
   - Handle result aggregation

4. **Task Router**
   - Match user requests to appropriate skills/subagents
   - Support custom subagent definitions
   - Prevent nested subagent spawning

### LangGraph Considerations:
- State management naturally fits StateGraph pattern
- Subagent isolation via separate graph instances
- Conditional routing for skill/subagent selection
- Checkpointing for long-running tasks

---

## Questions Answered

1. **What are skills?** - Model-invoked instruction sets with progressive loading
2. **What are subagents?** - Isolated agents with custom tools spawned via Task tool
3. **How to add custom skills?** - Create SKILL.md in `.claude/skills/` with frontmatter
4. **How to add custom subagents?** - Markdown in `.claude/agents/` or programmatically
5. **How to build dynamic agent system?** - Skill registry + subagent orchestrator pattern

---

## Next Steps
1. Create comprehensive docs.md
2. Implement LangChain/LangGraph solution:
   - Skill discovery and loading
   - Subagent spawning with tool restrictions
   - Dynamic routing based on task type
   - Parallel execution support

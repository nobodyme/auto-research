"""
Dynamic Agent System with Skills and Subagents

This implementation mirrors Claude Code's architecture for dynamic skills and subagents
using LangChain and LangGraph.

Architecture:
- Skill Registry: Discovers and loads skills from filesystem
- Subagent Registry: Manages subagent definitions and spawning
- Orchestrator: Routes requests to skills/subagents using LangGraph

Compatible with:
- langchain>=1.2.3
- langgraph>=1.0.5
"""

import os
import yaml
import asyncio
from pathlib import Path
from typing import TypedDict, Annotated, Literal, Optional, Any
from dataclasses import dataclass, field
import operator

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SkillDefinition:
    """Represents a loaded skill definition"""
    name: str
    description: str
    content: str
    allowed_tools: list[str] = field(default_factory=list)
    context: str = "inline"  # "inline" or "fork"
    path: Path = None

    @property
    def metadata_tokens(self) -> int:
        """Estimate tokens for metadata only (for progressive loading)"""
        return len(f"{self.name}: {self.description}".split()) * 2


@dataclass
class SubagentDefinition:
    """Represents a subagent definition"""
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"


class AgentState(TypedDict):
    """State passed through the LangGraph workflow"""
    messages: Annotated[list[BaseMessage], operator.add]
    skills_registry: dict[str, SkillDefinition]
    subagents_registry: dict[str, SubagentDefinition]
    active_skill: Optional[str]
    active_subagent: Optional[str]
    subagent_results: list[dict]
    pending_tasks: list[dict]
    route_decision: Optional[str]


# =============================================================================
# Skill Registry
# =============================================================================

class SkillRegistry:
    """
    Discovers and manages skills from the filesystem.

    Mirrors Claude Code's progressive loading:
    1. At startup: Load metadata only (~100 tokens/skill)
    2. On activation: Load full skill content
    3. On demand: Load referenced resources
    """

    def __init__(self, skill_paths: list[Path] = None):
        self.skill_paths = skill_paths or [
            Path(".claude/skills"),       # Project skills
            Path.home() / ".claude/skills"  # User skills
        ]
        self.skills: dict[str, SkillDefinition] = {}
        self._loaded_content: dict[str, str] = {}

    def discover_skills(self) -> dict[str, SkillDefinition]:
        """Discover skills from filesystem (metadata only for efficiency)"""
        for skill_path in self.skill_paths:
            if skill_path.exists():
                for skill_dir in skill_path.iterdir():
                    if skill_dir.is_dir():
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.exists():
                            skill = self._parse_skill_metadata(skill_file)
                            if skill and skill.name not in self.skills:
                                self.skills[skill.name] = skill
        return self.skills

    def _parse_skill_metadata(self, skill_file: Path) -> Optional[SkillDefinition]:
        """Parse only the YAML frontmatter for progressive loading"""
        try:
            content = skill_file.read_text()

            # Parse YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    markdown_content = parts[2].strip()

                    allowed_tools_str = frontmatter.get("allowed-tools", "")
                    if allowed_tools_str:
                        allowed_tools = [t.strip() for t in allowed_tools_str.split(",")]
                    else:
                        allowed_tools = []

                    return SkillDefinition(
                        name=frontmatter.get("name", skill_file.parent.name),
                        description=frontmatter.get("description", ""),
                        content=markdown_content,
                        allowed_tools=allowed_tools,
                        context=frontmatter.get("context", "inline"),
                        path=skill_file
                    )
        except Exception as e:
            print(f"Error parsing skill {skill_file}: {e}")
        return None

    def get_skill_descriptions(self) -> str:
        """Get skill metadata for system prompt (progressive loading - metadata only)"""
        descriptions = []
        for name, skill in self.skills.items():
            descriptions.append(f"- **{name}**: {skill.description}")
        return "\n".join(descriptions)

    def activate_skill(self, name: str) -> Optional[SkillDefinition]:
        """Load full skill content on activation"""
        if name in self.skills:
            skill = self.skills[name]
            return skill
        return None

    def match_skill(self, user_request: str, llm: ChatAnthropic) -> Optional[str]:
        """Use LLM to match user request to appropriate skill"""
        if not self.skills:
            return None

        prompt = f"""Given the user request and available skills, determine if a skill should be activated.

Available Skills:
{self.get_skill_descriptions()}

User Request: {user_request}

If a skill should be activated, respond with just the skill name.
If no skill is appropriate, respond with "none".
Only output the skill name or "none", nothing else."""

        response = llm.invoke([HumanMessage(content=prompt)])
        skill_name = response.content.strip().lower()

        if skill_name != "none" and skill_name in self.skills:
            return skill_name
        return None


# =============================================================================
# Subagent Registry
# =============================================================================

class SubagentRegistry:
    """
    Manages subagent definitions and spawning.

    Mirrors Claude Code's Task tool functionality:
    - Spawn agents with isolated context
    - Configure tool access per agent
    - Support parallel execution
    - Prevent nested subagent spawning
    """

    # Built-in subagent types (like Claude Code's Explore, Plan, etc.)
    BUILTIN_SUBAGENTS = {
        "explore": SubagentDefinition(
            name="explore",
            description="Fast codebase exploration agent for finding files, searching code, and answering questions about the codebase",
            system_prompt="""You are a fast, focused exploration agent. Your job is to:
1. Search for files matching patterns
2. Find code containing specific keywords
3. Answer questions about codebase structure

Be concise and efficient. Report findings clearly.""",
            tools=["read_file", "search_files", "list_directory"],
            model="claude-sonnet-4-20250514"
        ),
        "plan": SubagentDefinition(
            name="plan",
            description="Software architect agent for designing implementation plans and identifying critical files",
            system_prompt="""You are a software architect. Your job is to:
1. Analyze requirements
2. Design implementation strategies
3. Identify critical files and dependencies
4. Consider architectural trade-offs

Provide step-by-step plans with clear rationale.""",
            tools=["read_file", "search_files", "list_directory"],
            model="claude-sonnet-4-20250514"
        ),
        "general-purpose": SubagentDefinition(
            name="general-purpose",
            description="General-purpose agent for complex, multi-step tasks",
            system_prompt="""You are a general-purpose assistant capable of handling complex tasks.
Follow instructions carefully and complete tasks thoroughly.""",
            tools=[],  # Empty = inherit all tools
            model="claude-sonnet-4-20250514"
        )
    }

    def __init__(self, agent_paths: list[Path] = None):
        self.agent_paths = agent_paths or [
            Path(".claude/agents"),       # Project agents
            Path.home() / ".claude/agents"  # User agents
        ]
        self.subagents: dict[str, SubagentDefinition] = dict(self.BUILTIN_SUBAGENTS)

    def discover_subagents(self) -> dict[str, SubagentDefinition]:
        """Discover custom subagent definitions from filesystem"""
        for agent_path in self.agent_paths:
            if agent_path.exists():
                for agent_file in agent_path.glob("*.md"):
                    agent = self._parse_agent_definition(agent_file)
                    if agent:
                        self.subagents[agent.name] = agent
        return self.subagents

    def _parse_agent_definition(self, agent_file: Path) -> Optional[SubagentDefinition]:
        """Parse subagent definition from markdown file"""
        try:
            content = agent_file.read_text()

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    system_prompt = parts[2].strip()

                    tools_str = frontmatter.get("tools", "")
                    tools = [t.strip() for t in tools_str.split(",")] if tools_str else []

                    disallowed_str = frontmatter.get("disallowedTools", "")
                    disallowed = [t.strip() for t in disallowed_str.split(",")] if disallowed_str else []

                    return SubagentDefinition(
                        name=frontmatter.get("name", agent_file.stem),
                        description=frontmatter.get("description", ""),
                        system_prompt=system_prompt,
                        tools=tools,
                        disallowed_tools=disallowed,
                        model=frontmatter.get("model", "claude-sonnet-4-20250514")
                    )
        except Exception as e:
            print(f"Error parsing agent {agent_file}: {e}")
        return None

    def get_subagent_descriptions(self) -> str:
        """Get subagent descriptions for routing decisions"""
        descriptions = []
        for name, agent in self.subagents.items():
            descriptions.append(f"- **{name}**: {agent.description}")
        return "\n".join(descriptions)

    def spawn_subagent(
        self,
        name: str,
        task_prompt: str,
        available_tools: list[BaseTool],
        llm_factory: callable
    ) -> dict:
        """
        Spawn a subagent with isolated context.

        Key behaviors (matching Claude Code):
        - Fresh context window
        - Filtered tool access
        - Cannot spawn nested subagents
        """
        if name not in self.subagents:
            return {"error": f"Unknown subagent: {name}"}

        agent_def = self.subagents[name]

        # Filter tools based on agent definition
        filtered_tools = self._filter_tools(available_tools, agent_def)

        # CRITICAL: Remove Task tool from subagents (prevent nesting)
        filtered_tools = [t for t in filtered_tools if t.name != "spawn_subagent"]

        # Create isolated LLM instance
        llm = llm_factory(model=agent_def.model)

        # Execute subagent with isolated context
        result = self._execute_subagent(
            llm=llm,
            system_prompt=agent_def.system_prompt,
            task_prompt=task_prompt,
            tools=filtered_tools
        )

        return {
            "agent": name,
            "task": task_prompt,
            "result": result
        }

    def _filter_tools(
        self,
        available_tools: list[BaseTool],
        agent_def: SubagentDefinition
    ) -> list[BaseTool]:
        """Filter tools based on agent definition"""
        if agent_def.tools:
            # Explicit allowlist
            return [t for t in available_tools if t.name in agent_def.tools]
        elif agent_def.disallowed_tools:
            # Explicit denylist
            return [t for t in available_tools if t.name not in agent_def.disallowed_tools]
        else:
            # Inherit all tools
            return available_tools

    def _execute_subagent(
        self,
        llm: ChatAnthropic,
        system_prompt: str,
        task_prompt: str,
        tools: list[BaseTool]
    ) -> str:
        """Execute subagent in isolated context using LangGraph subgraph"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ]

        if tools:
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm

        # Simple execution loop
        response = llm_with_tools.invoke(messages)

        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                for t in tools:
                    if t.name == tool_call['name']:
                        result = t.invoke(tool_call['args'])
                        tool_results.append(f"{tool_call['name']}: {result}")

            messages.append(response)
            messages.append(HumanMessage(content=f"Tool results:\n" + "\n".join(tool_results)))
            response = llm_with_tools.invoke(messages)

        return response.content


# =============================================================================
# Tools
# =============================================================================

def create_tools(skill_registry: SkillRegistry, subagent_registry: SubagentRegistry):
    """Create tools for the agent"""

    @tool
    def activate_skill(skill_name: str) -> str:
        """Activate a skill to get detailed instructions for a specific task type.

        Args:
            skill_name: The name of the skill to activate
        """
        skill = skill_registry.activate_skill(skill_name)
        if skill:
            return f"""# Skill Activated: {skill.name}

{skill.content}

## Allowed Tools
{', '.join(skill.allowed_tools) if skill.allowed_tools else 'All tools available'}
"""
        return f"Skill '{skill_name}' not found"

    @tool
    def spawn_subagent(agent_type: str, task: str) -> str:
        """Spawn a specialized subagent to handle a specific task.

        Args:
            agent_type: The type of subagent to spawn (e.g., 'explore', 'plan', 'code-reviewer')
            task: The task description for the subagent
        """
        # This is a placeholder - actual spawning happens in the graph
        return f"SPAWN_SUBAGENT:{agent_type}:{task}"

    @tool
    def read_file(file_path: str) -> str:
        """Read the contents of a file.

        Args:
            file_path: Path to the file to read
        """
        try:
            path = Path(file_path)
            if path.exists():
                return path.read_text()[:10000]  # Limit for context
            return f"File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"

    @tool
    def search_files(pattern: str, directory: str = ".") -> str:
        """Search for files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py')
            directory: Directory to search in
        """
        try:
            path = Path(directory)
            matches = list(path.glob(pattern))[:20]  # Limit results
            return "\n".join(str(m) for m in matches) if matches else "No files found"
        except Exception as e:
            return f"Error searching: {e}"

    @tool
    def list_directory(directory: str = ".") -> str:
        """List contents of a directory.

        Args:
            directory: Directory to list
        """
        try:
            path = Path(directory)
            if path.is_dir():
                items = list(path.iterdir())[:50]
                return "\n".join(f"{'[DIR]' if i.is_dir() else '[FILE]'} {i.name}" for i in items)
            return f"Not a directory: {directory}"
        except Exception as e:
            return f"Error listing directory: {e}"

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file.

        Args:
            file_path: Path to write to
            content: Content to write
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

    return [activate_skill, spawn_subagent, read_file, search_files, list_directory, write_file]


# =============================================================================
# LangGraph Workflow (Updated for LangGraph 1.0.5+)
# =============================================================================

def create_dynamic_agent(
    api_key: str = None,
    skill_paths: list[Path] = None,
    agent_paths: list[Path] = None
):
    """
    Create a dynamic agent with skills and subagent support.

    This mirrors Claude Code's architecture:
    1. Skills are model-invoked based on user intent
    2. Subagents are spawned via Task tool for isolated execution
    3. Orchestrator coordinates everything

    Compatible with LangGraph 1.0.5+
    """

    # Initialize registries
    skill_registry = SkillRegistry(skill_paths)
    subagent_registry = SubagentRegistry(agent_paths)

    # Discover available skills and subagents
    skill_registry.discover_skills()
    subagent_registry.discover_subagents()

    # Create tools
    tools = create_tools(skill_registry, subagent_registry)

    # LLM factory for creating isolated instances
    def llm_factory(model: str = "claude-sonnet-4-20250514"):
        return ChatAnthropic(
            model=model,
            api_key=api_key
        )

    # Main LLM
    llm = llm_factory()

    # System prompt with skill/subagent metadata
    system_prompt = f"""You are a dynamic agent with access to skills and subagents.

## Available Skills
{skill_registry.get_skill_descriptions() or 'No skills loaded'}

When a user request matches a skill's purpose, use the activate_skill tool to load detailed instructions.

## Available Subagents
{subagent_registry.get_subagent_descriptions()}

Use the spawn_subagent tool to delegate complex or specialized tasks to subagents.
Subagents run in isolated contexts with their own tools - use them for:
- Parallel task execution
- Specialized analysis
- Context-heavy operations

## Guidelines
1. Match user requests to appropriate skills automatically
2. Delegate to subagents for complex multi-step tasks
3. Use tools efficiently
4. Provide clear, helpful responses
"""

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create tool node for executing tool calls
    tool_node = ToolNode(tools)

    # Define graph nodes
    def router(state: AgentState) -> AgentState:
        """Route incoming request - decide skill/subagent/direct"""
        messages = state["messages"]

        # Get last human message
        last_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_message = msg.content
                break

        if not last_message:
            return {**state, "route_decision": "respond"}

        # Check if a skill should be activated
        matched_skill = skill_registry.match_skill(last_message, llm)
        if matched_skill:
            return {**state, "active_skill": matched_skill, "route_decision": "skill"}
        else:
            return {**state, "route_decision": "agent"}

    def agent_node(state: AgentState) -> dict:
        """Main agent processing node"""
        messages = state["messages"]

        # Add system prompt
        full_messages = [SystemMessage(content=system_prompt)] + messages

        response = llm_with_tools.invoke(full_messages)

        # Check for subagent spawn requests
        pending_tasks = list(state.get("pending_tasks", []))
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'spawn_subagent':
                    pending_tasks.append({
                        "type": "subagent",
                        "agent": tool_call['args']['agent_type'],
                        "task": tool_call['args']['task']
                    })

        return {"messages": [response], "pending_tasks": pending_tasks}

    def skill_node(state: AgentState) -> dict:
        """Execute with activated skill"""
        skill_name = state.get("active_skill")
        if not skill_name:
            return state

        skill = skill_registry.activate_skill(skill_name)
        if not skill:
            return state

        # Augment system prompt with skill instructions
        skill_system_prompt = f"""{system_prompt}

## ACTIVE SKILL: {skill.name}
{skill.content}
"""
        messages = state["messages"]
        full_messages = [SystemMessage(content=skill_system_prompt)] + messages

        # Filter tools if skill restricts them
        if skill.allowed_tools:
            filtered_tools = [t for t in tools if t.name in skill.allowed_tools]
            skill_llm = llm.bind_tools(filtered_tools)
        else:
            skill_llm = llm_with_tools

        response = skill_llm.invoke(full_messages)
        return {"messages": [response], "active_skill": None}

    def subagent_node(state: AgentState) -> dict:
        """Execute pending subagent tasks"""
        pending = state.get("pending_tasks", [])
        results = list(state.get("subagent_results", []))

        for task in pending:
            if task["type"] == "subagent":
                result = subagent_registry.spawn_subagent(
                    name=task["agent"],
                    task_prompt=task["task"],
                    available_tools=tools,
                    llm_factory=llm_factory
                )
                results.append(result)

        return {"pending_tasks": [], "subagent_results": results}

    def tools_node(state: AgentState) -> dict:
        """Execute tool calls using ToolNode"""
        return tool_node.invoke(state)

    def should_continue(state: AgentState) -> Literal["tools", "subagent", "__end__"]:
        """Determine next step after agent node"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        # Check for pending subagent tasks
        if state.get("pending_tasks"):
            return "subagent"

        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Filter out subagent spawning (handled separately)
            non_spawn_calls = [t for t in last_message.tool_calls if t['name'] != 'spawn_subagent']
            if non_spawn_calls:
                return "tools"

        return "__end__"

    def route_from_router(state: AgentState) -> Literal["skill", "agent"]:
        """Route from router node"""
        return state.get("route_decision", "agent")

    # Build the graph using LangGraph 1.0.5+ API
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("agent", agent_node)
    workflow.add_node("skill", skill_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("subagent", subagent_node)

    # Set entry point using START (LangGraph 1.0.5+ pattern)
    workflow.add_edge(START, "router")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {"skill": "skill", "agent": "agent"}
    )

    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "subagent": "subagent", "__end__": END}
    )

    # Add edges for returning to agent
    workflow.add_edge("skill", "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("subagent", "agent")

    # Compile with memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app, skill_registry, subagent_registry


# =============================================================================
# Async Support for Parallel Subagent Execution
# =============================================================================

async def execute_subagents_parallel(
    subagent_registry: SubagentRegistry,
    tasks: list[dict],
    available_tools: list[BaseTool],
    llm_factory: callable
) -> list[dict]:
    """Execute multiple subagent tasks in parallel"""

    async def run_subagent(task: dict) -> dict:
        return subagent_registry.spawn_subagent(
            name=task["agent"],
            task_prompt=task["task"],
            available_tools=available_tools,
            llm_factory=llm_factory
        )

    # Run all tasks concurrently
    results = await asyncio.gather(*[run_subagent(t) for t in tasks])
    return list(results)


# =============================================================================
# Usage Example
# =============================================================================

def main():
    """Example usage of the dynamic agent"""

    # Create sample skills directory
    skills_dir = Path(".claude/skills/code-review")
    skills_dir.mkdir(parents=True, exist_ok=True)

    sample_skill = """---
name: code-review
description: Automated code review for quality, security, and best practices
allowed-tools: read_file, search_files, list_directory
---

# Code Review Skill

When reviewing code, follow these steps:

1. **Understand the changes**: Read the files that were modified
2. **Check for security issues**: Look for SQL injection, XSS, etc.
3. **Review code quality**: Check naming, structure, complexity
4. **Verify best practices**: Ensure coding standards are followed

## Output Format
Provide a structured review with:
- Summary of changes
- Issues found (Critical/Warning/Info)
- Recommendations
- Approval status (Approved/Changes Requested)
"""

    (skills_dir / "SKILL.md").write_text(sample_skill)

    # Create sample subagent
    agents_dir = Path(".claude/agents")
    agents_dir.mkdir(parents=True, exist_ok=True)

    sample_agent = """---
name: security-scanner
description: Scans code for security vulnerabilities and potential attack vectors
tools: read_file, search_files
model: claude-sonnet-4-20250514
---

You are a security expert specializing in code security analysis.

When scanning code:
1. Look for common vulnerabilities (OWASP Top 10)
2. Check for hardcoded secrets
3. Identify insecure dependencies
4. Review authentication/authorization logic

Provide findings in this format:
- Severity: Critical/High/Medium/Low
- Location: File and line number
- Description: What the issue is
- Remediation: How to fix it
"""

    (agents_dir / "security-scanner.md").write_text(sample_agent)

    # Create the dynamic agent
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return

    app, skill_registry, subagent_registry = create_dynamic_agent(api_key=api_key)

    print("Dynamic Agent initialized!")
    print(f"\nLoaded Skills: {list(skill_registry.skills.keys())}")
    print(f"Loaded Subagents: {list(subagent_registry.subagents.keys())}")

    # Example conversation
    config = {"configurable": {"thread_id": "demo-session"}}

    initial_state: AgentState = {
        "messages": [],
        "skills_registry": {},
        "subagents_registry": {},
        "active_skill": None,
        "active_subagent": None,
        "subagent_results": [],
        "pending_tasks": [],
        "route_decision": None
    }

    # Test with a request that should trigger the code-review skill
    print("\n--- Test 1: Skill Activation ---")
    test_state: AgentState = {
        **initial_state,
        "messages": [HumanMessage(content="Please review the code in this project for quality issues")]
    }

    result = app.invoke(test_state, config)
    print(f"Response: {result['messages'][-1].content[:500]}...")

    # Test with a request that should spawn a subagent
    print("\n--- Test 2: Subagent Spawning ---")
    test_state: AgentState = {
        **initial_state,
        "messages": [HumanMessage(content="Explore the codebase and find all Python files")]
    }

    result = app.invoke(test_state, config)
    print(f"Response: {result['messages'][-1].content[:500]}...")


if __name__ == "__main__":
    main()

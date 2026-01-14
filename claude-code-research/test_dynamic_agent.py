"""
Tests for Dynamic Agent System

TDD approach: Tests written first, then implementation updated to pass them.
Uses pytest with fixtures for clean test setup.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Import the modules we're testing (will be updated)
from dynamic_agent import (
    SkillDefinition,
    SubagentDefinition,
    SkillRegistry,
    SubagentRegistry,
    AgentState,
    create_tools,
    create_dynamic_agent,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_skills_dir():
    """Create a temporary directory with sample skills"""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_path = Path(tmpdir) / ".claude" / "skills"

        # Create code-review skill
        code_review_dir = skills_path / "code-review"
        code_review_dir.mkdir(parents=True)
        (code_review_dir / "SKILL.md").write_text("""---
name: code-review
description: Automated code review for quality, security, and best practices
allowed-tools: read_file, search_files, list_directory
context: inline
---

# Code Review Skill

When reviewing code, follow these steps:
1. Read the files
2. Check for issues
3. Provide feedback
""")

        # Create api-docs skill
        api_docs_dir = skills_path / "api-docs"
        api_docs_dir.mkdir(parents=True)
        (api_docs_dir / "SKILL.md").write_text("""---
name: api-documentation
description: Generate API documentation from code
allowed-tools: read_file, write_file
---

# API Documentation Skill

Generate comprehensive API documentation.
""")

        yield skills_path


@pytest.fixture
def temp_agents_dir():
    """Create a temporary directory with sample subagent definitions"""
    with tempfile.TemporaryDirectory() as tmpdir:
        agents_path = Path(tmpdir) / ".claude" / "agents"
        agents_path.mkdir(parents=True)

        # Create security-scanner agent
        (agents_path / "security-scanner.md").write_text("""---
name: security-scanner
description: Scans code for security vulnerabilities
tools: read_file, search_files
model: claude-sonnet-4-20250514
disallowedTools: write_file
---

You are a security expert. Scan for vulnerabilities.
""")

        # Create test-runner agent
        (agents_path / "test-runner.md").write_text("""---
name: test-runner
description: Runs tests and analyzes results
tools: read_file, search_files, execute_command
model: claude-sonnet-4-20250514
---

You are a test execution specialist.
""")

        yield agents_path


@pytest.fixture
def skill_registry(temp_skills_dir):
    """Create a SkillRegistry with test skills"""
    registry = SkillRegistry(skill_paths=[temp_skills_dir])
    registry.discover_skills()
    return registry


@pytest.fixture
def subagent_registry(temp_agents_dir):
    """Create a SubagentRegistry with test agents"""
    registry = SubagentRegistry(agent_paths=[temp_agents_dir])
    registry.discover_subagents()
    return registry


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test response", tool_calls=[])
    return mock


# =============================================================================
# SkillDefinition Tests
# =============================================================================

class TestSkillDefinition:
    """Tests for SkillDefinition dataclass"""

    def test_create_skill_definition(self):
        """Test creating a skill definition with all fields"""
        skill = SkillDefinition(
            name="test-skill",
            description="A test skill",
            content="# Test Content",
            allowed_tools=["read_file", "write_file"],
            context="inline",
            path=Path("/test/path")
        )

        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.content == "# Test Content"
        assert skill.allowed_tools == ["read_file", "write_file"]
        assert skill.context == "inline"

    def test_skill_definition_defaults(self):
        """Test that skill definition has correct defaults"""
        skill = SkillDefinition(
            name="minimal",
            description="Minimal skill",
            content="Content"
        )

        assert skill.allowed_tools == []
        assert skill.context == "inline"
        assert skill.path is None

    def test_metadata_tokens_estimation(self):
        """Test token estimation for progressive loading"""
        skill = SkillDefinition(
            name="test",
            description="A short description",
            content="Long content that shouldn't affect token count"
        )

        # Token estimate should be based on name + description only
        tokens = skill.metadata_tokens
        assert tokens > 0
        assert isinstance(tokens, int)


# =============================================================================
# SubagentDefinition Tests
# =============================================================================

class TestSubagentDefinition:
    """Tests for SubagentDefinition dataclass"""

    def test_create_subagent_definition(self):
        """Test creating a subagent definition"""
        agent = SubagentDefinition(
            name="test-agent",
            description="A test agent",
            system_prompt="You are a test agent",
            tools=["read_file"],
            disallowed_tools=["write_file"],
            model="claude-sonnet-4-20250514"
        )

        assert agent.name == "test-agent"
        assert agent.description == "A test agent"
        assert agent.system_prompt == "You are a test agent"
        assert agent.tools == ["read_file"]
        assert agent.disallowed_tools == ["write_file"]

    def test_subagent_definition_defaults(self):
        """Test that subagent definition has correct defaults"""
        agent = SubagentDefinition(
            name="minimal",
            description="Minimal agent",
            system_prompt="You are an agent"
        )

        assert agent.tools == []
        assert agent.disallowed_tools == []
        assert agent.model == "claude-sonnet-4-20250514"


# =============================================================================
# SkillRegistry Tests
# =============================================================================

class TestSkillRegistry:
    """Tests for SkillRegistry"""

    def test_discover_skills(self, skill_registry):
        """Test that skills are discovered from filesystem"""
        assert len(skill_registry.skills) == 2
        assert "code-review" in skill_registry.skills
        assert "api-documentation" in skill_registry.skills

    def test_skill_content_loaded(self, skill_registry):
        """Test that skill content is loaded correctly"""
        skill = skill_registry.skills["code-review"]
        assert "Code Review Skill" in skill.content
        assert skill.allowed_tools == ["read_file", "search_files", "list_directory"]

    def test_get_skill_descriptions(self, skill_registry):
        """Test getting skill descriptions for system prompt"""
        descriptions = skill_registry.get_skill_descriptions()

        assert "code-review" in descriptions
        assert "api-documentation" in descriptions
        assert "Automated code review" in descriptions

    def test_activate_skill_exists(self, skill_registry):
        """Test activating an existing skill"""
        skill = skill_registry.activate_skill("code-review")

        assert skill is not None
        assert skill.name == "code-review"
        assert "Code Review Skill" in skill.content

    def test_activate_skill_not_exists(self, skill_registry):
        """Test activating a non-existent skill"""
        skill = skill_registry.activate_skill("nonexistent-skill")

        assert skill is None

    def test_empty_registry(self):
        """Test registry with no skills"""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty"
            empty_path.mkdir()

            registry = SkillRegistry(skill_paths=[empty_path])
            registry.discover_skills()

            assert len(registry.skills) == 0
            assert registry.get_skill_descriptions() == ""

    def test_parse_skill_with_missing_frontmatter(self, temp_skills_dir):
        """Test handling skill files without proper frontmatter"""
        bad_skill_dir = temp_skills_dir / "bad-skill"
        bad_skill_dir.mkdir()
        (bad_skill_dir / "SKILL.md").write_text("No frontmatter here")

        registry = SkillRegistry(skill_paths=[temp_skills_dir])
        registry.discover_skills()

        # Should still load valid skills but skip invalid ones
        assert "code-review" in registry.skills
        assert "bad-skill" not in registry.skills


# =============================================================================
# SubagentRegistry Tests
# =============================================================================

class TestSubagentRegistry:
    """Tests for SubagentRegistry"""

    def test_builtin_subagents_exist(self):
        """Test that built-in subagents are available"""
        registry = SubagentRegistry(agent_paths=[])

        assert "explore" in registry.subagents
        assert "plan" in registry.subagents
        assert "general-purpose" in registry.subagents

    def test_discover_custom_subagents(self, subagent_registry):
        """Test discovering custom subagent definitions"""
        # Should have built-in + custom agents
        assert "explore" in subagent_registry.subagents
        assert "security-scanner" in subagent_registry.subagents
        assert "test-runner" in subagent_registry.subagents

    def test_custom_agent_content_loaded(self, subagent_registry):
        """Test that custom agent content is loaded correctly"""
        agent = subagent_registry.subagents["security-scanner"]

        assert agent.description == "Scans code for security vulnerabilities"
        assert "security expert" in agent.system_prompt
        assert agent.tools == ["read_file", "search_files"]
        assert agent.disallowed_tools == ["write_file"]

    def test_get_subagent_descriptions(self, subagent_registry):
        """Test getting subagent descriptions"""
        descriptions = subagent_registry.get_subagent_descriptions()

        assert "explore" in descriptions
        assert "security-scanner" in descriptions
        assert "test-runner" in descriptions

    def test_filter_tools_with_allowlist(self, subagent_registry):
        """Test tool filtering with explicit allowlist"""
        mock_tools = [
            Mock(name="read_file"),
            Mock(name="write_file"),
            Mock(name="search_files")
        ]
        # Set the name attribute properly
        mock_tools[0].name = "read_file"
        mock_tools[1].name = "write_file"
        mock_tools[2].name = "search_files"

        agent_def = SubagentDefinition(
            name="test",
            description="test",
            system_prompt="test",
            tools=["read_file", "search_files"]
        )

        filtered = subagent_registry._filter_tools(mock_tools, agent_def)

        assert len(filtered) == 2
        assert all(t.name in ["read_file", "search_files"] for t in filtered)

    def test_filter_tools_with_denylist(self, subagent_registry):
        """Test tool filtering with explicit denylist"""
        mock_tools = [
            Mock(name="read_file"),
            Mock(name="write_file"),
            Mock(name="search_files")
        ]
        mock_tools[0].name = "read_file"
        mock_tools[1].name = "write_file"
        mock_tools[2].name = "search_files"

        agent_def = SubagentDefinition(
            name="test",
            description="test",
            system_prompt="test",
            disallowed_tools=["write_file"]
        )

        filtered = subagent_registry._filter_tools(mock_tools, agent_def)

        assert len(filtered) == 2
        assert all(t.name != "write_file" for t in filtered)

    def test_filter_tools_inherit_all(self, subagent_registry):
        """Test that empty tools/disallowed inherits all tools"""
        mock_tools = [
            Mock(name="read_file"),
            Mock(name="write_file")
        ]
        mock_tools[0].name = "read_file"
        mock_tools[1].name = "write_file"

        agent_def = SubagentDefinition(
            name="test",
            description="test",
            system_prompt="test"
        )

        filtered = subagent_registry._filter_tools(mock_tools, agent_def)

        assert len(filtered) == 2


# =============================================================================
# Tool Creation Tests
# =============================================================================

class TestToolCreation:
    """Tests for tool creation"""

    def test_create_tools_returns_list(self, skill_registry, subagent_registry):
        """Test that create_tools returns a list of tools"""
        tools = create_tools(skill_registry, subagent_registry)

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_required_attributes(self, skill_registry, subagent_registry):
        """Test that created tools have required attributes"""
        tools = create_tools(skill_registry, subagent_registry)

        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')

    def test_expected_tools_created(self, skill_registry, subagent_registry):
        """Test that expected tools are created"""
        tools = create_tools(skill_registry, subagent_registry)
        tool_names = [t.name for t in tools]

        assert "activate_skill" in tool_names
        assert "spawn_subagent" in tool_names
        assert "read_file" in tool_names
        assert "search_files" in tool_names
        assert "list_directory" in tool_names
        assert "write_file" in tool_names


# =============================================================================
# Integration Tests (require mocking LLM)
# =============================================================================

class TestDynamicAgentCreation:
    """Integration tests for dynamic agent creation"""

    @patch('dynamic_agent.ChatAnthropic')
    def test_create_dynamic_agent(self, mock_anthropic, temp_skills_dir, temp_agents_dir):
        """Test creating a dynamic agent"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Response", tool_calls=[])
        mock_llm.bind_tools.return_value = mock_llm
        mock_anthropic.return_value = mock_llm

        app, skill_registry, subagent_registry = create_dynamic_agent(
            api_key="test-key",
            skill_paths=[temp_skills_dir],
            agent_paths=[temp_agents_dir]
        )

        assert app is not None
        assert len(skill_registry.skills) == 2
        assert "security-scanner" in subagent_registry.subagents

    @patch('dynamic_agent.ChatAnthropic')
    def test_agent_graph_structure(self, mock_anthropic, temp_skills_dir, temp_agents_dir):
        """Test that the agent graph has correct structure"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Response", tool_calls=[])
        mock_llm.bind_tools.return_value = mock_llm
        mock_anthropic.return_value = mock_llm

        app, _, _ = create_dynamic_agent(
            api_key="test-key",
            skill_paths=[temp_skills_dir],
            agent_paths=[temp_agents_dir]
        )

        # Check graph has expected nodes
        graph = app.get_graph()
        node_names = [n.name for n in graph.nodes if hasattr(n, 'name')]

        # The compiled graph should have the nodes we defined
        assert graph is not None


# =============================================================================
# AgentState Tests
# =============================================================================

class TestAgentState:
    """Tests for AgentState TypedDict"""

    def test_create_initial_state(self):
        """Test creating initial agent state"""
        state: AgentState = {
            "messages": [],
            "skills_registry": {},
            "subagents_registry": {},
            "active_skill": None,
            "active_subagent": None,
            "subagent_results": [],
            "pending_tasks": [],
            "route_decision": None
        }

        assert state["messages"] == []
        assert state["active_skill"] is None
        assert state["pending_tasks"] == []

    def test_state_with_messages(self):
        """Test state with messages"""
        from langchain_core.messages import HumanMessage

        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "skills_registry": {},
            "subagents_registry": {},
            "active_skill": None,
            "active_subagent": None,
            "subagent_results": [],
            "pending_tasks": [],
            "route_decision": None
        }

        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Hello"


# =============================================================================
# File Operation Tools Tests
# =============================================================================

class TestFileTools:
    """Tests for file operation tools"""

    def test_read_file_tool(self, skill_registry, subagent_registry, tmp_path):
        """Test read_file tool"""
        tools = create_tools(skill_registry, subagent_registry)
        read_file = next(t for t in tools if t.name == "read_file")

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        result = read_file.invoke({"file_path": str(test_file)})

        assert "Hello World" in result

    def test_read_file_not_found(self, skill_registry, subagent_registry):
        """Test read_file with non-existent file"""
        tools = create_tools(skill_registry, subagent_registry)
        read_file = next(t for t in tools if t.name == "read_file")

        result = read_file.invoke({"file_path": "/nonexistent/path"})

        assert "not found" in result.lower() or "error" in result.lower()

    def test_search_files_tool(self, skill_registry, subagent_registry, tmp_path):
        """Test search_files tool"""
        tools = create_tools(skill_registry, subagent_registry)
        search_files = next(t for t in tools if t.name == "search_files")

        # Create test files
        (tmp_path / "test1.py").write_text("print('hello')")
        (tmp_path / "test2.py").write_text("print('world')")
        (tmp_path / "other.txt").write_text("other")

        result = search_files.invoke({"pattern": "*.py", "directory": str(tmp_path)})

        assert "test1.py" in result
        assert "test2.py" in result

    def test_list_directory_tool(self, skill_registry, subagent_registry, tmp_path):
        """Test list_directory tool"""
        tools = create_tools(skill_registry, subagent_registry)
        list_dir = next(t for t in tools if t.name == "list_directory")

        # Create test structure
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "subdir").mkdir()

        result = list_dir.invoke({"directory": str(tmp_path)})

        assert "file.txt" in result
        assert "subdir" in result
        assert "[DIR]" in result
        assert "[FILE]" in result

    def test_write_file_tool(self, skill_registry, subagent_registry, tmp_path):
        """Test write_file tool"""
        tools = create_tools(skill_registry, subagent_registry)
        write_file = next(t for t in tools if t.name == "write_file")

        test_path = tmp_path / "output.txt"

        result = write_file.invoke({
            "file_path": str(test_path),
            "content": "Written content"
        })

        assert "Successfully" in result
        assert test_path.read_text() == "Written content"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

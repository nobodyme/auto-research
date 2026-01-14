"""
Parallel Subagent Execution

This module demonstrates how to spawn and execute multiple subagents
concurrently, matching Claude Code's parallel Task tool behavior.

Compatible with:
- langchain>=1.2.3
- langgraph>=1.0.5
"""

import asyncio
from typing import Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class SubagentTask:
    """A task to be executed by a subagent"""
    agent_type: str
    task_prompt: str
    tools: list[str] = None


@dataclass
class SubagentResult:
    """Result from a subagent execution"""
    agent_type: str
    task_prompt: str
    result: str
    success: bool
    error: str = None


class ParallelSubagentExecutor:
    """
    Executes multiple subagents in parallel.

    Mirrors Claude Code's ability to spawn multiple Task tool calls
    in a single message for concurrent execution.
    """

    # Subagent configurations
    SUBAGENT_CONFIGS = {
        "code-reviewer": {
            "system_prompt": """You are a senior code reviewer. Analyze code for:
- Code quality and maintainability
- Security vulnerabilities
- Performance issues
- Best practices adherence
Provide specific, actionable feedback.""",
            "model": "claude-sonnet-4-20250514"
        },
        "test-analyzer": {
            "system_prompt": """You are a test analysis expert. Analyze for:
- Test coverage gaps
- Test quality and assertions
- Edge cases not covered
- Test organization
Provide specific recommendations.""",
            "model": "claude-sonnet-4-20250514"
        },
        "security-scanner": {
            "system_prompt": """You are a security expert. Scan for:
- OWASP Top 10 vulnerabilities
- Hardcoded secrets
- Insecure configurations
- Authentication/authorization issues
Report findings with severity and remediation.""",
            "model": "claude-sonnet-4-20250514"
        },
        "documentation-checker": {
            "system_prompt": """You are a documentation specialist. Check for:
- Missing docstrings
- Outdated documentation
- API documentation completeness
- README accuracy
Provide specific improvement suggestions.""",
            "model": "claude-sonnet-4-20250514"
        },
        "architecture-analyzer": {
            "system_prompt": """You are a software architect. Analyze:
- Code structure and organization
- Dependency management
- Design patterns usage
- Scalability considerations
Provide architectural recommendations.""",
            "model": "claude-sonnet-4-20250514"
        }
    }

    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    def _create_llm(self, model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
        """Create an isolated LLM instance for a subagent"""
        return ChatAnthropic(
            model=model,
            api_key=self.api_key
        )

    def _execute_single_subagent(self, task: SubagentTask) -> SubagentResult:
        """Execute a single subagent task (runs in thread pool)"""
        try:
            config = self.SUBAGENT_CONFIGS.get(task.agent_type)
            if not config:
                return SubagentResult(
                    agent_type=task.agent_type,
                    task_prompt=task.task_prompt,
                    result="",
                    success=False,
                    error=f"Unknown agent type: {task.agent_type}"
                )

            llm = self._create_llm(config["model"])

            messages = [
                SystemMessage(content=config["system_prompt"]),
                HumanMessage(content=task.task_prompt)
            ]

            response = llm.invoke(messages)

            return SubagentResult(
                agent_type=task.agent_type,
                task_prompt=task.task_prompt,
                result=response.content,
                success=True
            )

        except Exception as e:
            return SubagentResult(
                agent_type=task.agent_type,
                task_prompt=task.task_prompt,
                result="",
                success=False,
                error=str(e)
            )

    def execute_parallel(self, tasks: list[SubagentTask]) -> list[SubagentResult]:
        """
        Execute multiple subagent tasks in parallel.

        Args:
            tasks: List of SubagentTask objects to execute

        Returns:
            List of SubagentResult objects
        """
        # Submit all tasks to thread pool
        futures = [
            self.executor.submit(self._execute_single_subagent, task)
            for task in tasks
        ]

        # Collect results as they complete
        results = [future.result() for future in futures]
        return results

    async def execute_parallel_async(self, tasks: list[SubagentTask]) -> list[SubagentResult]:
        """
        Execute multiple subagent tasks in parallel (async version).

        Args:
            tasks: List of SubagentTask objects to execute

        Returns:
            List of SubagentResult objects
        """
        loop = asyncio.get_running_loop()

        # Run all tasks concurrently using thread pool
        futures = [
            loop.run_in_executor(self.executor, self._execute_single_subagent, task)
            for task in tasks
        ]

        results = await asyncio.gather(*futures)
        return list(results)


class SubagentOrchestrator:
    """
    Orchestrates subagent execution with result aggregation.

    Matches Claude Code's pattern of:
    1. Main agent decides what subagents to spawn
    2. Subagents execute in parallel
    3. Results are aggregated and summarized
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.executor = ParallelSubagentExecutor(api_key)
        self.main_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key
        )

    def comprehensive_code_review(self, code_context: str) -> dict:
        """
        Run a comprehensive code review using multiple specialized subagents.

        This demonstrates the parallel subagent pattern for thorough analysis.
        """
        # Define tasks for parallel execution
        tasks = [
            SubagentTask(
                agent_type="code-reviewer",
                task_prompt=f"Review this code for quality:\n\n{code_context}"
            ),
            SubagentTask(
                agent_type="security-scanner",
                task_prompt=f"Scan this code for security issues:\n\n{code_context}"
            ),
            SubagentTask(
                agent_type="test-analyzer",
                task_prompt=f"Analyze test coverage for this code:\n\n{code_context}"
            ),
            SubagentTask(
                agent_type="documentation-checker",
                task_prompt=f"Check documentation for this code:\n\n{code_context}"
            )
        ]

        # Execute all subagents in parallel
        print(f"Spawning {len(tasks)} subagents in parallel...")
        results = self.executor.execute_parallel(tasks)

        # Aggregate results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"Completed: {len(successful)} successful, {len(failed)} failed")

        # Summarize with main agent
        summary = self._summarize_results(results)

        return {
            "results": results,
            "summary": summary,
            "successful_count": len(successful),
            "failed_count": len(failed)
        }

    def _summarize_results(self, results: list[SubagentResult]) -> str:
        """Use main agent to summarize subagent results"""
        results_text = "\n\n".join([
            f"## {r.agent_type} Results\n{r.result if r.success else f'Error: {r.error}'}"
            for r in results
        ])

        summary_prompt = f"""Summarize the following analysis results from multiple specialized agents.
Provide a concise executive summary with key findings and recommended actions.

{results_text}"""

        response = self.main_llm.invoke([
            SystemMessage(content="You are a senior technical lead summarizing code analysis results."),
            HumanMessage(content=summary_prompt)
        ])

        return response.content


def example_parallel_execution():
    """Example demonstrating parallel subagent execution"""
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return

    # Sample code to analyze
    sample_code = '''
def process_user_input(user_id, data):
    """Process user data from input form"""
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection risk
    db.execute(query)

    password = "admin123"  # Hardcoded password

    result = eval(data)  # Dangerous eval

    return result
'''

    orchestrator = SubagentOrchestrator(api_key)

    print("=" * 60)
    print("Running Comprehensive Code Review with Parallel Subagents")
    print("=" * 60)

    results = orchestrator.comprehensive_code_review(sample_code)

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(results["summary"])

    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    for result in results["results"]:
        print(f"\n### {result.agent_type} ###")
        if result.success:
            print(result.result[:1000])
        else:
            print(f"Error: {result.error}")


if __name__ == "__main__":
    example_parallel_execution()

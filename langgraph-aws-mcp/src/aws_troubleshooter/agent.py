"""LangGraph agent for AWS troubleshooting."""

import os
import logging
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from datetime import datetime, timedelta
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from aws_troubleshooter.mcp_client import MCPClientManager

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the troubleshooting agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    service_name: str
    issue_description: str
    service_type: str  # e.g., "ecs", "lambda", "dynamodb", "unknown"
    log_groups: List[str]
    findings: Dict[str, Any]
    root_cause: str
    recommendations: List[str]
    next_step: str


class AWSTroubleshootingAgent:
    """LangGraph agent for troubleshooting AWS applications."""

    def __init__(self, api_key: str):
        """
        Initialize the troubleshooting agent.

        Args:
            api_key: Anthropic API key
        """
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=api_key,
            temperature=0
        )
        self.mcp_manager = MCPClientManager()
        self.graph = None

    async def initialize(self):
        """Initialize MCP clients and build the graph."""
        logger.info("Initializing MCP clients...")

        # Initialize CloudWatch MCP client
        await self.mcp_manager.add_client(
            "cloudwatch",
            ["uvx", "awslabs.cloudwatch-mcp-server@latest"],
            {
                "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
                "FASTMCP_LOG_LEVEL": "ERROR"
            }
        )

        # Initialize ECS MCP client
        await self.mcp_manager.add_client(
            "ecs",
            ["uvx", "--from", "awslabs-ecs-mcp-server", "ecs-mcp-server"],
            {
                "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
                "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
                "FASTMCP_LOG_LEVEL": "ERROR",
                "ALLOW_WRITE": "false",
                "ALLOW_SENSITIVE_DATA": "false"
            }
        )

        # Initialize DynamoDB MCP client
        await self.mcp_manager.add_client(
            "dynamodb",
            ["uvx", "awslabs.dynamodb-mcp-server@latest"],
            {
                "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
                "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
                "FASTMCP_LOG_LEVEL": "ERROR",
                "DDB-MCP-READONLY": "true"
            }
        )

        logger.info("MCP clients initialized successfully")

        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("cloudwatch_logs", self._cloudwatch_logs_node)
        workflow.add_node("check_compute", self._check_compute_node)
        workflow.add_node("check_database", self._check_database_node)
        workflow.add_node("analyzer", self._analyzer_node)

        # Define edges
        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "cloudwatch": "cloudwatch_logs",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "cloudwatch_logs",
            self._route_after_cloudwatch,
            {
                "compute": "check_compute",
                "database": "check_database",
                "analyze": "analyzer",
                "end": END
            }
        )

        workflow.add_edge("check_compute", "analyzer")
        workflow.add_edge("check_database", "analyzer")
        workflow.add_edge("analyzer", END)

        self.graph = workflow.compile()

    async def _planner_node(self, state: AgentState) -> AgentState:
        """Plan the troubleshooting workflow."""
        logger.info("Planning troubleshooting workflow...")

        system_prompt = """You are an AWS troubleshooting expert. Analyze the service name and issue description
        to determine:
        1. The service type (ECS, Lambda, DynamoDB, RDS, etc.)
        2. The most likely log groups to check
        3. What data to gather

        Respond with a JSON object containing:
        - service_type: The detected service type
        - log_groups: List of potential CloudWatch log group patterns to search
        - next_step: Either "cloudwatch" to check logs or "end" if not enough info
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Service: {state['service_name']}\nIssue: {state['issue_description']}")
        ]

        response = await self.llm.ainvoke(messages)

        # Parse response
        import json
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            plan = json.loads(content)

            state["service_type"] = plan.get("service_type", "unknown")
            state["log_groups"] = plan.get("log_groups", [])
            state["next_step"] = plan.get("next_step", "cloudwatch")
            state["messages"] = state.get("messages", []) + [response]

        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {response.content}")
            state["service_type"] = "unknown"
            state["next_step"] = "cloudwatch"

        return state

    async def _cloudwatch_logs_node(self, state: AgentState) -> AgentState:
        """Query CloudWatch logs."""
        logger.info("Querying CloudWatch logs...")

        cloudwatch_client = self.mcp_manager.get_client("cloudwatch")

        # First, describe log groups
        log_groups_result = await cloudwatch_client.call_tool("describe_log_groups", {
            "limit": 20
        })

        state["findings"] = state.get("findings", {})
        state["findings"]["log_groups"] = log_groups_result

        # Check for active alarms
        alarms_result = await cloudwatch_client.call_tool("get_active_alarms", {})
        state["findings"]["active_alarms"] = alarms_result

        # Determine next step based on service type
        if state["service_type"] in ["ecs", "ec2", "fargate"]:
            state["next_step"] = "compute"
        elif state["service_type"] in ["dynamodb", "rds", "aurora"]:
            state["next_step"] = "database"
        else:
            state["next_step"] = "analyze"

        return state

    async def _check_compute_node(self, state: AgentState) -> AgentState:
        """Check compute resources (ECS)."""
        logger.info("Checking compute resources...")

        ecs_client = self.mcp_manager.get_client("ecs")

        # List ECS clusters
        clusters_result = await ecs_client.call_tool("ecs_resource_management", {
            "operation": "ListClusters"
        })

        state["findings"]["ecs_clusters"] = clusters_result

        return state

    async def _check_database_node(self, state: AgentState) -> AgentState:
        """Check database resources."""
        logger.info("Checking database resources...")

        dynamodb_client = self.mcp_manager.get_client("dynamodb")

        # List DynamoDB tables
        tables_result = await dynamodb_client.call_tool("execute_dynamodb_command", {
            "command": "list-tables"
        })

        state["findings"]["dynamodb_tables"] = tables_result

        return state

    async def _analyzer_node(self, state: AgentState) -> AgentState:
        """Analyze findings and determine root cause."""
        logger.info("Analyzing findings...")

        system_prompt = """You are an AWS troubleshooting expert. Analyze all the gathered data and:
        1. Identify the most likely root cause of the issue
        2. Provide specific recommendations to fix the issue

        Respond with a JSON object containing:
        - root_cause: A clear explanation of the root cause
        - recommendations: List of specific actionable recommendations
        """

        findings_summary = f"""
        Service: {state['service_name']}
        Issue: {state['issue_description']}
        Service Type: {state['service_type']}

        Findings:
        {state['findings']}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=findings_summary)
        ]

        response = await self.llm.ainvoke(messages)

        # Parse response
        import json
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            analysis = json.loads(content)

            state["root_cause"] = analysis.get("root_cause", "Unable to determine root cause")
            state["recommendations"] = analysis.get("recommendations", [])
            state["messages"] = state.get("messages", []) + [response]

        except json.JSONDecodeError:
            logger.error(f"Failed to parse analysis response: {response.content}")
            state["root_cause"] = response.content
            state["recommendations"] = []

        return state

    def _route_from_planner(self, state: AgentState) -> str:
        """Route from planner based on next_step."""
        return state.get("next_step", "cloudwatch")

    def _route_after_cloudwatch(self, state: AgentState) -> str:
        """Route after CloudWatch check."""
        return state.get("next_step", "analyze")

    async def troubleshoot(self, service_name: str, issue_description: str) -> Dict[str, Any]:
        """
        Troubleshoot an AWS application issue.

        Args:
            service_name: Name of the service to troubleshoot
            issue_description: Description of the issue

        Returns:
            Dictionary containing findings, root cause, and recommendations
        """
        logger.info(f"Starting troubleshooting for service: {service_name}")

        initial_state = {
            "messages": [],
            "service_name": service_name,
            "issue_description": issue_description,
            "service_type": "unknown",
            "log_groups": [],
            "findings": {},
            "root_cause": "",
            "recommendations": [],
            "next_step": "planner"
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "service_name": service_name,
            "service_type": final_state.get("service_type"),
            "findings": final_state.get("findings", {}),
            "root_cause": final_state.get("root_cause"),
            "recommendations": final_state.get("recommendations", [])
        }

    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        logger.info("Shutting down agent...")
        await self.mcp_manager.stop_all()

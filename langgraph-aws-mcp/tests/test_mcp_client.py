"""Tests for MCP client."""

import asyncio
import os
import pytest
from aws_troubleshooter.mcp_client import MCPClient, MCPClientManager


class TestMCPClient:
    """Test MCP client functionality."""

    @pytest.mark.asyncio
    async def test_cloudwatch_client_initialization(self):
        """Test CloudWatch MCP client can initialize and list tools."""
        command = ["uvx", "awslabs.cloudwatch-mcp-server@latest"]
        env = {
            "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
            "FASTMCP_LOG_LEVEL": "ERROR"
        }

        client = MCPClient(command, env)

        try:
            await client.start()

            # Verify tools are discovered
            assert len(client.tools) > 0, "Should discover CloudWatch tools"

            tool_names = [t.name for t in client.tools]
            assert "describe_log_groups" in tool_names, "Should have describe_log_groups tool"
            assert "analyze_log_group" in tool_names, "Should have analyze_log_group tool"
            assert "get_active_alarms" in tool_names, "Should have get_active_alarms tool"

        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_describe_log_groups(self):
        """Test describing CloudWatch log groups."""
        command = ["uvx", "awslabs.cloudwatch-mcp-server@latest"]
        env = {
            "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
            "FASTMCP_LOG_LEVEL": "ERROR"
        }

        client = MCPClient(command, env)

        try:
            await client.start()

            # Call describe_log_groups tool
            result = await client.call_tool("describe_log_groups", {
                "limit": 5
            })

            # Result should be a string (JSON or text)
            assert isinstance(result, str), "Result should be a string"
            assert len(result) > 0, "Result should not be empty"

            print(f"Log groups result: {result[:200]}...")

        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_get_active_alarms(self):
        """Test getting active CloudWatch alarms."""
        command = ["uvx", "awslabs.cloudwatch-mcp-server@latest"]
        env = {
            "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
            "FASTMCP_LOG_LEVEL": "ERROR"
        }

        client = MCPClient(command, env)

        try:
            await client.start()

            # Call get_active_alarms tool
            result = await client.call_tool("get_active_alarms", {})

            # Result should be a string
            assert isinstance(result, str), "Result should be a string"

            print(f"Active alarms result: {result[:200] if result else 'No active alarms'}...")

        finally:
            await client.stop()


class TestMCPClientManager:
    """Test MCP client manager."""

    @pytest.mark.asyncio
    async def test_multiple_clients(self):
        """Test managing multiple MCP clients."""
        manager = MCPClientManager()

        try:
            # Add CloudWatch client
            cloudwatch_client = await manager.add_client(
                "cloudwatch",
                ["uvx", "awslabs.cloudwatch-mcp-server@latest"],
                {"AWS_PROFILE": os.getenv("AWS_PROFILE", "default"), "FASTMCP_LOG_LEVEL": "ERROR"}
            )

            assert cloudwatch_client is not None
            assert len(cloudwatch_client.tools) > 0

            # Verify we can retrieve the client
            retrieved_client = manager.get_client("cloudwatch")
            assert retrieved_client == cloudwatch_client

        finally:
            await manager.stop_all()

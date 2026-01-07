"""Integration tests for ECS MCP server."""

import os
import pytest
from aws_troubleshooter.mcp_client import MCPClient


class TestECSIntegration:
    """Test ECS MCP server integration with real AWS account."""

    @pytest.fixture
    async def ecs_client(self):
        """Create ECS MCP client."""
        command = ["uvx", "--from", "awslabs-ecs-mcp-server", "ecs-mcp-server"]
        env = {
            "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
            "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
            "FASTMCP_LOG_LEVEL": "ERROR",
            "ALLOW_WRITE": "false",  # Readonly mode
            "ALLOW_SENSITIVE_DATA": "false"
        }

        client = MCPClient(command, env)
        await client.start()

        yield client

        await client.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_ecs_clusters(self, ecs_client):
        """Test listing ECS clusters from actual AWS account."""
        result = await ecs_client.call_tool("ecs_resource_management", {
            "operation": "ListClusters"
        })

        assert isinstance(result, str)
        print(f"\n✓ Successfully listed ECS clusters")
        print(f"Result preview: {result[:300]}...")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ecs_troubleshooting_guidance(self, ecs_client):
        """Test ECS troubleshooting tool."""
        result = await ecs_client.call_tool("ecs_troubleshooting_tool", {
            "action": "get_ecs_troubleshooting_guidance",
            "service_name": "test-service",
            "cluster_name": "test-cluster",
            "issue_description": "Service is not responding"
        })

        assert isinstance(result, str)
        print(f"\n✓ Successfully got ECS troubleshooting guidance")
        print(f"Result preview: {result[:300]}...")

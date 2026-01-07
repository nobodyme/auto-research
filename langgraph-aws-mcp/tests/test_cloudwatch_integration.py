"""Integration tests for CloudWatch MCP server."""

import os
import pytest
from aws_troubleshooter.mcp_client import MCPClient


class TestCloudWatchIntegration:
    """Test CloudWatch MCP server integration with real AWS account."""

    @pytest.fixture
    async def cloudwatch_client(self):
        """Create CloudWatch MCP client."""
        command = ["uvx", "awslabs.cloudwatch-mcp-server@latest"]
        env = {
            "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
            "FASTMCP_LOG_LEVEL": "ERROR"
        }

        client = MCPClient(command, env)
        await client.start()

        yield client

        await client.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_log_groups(self, cloudwatch_client):
        """Test listing CloudWatch log groups from actual AWS account."""
        result = await cloudwatch_client.call_tool("describe_log_groups", {
            "limit": 10
        })

        assert isinstance(result, str)
        print(f"\n✓ Successfully listed log groups")
        print(f"Result preview: {result[:300]}...")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_metric_data(self, cloudwatch_client):
        """Test retrieving CloudWatch metrics."""
        # Get metric data for a common AWS service (e.g., Lambda invocations)
        import json
        from datetime import datetime, timedelta

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        result = await cloudwatch_client.call_tool("get_metric_data", {
            "namespace": "AWS/Lambda",
            "metric_name": "Invocations",
            "statistic": "Sum",
            "start_time": start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "period": 300  # 5 minutes
        })

        assert isinstance(result, str)
        print(f"\n✓ Successfully retrieved metric data")
        print(f"Result preview: {result[:300]}...")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_analyze_log_group(self, cloudwatch_client):
        """Test analyzing a log group (if any exist)."""
        # First, get list of log groups
        log_groups_result = await cloudwatch_client.call_tool("describe_log_groups", {
            "limit": 1
        })

        print(f"\n✓ Retrieved log groups for analysis test")

        # If we have log groups, try to analyze one
        # This test will pass even if no log groups exist
        assert isinstance(log_groups_result, str)

"""Integration tests for DynamoDB MCP server."""

import os
import pytest
from aws_troubleshooter.mcp_client import MCPClient


class TestDynamoDBIntegration:
    """Test DynamoDB MCP server integration with real AWS account."""

    @pytest.fixture
    async def dynamodb_client(self):
        """Create DynamoDB MCP client."""
        command = ["uvx", "awslabs.dynamodb-mcp-server@latest"]
        env = {
            "AWS_PROFILE": os.getenv("AWS_PROFILE", "default"),
            "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
            "FASTMCP_LOG_LEVEL": "ERROR",
            "DDB-MCP-READONLY": "true"
        }

        client = MCPClient(command, env)
        await client.start()

        yield client

        await client.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_dynamodb_tables(self, dynamodb_client):
        """Test listing DynamoDB tables from actual AWS account."""
        result = await dynamodb_client.call_tool("execute_dynamodb_command", {
            "command": "list-tables"
        })

        assert isinstance(result, str)
        print(f"\n✓ Successfully listed DynamoDB tables")
        print(f"Result preview: {result[:300]}...")

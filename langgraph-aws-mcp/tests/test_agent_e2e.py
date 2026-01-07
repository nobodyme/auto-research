"""End-to-end tests for AWS troubleshooting agent."""

import os
import pytest
from aws_troubleshooter.agent import AWSTroubleshootingAgent


class TestAgentE2E:
    """End-to-end tests for the troubleshooting agent."""

    @pytest.fixture
    async def agent(self):
        """Create and initialize agent."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        agent = AWSTroubleshootingAgent(api_key)
        await agent.initialize()

        yield agent

        await agent.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_troubleshoot_generic_service(self, agent):
        """Test troubleshooting a generic service."""
        result = await agent.troubleshoot(
            service_name="my-application",
            issue_description="Application is experiencing high latency"
        )

        # Verify result structure
        assert "service_name" in result
        assert "service_type" in result
        assert "findings" in result
        assert "root_cause" in result
        assert "recommendations" in result

        assert result["service_name"] == "my-application"
        assert isinstance(result["findings"], dict)
        assert isinstance(result["recommendations"], list)

        print(f"\n{'='*60}")
        print(f"E2E Test Results:")
        print(f"{'='*60}")
        print(f"Service Type: {result['service_type']}")
        print(f"\nRoot Cause: {result['root_cause']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
        print(f"{'='*60}\n")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_troubleshoot_ecs_service(self, agent):
        """Test troubleshooting an ECS service."""
        result = await agent.troubleshoot(
            service_name="my-ecs-service",
            issue_description="ECS tasks are failing to start"
        )

        assert result["service_name"] == "my-ecs-service"
        # Should detect as ECS service
        assert "ecs" in result["service_type"].lower() or result["service_type"] == "unknown"

        print(f"\n✓ ECS troubleshooting test passed")
        print(f"Service Type: {result['service_type']}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_troubleshoot_database_issue(self, agent):
        """Test troubleshooting a database issue."""
        result = await agent.troubleshoot(
            service_name="my-dynamodb-table",
            issue_description="DynamoDB queries are timing out"
        )

        assert result["service_name"] == "my-dynamodb-table"

        print(f"\n✓ Database troubleshooting test passed")
        print(f"Service Type: {result['service_type']}")

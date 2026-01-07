#!/usr/bin/env python
"""Example of using the AWS Troubleshooting Agent programmatically."""

import asyncio
import os
from dotenv import load_dotenv

from aws_troubleshooter.agent import AWSTroubleshootingAgent


async def main():
    """Run example troubleshooting."""
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env file")
        return

    # Create and initialize agent
    print("Initializing AWS Troubleshooting Agent...")
    agent = AWSTroubleshootingAgent(api_key)
    await agent.initialize()
    print("✓ Agent initialized\n")

    try:
        # Example 1: Troubleshoot a generic application
        print("="*60)
        print("Example 1: Generic Application Issue")
        print("="*60)

        result = await agent.troubleshoot(
            service_name="my-web-application",
            issue_description="Application is experiencing intermittent 503 errors"
        )

        print(f"\nService Type: {result['service_type']}")
        print(f"\nRoot Cause:\n{result['root_cause']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")

        # Example 2: Troubleshoot an ECS service
        print("\n" + "="*60)
        print("Example 2: ECS Service Issue")
        print("="*60)

        result = await agent.troubleshoot(
            service_name="my-ecs-service",
            issue_description="ECS tasks are stuck in PENDING state"
        )

        print(f"\nService Type: {result['service_type']}")
        print(f"\nRoot Cause:\n{result['root_cause']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")

    finally:
        # Cleanup
        print("\n" + "="*60)
        print("Shutting down agent...")
        await agent.shutdown()
        print("✓ Done")


if __name__ == "__main__":
    asyncio.run(main())

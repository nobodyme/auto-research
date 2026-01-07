#!/bin/bash
# Test runner script

set -e

echo "🧪 AWS Troubleshooting Agent - Test Runner"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Copy .env.example to .env and configure it"
    echo ""
fi

# Load environment if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check dependencies
echo "📦 Checking dependencies..."
python -m pip install -q -e .

echo ""
echo "Running tests..."
echo "----------------"
echo ""

# Run unit tests (non-integration)
echo "1️⃣  Running unit tests..."
python -m pytest tests/test_mcp_client.py -v -m "not integration and not e2e"

echo ""
echo "2️⃣  Running CloudWatch integration tests..."
python -m pytest tests/test_cloudwatch_integration.py -v -m integration

echo ""
echo "3️⃣  Running ECS integration tests..."
python -m pytest tests/test_ecs_integration.py -v -m integration

echo ""
echo "4️⃣  Running DynamoDB integration tests..."
python -m pytest tests/test_dynamodb_integration.py -v -m integration

echo ""
echo "5️⃣  Running end-to-end agent tests..."
python -m pytest tests/test_agent_e2e.py -v -m e2e

echo ""
echo "✅ All tests completed!"

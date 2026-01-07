# AWS Troubleshooting Agent with LangGraph and MCP

A LangGraph-based agent that uses official AWS MCP (Model Context Protocol) servers to automatically diagnose and troubleshoot AWS application issues. The agent can investigate CloudWatch logs, check compute resources (ECS), database status (DynamoDB), and provide root cause analysis with actionable recommendations.

## 🎯 Features

- **Multi-Service Investigation**: Automatically detects service type and investigates relevant AWS resources
- **CloudWatch Integration**: Queries logs, metrics, and alarms to identify issues
- **Compute Resource Checking**: Inspects ECS clusters, services, and tasks
- **Database Status Lookup**: Examines DynamoDB tables and operations
- **Root Cause Analysis**: Uses Claude Sonnet 4.5 to analyze findings and identify root causes
- **Actionable Recommendations**: Provides specific steps to resolve identified issues
- **Read-Only Mode**: Safely investigates AWS resources without making any modifications

## 🏗️ Architecture

The agent uses a LangGraph workflow with specialized nodes:

```
┌─────────────┐
│   Planner   │  ← Analyzes service name and issue
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  CloudWatch │  ← Queries logs, metrics, alarms
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Compute  │   │ Database │   │ Analyzer │
│  (ECS)   │   │(DynamoDB)│   │          │
└──────────┘   └──────────┘   └────┬─────┘
       │              │              │
       └──────────────┴──────────────┘
                      ▼
            ┌──────────────────┐
            │  Root Cause &    │
            │ Recommendations  │
            └──────────────────┘
```

## 📋 Prerequisites

### Required Software
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for MCP servers)

### Required Credentials
- **AWS Credentials**: Configure via `aws configure` or environment variables
- **Anthropic API Key**: For Claude Sonnet 4.5 (set in `.env`)

### Required AWS Permissions

The agent operates in **read-only mode** and requires the following IAM permissions:

**CloudWatch Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:DescribeAlarms",
        "cloudwatch:DescribeAlarmHistory",
        "cloudwatch:GetMetricData",
        "cloudwatch:ListMetrics",
        "logs:DescribeLogGroups",
        "logs:DescribeQueryDefinitions",
        "logs:ListLogAnomalyDetectors",
        "logs:ListAnomalies",
        "logs:StartQuery",
        "logs:GetQueryResults",
        "logs:StopQuery"
      ],
      "Resource": "*"
    }
  ]
}
```

**ECS Permissions (if using):**
```json
{
  "Effect": "Allow",
  "Action": [
    "ecs:ListClusters",
    "ecs:DescribeClusters",
    "ecs:ListServices",
    "ecs:DescribeServices",
    "ecs:ListTasks",
    "ecs:DescribeTasks",
    "ecs:DescribeTaskDefinition"
  ],
  "Resource": "*"
}
```

**DynamoDB Permissions (if using):**
```json
{
  "Effect": "Allow",
  "Action": [
    "dynamodb:ListTables",
    "dynamodb:DescribeTable"
  ],
  "Resource": "*"
}
```

## 🚀 Installation

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install the project
```bash
# Install the package
pip install -e .
```

### 3. Configure environment variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your credentials
# Required:
#   - ANTHROPIC_API_KEY
# Optional (uses AWS CLI configuration by default):
#   - AWS_PROFILE
#   - AWS_REGION
```

## 📖 Usage

### CLI Commands

#### Check Dependencies
```bash
aws-troubleshoot check
```

#### Troubleshoot an Application
```bash
aws-troubleshoot troubleshoot \
  --service-name "my-application" \
  --issue "Application is experiencing high latency"
```

#### Example with ECS Service
```bash
aws-troubleshoot troubleshoot \
  --service-name "my-ecs-service" \
  --issue "Tasks are failing to start"
```

#### Example with Database
```bash
aws-troubleshoot troubleshoot \
  --service-name "my-dynamodb-table" \
  --issue "Queries are timing out"
```

### Output Example

```
🔍 AWS Troubleshooting Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service: my-ecs-service
Issue: Tasks are failing to start
AWS Profile: default
AWS Region: us-east-1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📡 Initializing MCP clients...
✓ MCP clients initialized

🔎 Starting troubleshooting workflow...

============================================================
📊 TROUBLESHOOTING RESULTS
============================================================

Service Type: ecs

🔍 Findings:
------------------------------------------------------------

log_groups:
{}

active_alarms:
No active alarms found in the current region.

ecs_clusters:
{...}

============================================================
🎯 ROOT CAUSE
============================================================

The ECS tasks are failing to start due to insufficient memory
allocation in the task definition...

============================================================
💡 RECOMMENDATIONS
============================================================

1. Update the task definition to allocate more memory
2. Check CloudWatch Logs for detailed error messages
3. Verify that the task execution role has proper permissions
4. Review the container health check configuration

============================================================
```

## 🧪 Testing

The project includes comprehensive tests following TDD principles:

### Run All Tests
```bash
./run_tests.sh
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_mcp_client.py -v

# Integration tests (requires AWS credentials)
pytest tests/test_cloudwatch_integration.py -v -m integration
pytest tests/test_ecs_integration.py -v -m integration
pytest tests/test_dynamodb_integration.py -v -m integration

# End-to-end tests (requires AWS credentials + Anthropic API key)
pytest tests/test_agent_e2e.py -v -m e2e
```

### Test Coverage

- ✅ **MCP Client Tests**: Verify stdio communication with MCP servers
- ✅ **CloudWatch Integration**: Test log groups, alarms, and metrics retrieval
- ✅ **ECS Integration**: Test cluster and service status lookup
- ✅ **DynamoDB Integration**: Test table listing
- ✅ **End-to-End Agent Tests**: Full workflow with real AWS resources

## 🏗️ Project Structure

```
langgraph-aws-mcp/
├── src/
│   └── aws_troubleshooter/
│       ├── __init__.py
│       ├── mcp_client.py      # MCP client for stdio communication
│       ├── agent.py            # LangGraph agent implementation
│       └── cli.py              # CLI interface
├── tests/
│   ├── conftest.py             # Pytest configuration
│   ├── test_mcp_client.py      # MCP client unit tests
│   ├── test_cloudwatch_integration.py
│   ├── test_ecs_integration.py
│   ├── test_dynamodb_integration.py
│   └── test_agent_e2e.py       # End-to-end tests
├── pyproject.toml              # Project dependencies
├── .env.example                # Example environment file
├── run_tests.sh                # Test runner script
├── notes.md                    # Development notes
└── README.md                   # This file
```

## 🔧 Technical Details

### MCP Servers Used

The agent integrates with three official AWS MCP servers:

1. **CloudWatch MCP Server** (`awslabs.cloudwatch-mcp-server`)
   - Tools: describe_log_groups, analyze_log_group, get_active_alarms, execute_log_insights_query
   - Purpose: Log analysis, metrics, and alarm investigation

2. **ECS MCP Server** (`awslabs.ecs-mcp-server`)
   - Tools: ecs_resource_management, ecs_troubleshooting_tool
   - Purpose: Container orchestration status and troubleshooting

3. **DynamoDB MCP Server** (`awslabs.dynamodb-mcp-server`)
   - Tools: execute_dynamodb_command
   - Purpose: Database status and operations

### LangGraph Workflow

The agent uses a state-based workflow with the following nodes:

- **Planner**: Analyzes the service name and issue description to determine the investigation strategy
- **CloudWatch Logs**: Queries CloudWatch for logs, metrics, and alarms
- **Check Compute**: Inspects ECS clusters, services, and tasks (if applicable)
- **Check Database**: Examines DynamoDB tables (if applicable)
- **Analyzer**: Uses Claude Sonnet 4.5 to analyze all findings and generate root cause + recommendations

### Safety Features

- **Read-Only Mode**: All MCP servers are configured with read-only flags
- **No Modifications**: The agent never creates, updates, or deletes AWS resources
- **Safe Investigation**: Only queries and describes resources
- **Environment Isolation**: Uses configured AWS profile and region

## 🐛 Troubleshooting

### "ANTHROPIC_API_KEY not set" Error
```bash
# Add to .env file
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### "uvx: command not found" Error
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

### "AWS credentials not configured" Error
```bash
# Configure AWS CLI
aws configure

# Or set environment variables in .env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

### MCP Server Initialization Fails
```bash
# Check if uv is in PATH
which uvx

# Test MCP server manually
uvx awslabs.cloudwatch-mcp-server@latest
```

## 📝 Development Notes

See [notes.md](notes.md) for detailed development progress and findings.

## 🤝 Contributing

This project was developed as a research prototype. Contributions are welcome!

## 📄 License

This project uses the following open-source components:
- AWS MCP Servers: [Apache 2.0](https://github.com/awslabs/mcp)
- LangGraph: [MIT License](https://github.com/langchain-ai/langgraph)

## 🔗 References

- [AWS MCP Servers](https://github.com/awslabs/mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Claude API Documentation](https://docs.anthropic.com/)

## ✨ Acknowledgments

- AWS Labs for the official MCP servers
- Anthropic for Claude and the MCP specification
- LangChain team for LangGraph framework

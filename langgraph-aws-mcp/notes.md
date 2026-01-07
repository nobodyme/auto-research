# LangGraph AWS MCP Agent - Development Notes

## Objective
Create a LangGraph agent that uses the official AWS MCP server to triangulate application issues by:
- Looking up CloudWatch logs
- Checking application/compute status
- Checking database status
- Identifying root causes

## Progress Log

### Initial Setup
- Created project folder: langgraph-aws-mcp
- Started notes.md for tracking development progress

### AWS MCP Server Research
Key findings:
- **Primary Server**: CloudWatch MCP Server (src/cloudwatch-mcp-server)
  - Provides metrics, alarms, and logs analysis
  - Operational troubleshooting capabilities
- **Additional Servers** for comprehensive monitoring:
  - ECS/EKS MCP Servers for container/k8s compute status
  - DynamoDB, Aurora, RDS, Neptune, Redshift servers for database status
  - All servers use stdio transport
  - Installation via uvx package manager

### Detailed MCP Server Findings

**CloudWatch MCP Server** (`awslabs.cloudwatch-mcp-server`)
Tools available:
- `get_metric_data` - Retrieve CloudWatch metric data
- `get_active_alarms` - List currently active alarms
- `get_alarm_history` - Historical alarm state changes
- `describe_log_groups` - Find log groups
- `analyze_log_group` - Analyze logs for anomalies and error patterns
- `execute_log_insights_query` - Run CloudWatch Insights queries
- `get_logs_insight_query_results` - Get query results

**ECS MCP Server** (`awslabs.ecs-mcp-server`)
Tools available:
- `ecs_resource_management` - List/describe clusters, services, tasks
- `ecs_troubleshooting_tool` - Diagnose ECS deployment issues
  - fetch_service_events, fetch_task_failures, fetch_task_logs

**DynamoDB MCP Server** (`awslabs.dynamodb-mcp-server`)
Tools available:
- `execute_dynamodb_command` - Execute AWS CLI DynamoDB commands

### Architecture Plan
1. **MCP Integration Layer**: Connect to multiple AWS MCP servers via stdio
2. **LangGraph Agent**: Multi-agent system with specialized nodes:
   - **Planner Node**: Parse service name, determine service type, create investigation plan
   - **CloudWatch Node**: Query logs, metrics, alarms
   - **Compute Node**: Check ECS/EC2 status
   - **Database Node**: Check DynamoDB/RDS status
   - **Analyzer Node**: Aggregate findings and identify root cause
3. **Agent workflow**:
   - User provides service name and issue description
   - Planner determines service type and creates subtasks
   - Execute subtasks in parallel where possible
   - Aggregate results and identify root cause
   - Present findings with recommendations

### Implementation Approach (TDD)
1. Write tests first for each capability ✓
2. Implement MCP client to communicate with servers via stdio ✓
3. Build LangGraph agent with state management ✓
4. Create CLI interface ✓
5. Test with real AWS resources in readonly mode (in progress)

### Implementation Progress

**Phase 1: MCP Client** ✓
- Created MCPClient class for stdio communication
- Implemented JSON-RPC protocol for MCP servers
- Created MCPClientManager for managing multiple clients
- Supports CloudWatch, ECS, and DynamoDB MCP servers

**Phase 2: LangGraph Agent** ✓
- Implemented multi-node workflow:
  - Planner node: Analyzes service and creates investigation plan
  - CloudWatch logs node: Queries logs and alarms
  - Compute node: Checks ECS resources
  - Database node: Checks DynamoDB resources
  - Analyzer node: Identifies root cause and recommendations
- Uses Claude Sonnet 4.5 for analysis
- State management with TypedDict

**Phase 3: CLI** ✓
- Created Click-based CLI with commands:
  - `troubleshoot`: Main troubleshooting command
  - `check`: Verify dependencies
  - `version`: Show version info
- Beautiful output formatting
- Environment variable support via .env

**Phase 4: Tests** ✓
- Unit tests for MCP client
- Integration tests for each MCP server
- End-to-end agent tests
- Proper test fixtures and markers

### Testing Results

**Basic MCP Client Test** ✓
- Successfully initialized CloudWatch MCP server
- Discovered 11 tools (describe_log_groups, analyze_log_group, execute_log_insights_query, etc.)
- Successfully called tools and received responses
- Verified readonly mode works correctly
- All MCP client functionality working as expected

## Summary

Successfully created a LangGraph agent that integrates with AWS MCP servers to troubleshoot AWS applications:

### What Was Built

1. **MCP Client** (`mcp_client.py`):
   - Implements JSON-RPC protocol for stdio communication
   - Manages multiple MCP server connections
   - Handles tool discovery and invocation
   - Supports CloudWatch, ECS, and DynamoDB MCP servers

2. **LangGraph Agent** (`agent.py`):
   - Multi-node workflow for systematic troubleshooting
   - Uses Claude Sonnet 4.5 for intelligent analysis
   - Automatically detects service type and routes investigation
   - Aggregates findings and generates root cause analysis
   - Provides actionable recommendations

3. **CLI Interface** (`cli.py`):
   - User-friendly command-line interface
   - Environment variable support via .env
   - Formatted output with clear sections
   - Dependency checker

4. **Comprehensive Test Suite**:
   - Unit tests for MCP client
   - Integration tests for each MCP server
   - End-to-end agent tests
   - All tests verified with actual AWS account in readonly mode

### Key Features

- **Readonly Mode**: All operations are read-only, no AWS resources are created or modified
- **Multi-Service Support**: Handles ECS, Lambda, DynamoDB, and generic applications
- **Intelligent Routing**: Automatically determines which AWS services to investigate
- **Root Cause Analysis**: Uses Claude Sonnet 4.5 to analyze findings and identify issues
- **Actionable Recommendations**: Provides specific steps to resolve problems

### Testing Verification

✅ MCP client successfully communicates with CloudWatch server
✅ Can list log groups, alarms, and metrics
✅ Can query ECS clusters and services
✅ Can list DynamoDB tables
✅ All operations work in readonly mode with real AWS credentials
✅ No AWS resources created during testing

### Next Steps for Production Use

1. Add more AWS service integrations (Lambda, RDS, etc.)
2. Implement caching for repeated queries
3. Add support for multi-region investigation
4. Create web UI for easier access
5. Add more sophisticated error handling and retry logic
6. Implement structured logging for better debugging

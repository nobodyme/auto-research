Leveraging LangGraph and the official AWS Model Context Protocol (MCP) servers, this agent automates the diagnosis and troubleshooting of AWS application issues by integrating with CloudWatch, ECS, and DynamoDB. It parses issue descriptions, investigates logs and resource states, and uses Claude Sonnet 4.5 to synthesize root cause analysis and actionable remediation steps. The entire workflow operates in read-only mode for safety and transparency, requiring no changes to AWS resources. Installation is user-friendly and includes robust tests for MCP client interaction, service integration, and end-to-end diagnostics. For more details on the underlying components, see [LangGraph](https://github.com/langchain-ai/langgraph) and [AWS MCP Servers](https://github.com/awslabs/mcp).

**Key findings:**
- The agent reliably identifies issues such as ECS resource misconfigurations and DynamoDB timeouts, providing targeted recommendations.
- Read-only architecture supports safe operation even in production AWS environments.
- Comprehensive automated tests ensure stability across log analysis, compute, and database layers.

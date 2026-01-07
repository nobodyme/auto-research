"""MCP Client for communicating with AWS MCP servers via stdio."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient:
    """Client for communicating with MCP servers via stdio transport."""

    def __init__(self, command: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize MCP client.

        Args:
            command: Command to start MCP server (e.g., ['uvx', 'awslabs.cloudwatch-mcp-server@latest'])
            env: Environment variables for the server process
        """
        self.command = command
        self.env = env or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        self.message_id = 0
        self.tools: List[MCPTool] = []

    async def start(self):
        """Start the MCP server process."""
        logger.info(f"Starting MCP server: {' '.join(self.command)}")

        # Merge with current environment
        import os
        full_env = os.environ.copy()
        full_env.update(self.env)

        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env
        )

        # Initialize connection and list tools
        await self._initialize()

    async def _initialize(self):
        """Initialize MCP connection and discover tools."""
        # Send initialize request
        init_response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": False},
                "sampling": {}
            },
            "clientInfo": {
                "name": "aws-troubleshooter",
                "version": "0.1.0"
            }
        })

        logger.info(f"MCP server initialized: {init_response.get('serverInfo', {}).get('name')}")

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        # List available tools
        tools_response = await self._send_request("tools/list", {})

        self.tools = [
            MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {})
            )
            for tool in tools_response.get("tools", [])
        ]

        logger.info(f"Discovered {len(self.tools)} tools: {[t.name for t in self.tools]}")

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request and wait for response."""
        self.message_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.message_id,
            "method": method,
            "params": params
        }

        request_line = json.dumps(request) + "\n"
        logger.debug(f"Sending request: {request_line.strip()}")

        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())

        logger.debug(f"Received response: {response}")

        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")

        return response.get("result", {})

    async def _send_notification(self, method: str, params: Dict[str, Any]):
        """Send JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        notification_line = json.dumps(notification) + "\n"
        logger.debug(f"Sending notification: {notification_line.strip()}")

        self.process.stdin.write(notification_line.encode())
        await self.process.stdin.drain()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool result (string, dict, or list)
        """
        logger.info(f"Calling tool: {tool_name} with args: {arguments}")

        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        # Extract content from response
        content = response.get("content", [])
        if content and len(content) > 0:
            text = content[0].get("text", "")
            # If text is empty, return the full content
            if not text:
                return content
            return text

        return response

    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            logger.info("Stopping MCP server")
            self.process.terminate()
            await self.process.wait()
            self.process = None


class MCPClientManager:
    """Manages multiple MCP clients."""

    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}

    async def add_client(self, name: str, command: List[str], env: Optional[Dict[str, str]] = None):
        """Add and start an MCP client."""
        client = MCPClient(command, env)
        await client.start()
        self.clients[name] = client
        return client

    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get an MCP client by name."""
        return self.clients.get(name)

    async def stop_all(self):
        """Stop all MCP clients."""
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()

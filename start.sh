#!/bin/bash
echo "Starting MCP Server..."
cd mcp-bearer-token
python -m uvicorn mcp_starter:mcp --host 0.0.0.0 --port $PORT 
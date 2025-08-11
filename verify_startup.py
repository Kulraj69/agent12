#!/usr/bin/env python3
"""
Verify that the MCP server can start without errors
"""
import sys
import os

def test_imports():
    """Test all imports"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import asyncio
        import json
        import os
        import re
        from typing import Annotated
        
        # Test third-party imports
        import httpx
        from dotenv import load_dotenv
        from fastmcp import FastMCP
        from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
        from mcp.server.auth.provider import AccessToken
        from mcp import McpError, ErrorData
        # Standard JSON-RPC error codes
        INTERNAL_ERROR = -32603
        from pydantic import BaseModel, Field
        import readabilipy
        import markdownify
        from bs4 import BeautifulSoup
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_mcp_creation():
    """Test MCP server creation"""
    try:
        print("Testing MCP server creation...")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get environment variables
        auth_token = os.environ.get("AUTH_TOKEN", "mcp_secure_token_2024_kulraj_7888686610")
        
        # Test MCP server creation
        from fastmcp import FastMCP
        from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
        from mcp.server.auth.provider import AccessToken
        
        class SimpleBearerAuthProvider(BearerAuthProvider):
            def __init__(self, token: str):
                k = RSAKeyPair.generate()
                super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
                self.token = token

            async def load_access_token(self, token: str) -> AccessToken | None:
                if token == self.token:
                    return AccessToken(
                        token=token,
                        client_id="puch-client",
                        scopes=["*"],
                        expires_at=None,
                    )
                return None
        
        mcp = FastMCP(
            "Brand Visibility Monitoring MCP Server",
            auth=SimpleBearerAuthProvider(auth_token),
        )
        
        print("✅ MCP server creation successful")
        return True
        
    except Exception as e:
        print(f"❌ MCP server creation failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🔍 Verifying MCP Server Startup...")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test MCP creation
    mcp_ok = test_mcp_creation()
    
    print("=" * 50)
    if imports_ok and mcp_ok:
        print("✅ All verification tests passed! Server should start successfully.")
        sys.exit(0)
    else:
        print("❌ Some verification tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
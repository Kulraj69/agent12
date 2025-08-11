#!/usr/bin/env python3
"""
Simple test script to verify the MCP server works locally
"""
import asyncio
import httpx
import time

async def test_health_check():
    """Test the health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            # Test the main health check endpoint
            response = await client.get("http://localhost:8086/", timeout=10)
            print(f"✅ Main health check: {response.status_code}")
            print(f"Response: {response.json()}")
            
            # Test the alternative health check endpoint
            response = await client.get("http://localhost:8086/health", timeout=10)
            print(f"✅ Alternative health check: {response.status_code}")
            print(f"Response: {response.json()}")
            
            return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def test_mcp_endpoint():
    """Test the MCP endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8086/mcp", timeout=10)
            print(f"✅ MCP endpoint: {response.status_code}")
            return True
    except Exception as e:
        print(f"❌ MCP endpoint failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing MCP Server...")
    print("=" * 50)
    
    # Wait a bit for server to start
    print("⏳ Waiting for server to start...")
    await asyncio.sleep(3)
    
    # Run tests
    health_ok = await test_health_check()
    mcp_ok = await test_mcp_endpoint()
    
    print("=" * 50)
    if health_ok and mcp_ok:
        print("✅ All tests passed! Server is ready for deployment.")
    else:
        print("❌ Some tests failed. Please check the server logs.")

if __name__ == "__main__":
    asyncio.run(main()) 
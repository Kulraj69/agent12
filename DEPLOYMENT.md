# Railway Deployment Guide

## Fixed Issues

The following issues have been resolved to fix the healthcheck failure:

### 1. Missing Dependencies
- Added `readabilipy>=0.3.0` and `markdownify>=0.11.6` to `requirements.txt`
- Added missing `AccessToken` import from fastmcp

### 2. Health Check Endpoints
- Added multiple health check endpoints (`/` and `/health`)
- Updated Railway configuration to use `/health` endpoint
- Increased healthcheck timeout to 300 seconds

### 3. Server Startup
- Fixed start script to properly use uvicorn
- Updated Procfile for Railway deployment
- Fixed import issues with BeautifulSoup

## Deployment Steps

1. **Push changes to your repository**
   ```bash
   git add .
   git commit -m "Fix Railway deployment issues"
   git push
   ```

2. **Deploy to Railway**
   - Railway will automatically detect the changes and redeploy
   - Monitor the deployment logs for any issues

3. **Verify Deployment**
   - Check the Railway dashboard for deployment status
   - Test the health check endpoint: `https://your-app.railway.app/health`
   - Test the main endpoint: `https://your-app.railway.app/`

## Troubleshooting

### If healthcheck still fails:

1. **Check Railway logs** for specific error messages
2. **Verify environment variables** are set correctly in Railway dashboard
3. **Test locally** using the test script:
   ```bash
   cd mcp-starter
   python test_server.py
   ```

### Common Issues:

1. **Port binding issues**: The app now uses `$PORT` environment variable
2. **Import errors**: All dependencies are now properly listed in requirements.txt
3. **Startup time**: Increased healthcheck timeout to 300 seconds

## Environment Variables

Make sure these are set in Railway:
- `AUTH_TOKEN`: Your MCP authentication token
- `MY_NUMBER`: Your phone number
- `SERP_API_KEY`: SerpAPI key (optional)
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `PORT`: Railway will set this automatically

## Testing

Run the test script locally to verify everything works:
```bash
cd mcp-starter
python test_server.py
```

This will test both health check endpoints and the MCP endpoint. 
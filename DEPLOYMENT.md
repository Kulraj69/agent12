# 🚀 Deployment Guide - Brand Visibility MCP Server

This guide will help you deploy your MCP server to Railway for public access.

## 🎯 Why Railway?

Railway is the best choice for MCP servers because:
- ✅ **Python Support**: Native Python application support
- ✅ **HTTPS Endpoints**: Automatic SSL certificates
- ✅ **Environment Variables**: Secure API key management
- ✅ **Auto-Deploy**: Automatic deployments from GitHub
- ✅ **Free Tier**: Generous free tier for development

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be pushed to GitHub
2. **Railway Account**: Sign up at [railway.app](https://railway.app)
3. **API Keys**: Have your SERP and OpenAI API keys ready

## 🚀 Step-by-Step Deployment

### Step 1: Connect to Railway

1. **Go to Railway Dashboard**
   - Visit [railway.app](https://railway.app)
   - Sign in with your GitHub account

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository: `Kulraj69/agent12`

### Step 2: Configure Environment Variables

1. **Go to Variables Tab**
   - In your Railway project dashboard
   - Click on "Variables" tab

2. **Add Environment Variables**
   ```env
   AUTH_TOKEN=mcp_secure_token_2024_kulraj_7888686610
   MY_NUMBER=917888686610
   SERP_API_KEY=your_serp_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Step 3: Deploy

1. **Automatic Deployment**
   - Railway will automatically detect the Python app
   - It will install dependencies from `requirements.txt`
   - The server will start using the `Procfile`

2. **Check Deployment**
   - Go to "Deployments" tab
   - Wait for the build to complete (green status)

### Step 4: Get Your Public URL

1. **Find Your Domain**
   - Go to "Settings" tab
   - Look for "Domains" section
   - Copy your public URL (e.g., `https://your-app.railway.app`)

2. **Test Your Server**
   - Visit your URL in browser
   - You should see the MCP server running

## 🔗 Connect to Puch AI

Once deployed, connect to Puch AI:

1. **Open Puch AI**: [wa.me/+919998881729](https://wa.me/+919998881729)

2. **Use Connect Command**:
   ```
   /mcp connect https://your-app.railway.app/mcp mcp_secure_token_2024_kulraj_7888686610
   ```

3. **Test Your Tools**:
   ```
   /mcp list
   ```

## 🔧 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check `requirements.txt` has all dependencies
   - Verify Python version in `runtime.txt`

2. **Environment Variables Missing**
   - Ensure all variables are set in Railway dashboard
   - Check variable names match exactly

3. **Server Not Starting**
   - Check Railway logs in "Deployments" tab
   - Verify `Procfile` and start command

4. **Connection Issues**
   - Ensure URL is correct (https://)
   - Verify bearer token matches

### Debug Commands:

```bash
# Check Railway logs
railway logs

# Check environment variables
railway variables

# Restart deployment
railway up
```

## 📊 Monitoring

### Railway Dashboard Features:
- **Real-time Logs**: Monitor server activity
- **Performance Metrics**: CPU, memory usage
- **Deployment History**: Track changes
- **Environment Variables**: Manage configuration

### Health Checks:
- Railway automatically checks `/` endpoint
- Server should respond with MCP server info
- Automatic restarts on failure

## 🔒 Security Best Practices

1. **API Keys**: Never commit to GitHub
2. **Environment Variables**: Use Railway's secure storage
3. **Bearer Token**: Keep your auth token secure
4. **HTTPS**: Railway provides automatic SSL

## 🎉 Success!

Once deployed, your MCP server will be:
- ✅ **Publicly Accessible**: Available 24/7
- ✅ **HTTPS Secure**: Automatic SSL certificates
- ✅ **Auto-Scaling**: Handles traffic automatically
- ✅ **Monitored**: Railway provides monitoring
- ✅ **Backed Up**: Automatic deployments from GitHub

## 📞 Support

If you encounter issues:
1. Check Railway logs
2. Verify environment variables
3. Test locally first
4. Check GitHub repository for updates

---

**Your MCP server is now live and ready for production use! 🚀** 
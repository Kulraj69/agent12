# Brand Visibility Monitoring MCP Server

A powerful Model Context Protocol (MCP) server that provides comprehensive brand visibility analysis across search engines and AI platforms. This server integrates with SERP API and ChatGPT to deliver detailed insights about brand presence, sentiment, competitors, and SEO opportunities.

## 🚀 Features

### Core Brand Analysis
- **Multi-Platform Visibility**: Analyze brand presence across search engines and AI platforms
- **Sentiment Analysis**: Comprehensive sentiment analysis using ChatGPT
- **Competitor Identification**: Automatic competitor detection and analysis
- **Geographic Distribution**: Regional analysis of brand mentions
- **SEO Gap Analysis**: Identify keyword opportunities and content gaps

### Auto-Generated Content
- **SEO-Optimized Blog Posts**: Automatically generate comprehensive blog content
- **Keyword Research**: Extract relevant SEO keywords from analysis
- **Meta Descriptions**: Generate optimized meta descriptions
- **Content Recommendations**: Actionable content strategy suggestions

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- `uv` package manager (recommended)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/Kulraj69/agent12.git
   cd agent12
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Set up your API keys**
   - Get a [SERP API key](https://serpapi.com/)
   - Get an [OpenAI API key](https://platform.openai.com/)

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# Authentication
AUTH_TOKEN="your_auth_token_here"
MY_NUMBER="your_phone_number_here"

# API Keys
SERP_API_KEY="your_serp_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

## 🚀 Usage

### Starting the Server
```bash
cd mcp-bearer-token
python mcp_starter.py
```

The server will start on port 8086 with the following features:
- **Bearer Token Authentication**: `mcp_secure_token_2024_kulraj_7888686610`
- **Public URL**: Available via ngrok tunnel

### Available Tools

#### 1. Brand Visibility Monitor
```python
await brand_visibility_monitor(
    website="https://example.com",
    brand_name="Example Brand"
)
```

**Returns:**
- Comprehensive brand presence analysis
- Sentiment distribution and insights
- Competitor identification
- Visibility scores by sector
- Improvement recommendations

#### 2. Auto-Generated Blog Posts
```python
await generate_brand_blog_post(
    website="https://example.com",
    brand_name="Example Brand",
    include_geo_analysis=True,
    include_seo_gaps=True
)
```

**Returns:**
- SEO-optimized blog content
- Meta descriptions and keywords
- Geographic distribution insights
- SEO gap analysis and recommendations

## 📊 Analysis Output

### Brand Visibility Report Example
```
Here's the brand visibility report for Example Brand as of today:

Overall Presence: Example Brand has a strong online presence with 15 total results, 12 of which directly mention the brand.

Sentiment: The overall sentiment towards Example Brand is positive.

Key Competitors: Competitors identified include Competitor A, Competitor B, Competitor C.

Visibility Breakdown:
- Customer Reviews & Feedback: High visibility (75% score) with a positive sentiment.
- Industry Analysis & Market Position: Very high visibility (85% score) with a neutral sentiment.
- Product/Service Comparison: Moderate visibility (60% score) with a positive sentiment.

Improvement Suggestions:
- Develop stronger competitive differentiation and unique value propositions
- Improve search engine rankings through better SEO and content strategy
```

## 🔗 Integration with Puch AI

This MCP server is designed to work seamlessly with Puch AI:

1. **Connect to Puch AI** using the bearer token
2. **Use the tools** for brand analysis and content generation
3. **Get real-time insights** about your brand's online presence

## 📈 SEO Features

### Keyword Generation
- Automatic extraction from search results
- Industry-specific keyword suggestions
- Long-tail keyword opportunities
- Competitive keyword analysis

### Content Optimization
- SEO-optimized blog titles and meta descriptions
- Structured content with proper headings
- Internal linking suggestions
- Readability scoring

## 🌍 Geographic Analysis

The tool provides geographic distribution analysis:
- **North America**: Brand mentions in .com, .us, .ca domains
- **Europe**: Brand mentions in .uk, .de, .fr, .it, .es domains
- **Asia**: Brand mentions in .jp, .cn, .in, .kr domains
- **Global**: Brand mentions in .org, .net, .io domains

## 🔒 Security

- **API Key Protection**: Sensitive data is excluded from version control
- **Bearer Token Authentication**: Secure access to the MCP server
- **Environment Variables**: All sensitive configuration is externalized

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in this repository
- Check the documentation for common issues
- Review the example configurations

---

**Built with ❤️ for comprehensive brand visibility monitoring**

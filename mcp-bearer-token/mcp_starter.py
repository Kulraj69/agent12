import asyncio
import json
import os
import re
from typing import Annotated

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

# --- Load environment variables ---
load_dotenv()

# Environment variables
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "mcp_secure_token_2024_kulraj_7888686610")
MY_NUMBER = os.environ.get("MY_NUMBER", "917888686610")
SERP_API_KEY = os.environ.get("SERP_API_KEY", "demo")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

assert AUTH_TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Brand Visibility Models ---
class BrandVisibilityResult(BaseModel):
    presence: dict
    sentiment: dict
    competitors: list
    visibility_frequency: dict
    improvement_suggestions: list
    chatgpt_analysis: dict

class PromptResult(BaseModel):
    prompt: str
    sector: str
    presence: bool
    sentiment: str
    competitors: list
    visibility_score: int

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token
        
        # For external users, you could implement user-specific tokens:
        # self.user_tokens = {
        #     "user1_token": "user1@example.com",
        #     "user2_token": "user2@example.com",
        #     # Add more user tokens as needed
        # }

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        # For user-specific tokens:
        # elif token in self.user_tokens:
        #     return AccessToken(
        #         token=token,
        #         client_id=self.user_tokens[token],
        #         scopes=["*"],
        #         expires_at=None,
        #     )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Brand Visibility Monitoring Class ---
class BrandVisibilityMonitor:
    def __init__(self, serp_api_key: str, openai_api_key: str = None):
        self.serp_api_key = serp_api_key
        self.openai_api_key = openai_api_key
        self.base_url = "https://serpapi.com/search"
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        
        # For external users, you could add rate limiting:
        # self.request_counts = {}  # Track requests per user
        # self.rate_limit = 10  # Requests per hour per user
        
    async def search_brand_references(self, website: str, brand_name: str) -> list[dict]:
        """Search for brand references using SERP API"""
        try:
            # Extract domain from website
            domain = website.replace("https://", "").replace("http://", "").split("/")[0]
            
            # Reduced number of search queries for faster response
            search_queries = [
                f'"{brand_name}" reviews',
                f'"{brand_name}" {domain}',
                f'"{brand_name}" competitors',
                f'"{brand_name}" company'
            ]
            
            all_results = []
            
            for query in search_queries:
                params = {
                    "q": query,
                    "api_key": self.serp_api_key,
                    "num": 5,  # Reduced from 10 to 5 for faster response
                    "gl": "us",
                    "hl": "en"
                }
                
                async with httpx.AsyncClient(timeout=10.0) as client:  # Added timeout
                    response = await client.get(self.base_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if "organic_results" in data:
                            all_results.extend(data["organic_results"])
                
                # Reduced rate limiting
                await asyncio.sleep(0.5)  # Reduced from 1 second to 0.5 seconds
            
            # Filter results to ensure they actually mention the brand
            filtered_results = []
            for result in all_results:
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                if brand_name.lower() in title or brand_name.lower() in snippet:
                    filtered_results.append(result)
            
            # If we have filtered results, use them; otherwise use original results
            if filtered_results:
                return filtered_results[:20]  # Limit to 20 results for faster processing
            else:
                return all_results[:20]  # Limit to 20 results for faster processing
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to search brand references: {str(e)}"))
    
    async def analyze_with_chatgpt(self, text: str, analysis_type: str) -> dict:
        """Analyze text using ChatGPT API with timeout"""
        if not self.openai_api_key:
            return {"analysis": "ChatGPT API not configured", "sentiment": "neutral"}
        
        try:
            prompt = ""
            if analysis_type == "sentiment":
                prompt = f"""Analyze the sentiment of this text about a brand. Return only a JSON object with:
                - sentiment: "positive", "negative", or "neutral"
                - confidence: 0-100
                
                Text: {text[:500]}"""  # Limit text length for faster processing
            elif analysis_type == "competitors":
                prompt = f"""Extract competitor brands mentioned in this text. Return only a JSON object with:
                - competitors: list of competitor brand names
                
                Text: {text[:500]}"""  # Limit text length for faster processing
            elif analysis_type == "brand_analysis":
                prompt = f"""Analyze this brand-related content. Return only a JSON object with:
                - key_insights: list of main insights
                - strengths: list of brand strengths mentioned
                - weaknesses: list of brand weaknesses mentioned
                
                Text: {text[:500]}"""  # Limit text length for faster processing
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,  # Reduced from 500 for faster response
                "temperature": 0.3
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:  # Added timeout
                response = await client.post(self.openai_url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except:
                        return {"analysis": content, "error": "Failed to parse JSON"}
                else:
                    return {"error": f"OpenAI API error: {response.status_code}"}
                    
        except Exception as e:
            return {"error": f"ChatGPT analysis failed: {str(e)}"}
    
    def generate_chatgpt_prompts(self, brand_name: str, website: str) -> list[dict]:
        """Generate targeted prompts for ChatGPT analysis"""
        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        
        prompts = [
            {
                "sector": "Customer Reviews & Feedback",
                "prompt": f"What are people saying about {brand_name} ({website})? Recent reviews, customer feedback, user experiences, and overall satisfaction ratings."
            },
            {
                "sector": "Industry Analysis & Market Position",
                "prompt": f"How does {brand_name} ({website}) position itself in the market? Industry analysis, market share, competitive landscape, and business strategy."
            },
            {
                "sector": "Product/Service Comparison",
                "prompt": f"How does {brand_name} ({website}) compare to competitors? Product features, pricing, quality, and value proposition analysis."
            },
            {
                "sector": "Brand Reputation & Trust",
                "prompt": f"What is the reputation and trust level of {brand_name} ({website})? Brand perception, credibility, reliability, and customer trust factors."
            },
            {
                "sector": "Business Performance & Growth",
                "prompt": f"How is {brand_name} ({website}) performing? Recent news, business metrics, growth trends, financial performance, and future outlook."
            }
        ]
        
        return prompts
    
    def analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        text_lower = text.lower()
        positive_words = ["excellent", "great", "amazing", "best", "outstanding", "recommend", "love", "fantastic", "superior"]
        negative_words = ["terrible", "awful", "worst", "avoid", "hate", "disappointing", "poor", "bad", "frustrating"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def extract_competitors(self, text: str, brand_name: str) -> list[str]:
        """Extract competitor mentions from text"""
        # Improved competitor extraction to avoid false positives
        competitors = []
        text_lower = text.lower()
        brand_name_lower = brand_name.lower()
        
        # Common competitor indicators
        competitor_indicators = ["vs", "versus", "compared to", "alternative to", "competitor", "rival", "competes with", "alternative"]
        
        # Split text into sentences for better analysis
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains competitor indicators
            has_indicator = any(indicator in sentence_lower for indicator in competitor_indicators)
            
            if has_indicator:
                # Extract potential competitor names (more sophisticated)
                words = sentence.split()
                for i, word in enumerate(words):
                    # Look for capitalized words that could be brand names
                    if (word[0].isupper() and 
                        len(word) > 2 and 
                        word.lower() != brand_name_lower and
                        not word.lower() in ['the', 'and', 'or', 'for', 'with', 'from', 'this', 'that', 'they', 'their']):
                        
                        # Check if it's likely a brand name (not a common word)
                        potential_competitor = word.strip(".,!?;:")
                        
                        # Additional filtering to avoid common words
                        common_words = ['company', 'business', 'service', 'product', 'solution', 'platform', 'software', 'technology']
                        if (potential_competitor.lower() not in common_words and
                            potential_competitor.lower() not in [brand_name_lower]):
                            competitors.append(potential_competitor)
        
        # Remove duplicates and limit results
        unique_competitors = list(set(competitors))
        return unique_competitors[:5]  # Return top 5 unique competitors
    
    def calculate_visibility_score(self, results: list[dict], brand_name: str) -> int:
        """Calculate visibility frequency score"""
        brand_mentions = 0
        total_results = len(results)
        
        for result in results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            if brand_name.lower() in title or brand_name.lower() in snippet:
                brand_mentions += 1
        
        return int((brand_mentions / total_results) * 100) if total_results > 0 else 0

    def generate_seo_keywords(self, brand_name: str, website: str, search_results: list) -> list[str]:
        """Generate SEO keywords from search results and brand analysis"""
        keywords = []
        
        # Extract domain for keyword generation
        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        
        # Brand-specific keywords
        keywords.extend([
            brand_name.lower(),
            f"{brand_name} reviews",
            f"{brand_name} {domain}",
            f"{brand_name} alternatives",
            f"{brand_name} competitors",
            f"{brand_name} pricing",
            f"{brand_name} features"
        ])
        
        # Extract keywords from search results
        for result in search_results[:5]:  # Use top 5 results
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            # Extract meaningful words (3+ characters, not common stop words)
            stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "a", "an", "as", "so", "than", "too", "very", "just", "now", "then", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "such", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once"}
            
            words = title.split() + snippet.split()
            for word in words:
                word = word.strip(".,!?;:()[]{}'\"")
                if len(word) >= 3 and word not in stop_words and word.isalpha():
                    keywords.append(word)
        
        # Add industry-specific keywords
        industry_keywords = [
            "brand visibility",
            "online presence",
            "digital marketing",
            "seo optimization",
            "brand reputation",
            "customer reviews",
            "market analysis",
            "competitive analysis",
            "brand monitoring",
            "sentiment analysis"
        ]
        keywords.extend(industry_keywords)
        
        # Remove duplicates and limit to top keywords
        unique_keywords = list(set(keywords))
        return unique_keywords[:15]  # Return top 15 keywords
    
    def generate_geo_heatmap_data(self, search_results: list) -> dict:
        """Generate geographic distribution data for brand mentions"""
        # This would typically use geolocation data from search results
        # For now, we'll create a simulated heatmap based on result sources
        geo_data = {
            "regions": {
                "North America": len([r for r in search_results if any(domain in r.get("link", "") for domain in [".com", ".us", ".ca"])]),
                "Europe": len([r for r in search_results if any(domain in r.get("link", "") for domain in [".uk", ".de", ".fr", ".it", ".es"])]),
                "Asia": len([r for r in search_results if any(domain in r.get("link", "") for domain in [".jp", ".cn", ".in", ".kr"])]),
                "Global": len([r for r in search_results if any(domain in r.get("link", "") for domain in [".org", ".net", ".io"])])
            },
            "top_countries": ["United States", "United Kingdom", "Canada", "Germany", "India"],
            "total_mentions": len(search_results)
        }
        return geo_data
    
    def identify_seo_gaps(self, brand_name: str, search_results: list, competitors: list) -> list[dict]:
        """Identify SEO keyword gaps and opportunities"""
        gaps = []
        
        # Analyze search result titles and snippets for missing keywords
        all_text = " ".join([r.get("title", "") + " " + r.get("snippet", "") for r in search_results])
        all_text_lower = all_text.lower()
        
        # Common SEO gaps to check
        gap_opportunities = [
            {
                "type": "long_tail_keywords",
                "keywords": [f"{brand_name} best practices", f"{brand_name} tutorial", f"{brand_name} guide"],
                "priority": "high"
            },
            {
                "type": "comparison_keywords", 
                "keywords": [f"{brand_name} vs competitors", f"{brand_name} alternatives", f"{brand_name} comparison"],
                "priority": "medium"
            },
            {
                "type": "problem_solving_keywords",
                "keywords": [f"{brand_name} solutions", f"{brand_name} problems", f"{brand_name} help"],
                "priority": "medium"
            },
            {
                "type": "industry_keywords",
                "keywords": [f"{brand_name} industry", f"{brand_name} market", f"{brand_name} trends"],
                "priority": "low"
            }
        ]
        
        for opportunity in gap_opportunities:
            missing_keywords = []
            for keyword in opportunity["keywords"]:
                if keyword.lower() not in all_text_lower:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                gaps.append({
                    "type": opportunity["type"],
                    "missing_keywords": missing_keywords,
                    "priority": opportunity["priority"],
                    "recommendation": f"Create content targeting: {', '.join(missing_keywords[:3])}"
                })
        
        return gaps
    
    def generate_blog_content(self, brand_name: str, website: str, analysis_data: dict) -> dict:
        """Generate SEO-optimized blog post from brand visibility analysis"""
        
        # Extract data from analysis
        presence_data = analysis_data.get("presence", {})
        sentiment_data = analysis_data.get("sentiment", {})
        competitors = analysis_data.get("competitors", [])
        search_results = analysis_data.get("search_results", [])
        visibility_scores = analysis_data.get("visibility_scores", {})
        improvement_suggestions = analysis_data.get("improvement_suggestions", [])
        chatgpt_analysis = analysis_data.get("chatgpt_analysis", {})
        
        # Generate SEO keywords
        seo_keywords = self.generate_seo_keywords(brand_name, website, search_results)
        
        # Generate geographic data
        geo_data = self.generate_geo_heatmap_data(search_results)
        
        # Identify SEO gaps
        seo_gaps = self.identify_seo_gaps(brand_name, search_results, competitors)
        
        # Create blog content
        blog_title = f"{brand_name} Brand Visibility Analysis: Complete Digital Presence Report"
        
        # Meta description
        meta_description = f"Comprehensive {brand_name} brand visibility analysis covering online presence, sentiment analysis, competitor insights, and SEO optimization recommendations. Discover how {brand_name} performs across digital platforms."
        
        # Blog content sections
        introduction = f"""
# {brand_name} Brand Visibility Analysis: Complete Digital Presence Report

In today's digital landscape, maintaining a strong online presence is crucial for brand success. This comprehensive analysis examines {brand_name}'s digital footprint across search engines, social media, and AI platforms to provide actionable insights for improving brand visibility and market positioning.

## Executive Summary

{brand_name} currently demonstrates a {presence_data.get('presence_strength', 'moderate')} online presence with {presence_data.get('total_results', 0)} total search results, {presence_data.get('brand_mentions', 0)} of which directly mention the brand. The overall sentiment analysis reveals a {sentiment_data.get('overall_sentiment', 'neutral')} perception among online audiences.
"""
        
        # Brand Presence Analysis
        presence_section = f"""
## Brand Presence Analysis

### Search Engine Visibility
- **Total Search Results**: {presence_data.get('total_results', 0)}
- **Brand Mentions**: {presence_data.get('brand_mentions', 0)}
- **Top Ranking Positions**: {', '.join(map(str, presence_data.get('top_ranking_positions', [])))}
- **Presence Strength**: {presence_data.get('presence_strength', 'moderate').title()}

### Geographic Distribution
Our analysis reveals {brand_name}'s global reach across multiple regions:
"""
        
        for region, count in geo_data["regions"].items():
            if count > 0:
                presence_section += f"- **{region}**: {count} mentions\n"
        
        # Sentiment Analysis
        sentiment_section = f"""
## Sentiment Analysis

### Overall Sentiment: {sentiment_data.get('overall_sentiment', 'neutral').title()}

**Sentiment Distribution:**
- Positive Mentions: {sentiment_data.get('distribution', {}).get('positive', 0)}
- Neutral Mentions: {sentiment_data.get('distribution', {}).get('neutral', 0)}
- Negative Mentions: {sentiment_data.get('distribution', {}).get('negative', 0)}

### Key Insights from AI Analysis
"""
        
        if chatgpt_analysis.get("strengths"):
            sentiment_section += f"**Strengths:** {', '.join(chatgpt_analysis['strengths'][:3])}\n\n"
        
        if chatgpt_analysis.get("weaknesses"):
            sentiment_section += f"**Areas for Improvement:** {', '.join(chatgpt_analysis['weaknesses'][:3])}\n\n"
        
        # Competitor Analysis
        competitor_section = f"""
## Competitive Landscape

### Key Competitors Identified
{', '.join(competitors[:8])}

### Market Position Analysis
{brand_name} operates in a competitive landscape with {len(competitors)} identified competitors. Understanding this competitive environment is crucial for strategic positioning and differentiation.
"""
        
        # Visibility Breakdown
        visibility_section = """
## Visibility Breakdown by Sector

### Sector Performance Analysis
"""
        
        for sector, data in visibility_scores.items():
            score_level = "Very High" if data.get("visibility_score", 0) >= 80 else "High" if data.get("visibility_score", 0) >= 60 else "Moderate" if data.get("visibility_score", 0) >= 40 else "Low"
            visibility_section += f"- **{sector}**: {score_level} visibility ({data.get('visibility_score', 0)}% score) with {data.get('sentiment', 'neutral')} sentiment\n"
        
        # SEO Analysis
        seo_section = f"""
## SEO Analysis & Keyword Opportunities

### Primary SEO Keywords
{', '.join(seo_keywords[:10])}

### Identified SEO Gaps
"""
        
        for gap in seo_gaps[:5]:
            seo_section += f"- **{gap['type'].replace('_', ' ').title()}**: {gap['recommendation']}\n"
        
        # Recommendations
        recommendations_section = """
## Strategic Recommendations

### Immediate Actions
"""
        
        for suggestion in improvement_suggestions[:5]:
            recommendations_section += f"- {suggestion}\n"
        
        # Conclusion
        conclusion = f"""
## Conclusion

{brand_name}'s digital presence analysis reveals both opportunities and challenges in the current market landscape. By implementing the recommended strategies, {brand_name} can significantly improve its online visibility, enhance brand perception, and strengthen its competitive position.

### Next Steps
1. **Implement SEO Optimization**: Focus on identified keyword gaps
2. **Enhance Content Strategy**: Create targeted content for underserved topics
3. **Monitor Performance**: Establish regular tracking and reporting
4. **Competitive Analysis**: Continuously monitor competitor activities
5. **Sentiment Management**: Address negative feedback and amplify positive mentions

---

*This analysis was generated using advanced AI-powered brand monitoring tools and search engine data to provide comprehensive insights into {brand_name}'s digital presence.*
"""
        
        # Combine all sections
        full_content = introduction + presence_section + sentiment_section + competitor_section + visibility_section + seo_section + recommendations_section + conclusion
        
        # Create blog post structure
        blog_post = {
            "title": blog_title,
            "meta_description": meta_description,
            "content": full_content,
            "seo_data": {
                "keywords": seo_keywords,
                "geo_distribution": geo_data,
                "seo_gaps": seo_gaps,
                "word_count": len(full_content.split()),
                "readability_score": "Intermediate"
            },
            "analysis_summary": {
                "brand_name": brand_name,
                "website": website,
                "analysis_date": "as of today",
                "key_metrics": {
                    "total_results": presence_data.get('total_results', 0),
                    "brand_mentions": presence_data.get('brand_mentions', 0),
                    "sentiment": sentiment_data.get('overall_sentiment', 'neutral'),
                    "competitors_count": len(competitors),
                    "seo_gaps_identified": len(seo_gaps)
                }
            }
        }
        
        return blog_post

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Brand Visibility Monitoring MCP Server",
    auth=SimpleBearerAuthProvider(AUTH_TOKEN),
)

# Add a simple health check endpoint
@mcp.custom_route("/", methods=["GET"])
async def health_check(request):
    """Health check endpoint for Railway"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "healthy",
        "service": "Brand Visibility Monitoring MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "health": "/"
        }
    })

# Add an additional health check endpoint for Railway
@mcp.custom_route("/health", methods=["GET"])
async def health_check_alt(request):
    """Alternative health check endpoint for Railway"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "healthy",
        "service": "Brand Visibility Monitoring MCP Server",
        "version": "1.0.0"
    })

# Initialize brand visibility monitor
brand_monitor = BrandVisibilityMonitor(SERP_API_KEY, OPENAI_API_KEY)

# --- Tool: validate (required by Puch) ---
ValidateDescription = RichToolDescription(
    description="Validate that the MCP server is working correctly.",
    use_when="Use this to test if the MCP server is functioning properly.",
    side_effects="Returns a simple validation message confirming the server is operational.",
)

@mcp.tool(description=ValidateDescription.model_dump_json())
async def validate() -> str:
    """
    Validate that the MCP server is working correctly.
    Returns your phone number in the required format for Puch AI.
    """
    return MY_NUMBER

# --- Tool: Brand Visibility Monitor ---
BrandVisibilityDescription = RichToolDescription(
    description="Monitor brand visibility across ChatGPT by analyzing search results and generating targeted prompts for comprehensive brand analysis.",
    use_when="Use this to analyze brand presence, sentiment, competitors, and visibility across different sectors using SERP search results and ChatGPT analysis.",
    side_effects="Returns structured analysis with presence metrics, sentiment analysis, competitor identification, and improvement suggestions.",
)

@mcp.tool(description=BrandVisibilityDescription.model_dump_json())
async def brand_visibility_monitor(
    website: Annotated[str, Field(description="The website URL to analyze (e.g., 'https://example.com')")],
    brand_name: Annotated[str, Field(description="The brand name to analyze")],
) -> str:
    """
    Monitor brand visibility across ChatGPT by analyzing search results and generating targeted prompts.
    Returns structured JSON with comprehensive brand analysis from SERP and ChatGPT platforms.
    """
    try:
        # Search for brand references
        search_results = await brand_monitor.search_brand_references(website, brand_name)
        
        # Generate ChatGPT prompts
        prompts = brand_monitor.generate_chatgpt_prompts(brand_name, website)
        
        # Analyze results
        presence_data = {
            "total_results": len(search_results),
            "brand_mentions": 0,
            "top_ranking_positions": []
        }
        
        sentiment_data = {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "overall_sentiment": "neutral"
        }
        
        all_competitors = []
        visibility_scores = {}
        chatgpt_analysis = {}
        
        # Analyze each search result (limit to first 10 for speed)
        for i, result in enumerate(search_results[:10]):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            full_text = f"{title} {snippet}"
            
            # Check presence
            if brand_name.lower() in full_text.lower():
                presence_data["brand_mentions"] += 1
                if i < 10:  # Top 10 results
                    presence_data["top_ranking_positions"].append(i + 1)
            
            # Use basic sentiment analysis for speed (skip ChatGPT for individual results)
            sentiment = brand_monitor.analyze_sentiment(full_text)
            sentiment_data[sentiment] += 1
            
            # Extract competitors using basic method for speed
            competitors = brand_monitor.extract_competitors(full_text, brand_name)
            all_competitors.extend(competitors)
        
        # Determine overall sentiment
        if sentiment_data["positive"] > sentiment_data["negative"]:
            sentiment_data["overall_sentiment"] = "positive"
        elif sentiment_data["negative"] > sentiment_data["positive"]:
            sentiment_data["overall_sentiment"] = "negative"
        
        # Get unique competitors
        unique_competitors = list(set(all_competitors))[:10]
        
        # Analyze prompts with ChatGPT (limited for speed)
        for i, prompt_data in enumerate(prompts[:2]):
            sector = prompt_data["sector"]
            prompt_text = prompt_data["prompt"]
            
            # Use ChatGPT for comprehensive analysis (only for first 2 prompts)
            if brand_monitor.openai_api_key:
                try:
                    analysis = await brand_monitor.analyze_with_chatgpt(prompt_text, "brand_analysis")
                    chatgpt_analysis[sector] = analysis
                except:
                    chatgpt_analysis[sector] = {"error": "ChatGPT analysis failed"}
            
            # Basic analysis
            presence = brand_name.lower() in prompt_text.lower()
            sentiment = brand_monitor.analyze_sentiment(prompt_text)
            competitors = brand_monitor.extract_competitors(prompt_text, brand_name)
            visibility_score = brand_monitor.calculate_visibility_score(search_results, brand_name)
            
            visibility_scores[sector] = {
                "presence": presence,
                "sentiment": sentiment,
                "competitors": competitors,
                "visibility_score": visibility_score
            }
        
        # Add remaining prompts with basic analysis only
        for i, prompt_data in enumerate(prompts[2:], 2):
            sector = prompt_data["sector"]
            prompt_text = prompt_data["prompt"]
            
            presence = brand_name.lower() in prompt_text.lower()
            sentiment = brand_monitor.analyze_sentiment(prompt_text)
            competitors = brand_monitor.extract_competitors(prompt_text, brand_name)
            visibility_score = brand_monitor.calculate_visibility_score(search_results, brand_name)
            
            visibility_scores[sector] = {
                "presence": presence,
                "sentiment": sentiment,
                "competitors": competitors,
                "visibility_score": visibility_score
            }
        
        # Generate improvement suggestions
        improvement_suggestions = []
        
        if presence_data["brand_mentions"] < len(search_results) * 0.3:
            improvement_suggestions.append("Increase brand mention frequency in online content and SEO optimization")
        
        if sentiment_data["negative"] > sentiment_data["positive"]:
            improvement_suggestions.append("Address negative sentiment through customer service improvements and reputation management")
        
        if len(unique_competitors) > 5:
            improvement_suggestions.append("Develop stronger competitive differentiation and unique value propositions")
        
        if presence_data["top_ranking_positions"]:
            avg_position = sum(presence_data["top_ranking_positions"]) / len(presence_data["top_ranking_positions"])
            if avg_position > 5:
                improvement_suggestions.append("Improve search engine rankings through better SEO and content strategy")
        
        if not improvement_suggestions:
            improvement_suggestions.append("Maintain current brand visibility and continue monitoring for opportunities")
        
        # Calculate overall visibility metrics
        overall_visibility_score = brand_monitor.calculate_visibility_score(search_results, brand_name)
        presence_strength = "very strong" if presence_data["brand_mentions"] >= len(search_results) * 0.7 else "strong" if presence_data["brand_mentions"] >= len(search_results) * 0.4 else "moderate" if presence_data["brand_mentions"] >= len(search_results) * 0.2 else "weak"
        
        # Create comprehensive report
        report = {
            "brand": brand_name,
            "website": website,
            "analysis_date": "as of today",
            "overall_presence": {
                "description": f"{brand_name} has a {presence_strength} online presence with {presence_data['total_results']} total results, {presence_data['brand_mentions']} of which directly mention the brand.",
                "total_results": presence_data["total_results"],
                "brand_mentions": presence_data["brand_mentions"],
                "top_ranking_positions": presence_data["top_ranking_positions"],
                "presence_strength": presence_strength
            },
            "sentiment": {
                "overall": sentiment_data["overall_sentiment"],
                "distribution": {
                    "positive": sentiment_data["positive"],
                    "neutral": sentiment_data["neutral"],
                    "negative": sentiment_data["negative"]
                }
            },
            "competitors": {
                "key_competitors": unique_competitors,
                "total_identified": len(unique_competitors)
            },
            "visibility_breakdown": {},
            "improvement_suggestions": improvement_suggestions,
            "chatgpt_analysis": {
                "highlights": {},
                "strengths": [],
                "weaknesses": []
            },
            "technical_metrics": {
                "overall_visibility_score": overall_visibility_score,
                "search_coverage": f"{presence_data['brand_mentions']}/{presence_data['total_results']} results mention the brand",
                "sentiment_ratio": f"{sentiment_data['positive']}:{sentiment_data['neutral']}:{sentiment_data['negative']}"
            }
        }
        
        # Add visibility breakdown for each sector
        for sector, data in visibility_scores.items():
            score_level = "very high" if data["visibility_score"] >= 80 else "high" if data["visibility_score"] >= 60 else "moderate" if data["visibility_score"] >= 40 else "low"
            report["visibility_breakdown"][sector] = {
                "visibility": f"{score_level} visibility ({data['visibility_score']}% score)",
                "sentiment": data["sentiment"],
                "presence": data["presence"],
                "competitors": data["competitors"]
            }
        
        # Extract ChatGPT insights
        if chatgpt_analysis:
            for sector, analysis in chatgpt_analysis.items():
                if "error" not in analysis:
                    if "strengths" in analysis:
                        report["chatgpt_analysis"]["strengths"].extend(analysis.get("strengths", []))
                    if "weaknesses" in analysis:
                        report["chatgpt_analysis"]["weaknesses"].extend(analysis.get("weaknesses", []))
                    if "key_insights" in analysis:
                        report["chatgpt_analysis"]["highlights"][sector] = analysis.get("key_insights", [])
        
        # Format the report as a comprehensive text summary
        summary = f"""Here's the brand visibility report for {brand_name} {report['analysis_date']}:

Overall Presence: {brand_name} has a {presence_strength} online presence with {presence_data['total_results']} total results, {presence_data['brand_mentions']} of which directly mention the brand. It consistently ranks highly in search results, occupying the top {len(presence_data['top_ranking_positions'])} positions.

Sentiment: The overall sentiment towards {brand_name} is {sentiment_data['overall_sentiment']}.

Key Competitors: Competitors identified include {', '.join(unique_competitors[:8])}.

Visibility Breakdown:
"""
        
        for sector, data in report["visibility_breakdown"].items():
            summary += f"- {sector}: {data['visibility']} with a {data['sentiment']} sentiment.\n"
        
        summary += f"""
Improvement Suggestions:
"""
        for suggestion in improvement_suggestions:
            summary += f"- {suggestion}\n"
        
        if report["chatgpt_analysis"]["strengths"] or report["chatgpt_analysis"]["weaknesses"]:
            summary += f"""
ChatGPT Analysis Highlights:
"""
            if report["chatgpt_analysis"]["strengths"]:
                summary += f"- Strengths: {', '.join(set(report['chatgpt_analysis']['strengths'][:3]))}.\n"
            if report["chatgpt_analysis"]["weaknesses"]:
                summary += f"- Weaknesses: {', '.join(set(report['chatgpt_analysis']['weaknesses'][:3]))}.\n"
        
        summary += f"""
Would you like me to delve deeper into any specific area of this report, or perhaps compare {brand_name} to one of its competitors?"""
        
        # Return both the structured data and the formatted summary
        result = {
            "formatted_summary": summary,
            "detailed_data": report,
            "raw_analysis": {
                "presence": presence_data,
                "sentiment": sentiment_data,
                "competitors": unique_competitors,
                "visibility_scores": visibility_scores,
                "chatgpt_analysis": chatgpt_analysis
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to monitor brand visibility: {str(e)}"))

# --- Tool: Auto-Generated Blog Posts ---
BlogPostDescription = RichToolDescription(
    description="Generate SEO-optimized blog posts from brand visibility analysis results, including sentiment analysis, competitor insights, and SEO recommendations.",
    use_when="Use this to create comprehensive blog content summarizing brand visibility analysis with SEO optimization, geographic insights, and actionable recommendations.",
    side_effects="Returns a complete blog post with SEO keywords, meta descriptions, and structured content based on brand analysis data.",
)

@mcp.tool(description=BlogPostDescription.model_dump_json())
async def generate_brand_blog_post(
    website: Annotated[str, Field(description="The website URL to analyze (e.g., 'https://example.com')")],
    brand_name: Annotated[str, Field(description="The brand name to analyze")],
    include_geo_analysis: Annotated[bool, Field(description="Include geographic distribution analysis in the blog")] = True,
    include_seo_gaps: Annotated[bool, Field(description="Include SEO gap analysis and keyword opportunities")] = True,
) -> str:
    """
    Generate an SEO-optimized blog post from brand visibility analysis.
    Returns comprehensive blog content with SEO keywords, geographic insights, and strategic recommendations.
    """
    try:
        # First, run the brand visibility analysis
        visibility_result = await brand_visibility_monitor(website, brand_name)
        visibility_data = json.loads(visibility_result)
        
        # Extract the raw analysis data
        raw_analysis = visibility_data.get("raw_analysis", {})
        
        # Prepare analysis data for blog generation
        analysis_data = {
            "presence": raw_analysis.get("presence", {}),
            "sentiment": raw_analysis.get("sentiment", {}),
            "competitors": raw_analysis.get("competitors", []),
            "search_results": [],  # We'll need to get this from the search
            "visibility_scores": raw_analysis.get("visibility_scores", {}),
            "improvement_suggestions": [],  # We'll extract from the detailed data
            "chatgpt_analysis": raw_analysis.get("chatgpt_analysis", {})
        }
        
        # Get search results by running a quick search
        search_results = await brand_monitor.search_brand_references(website, brand_name)
        analysis_data["search_results"] = search_results
        
        # Extract improvement suggestions from the detailed data
        detailed_data = visibility_data.get("detailed_data", {})
        analysis_data["improvement_suggestions"] = detailed_data.get("improvement_suggestions", [])
        
        # Generate the blog post
        blog_post = brand_monitor.generate_blog_content(brand_name, website, analysis_data)
        
        # Create the final result
        result = {
            "blog_post": blog_post,
            "analysis_summary": {
                "brand_name": brand_name,
                "website": website,
                "generation_date": "as of today",
                "blog_metrics": {
                    "title": blog_post["title"],
                    "word_count": blog_post["seo_data"]["word_count"],
                    "keywords_count": len(blog_post["seo_data"]["keywords"]),
                    "seo_gaps_identified": len(blog_post["seo_data"]["seo_gaps"]),
                    "geo_regions_analyzed": len(blog_post["seo_data"]["geo_distribution"]["regions"])
                }
            },
            "seo_optimization": {
                "primary_keywords": blog_post["seo_data"]["keywords"][:5],
                "meta_description": blog_post["meta_description"],
                "readability_score": blog_post["seo_data"]["readability_score"],
                "recommended_actions": [
                    "Publish the blog post with the provided SEO keywords",
                    "Use the meta description for social media sharing",
                    "Implement the identified SEO gap strategies",
                    "Monitor performance using the suggested keywords",
                    "Create follow-up content based on competitor analysis"
                ]
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to generate blog post: {str(e)}"))

# --- Run MCP Server ---
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from Railway environment variable or default to 8086
    port = int(os.environ.get("PORT", 8086))
    
    print(f"🚀 Starting MCP server on http://0.0.0.0:{port}")
    print(f"🔑 Bearer Token: {AUTH_TOKEN}")
    print(f"📱 Your Number: {MY_NUMBER}")
    print(f"🌐 Public URL: Will be available after Railway deployment")
    print("=" * 50)
    
    # For Railway deployment, use the FastMCP HTTP app
    app = mcp.http_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

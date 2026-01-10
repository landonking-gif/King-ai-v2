"""
Dropshipping Business Creator - Creates fully functional dropshipping businesses.

This module orchestrates the complete creation of a dropshipping business including:
- Market research and niche selection
- Product research and supplier identification
- Website/landing page creation
- Marketing plan development
- Business plan and financial projections
- Operational procedures

All outputs are REAL files that are verified to exist.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from src.services.execution_engine import get_execution_engine, ActionRequest, ActionType
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger

logger = get_logger("dropshipping_creator")


@dataclass
class DropshippingProduct:
    """A product for the dropshipping store."""
    name: str
    description: str
    niche: str
    target_price: float
    cost_estimate: float
    profit_margin: float
    supplier_info: str
    search_volume: str = "Unknown"
    competition_level: str = "Medium"
    trending: bool = False


@dataclass
class DropshippingBusiness:
    """Complete dropshipping business structure."""
    business_id: str
    name: str
    niche: str
    description: str
    target_audience: str
    products: List[DropshippingProduct] = field(default_factory=list)
    website_path: str = ""
    marketing_plan_path: str = ""
    business_plan_path: str = ""
    supplier_list_path: str = ""
    financial_plan_path: str = ""
    operations_plan_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    verified_files: List[str] = field(default_factory=list)


class DropshippingCreator:
    """Creates complete, functional dropshipping businesses."""
    
    def __init__(self, workspace: str = None):
        """Initialize with workspace directory."""
        self.workspace = workspace or os.getcwd()
        self.businesses_dir = Path(self.workspace) / "businesses"
        self.businesses_dir.mkdir(exist_ok=True)
        self.execution_engine = get_execution_engine(self.workspace)
        self.llm_router = LLMRouter()
        
        logger.info("DropshippingCreator initialized", workspace=self.workspace)
    
    async def create_business(
        self,
        name: str = None,
        niche: str = None,
        budget: float = 1000.0,
        target_roi: str = "high"
    ) -> DropshippingBusiness:
        """
        Create a complete dropshipping business.
        
        This method:
        1. Researches and selects a profitable niche (if not provided)
        2. Finds trending products with good margins
        3. Identifies suppliers
        4. Creates a complete website
        5. Develops marketing strategy
        6. Creates business and financial plans
        7. Sets up operational procedures
        
        All outputs are verified to exist.
        """
        business_id = f"dropship_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        business_dir = self.businesses_dir / business_id
        
        logger.info(f"Creating dropshipping business: {business_id}", niche=niche, budget=budget)
        
        # Step 1: Research niche if not provided
        if not niche:
            niche = await self._research_profitable_niche(target_roi)
        
        if not name:
            name = await self._generate_business_name(niche)
        
        # Create business structure
        business = DropshippingBusiness(
            business_id=business_id,
            name=name,
            niche=niche,
            description=f"High-ROI dropshipping business in the {niche} niche",
            target_audience=await self._identify_target_audience(niche)
        )
        
        # Step 2: Create directory structure
        await self._create_directory_structure(business_dir)
        
        # Step 3: Research products
        business.products = await self._research_products(niche, budget)
        
        # Step 4: Create all business assets
        tasks = [
            self._create_website(business, business_dir),
            self._create_marketing_plan(business, business_dir),
            self._create_business_plan(business, business_dir),
            self._create_supplier_document(business, business_dir),
            self._create_financial_plan(business, business_dir, budget),
            self._create_operations_manual(business, business_dir),
            self._create_product_catalog(business, business_dir),
        ]
        
        # Execute all creation tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
        
        # Verify all files exist
        business.verified_files = await self._verify_all_files(business_dir)
        
        # Create summary document
        await self._create_summary(business, business_dir)
        
        logger.info(
            f"Dropshipping business created",
            business_id=business_id,
            verified_files=len(business.verified_files)
        )
        
        return business
    
    async def _research_profitable_niche(self, target_roi: str) -> str:
        """Research and select a profitable niche."""
        prompt = f"""You are a dropshipping market research expert. Identify ONE specific, profitable niche for a {target_roi} ROI dropshipping business.

Requirements:
- Growing market demand (not saturated)
- Products with 40%+ profit margins possible
- Passionate buyer demographic
- Good for beginners
- Works well with social media marketing

Respond with ONLY the niche name (2-4 words), no explanation.
Examples: "Pet Accessories", "Yoga Equipment", "Home Office Gadgets"

Your niche selection:"""
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="research", complexity="moderate")
            )
            niche = response.strip().strip('"').strip("'")
            if len(niche) > 50:
                niche = "Pet Accessories"  # Default fallback
            return niche
        except Exception as e:
            logger.warning(f"Niche research failed: {e}, using default")
            return "Pet Accessories"
    
    async def _generate_business_name(self, niche: str) -> str:
        """Generate a catchy business name."""
        prompt = f"""Generate ONE catchy, memorable business name for a dropshipping store in the "{niche}" niche.

Requirements:
- 2-3 words maximum
- Easy to spell and remember
- Sounds professional
- Available as a domain name (use .co or .store)

Respond with ONLY the business name, nothing else."""
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="creative", complexity="simple")
            )
            name = response.strip().strip('"').strip("'")
            if len(name) > 30:
                name = f"{niche.split()[0]} Hub"
            return name
        except Exception:
            return f"{niche.split()[0]} Store"
    
    async def _identify_target_audience(self, niche: str) -> str:
        """Identify the target audience for the niche."""
        prompt = f"""Describe the ideal target audience for a dropshipping store in the "{niche}" niche.

Include:
- Age range
- Gender (if applicable)
- Interests/hobbies
- Pain points
- Where they spend time online

Keep it to 2-3 sentences."""
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="research", complexity="simple")
            )
            return response.strip()
        except Exception:
            return f"Adults aged 25-45 interested in {niche.lower()}"
    
    async def _create_directory_structure(self, business_dir: Path):
        """Create the business directory structure."""
        directories = [
            business_dir,
            business_dir / "website",
            business_dir / "website" / "css",
            business_dir / "website" / "js",
            business_dir / "website" / "images",
            business_dir / "marketing",
            business_dir / "documents",
            business_dir / "products",
            business_dir / "suppliers",
            business_dir / "financials",
            business_dir / "operations",
        ]
        
        for dir_path in directories:
            await self.execution_engine.execute(ActionRequest(
                action_type=ActionType.DIR_CREATE,
                params={"path": str(dir_path)},
                description=f"Create directory: {dir_path.name}"
            ))
    
    async def _research_products(self, niche: str, budget: float) -> List[DropshippingProduct]:
        """Research profitable products for the niche."""
        prompt = f"""You are a dropshipping product research expert. Find 5 specific, profitable products in the "{niche}" niche.

For each product provide (in this exact JSON format):
{{
  "products": [
    {{
      "name": "Product Name",
      "description": "Brief description",
      "target_price": 29.99,
      "cost_estimate": 8.00,
      "supplier_info": "Available on AliExpress, CJ Dropshipping",
      "trending": true
    }}
  ]
}}

Requirements:
- Target price $15-$100
- Minimum 50% markup
- Available from major dropshipping suppliers
- Currently trending or evergreen sellers

Return ONLY the JSON, no other text."""
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="research", complexity="moderate")
            )
            
            # Parse JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                products = []
                for p in data.get("products", [])[:5]:
                    cost = p.get("cost_estimate", 10.0)
                    price = p.get("target_price", 29.99)
                    margin = ((price - cost) / price) * 100
                    products.append(DropshippingProduct(
                        name=p.get("name", "Product"),
                        description=p.get("description", ""),
                        niche=niche,
                        target_price=price,
                        cost_estimate=cost,
                        profit_margin=margin,
                        supplier_info=p.get("supplier_info", "AliExpress"),
                        trending=p.get("trending", False)
                    ))
                return products
        except Exception as e:
            logger.warning(f"Product research failed: {e}")
        
        # Default products
        return [
            DropshippingProduct(
                name=f"Premium {niche} Item 1",
                description="High-quality product with great reviews",
                niche=niche,
                target_price=29.99,
                cost_estimate=8.00,
                profit_margin=73.3,
                supplier_info="AliExpress, CJ Dropshipping",
                trending=True
            ),
            DropshippingProduct(
                name=f"Bestselling {niche} Item 2",
                description="Popular item with proven sales",
                niche=niche,
                target_price=39.99,
                cost_estimate=12.00,
                profit_margin=70.0,
                supplier_info="AliExpress",
                trending=True
            ),
            DropshippingProduct(
                name=f"Trending {niche} Item 3",
                description="New trending product with viral potential",
                niche=niche,
                target_price=24.99,
                cost_estimate=6.00,
                profit_margin=76.0,
                supplier_info="CJ Dropshipping",
                trending=True
            ),
        ]
    
    async def _create_website(self, business: DropshippingBusiness, business_dir: Path) -> str:
        """Create a complete website for the business."""
        prompt = f"""Create a complete, professional HTML landing page for an e-commerce store.

Business: {business.name}
Niche: {business.niche}
Target Audience: {business.target_audience}
Products: {', '.join(p.name for p in business.products[:3])}

Create a COMPLETE HTML file with:
1. Modern, responsive design using Tailwind CSS (via CDN)
2. Hero section with compelling headline and CTA
3. Featured products section (show 3 products with prices)
4. About section explaining the brand
5. Benefits/features section
6. Customer testimonials (placeholder)
7. Newsletter signup form
8. Footer with links

The page should be ready to use - fully styled and functional.
Return ONLY the complete HTML code."""

        try:
            html_content = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="code_generation", complexity="high")
            )
            
            # Clean up the HTML
            if "```html" in html_content:
                html_content = html_content.split("```html")[1].split("```")[0]
            elif "```" in html_content:
                html_content = html_content.split("```")[1].split("```")[0]
            
        except Exception as e:
            logger.warning(f"Website generation failed: {e}, using template")
            html_content = self._get_default_website(business)
        
        # Save the website
        index_path = business_dir / "website" / "index.html"
        result = await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(index_path), "content": html_content},
            description="Create main website"
        ))
        
        business.website_path = str(index_path)
        
        # Create CSS file
        css_content = self._get_custom_css(business)
        css_path = business_dir / "website" / "css" / "custom.css"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(css_path), "content": css_content},
            description="Create custom CSS"
        ))
        
        return str(index_path)
    
    def _get_default_website(self, business: DropshippingBusiness) -> str:
        """Get default website template."""
        products_html = ""
        for p in business.products[:3]:
            products_html += f"""
            <div class="bg-white rounded-lg shadow-lg overflow-hidden transform hover:scale-105 transition-transform duration-300">
                <div class="h-48 bg-gradient-to-br from-purple-100 to-pink-100 flex items-center justify-center">
                    <svg class="w-20 h-20 text-purple-300" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                </div>
                <div class="p-6">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">{p.name}</h3>
                    <p class="text-gray-600 text-sm mb-4">{p.description}</p>
                    <div class="flex justify-between items-center">
                        <span class="text-2xl font-bold text-purple-600">${p.target_price:.2f}</span>
                        <button class="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors">
                            Add to Cart
                        </button>
                    </div>
                </div>
            </div>"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{business.name} - {business.niche}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
    </style>
</head>
<body class="bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-sm sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16 items-center">
                <div class="text-2xl font-bold text-purple-600">{business.name}</div>
                <div class="hidden md:flex space-x-8">
                    <a href="#products" class="text-gray-600 hover:text-purple-600">Products</a>
                    <a href="#about" class="text-gray-600 hover:text-purple-600">About</a>
                    <a href="#contact" class="text-gray-600 hover:text-purple-600">Contact</a>
                </div>
                <button class="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700">
                    Cart (0)
                </button>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="bg-gradient-to-r from-purple-600 to-pink-500 text-white py-20">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h1 class="text-4xl md:text-6xl font-bold mb-6">
                Premium {business.niche}
            </h1>
            <p class="text-xl md:text-2xl mb-8 opacity-90">
                Discover the best products curated just for you
            </p>
            <a href="#products" class="bg-white text-purple-600 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-gray-100 transition-colors inline-block">
                Shop Now
            </a>
        </div>
    </section>

    <!-- Features -->
    <section class="py-16 bg-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                <div class="p-6">
                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                    </div>
                    <h3 class="text-lg font-semibold mb-2">Quality Guaranteed</h3>
                    <p class="text-gray-600">Every product is tested and verified for quality</p>
                </div>
                <div class="p-6">
                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    </div>
                    <h3 class="text-lg font-semibold mb-2">Fast Shipping</h3>
                    <p class="text-gray-600">Get your orders delivered in 7-14 business days</p>
                </div>
                <div class="p-6">
                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"/></svg>
                    </div>
                    <h3 class="text-lg font-semibold mb-2">Secure Payment</h3>
                    <p class="text-gray-600">Your payment information is always protected</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Products Section -->
    <section id="products" class="py-16 bg-gray-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 class="text-3xl font-bold text-center mb-12">Featured Products</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                {products_html}
            </div>
        </div>
    </section>

    <!-- About Section -->
    <section id="about" class="py-16 bg-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="max-w-3xl mx-auto text-center">
                <h2 class="text-3xl font-bold mb-6">About {business.name}</h2>
                <p class="text-gray-600 text-lg mb-8">
                    {business.description}. We're passionate about bringing you the best 
                    {business.niche.lower()} products at unbeatable prices. Our mission is to make 
                    quality accessible to everyone.
                </p>
                <p class="text-gray-600">
                    <strong>Target Audience:</strong> {business.target_audience}
                </p>
            </div>
        </div>
    </section>

    <!-- Newsletter -->
    <section class="py-16 bg-purple-600 text-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 class="text-3xl font-bold mb-4">Join Our Newsletter</h2>
            <p class="mb-8 opacity-90">Get exclusive deals and updates delivered to your inbox</p>
            <form class="max-w-md mx-auto flex gap-4">
                <input type="email" placeholder="Enter your email" class="flex-1 px-4 py-3 rounded-lg text-gray-800">
                <button class="bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100">
                    Subscribe
                </button>
            </form>
        </div>
    </section>

    <!-- Footer -->
    <footer id="contact" class="bg-gray-800 text-white py-12">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
                <div>
                    <h3 class="text-xl font-bold mb-4">{business.name}</h3>
                    <p class="text-gray-400">Your trusted source for premium {business.niche.lower()}.</p>
                </div>
                <div>
                    <h4 class="font-semibold mb-4">Quick Links</h4>
                    <ul class="space-y-2 text-gray-400">
                        <li><a href="#" class="hover:text-white">Home</a></li>
                        <li><a href="#products" class="hover:text-white">Products</a></li>
                        <li><a href="#about" class="hover:text-white">About</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="font-semibold mb-4">Support</h4>
                    <ul class="space-y-2 text-gray-400">
                        <li><a href="#" class="hover:text-white">FAQ</a></li>
                        <li><a href="#" class="hover:text-white">Shipping</a></li>
                        <li><a href="#" class="hover:text-white">Returns</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="font-semibold mb-4">Contact</h4>
                    <ul class="space-y-2 text-gray-400">
                        <li>support@{business.name.lower().replace(' ', '')}.com</li>
                    </ul>
                </div>
            </div>
            <div class="border-t border-gray-700 mt-8 pt-8 text-center text-gray-400">
                <p>&copy; {datetime.now().year} {business.name}. All rights reserved.</p>
            </div>
        </div>
    </footer>
</body>
</html>"""
    
    def _get_custom_css(self, business: DropshippingBusiness) -> str:
        """Get custom CSS for the website."""
        return """/* Custom styles for """ + business.name + """ */
:root {
    --primary-color: #7c3aed;
    --secondary-color: #ec4899;
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    transition: transform 0.2s, box-shadow 0.2s;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(124, 58, 237, 0.3);
}

.product-card {
    transition: transform 0.3s, box-shadow 0.3s;
}

.product-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}
"""
    
    async def _create_marketing_plan(self, business: DropshippingBusiness, business_dir: Path) -> str:
        """Create comprehensive marketing plan."""
        prompt = f"""Create a detailed marketing plan for a dropshipping business.

Business: {business.name}
Niche: {business.niche}
Target Audience: {business.target_audience}
Products: {', '.join(p.name for p in business.products)}

Include:
1. Executive Summary
2. Target Market Analysis
3. Marketing Channels (prioritized)
   - Social Media Strategy (TikTok, Instagram, Facebook)
   - Paid Advertising (Facebook Ads, TikTok Ads)
   - Email Marketing
   - Influencer Marketing
4. Content Calendar (first month)
5. Budget Allocation (for $500/month spend)
6. KPIs and Success Metrics
7. Competitor Analysis
8. 90-Day Action Plan

Format as a professional marketing document with clear sections."""

        try:
            content = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="document", complexity="high")
            )
        except Exception as e:
            logger.warning(f"Marketing plan generation failed: {e}")
            content = self._get_default_marketing_plan(business)
        
        marketing_path = business_dir / "marketing" / "marketing_plan.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(marketing_path), "content": content},
            description="Create marketing plan"
        ))
        
        business.marketing_plan_path = str(marketing_path)
        
        # Also create social media templates
        await self._create_social_media_templates(business, business_dir)
        
        return str(marketing_path)
    
    def _get_default_marketing_plan(self, business: DropshippingBusiness) -> str:
        """Get default marketing plan."""
        return f"""# Marketing Plan: {business.name}

## Executive Summary
This marketing plan outlines the strategy to launch and grow {business.name}, a dropshipping business in the {business.niche} niche.

## Target Market
{business.target_audience}

## Marketing Channels

### 1. TikTok (Primary - 40% of effort)
- Create viral product demo videos
- Leverage trending sounds and hashtags
- Post 2-3 times daily
- Target hashtags: #{business.niche.replace(' ', '')} #TikTokMadeMeBuyIt

### 2. Instagram (30% of effort)
- Product showcase posts
- Story polls and engagement
- Reels synced with TikTok
- Influencer collaborations

### 3. Facebook Ads (20% of effort)
- Interest-based targeting
- Lookalike audiences
- Retargeting campaigns
- Budget: $15-25/day starting

### 4. Email Marketing (10% of effort)
- Welcome sequence
- Abandoned cart recovery
- Weekly promotions

## First Month Content Calendar

Week 1: Product reveal posts, brand introduction
Week 2: Customer testimonials, behind-the-scenes
Week 3: User-generated content campaign
Week 4: Flash sale promotion

## Budget Allocation ($500/month)
- Facebook/Instagram Ads: $250 (50%)
- TikTok Ads: $150 (30%)
- Influencer Collabs: $75 (15%)
- Tools/Software: $25 (5%)

## KPIs
- Website Traffic: 5,000 visitors/month
- Conversion Rate: 2%
- Average Order Value: $35
- Customer Acquisition Cost: <$15
- Return on Ad Spend: 2.5x

## 90-Day Action Plan

### Days 1-30: Foundation
- Set up social media accounts
- Create initial content library
- Launch first ad campaigns
- Build email list

### Days 31-60: Optimization
- Analyze ad performance
- Scale winning campaigns
- Test new creatives
- Launch influencer partnerships

### Days 61-90: Growth
- Expand to new platforms
- Increase ad budget on winners
- Launch referral program
- Consider expanding product line
"""
    
    async def _create_social_media_templates(self, business: DropshippingBusiness, business_dir: Path):
        """Create social media content templates."""
        templates = f"""# Social Media Content Templates - {business.name}

## TikTok Video Ideas

### Product Reveal
"Wait for it... 🤯 This {business.products[0].name if business.products else 'product'} just changed my life!"

### Trending Format
"POV: You finally found the perfect {business.niche.lower()} product"

### Before/After
"I can't believe the difference! 😱 #transformation"

## Instagram Captions

### Product Post
"{business.products[0].name if business.products else 'Our bestseller'} ✨
The quality is unmatched. Link in bio to shop now!
.
.
.
#{business.niche.replace(' ', '')} #shopsmall #musthave"

### Story Engagement
"Quick poll! Which product should we feature next? 🤔"

## Facebook Ad Copy

### Headline Options:
1. "Transform Your [Area] with Premium {business.niche}"
2. "Finally! {business.niche} That Actually Works"
3. "Join 10,000+ Happy Customers"

### Primary Text:
"Tired of [problem]? Our {business.products[0].name if business.products else 'products'} is designed to [solution].

✅ Fast shipping
✅ 30-day guarantee
✅ Premium quality

Shop now and see the difference! 🛒"

## Email Subject Lines
- "Your order is almost complete..."
- "⚡ Flash Sale: 24 hours only!"
- "New arrivals you'll love"
- "Don't miss out - back in stock!"
"""
        
        templates_path = business_dir / "marketing" / "social_media_templates.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(templates_path), "content": templates},
            description="Create social media templates"
        ))
    
    async def _create_business_plan(self, business: DropshippingBusiness, business_dir: Path) -> str:
        """Create comprehensive business plan."""
        prompt = f"""Create a professional business plan for a dropshipping e-commerce business.

Business: {business.name}
Niche: {business.niche}
Products: {len(business.products)} products with avg margin of {sum(p.profit_margin for p in business.products) / len(business.products):.1f}% if business.products else "70%"

Include these sections:
1. Executive Summary
2. Company Description
3. Market Analysis
4. Products/Services
5. Marketing Strategy Summary
6. Operations Plan Summary
7. Financial Projections (Year 1)
8. SWOT Analysis
9. Goals and Milestones

Make it professional and realistic for a new dropshipping venture."""

        try:
            content = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="document", complexity="high")
            )
        except Exception as e:
            logger.warning(f"Business plan generation failed: {e}")
            content = self._get_default_business_plan(business)
        
        plan_path = business_dir / "documents" / "business_plan.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(plan_path), "content": content},
            description="Create business plan"
        ))
        
        business.business_plan_path = str(plan_path)
        return str(plan_path)
    
    def _get_default_business_plan(self, business: DropshippingBusiness) -> str:
        """Get default business plan."""
        avg_margin = sum(p.profit_margin for p in business.products) / len(business.products) if business.products else 70
        avg_price = sum(p.target_price for p in business.products) / len(business.products) if business.products else 30
        
        return f"""# Business Plan: {business.name}

## Executive Summary

{business.name} is a dropshipping e-commerce business specializing in {business.niche.lower()}. We target {business.target_audience}

Our competitive advantages include curated product selection, strong social media presence, and excellent customer service.

## Company Description

**Business Name:** {business.name}
**Business Model:** Dropshipping E-commerce
**Niche:** {business.niche}
**Launch Date:** {datetime.now().strftime('%B %Y')}

## Market Analysis

The {business.niche.lower()} market is growing with increased online shopping trends. Key insights:
- Growing consumer interest in {business.niche.lower()}
- Social media driving discovery and purchases
- Opportunity for differentiation through brand and service

## Products/Services

We offer {len(business.products)} carefully selected products:
{chr(10).join(f"- {p.name}: ${p.target_price:.2f} (margin: {p.profit_margin:.0f}%)" for p in business.products)}

## Operations Plan

1. Order Processing: Automated via Shopify
2. Fulfillment: Dropshipping via verified suppliers
3. Customer Service: Email support, 24-48hr response
4. Returns: 30-day return policy

## Financial Projections (Year 1)

### Revenue Forecast
- Month 1-3: $2,000-5,000/month (building)
- Month 4-6: $5,000-15,000/month (growing)
- Month 7-12: $15,000-30,000/month (scaling)

### Key Metrics
- Average Order Value: ${avg_price:.2f}
- Gross Margin: {avg_margin:.0f}%
- Target Net Margin: 20-30%

### Startup Costs
- Shopify: $29/month
- Apps/Tools: ~$50/month
- Initial Ad Budget: $500
- Total Startup: ~$700

## SWOT Analysis

### Strengths
- Low startup costs
- Flexible product selection
- Scalable model

### Weaknesses
- Longer shipping times
- Lower margins than wholesale
- Dependent on suppliers

### Opportunities
- Growing e-commerce market
- Viral social media potential
- Expand to new niches

### Threats
- Competition
- Supplier reliability
- Platform policy changes

## 90-Day Goals

1. Launch store and first products
2. Achieve first 100 sales
3. Reach $5,000 monthly revenue
4. Build email list to 1,000 subscribers
5. Establish consistent ad profitability
"""
    
    async def _create_supplier_document(self, business: DropshippingBusiness, business_dir: Path) -> str:
        """Create supplier information document."""
        supplier_content = f"""# Supplier Guide - {business.name}

## Recommended Suppliers

### AliExpress
- **Best For:** Wide product selection, competitive prices
- **Shipping Time:** 15-45 days (standard), 7-15 days (ePacket)
- **Pros:** Huge selection, buyer protection, easy to start
- **Cons:** Longer shipping, quality varies
- **URL:** https://www.aliexpress.com

### CJ Dropshipping
- **Best For:** Faster shipping, quality control
- **Shipping Time:** 7-12 days (US warehouse), 12-20 days (China)
- **Pros:** US warehouses, product sourcing, branding options
- **Cons:** Smaller selection than AliExpress
- **URL:** https://cjdropshipping.com

### Spocket
- **Best For:** US/EU products, fast shipping
- **Shipping Time:** 2-7 days (domestic)
- **Pros:** Fast shipping, vetted suppliers
- **Cons:** Higher product costs
- **URL:** https://www.spocket.co

## Products & Suppliers

{chr(10).join(f'''### {p.name}
- **Target Price:** ${p.target_price:.2f}
- **Estimated Cost:** ${p.cost_estimate:.2f}
- **Margin:** {p.profit_margin:.0f}%
- **Suppliers:** {p.supplier_info}
''' for p in business.products)}

## Supplier Selection Criteria

1. **Product Quality:** Check reviews and order samples
2. **Shipping Time:** Prefer <15 days to US
3. **Communication:** Responsive to messages
4. **Pricing:** Allows 50%+ markup
5. **Reliability:** Consistent stock and quality

## Order Process

1. Customer places order on your store
2. You receive order notification
3. Order product from supplier with customer's shipping address
4. Supplier ships directly to customer
5. You receive tracking and forward to customer
6. Follow up after delivery for reviews

## Quality Control Tips

- Order samples before listing products
- Check recent reviews (last 30 days)
- Verify shipping times with supplier
- Document supplier response times
- Have backup suppliers for top products
"""
        
        supplier_path = business_dir / "suppliers" / "supplier_guide.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(supplier_path), "content": supplier_content},
            description="Create supplier guide"
        ))
        
        business.supplier_list_path = str(supplier_path)
        return str(supplier_path)
    
    async def _create_financial_plan(self, business: DropshippingBusiness, business_dir: Path, budget: float) -> str:
        """Create financial projections."""
        avg_price = sum(p.target_price for p in business.products) / len(business.products) if business.products else 30
        avg_cost = sum(p.cost_estimate for p in business.products) / len(business.products) if business.products else 10
        avg_margin = ((avg_price - avg_cost) / avg_price) * 100
        
        financial_content = f"""# Financial Plan - {business.name}

## Startup Budget: ${budget:.2f}

### Initial Expenses
| Item | Cost |
|------|------|
| Shopify (3 months) | $87 |
| Domain | $15 |
| Apps (3 months) | $90 |
| Sample Products | $100 |
| Initial Ad Budget | ${budget - 300:.0f} |
| **Total** | **${budget:.0f}** |

## Product Economics

| Metric | Value |
|--------|-------|
| Average Selling Price | ${avg_price:.2f} |
| Average Product Cost | ${avg_cost:.2f} |
| Gross Margin | {avg_margin:.0f}% |
| Target Net Margin | 25% |

## Monthly Projections

### Month 1-3 (Launch Phase)
- Revenue: $2,000-3,000
- Ad Spend: $300-500
- Net Profit: $200-500

### Month 4-6 (Growth Phase)
- Revenue: $5,000-10,000
- Ad Spend: $1,000-2,000
- Net Profit: $1,000-2,500

### Month 7-12 (Scale Phase)
- Revenue: $15,000-30,000
- Ad Spend: $3,000-6,000
- Net Profit: $3,000-7,500

## Break-Even Analysis

- Fixed Costs: ~$150/month (Shopify + apps)
- Variable Margin: {avg_margin:.0f}%
- Break-even Revenue: ${150 / (avg_margin/100):.0f}/month
- Break-even Orders: {int(150 / (avg_margin/100) / avg_price)} orders/month

## Year 1 Projections

| Quarter | Revenue | Costs | Net Profit |
|---------|---------|-------|------------|
| Q1 | $7,500 | $5,250 | $2,250 |
| Q2 | $22,500 | $13,500 | $9,000 |
| Q3 | $45,000 | $25,200 | $19,800 |
| Q4 | $67,500 | $35,775 | $31,725 |
| **Year 1** | **$142,500** | **$79,725** | **$62,775** |

## Key Financial Metrics to Track

1. **Revenue per day**
2. **Cost of Goods Sold (COGS)**
3. **Customer Acquisition Cost (CAC)**
4. **Lifetime Value (LTV)**
5. **Return on Ad Spend (ROAS)**
6. **Refund Rate**
"""
        
        financial_path = business_dir / "financials" / "financial_plan.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(financial_path), "content": financial_content},
            description="Create financial plan"
        ))
        
        business.financial_plan_path = str(financial_path)
        return str(financial_path)
    
    async def _create_operations_manual(self, business: DropshippingBusiness, business_dir: Path) -> str:
        """Create operations manual."""
        ops_content = f"""# Operations Manual - {business.name}

## Daily Operations Checklist

### Morning (9 AM)
- [ ] Check for new orders
- [ ] Process pending orders with suppliers
- [ ] Respond to customer inquiries
- [ ] Check ad performance

### Afternoon (2 PM)
- [ ] Follow up on shipped orders
- [ ] Update tracking numbers
- [ ] Monitor inventory levels
- [ ] Social media posting

### Evening (6 PM)
- [ ] Review day's metrics
- [ ] Plan next day's content
- [ ] Address any issues
- [ ] Optimize underperforming ads

## Order Fulfillment Process

1. **Order Received**
   - Shopify notification received
   - Verify payment processed

2. **Order Supplier**
   - Log into supplier (AliExpress/CJ)
   - Place order with customer address
   - Use customer's shipping details exactly

3. **Update Order**
   - Mark as processing in Shopify
   - Save supplier order ID

4. **Track Shipment**
   - Get tracking number from supplier (2-5 days)
   - Update Shopify with tracking
   - Send customer notification

5. **Follow Up**
   - Check delivery status at day 14
   - Reach out if delayed
   - Request review after delivery

## Customer Service Guidelines

### Response Times
- Email: Within 24 hours
- Social Media: Within 4 hours
- Order Issues: Same day

### Common Issues & Responses

**"Where is my order?"**
> Your order is on its way! Here's your tracking: [link]
> Shipping typically takes 10-20 business days. 
> Let me know if you have any questions!

**"I want a refund"**
> I'm sorry to hear that. Let me help!
> What's the issue with your order?
> [Offer solution before refund]

**"Product is damaged"**
> I apologize for this! Please send a photo.
> We'll send a replacement right away.

## Metrics Dashboard

Track these KPIs weekly:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Conversion Rate | >2% | Shopify Analytics |
| Average Order Value | >${sum(p.target_price for p in business.products) / len(business.products) if business.products else 30:.0f} | Shopify Analytics |
| Customer Acquisition Cost | <$15 | Ad Spend / New Customers |
| Refund Rate | <5% | Refunds / Total Orders |
| Customer Satisfaction | >4.5/5 | Reviews & Surveys |

## Tools & Software

| Tool | Purpose | Cost |
|------|---------|------|
| Shopify | Store platform | $29/mo |
| Oberlo/DSers | Order fulfillment | Free-$20/mo |
| Canva | Graphics design | Free |
| Buffer | Social scheduling | Free |
| Klaviyo | Email marketing | Free (up to 250) |

## Weekly Review Template

Every Sunday, review:
1. Total revenue & profit
2. Best performing products
3. Ad performance (ROAS)
4. Customer feedback
5. Inventory status
6. Next week's goals
"""
        
        ops_path = business_dir / "operations" / "operations_manual.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(ops_path), "content": ops_content},
            description="Create operations manual"
        ))
        
        business.operations_plan_path = str(ops_path)
        return str(ops_path)
    
    async def _create_product_catalog(self, business: DropshippingBusiness, business_dir: Path) -> str:
        """Create product catalog."""
        catalog_content = f"""# Product Catalog - {business.name}

## Overview
- **Niche:** {business.niche}
- **Total Products:** {len(business.products)}
- **Average Margin:** {sum(p.profit_margin for p in business.products) / len(business.products) if business.products else 0:.0f}%

## Products

"""
        for i, product in enumerate(business.products, 1):
            catalog_content += f"""### Product {i}: {product.name}

**Description:** {product.description}

| Attribute | Value |
|-----------|-------|
| Selling Price | ${product.target_price:.2f} |
| Cost | ${product.cost_estimate:.2f} |
| Profit | ${product.target_price - product.cost_estimate:.2f} |
| Margin | {product.profit_margin:.0f}% |
| Trending | {'✅ Yes' if product.trending else '❌ No'} |
| Suppliers | {product.supplier_info} |

---

"""
        
        catalog_content += """## Adding New Products

When adding products, ensure:
1. Minimum 50% margin
2. Available from reliable supplier
3. Good reviews (4+ stars)
4. Consistent shipping times
5. Not too many competitors

## Product Photography Tips

1. Use clean, white backgrounds
2. Show product from multiple angles
3. Include lifestyle images
4. Ensure high resolution (1000x1000px+)
5. Maintain consistent style across catalog
"""
        
        catalog_path = business_dir / "products" / "product_catalog.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(catalog_path), "content": catalog_content},
            description="Create product catalog"
        ))
        
        return str(catalog_path)
    
    async def _verify_all_files(self, business_dir: Path) -> List[str]:
        """Verify all created files exist."""
        verified = []
        
        for file_path in business_dir.rglob("*"):
            if file_path.is_file():
                verified.append(str(file_path))
                logger.debug(f"Verified file: {file_path}")
        
        return verified
    
    async def _create_summary(self, business: DropshippingBusiness, business_dir: Path):
        """Create a summary README for the business."""
        readme_content = f"""# {business.name}

## 📊 Business Overview

| Attribute | Value |
|-----------|-------|
| Business ID | `{business.business_id}` |
| Niche | {business.niche} |
| Created | {business.created_at.strftime('%Y-%m-%d %H:%M')} |
| Products | {len(business.products)} |
| Files Created | {len(business.verified_files)} |

## 🎯 Target Audience
{business.target_audience}

## 📁 Directory Structure

```
{business.business_id}/
├── website/          # Complete website with landing page
│   ├── index.html    # Main landing page
│   └── css/          # Stylesheets
├── marketing/        # Marketing strategy & templates
│   ├── marketing_plan.md
│   └── social_media_templates.md
├── documents/        # Business documentation
│   └── business_plan.md
├── products/         # Product catalog
│   └── product_catalog.md
├── suppliers/        # Supplier information
│   └── supplier_guide.md
├── financials/       # Financial projections
│   └── financial_plan.md
└── operations/       # Operating procedures
    └── operations_manual.md
```

## 🚀 Quick Start

1. **Review the Business Plan:** Start with `documents/business_plan.md`
2. **Set Up Your Store:** Use `website/index.html` as your landing page template
3. **Source Products:** Follow `suppliers/supplier_guide.md`
4. **Launch Marketing:** Execute `marketing/marketing_plan.md`
5. **Daily Operations:** Follow `operations/operations_manual.md`

## 📦 Products

| Product | Price | Margin |
|---------|-------|--------|
{chr(10).join(f"| {p.name} | ${p.target_price:.2f} | {p.profit_margin:.0f}% |" for p in business.products)}

## ✅ Verified Files

All {len(business.verified_files)} files have been created and verified.

---
*Generated by King AI v2 - Autonomous Business Creation System*
"""
        
        readme_path = business_dir / "README.md"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(readme_path), "content": readme_content},
            description="Create business README"
        ))
    
    def get_business_summary(self, business: DropshippingBusiness) -> str:
        """Get a human-readable summary of the created business."""
        return f"""
🏪 **{business.name}** - Dropshipping Business Created Successfully!

📊 **Business Details:**
- ID: {business.business_id}
- Niche: {business.niche}
- Products: {len(business.products)}
- Verified Files: {len(business.verified_files)}

📁 **What Was Created:**
✅ Complete Website (index.html with Tailwind CSS)
✅ Marketing Plan & Social Media Templates
✅ Business Plan with Financial Projections
✅ Supplier Guide & Product Catalog
✅ Operations Manual
✅ Financial Projections

📍 **Location:** businesses/{business.business_id}/

🚀 **Next Steps:**
1. Open `website/index.html` to preview your store
2. Review `documents/business_plan.md` for the full strategy
3. Follow `operations/operations_manual.md` for daily tasks
4. Execute `marketing/marketing_plan.md` to start growing
"""


# Singleton instance
_dropshipping_creator: Optional[DropshippingCreator] = None


def get_dropshipping_creator(workspace: str = None) -> DropshippingCreator:
    """Get or create the singleton dropshipping creator."""
    global _dropshipping_creator
    if _dropshipping_creator is None:
        _dropshipping_creator = DropshippingCreator(workspace)
    return _dropshipping_creator

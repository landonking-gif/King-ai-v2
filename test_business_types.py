#!/usr/bin/env python3
"""
Test script to verify King AI can create different types of businesses.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.master_ai.brain import MasterAI

async def test_business_creation():
    """Test creating different types of businesses."""
    print("🚀 Testing King AI Business Creation Capabilities")
    print("=" * 60)

    # Initialize Master AI
    ai = MasterAI()

    test_cases = [
        "Create a SaaS business for project management",
        "Start a consulting business for small businesses",
        "Build an ecommerce store selling handmade crafts",
        "Launch a content business with a blog about cooking",
        "Create a subscription box service for pet owners",
        "Start a marketplace for freelance designers",
        "Build an agency that helps businesses with social media",
        "Create a dropshipping business selling fitness equipment"
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case}")
        print("-" * 40)

        try:
            # Process the request
            response = await ai.process_input(test_case)

            print("✅ Response received"            print(f"📝 Response: {response.response[:200]}...")

            if response.metadata:
                business_id = response.metadata.get("business_id")
                if business_id:
                    print(f"🏢 Business ID: {business_id}")

                    # Check if files were created
                    business_dir = Path("businesses") / business_id
                    if business_dir.exists():
                        files = list(business_dir.rglob("*"))
                        print(f"📁 Files created: {len([f for f in files if f.is_file])}")
                    else:
                        print("❌ Business directory not found")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_business_creation())
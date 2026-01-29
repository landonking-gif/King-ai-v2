#!/usr/bin/env python3
"""
Example: Ralph Loop with Memory Integration

Demonstrates how Ralph learns from experience using the Memory Service.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add ralph to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_integration import RalphMemoryClient


async def example_diary_entry():
    """Example: Write a diary entry after an attempt"""
    print("="*60)
    print("Example 1: Diary Entry")
    print("="*60)
    
    client = RalphMemoryClient(memory_service_url="http://localhost:8002")
    
    # Simulate a failed attempt
    memory_id = await client.diary(
        story_id="US-101",
        story_title="Add JWT authentication",
        attempt_number=1,
        success=False,
        changes_made=0,
        prompt_used="Implement JWT auth with refresh tokens...",
        code_generated=None,
        error="Copilot didn't generate any code",
        quality_checks=None,
        context={"tech_stack": "Python FastAPI"}
    )
    
    print(f"✅ Diary entry created: {memory_id}")
    print()


async def example_reflection():
    """Example: Reflect after story completion"""
    print("="*60)
    print("Example 2: Reflection")
    print("="*60)
    
    client = RalphMemoryClient(memory_service_url="http://localhost:8002")
    
    # Simulate story with 3 attempts
    all_attempts = [
        {
            "attempt": 1,
            "success": False,
            "changes_made": 0,
            "error": "No code generated"
        },
        {
            "attempt": 2,
            "success": False,
            "changes_made": 2,
            "error": "Quality checks failed"
        },
        {
            "attempt": 3,
            "success": True,
            "changes_made": 4,
            "error": None
        }
    ]
    
    reflection = await client.reflect(
        story_id="US-101",
        story_title="Add JWT authentication",
        total_attempts=3,
        final_success=True,
        all_attempts=all_attempts,
        files_changed=["src/auth.py", "src/middleware.py", "tests/test_auth.py"],
        commit_sha="abc123def"
    )
    
    print("\n📊 Reflection Summary:")
    print(f"  Story: {reflection['story']['title']}")
    print(f"  Attempts: {reflection['story']['total_attempts']}")
    print(f"  Success: {reflection['story']['final_success']}")
    print(f"\n  Key Insights:")
    for insight in reflection['analysis']['insights'][:3]:
        print(f"    • {insight}")
    print()


async def example_query_learnings():
    """Example: Query past learnings"""
    print("="*60)
    print("Example 3: Query Past Learnings")
    print("="*60)
    
    client = RalphMemoryClient(memory_service_url="http://localhost:8002")
    
    learnings = await client.query_past_learnings(
        story_title="Implement API endpoint",
        story_description="REST API with authentication and rate limiting",
        top_k=3
    )
    
    if learnings:
        print(f"\n✨ Found {len(learnings)} relevant past experience(s):")
        for i, learning in enumerate(learnings, 1):
            print(f"\n  {i}. {learning['artifact_type']} (similarity: {learning['similarity']:.2f})")
            
            if learning['artifact_type'] == 'reflection':
                recommendations = learning['content'].get('learnings', {}).get('recommendations', [])
                if recommendations:
                    print(f"     Recommendations:")
                    for rec in recommendations[:2]:
                        print(f"       - {rec}")
    else:
        print("\n  No past learnings found (memory is empty)")
    
    print()


async def example_full_workflow():
    """Example: Complete workflow with memory integration"""
    print("="*60)
    print("Example 4: Full Workflow")
    print("="*60)
    print("\nSimulating Ralph implementing a story with memory integration...")
    print()
    
    client = RalphMemoryClient(memory_service_url="http://localhost:8002")
    
    story_id = "US-202"
    story_title = "Add password reset functionality"
    
    # Step 1: Query past learnings
    print("1️⃣  Querying past learnings...")
    past_learnings = await client.query_past_learnings(
        story_title=story_title,
        story_description="Email-based password reset with secure tokens",
        top_k=3
    )
    
    if past_learnings:
        print(f"   Found {len(past_learnings)} relevant experience(s)")
    else:
        print("   No past experiences found")
    
    # Step 2: Attempt 1 (fails)
    print("\n2️⃣  Attempt 1...")
    await client.diary(
        story_id=story_id,
        story_title=story_title,
        attempt_number=1,
        success=False,
        changes_made=0,
        prompt_used="Add password reset...",
        code_generated=None,
        error="No code generated",
        quality_checks=None
    )
    print("   ❌ Failed - No code generated")
    
    # Step 3: Attempt 2 (fails)
    print("\n3️⃣  Attempt 2...")
    await client.diary(
        story_id=story_id,
        story_title=story_title,
        attempt_number=2,
        success=False,
        changes_made=3,
        prompt_used="Add password reset with email service...",
        code_generated="...some code...",
        error="Quality checks failed",
        quality_checks=[("Python syntax", False)]
    )
    print("   ❌ Failed - Quality checks failed")
    
    # Step 4: Attempt 3 (succeeds!)
    print("\n4️⃣  Attempt 3...")
    await client.diary(
        story_id=story_id,
        story_title=story_title,
        attempt_number=3,
        success=True,
        changes_made=5,
        prompt_used="Add password reset with email service and validation...",
        code_generated="...complete code...",
        error=None,
        quality_checks=[("Python syntax", True), ("Type checks", True)]
    )
    print("   ✅ Success - All quality checks passed!")
    
    # Step 5: Reflect
    print("\n5️⃣  Reflecting on the journey...")
    all_attempts = [
        {"attempt": 1, "success": False, "changes_made": 0, "error": "No code generated"},
        {"attempt": 2, "success": False, "changes_made": 3, "error": "Quality checks failed"},
        {"attempt": 3, "success": True, "changes_made": 5, "error": None}
    ]
    
    reflection = await client.reflect(
        story_id=story_id,
        story_title=story_title,
        total_attempts=3,
        final_success=True,
        all_attempts=all_attempts,
        files_changed=["src/auth/reset.py", "src/services/email.py"],
        commit_sha="def456"
    )
    
    print("\n📈 Learning Summary:")
    print(f"   What worked: {reflection['learnings']['what_worked']}")
    print(f"   What didn't: {reflection['learnings']['what_didnt_work']}")
    print(f"   Recommendations: {reflection['learnings']['recommendations'][:2]}")
    print()


async def main():
    """Run all examples"""
    print()
    print("🧪 Ralph Memory Integration Examples")
    print()
    print("These examples demonstrate how Ralph uses the Memory Service")
    print("to learn from experience and improve over time.")
    print()
    
    try:
        await example_diary_entry()
        await example_reflection()
        await example_query_learnings()
        await example_full_workflow()
        
        print("="*60)
        print("✅ All examples completed!")
        print("="*60)
        print()
        print("💡 Key Takeaways:")
        print("  • Diary entries log every attempt")
        print("  • Reflections analyze patterns after completion")
        print("  • Past learnings inform future attempts")
        print("  • Memory accumulates across all Ralph sessions")
        print()
        print("🚀 Next steps:")
        print("  1. Start Memory Service: python -m memory-service.service.main")
        print("  2. Run Ralph: python scripts/ralph/ralph.py")
        print("  3. Watch Ralph learn and improve!")
        print()
        
    except Exception as e:
        print()
        print("="*60)
        print("❌ Error running examples")
        print("="*60)
        print(f"\n{e}")
        print()
        print("Make sure Memory Service is running:")
        print("  cd king-ai-v3/agentic-framework-main/memory-service")
        print("  python -m service.main")
        print()


if __name__ == "__main__":
    asyncio.run(main())

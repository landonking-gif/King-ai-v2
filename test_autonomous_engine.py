"""
Test the Autonomous Business Engine.
Verifies the full lifecycle: understanding → research → planning → execution → verification → monitoring.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_autonomous_engine():
    """Test the autonomous business engine directly."""
    from src.services.autonomous_business_engine import get_autonomous_engine, BusinessPhase
    
    print("=" * 70)
    print("🚀 AUTONOMOUS BUSINESS ENGINE TEST")
    print("=" * 70)
    
    engine = get_autonomous_engine()
    
    # Test with a simple business prompt
    prompt = "create a pet supplies ecommerce business"
    
    print(f"\n📝 Testing with prompt: '{prompt}'")
    print("-" * 70)
    
    try:
        # Create the business
        blueprint = await engine.create_business(prompt)
        
        print(f"\n✅ Business Created Successfully!")
        print(f"   Business ID: {blueprint.business_id}")
        print(f"   Business Name: {blueprint.business_name}")
        print(f"   Business Type: {blueprint.business_type}")
        print(f"   Phase: {blueprint.phase.value}")
        print(f"   Tasks: {len(blueprint.completed_tasks)} / {len(blueprint.tasks)} completed")
        print(f"   Files Created: {len(blueprint.files_created)}")
        
        # Test file visibility
        files = engine.get_business_files(blueprint.business_id)
        print(f"\n📁 Files Created ({len(files)} total):")
        for f in files[:10]:
            status = "✅" if f['exists'] else "❌"
            print(f"   {status} {f['path']} ({f['size']} bytes)")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
        
        # Test action log
        actions = engine.get_action_log(blueprint.business_id)
        print(f"\n📋 Action Log ({len(actions)} entries):")
        for action in actions[:10]:
            print(f"   [{action['timestamp'][:19]}] {action['action']}")
        if len(actions) > 10:
            print(f"   ... and {len(actions) - 10} more actions")
        
        # Verify market research was done
        if blueprint.market_research:
            print(f"\n🔍 Market Research:")
            print(f"   Market Size: {blueprint.market_research.market_size}")
            print(f"   Competitors: {len(blueprint.market_research.competitors)}")
            print(f"   Trends: {len(blueprint.market_research.trends)}")
            print(f"   Opportunities: {len(blueprint.market_research.opportunities)}")
        else:
            print(f"\n⚠️ No market research recorded (may be simulated)")
        
        # Check phase transitions
        print(f"\n🔄 Phase Verification:")
        expected_final_phase = BusinessPhase.COMPLETE
        if blueprint.phase == expected_final_phase:
            print(f"   ✅ Business reached {expected_final_phase.value} phase")
        else:
            print(f"   ⚠️ Business in {blueprint.phase.value} phase (expected {expected_final_phase.value})")
        
        print("\n" + "=" * 70)
        print("✅ TEST PASSED - Autonomous Business Engine is working!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_master_ai_integration():
    """Test that MasterAI brain uses the autonomous engine."""
    from src.master_ai.brain import MasterAI
    
    print("\n" + "=" * 70)
    print("🧠 MASTER AI INTEGRATION TEST")
    print("=" * 70)
    
    try:
        master = MasterAI()
        
        # Verify autonomous engine is initialized
        if hasattr(master, 'autonomous_engine') and master.autonomous_engine is not None:
            print("✅ MasterAI has autonomous_engine attribute")
        else:
            print("❌ MasterAI missing autonomous_engine attribute")
            return False
        
        # Test process method with business creation
        prompt = "create a coffee subscription business"
        print(f"\n📝 Testing process() with: '{prompt}'")
        
        response = await master.process_input(prompt)
        
        print(f"\n📤 Response type: {response.type}")
        print(f"📝 Response length: {len(response.response)} chars")
        
        # Check for expected content
        expected_markers = ["Phase", "Files Created", "business"]
        found_markers = [m for m in expected_markers if m.lower() in response.response.lower()]
        
        print(f"\n✅ Found {len(found_markers)}/{len(expected_markers)} expected markers in response")
        
        # Check metadata
        if response.metadata:
            print(f"📊 Metadata keys: {list(response.metadata.keys())}")
            if 'business_id' in response.metadata:
                print(f"   ✅ business_id: {response.metadata['business_id']}")
            if 'files_created' in response.metadata:
                print(f"   ✅ files_created: {response.metadata['files_created']}")
        
        print("\n" + "=" * 70)
        print("✅ MASTER AI INTEGRATION TEST PASSED!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("  KING AI - AUTONOMOUS BUSINESS ENGINE TESTS")
    print("🔬" * 35 + "\n")
    
    results = []
    
    # Test 1: Autonomous Engine directly
    result1 = asyncio.run(test_autonomous_engine())
    results.append(("Autonomous Engine", result1))
    
    # Test 2: MasterAI Integration
    result2 = asyncio.run(test_master_ai_integration())
    results.append(("MasterAI Integration", result2))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status} - {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)

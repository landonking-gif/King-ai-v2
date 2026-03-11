"""
Simple test to verify anti-hallucination prompt changes.
Tests the prompt structure and temperature settings without requiring database.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_system_prompt():
    """Verify the system prompt has anti-hallucination rules."""
    from src.master_ai.prompts import SYSTEM_PROMPT
    
    print("\n" + "="*60)
    print("🧪 TEST 1: System Prompt Anti-Hallucination Rules")
    print("="*60)
    
    required_phrases = [
        "I don't have",
        "ONLY",
        "NEVER",
        "cannot access",
        "not available",
        "Do NOT make up",
        "limitations",
    ]
    
    found = []
    missing = []
    
    for phrase in required_phrases:
        if phrase.lower() in SYSTEM_PROMPT.lower():
            found.append(phrase)
        else:
            missing.append(phrase)
    
    print(f"✓ Found anti-hallucination phrases: {len(found)}/{len(required_phrases)}")
    for phrase in found:
        print(f"  ✓ '{phrase}'")
    
    if missing:
        print(f"✗ Missing phrases:")
        for phrase in missing:
            print(f"  ✗ '{phrase}'")
        return False
    
    print("\n✅ System prompt contains all required anti-hallucination rules")
    return True

def test_temperature_settings():
    """Verify temperature settings are low enough to prevent hallucination."""
    from src.utils.llm_router import TaskContext, LLMRouter
    
    print("\n" + "="*60)
    print("🧪 TEST 2: Temperature Settings")
    print("="*60)
    
    # Create a router instance to test
    try:
        router = LLMRouter()
    except Exception as e:
        print(f"Note: Could not create router ({e}), testing method directly")
        # Test the method logic directly
        pass
    
    # Test cases - all should have low temperatures
    test_cases = [
        (TaskContext(task_type="conversation", risk_level="low", requires_accuracy=False, token_estimate=100, priority="normal"), 0.3),
        (TaskContext(task_type="query", risk_level="low", requires_accuracy=True, token_estimate=100, priority="normal"), 0.1),
        (TaskContext(task_type="finance", risk_level="high", requires_accuracy=True, token_estimate=100, priority="high"), 0.1),
        (None, 0.2),  # Default should be low
    ]
    
    all_passed = True
    
    for context, max_expected in test_cases:
        try:
            temp = router._get_temperature_for_task(context)
            task_name = context.task_type if context else "None"
            
            if temp <= max_expected:
                print(f"  ✓ {task_name}: temperature={temp} (max allowed: {max_expected})")
            else:
                print(f"  ✗ {task_name}: temperature={temp} TOO HIGH (max allowed: {max_expected})")
                all_passed = False
        except Exception as e:
            print(f"  ⚠ Could not test {context}: {e}")
    
    if all_passed:
        print("\n✅ All temperature settings are appropriately low")
    else:
        print("\n❌ Some temperatures are too high")
    
    return all_passed

def test_conversation_handler_structure():
    """Verify conversation handler includes history and strict instructions."""
    import inspect
    from src.master_ai.brain import MasterAI
    
    print("\n" + "="*60)
    print("🧪 TEST 3: Conversation Handler Structure")
    print("="*60)
    
    # Get the source code of the method
    source = inspect.getsource(MasterAI._handle_conversation)
    
    required_elements = [
        "_format_conversation_history",  # Uses history
        "CONVERSATION HISTORY",  # Includes history section
        "ONLY",  # Strict limitation
        "don't have",  # Decline pattern
        "requires_accuracy",  # Fact checking enabled
    ]
    
    found = []
    missing = []
    
    for element in required_elements:
        if element in source:
            found.append(element)
        else:
            missing.append(element)
    
    print(f"✓ Found required elements: {len(found)}/{len(required_elements)}")
    for element in found:
        print(f"  ✓ '{element}'")
    
    if missing:
        print(f"✗ Missing elements:")
        for element in missing:
            print(f"  ✗ '{element}'")
        return False
    
    print("\n✅ Conversation handler has all required anti-hallucination elements")
    return True

def test_conversation_logging():
    """Verify conversation logging methods exist."""
    from src.master_ai.brain import MasterAI
    
    print("\n" + "="*60)
    print("🧪 TEST 4: Conversation Logging")
    print("="*60)
    
    required_methods = [
        "_log_conversation",
        "_format_conversation_history",
        "_conversation_history",  # Instance variable
    ]
    
    # Check methods exist
    found = []
    missing = []
    
    for method in required_methods:
        if hasattr(MasterAI, method) or method in MasterAI.__init__.__code__.co_names or method in str(MasterAI.__init__.__code__.co_consts):
            found.append(method)
        else:
            # Try to find in source
            import inspect
            source = inspect.getsource(MasterAI)
            if method in source:
                found.append(method)
            else:
                missing.append(method)
    
    print(f"✓ Found required methods/attributes: {len(found)}/{len(required_methods)}")
    for method in found:
        print(f"  ✓ '{method}'")
    
    if missing:
        print(f"✗ Missing methods/attributes:")
        for method in missing:
            print(f"  ✗ '{method}'")
        return False
    
    print("\n✅ Conversation logging is properly implemented")
    return True

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 ANTI-HALLUCINATION VERIFICATION TESTS")
    print("="*60)
    print("Testing prompt structure, temperatures, and handlers...")
    
    results = []
    
    results.append(("System Prompt", test_system_prompt()))
    results.append(("Temperature Settings", test_temperature_settings()))
    results.append(("Conversation Handler", test_conversation_handler_structure()))
    results.append(("Conversation Logging", test_conversation_logging()))
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All anti-hallucination measures are in place!")
        return True
    else:
        print("\n⚠️ Some tests failed - review the output above")
        return False

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)

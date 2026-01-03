"""
Custom Skill Example

Demonstrates how to create a custom skill from scratch.
This example shows the complete structure of a skill including:
- skill.yaml (metadata)
- schema.json (input/output validation)
- handler.py (implementation)
- test_handler.py (tests)
"""

import sys
from pathlib import Path

# Add skill to path
skill_path = Path(__file__).parent / "my_custom_skill"
sys.path.insert(0, str(skill_path.parent))


def main():
    """Demonstrate the custom sentiment analysis skill."""
    print("=" * 60)
    print("Custom Skill Example")
    print("=" * 60)

    print("\n📦 Skill: sentiment_analyzer")
    print("   Location: my_custom_skill/")
    print("   Type: NLP - Sentiment Analysis")
    print("   Safety: No flags (read-only operation)")

    # Import the skill handler
    try:
        from my_custom_skill.handler import analyze_sentiment

        print("\n✓ Skill loaded successfully")

    except ImportError as e:
        print(f"\n❌ Error loading skill: {e}")
        return

    # Test texts with different sentiments
    test_cases = [
        {
            "text": "This is an excellent product! I absolutely love it!",
            "expected": "positive"
        },
        {
            "text": "Terrible experience. Very disappointed and frustrated.",
            "expected": "negative"
        },
        {
            "text": "The product arrived on time. It has standard features.",
            "expected": "neutral"
        },
        {
            "text": "The service was great but the quality was poor.",
            "expected": "neutral"  # Mixed sentiment
        }
    ]

    print("\n🔄 Running Skill Tests:\n")

    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        expected = test_case["expected"]

        print(f"[Test {i}/4] Input: \"{text[:50]}...\"")

        # Execute the skill
        result = analyze_sentiment(text)

        # Display results
        sentiment = result["sentiment"]
        polarity = result["polarity"]
        confidence = result["confidence"]
        word_count = result["word_count"]

        status = "✓" if sentiment == expected else "⚠"
        print(f"  {status} Sentiment: {sentiment} (expected: {expected})")
        print(f"  Polarity: {polarity:+.3f} | Confidence: {confidence:.1%}")
        print(f"  Words analyzed: {word_count}\n")

    # Run unit tests
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    print()

    try:
        from my_custom_skill.test_handler import (
            test_positive_sentiment,
            test_negative_sentiment,
            test_neutral_sentiment,
            test_empty_text,
            test_output_schema
        )

        tests = [
            ("Positive sentiment", test_positive_sentiment),
            ("Negative sentiment", test_negative_sentiment),
            ("Neutral sentiment", test_neutral_sentiment),
            ("Empty text handling", test_empty_text),
            ("Output schema validation", test_output_schema)
        ]

        for name, test_func in tests:
            try:
                test_func()
                print(f"✓ {name}")
            except AssertionError as e:
                print(f"✗ {name}: {e}")

    except Exception as e:
        print(f"❌ Error running tests: {e}")

    # Show skill files
    print("\n=" * 60)
    print("Skill Structure")
    print("=" * 60)
    print("\nmy_custom_skill/")
    print("├── skill.yaml          # Skill metadata")
    print("├── schema.json         # Input/output schemas")
    print("├── handler.py          # Implementation")
    print("└── test_handler.py     # Unit tests")

    print("\n💡 What you learned:")
    print("   ✓ How to structure a custom skill")
    print("   ✓ How to define input/output schemas")
    print("   ✓ How to implement a skill handler")
    print("   ✓ How to write unit tests for skills")
    print("   ✓ How safety flags work")

    print("\n📝 Key Files:")
    print("   - skill.yaml: Metadata, safety flags, handler reference")
    print("   - schema.json: JSON Schema for validation")
    print("   - handler.py: Python implementation")
    print("   - test_handler.py: Unit tests")

    print("\nNext steps:")
    print("  - Copy my_custom_skill/ to code-exec/skills/ to use in framework")
    print("  - See examples/04-mcp-integration/ for external tool integration")
    print("  - Read docs/skills.md for advanced skill development")


if __name__ == "__main__":
    main()

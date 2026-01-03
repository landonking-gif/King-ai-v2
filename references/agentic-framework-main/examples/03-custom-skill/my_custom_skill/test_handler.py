"""
Tests for sentiment_analyzer skill
"""

from handler import analyze_sentiment


def test_positive_sentiment():
    """Test positive sentiment detection."""
    text = "This is an excellent product! I love it and it's amazing!"
    result = analyze_sentiment(text)

    assert result["sentiment"] == "positive"
    assert result["polarity"] > 0
    assert 0 <= result["confidence"] <= 1
    assert result["word_count"] > 0


def test_negative_sentiment():
    """Test negative sentiment detection."""
    text = "This is terrible and awful. Very disappointing experience."
    result = analyze_sentiment(text)

    assert result["sentiment"] == "negative"
    assert result["polarity"] < 0
    assert 0 <= result["confidence"] <= 1
    assert result["word_count"] > 0


def test_neutral_sentiment():
    """Test neutral sentiment detection."""
    text = "The product arrived on Tuesday. It has standard features."
    result = analyze_sentiment(text)

    assert result["sentiment"] == "neutral"
    assert -0.3 <= result["polarity"] <= 0.3
    assert 0 <= result["confidence"] <= 1
    assert result["word_count"] > 0


def test_empty_text():
    """Test handling of empty text."""
    text = ""
    result = analyze_sentiment(text)

    assert result["sentiment"] == "neutral"
    assert result["polarity"] == 0.0
    assert result["confidence"] == 0.0
    assert result["word_count"] == 0


def test_output_schema():
    """Test that output matches schema."""
    text = "Test text"
    result = analyze_sentiment(text)

    # Check all required fields are present
    assert "sentiment" in result
    assert "polarity" in result
    assert "confidence" in result
    assert "word_count" in result

    # Check types
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["polarity"], (int, float))
    assert isinstance(result["confidence"], (int, float))
    assert isinstance(result["word_count"], int)

    # Check ranges
    assert result["sentiment"] in ["positive", "negative", "neutral"]
    assert -1.0 <= result["polarity"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["word_count"] >= 0


if __name__ == "__main__":
    print("Running tests...")
    test_positive_sentiment()
    print("✓ test_positive_sentiment passed")

    test_negative_sentiment()
    print("✓ test_negative_sentiment passed")

    test_neutral_sentiment()
    print("✓ test_neutral_sentiment passed")

    test_empty_text()
    print("✓ test_empty_text passed")

    test_output_schema()
    print("✓ test_output_schema passed")

    print("\nAll tests passed!")

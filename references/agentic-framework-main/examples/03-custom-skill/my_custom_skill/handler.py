"""
Sentiment Analysis Skill Handler

A simple sentiment analyzer that classifies text as positive, negative, or neutral.
This is a basic implementation using keyword matching for demonstration purposes.
For production, use a proper NLP library like NLTK or transformers.
"""

from typing import Dict, Any


# Positive and negative word lists (simplified for demo)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
    'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'happy', 'glad',
    'joy', 'success', 'win', 'winning', 'benefit', 'advantage', 'improve'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'poor',
    'fail', 'failure', 'problem', 'issue', 'difficult', 'hard', 'sad', 'angry',
    'frustrating', 'disappointing', 'loss', 'lose', 'disadvantage', 'worse'
}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of the given text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary containing sentiment, polarity, confidence, and word_count
    """
    # Normalize text
    words = text.lower().split()
    word_count = len(words)

    # Count positive and negative words
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

    # Calculate polarity (-1 to 1)
    if word_count == 0:
        polarity = 0.0
        sentiment = "neutral"
        confidence = 0.0
    else:
        polarity = (positive_count - negative_count) / word_count

        # Scale to -1 to 1 range
        polarity = max(-1.0, min(1.0, polarity * 5))  # Amplify for demo

        # Classify sentiment
        if polarity > 0.2:
            sentiment = "positive"
        elif polarity < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Calculate confidence based on how many sentiment words found
        sentiment_word_ratio = (positive_count + negative_count) / word_count
        confidence = min(0.95, sentiment_word_ratio * 2)  # Cap at 95%

        # If very few sentiment words, lower confidence
        if positive_count + negative_count < 2:
            confidence = max(0.3, confidence)

    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 3),
        "confidence": round(confidence, 3),
        "word_count": word_count
    }


# For testing
if __name__ == "__main__":
    test_texts = [
        "This is a great product! I love it!",
        "Terrible experience, very disappointed.",
        "The product works as expected.",
    ]

    for text in test_texts:
        result = analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Result: {result}")

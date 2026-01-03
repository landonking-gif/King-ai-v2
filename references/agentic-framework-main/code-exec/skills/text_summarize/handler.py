"""
Text summarization skill handler.
Module: code-exec/skills/text_summarize/handler.py
"""

import re
from typing import Dict, Any


def summarize(
    text: str, max_sentences: int = 5, style: str = "concise"
) -> Dict[str, Any]:
    """
    Summarize text by extracting key sentences.

    This is a simple extractive summarization that scores sentences
    by keyword frequency and position. For production, use a proper
    NLP model like BART or T5.

    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary
        style: Summary style (concise, detailed, bullet-points)

    Returns:
        Dictionary with summary and metadata
    """
    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    original_length = len(text)

    # If text is short, return as-is
    if len(sentences) <= max_sentences:
        summary = text.strip()
        return {
            "summary": summary,
            "original_length": original_length,
            "summary_length": len(summary),
            "compression_ratio": 1.0,
            "sentences_extracted": len(sentences),
        }

    # Score sentences based on position and keyword frequency
    word_freq = _calculate_word_frequency(text)
    sentence_scores = _score_sentences(sentences, word_freq)

    # Select top sentences
    top_indices = sorted(
        range(len(sentence_scores)),
        key=lambda i: sentence_scores[i],
        reverse=True,
    )[:max_sentences]

    # Sort selected sentences by original order
    top_indices.sort()
    selected_sentences = [sentences[i] for i in top_indices]

    # Format based on style
    if style == "bullet-points":
        summary = "\n• " + "\n• ".join(selected_sentences)
    elif style == "detailed":
        summary = " ".join(selected_sentences) + "."
    else:  # concise
        summary = " ".join(selected_sentences[:max_sentences - 1])
        if max_sentences > 0:
            summary += ". " + selected_sentences[-1] + "."

    summary_length = len(summary)
    compression_ratio = summary_length / original_length if original_length > 0 else 0.0

    return {
        "summary": summary.strip(),
        "original_length": original_length,
        "summary_length": summary_length,
        "compression_ratio": round(compression_ratio, 3),
        "sentences_extracted": len(selected_sentences),
    }


def _calculate_word_frequency(text: str) -> Dict[str, int]:
    """Calculate word frequency for scoring."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())

    # Remove common stop words
    stop_words = {
        "the", "and", "is", "in", "to", "of", "a", "that", "it", "for",
        "as", "on", "with", "was", "at", "by", "from", "this", "be", "are",
        "or", "an", "has", "have", "been", "had", "not", "but", "they",
        "their", "were", "which", "will", "would", "can", "about", "all",
    }

    word_freq: Dict[str, int] = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    return word_freq


def _score_sentences(sentences: list, word_freq: Dict[str, int]) -> list:
    """Score sentences based on word frequency and position."""
    scores = []

    for i, sentence in enumerate(sentences):
        words = re.findall(r"\b[a-z]{3,}\b", sentence.lower())

        # Base score: sum of word frequencies
        score = sum(word_freq.get(word, 0) for word in words)

        # Normalize by sentence length
        if len(words) > 0:
            score = score / len(words)

        # Boost first and last sentences
        if i == 0:
            score *= 1.5
        elif i == len(sentences) - 1:
            score *= 1.3

        scores.append(score)

    return scores

"""
SEO Utilities - Tools for SEO analysis and optimization.
Provides keyword analysis, readability scoring, and SEO recommendations.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
import math


@dataclass
class KeywordAnalysis:
    """Analysis of keyword usage in content."""
    keyword: str
    count: int
    density: float  # Percentage
    in_title: bool
    in_headings: bool
    in_first_paragraph: bool
    in_meta_description: bool


@dataclass
class ReadabilityScore:
    """Readability metrics for content."""
    flesch_reading_ease: float  # 0-100, higher is easier
    flesch_kincaid_grade: float  # Grade level
    avg_sentence_length: float
    avg_word_length: float
    complex_word_percentage: float
    
    @property
    def reading_level(self) -> str:
        """Get human-readable reading level."""
        if self.flesch_reading_ease >= 80:
            return "Easy (6th grade)"
        elif self.flesch_reading_ease >= 60:
            return "Standard (8th-9th grade)"
        elif self.flesch_reading_ease >= 40:
            return "Difficult (College)"
        else:
            return "Very Difficult (Professional)"


@dataclass
class SEOScore:
    """Complete SEO analysis score."""
    overall_score: float  # 0-100
    title_score: float
    meta_score: float
    content_score: float
    keyword_score: float
    readability_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "title_score": self.title_score,
            "meta_score": self.meta_score,
            "content_score": self.content_score,
            "keyword_score": self.keyword_score,
            "readability_score": self.readability_score,
            "issues": self.issues,
            "recommendations": self.recommendations
        }


class SEOAnalyzer:
    """
    Analyzes content for SEO optimization.
    """
    
    # Ideal ranges for SEO metrics
    IDEAL_TITLE_LENGTH = (50, 60)
    IDEAL_META_LENGTH = (150, 160)
    IDEAL_KEYWORD_DENSITY = (1.0, 3.0)
    IDEAL_CONTENT_LENGTH = 1500
    
    def __init__(self):
        """Initialize SEO analyzer."""
        # Common stop words to exclude from keyword analysis
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'as', 'if', 'when', 'than'
        }
    
    def analyze(
        self,
        content: str,
        title: str = "",
        meta_description: str = "",
        target_keywords: List[str] = None
    ) -> SEOScore:
        """
        Perform complete SEO analysis.
        
        Args:
            content: Main content text
            title: Page/post title
            meta_description: Meta description
            target_keywords: Target keywords to optimize for
            
        Returns:
            Complete SEO score
        """
        issues = []
        recommendations = []
        
        # Title analysis
        title_score, title_issues = self._analyze_title(title, target_keywords)
        issues.extend(title_issues)
        
        # Meta description analysis
        meta_score, meta_issues = self._analyze_meta(meta_description, target_keywords)
        issues.extend(meta_issues)
        
        # Content analysis
        content_score, content_issues = self._analyze_content(content)
        issues.extend(content_issues)
        
        # Keyword analysis
        keyword_score = 100
        if target_keywords:
            keyword_score, keyword_issues = self._analyze_keywords(
                content, title, meta_description, target_keywords
            )
            issues.extend(keyword_issues)
        
        # Readability
        readability = self.calculate_readability(content)
        readability_score = min(100, readability.flesch_reading_ease)
        
        if readability.flesch_reading_ease < 50:
            recommendations.append("Consider simplifying language for better readability")
        
        # Calculate overall score
        overall_score = (
            title_score * 0.15 +
            meta_score * 0.15 +
            content_score * 0.30 +
            keyword_score * 0.25 +
            readability_score * 0.15
        )
        
        # Generate recommendations
        recommendations.extend(self._generate_recommendations(
            title_score, meta_score, content_score, keyword_score, content
        ))
        
        return SEOScore(
            overall_score=overall_score,
            title_score=title_score,
            meta_score=meta_score,
            content_score=content_score,
            keyword_score=keyword_score,
            readability_score=readability_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _analyze_title(
        self,
        title: str,
        keywords: List[str] = None
    ) -> tuple[float, List[str]]:
        """Analyze title for SEO."""
        score = 100
        issues = []
        
        if not title:
            return 0, ["Missing title"]
        
        title_len = len(title)
        
        if title_len < self.IDEAL_TITLE_LENGTH[0]:
            score -= 20
            issues.append(f"Title too short ({title_len} chars, ideal: 50-60)")
        elif title_len > self.IDEAL_TITLE_LENGTH[1]:
            score -= 15
            issues.append(f"Title too long ({title_len} chars, ideal: 50-60)")
        
        # Check for keyword in title
        if keywords:
            title_lower = title.lower()
            has_keyword = any(kw.lower() in title_lower for kw in keywords)
            if not has_keyword:
                score -= 25
                issues.append("Target keyword not in title")
        
        return max(0, score), issues
    
    def _analyze_meta(
        self,
        meta: str,
        keywords: List[str] = None
    ) -> tuple[float, List[str]]:
        """Analyze meta description for SEO."""
        score = 100
        issues = []
        
        if not meta:
            return 0, ["Missing meta description"]
        
        meta_len = len(meta)
        
        if meta_len < self.IDEAL_META_LENGTH[0]:
            score -= 20
            issues.append(f"Meta description too short ({meta_len} chars)")
        elif meta_len > self.IDEAL_META_LENGTH[1]:
            score -= 15
            issues.append(f"Meta description too long ({meta_len} chars)")
        
        # Check for keyword in meta
        if keywords:
            meta_lower = meta.lower()
            has_keyword = any(kw.lower() in meta_lower for kw in keywords)
            if not has_keyword:
                score -= 20
                issues.append("Target keyword not in meta description")
        
        return max(0, score), issues
    
    def _analyze_content(self, content: str) -> tuple[float, List[str]]:
        """Analyze content quality for SEO."""
        score = 100
        issues = []
        
        word_count = len(content.split())
        
        if word_count < 300:
            score -= 40
            issues.append(f"Content too short ({word_count} words, minimum 300)")
        elif word_count < self.IDEAL_CONTENT_LENGTH:
            score -= 20
            issues.append(f"Content could be longer ({word_count} words, ideal: 1500+)")
        
        # Check for headings
        heading_pattern = r'^#{1,6}\s|<h[1-6]>'
        has_headings = bool(re.search(heading_pattern, content, re.MULTILINE | re.IGNORECASE))
        if not has_headings and word_count > 500:
            score -= 15
            issues.append("Content lacks headings for structure")
        
        # Check for paragraphs (content structure)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3 and word_count > 300:
            score -= 10
            issues.append("Content needs more paragraph breaks")
        
        return max(0, score), issues
    
    def _analyze_keywords(
        self,
        content: str,
        title: str,
        meta: str,
        keywords: List[str]
    ) -> tuple[float, List[str]]:
        """Analyze keyword usage."""
        score = 100
        issues = []
        
        content_lower = content.lower()
        word_count = len(content.split())
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            count = content_lower.count(kw_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            
            if count == 0:
                score -= 30
                issues.append(f"Keyword '{keyword}' not found in content")
            elif density < self.IDEAL_KEYWORD_DENSITY[0]:
                score -= 10
                issues.append(f"Keyword '{keyword}' density too low ({density:.1f}%)")
            elif density > self.IDEAL_KEYWORD_DENSITY[1]:
                score -= 15
                issues.append(f"Keyword '{keyword}' may be over-optimized ({density:.1f}%)")
        
        return max(0, score), issues
    
    def calculate_readability(self, text: str) -> ReadabilityScore:
        """Calculate readability metrics."""
        # Clean and split text
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not sentences or not words:
            return ReadabilityScore(0, 0, 0, 0, 0)
        
        # Count syllables (simplified)
        def count_syllables(word: str) -> int:
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            prev_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            
            if word.endswith('e'):
                count -= 1
            
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in words)
        total_words = len(words)
        total_sentences = len(sentences)
        
        # Calculate metrics
        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        avg_word_length = sum(len(w) for w in words) / total_words
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_percentage = (complex_words / total_words) * 100
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        flesch_kincaid = max(0, flesch_kincaid)
        
        return ReadabilityScore(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            complex_word_percentage=complex_percentage
        )
    
    def extract_keywords(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract potential keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Get top keywords
        keywords = []
        for word, count in word_freq.most_common(top_n):
            density = (count / len(words)) * 100 if words else 0
            keywords.append({
                "keyword": word,
                "count": count,
                "density": round(density, 2)
            })
        
        return keywords
    
    def _generate_recommendations(
        self,
        title_score: float,
        meta_score: float,
        content_score: float,
        keyword_score: float,
        content: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if title_score < 80:
            recommendations.append("Optimize title: include target keyword, keep 50-60 characters")
        
        if meta_score < 80:
            recommendations.append("Improve meta description: include keyword, use 150-160 characters")
        
        if content_score < 80:
            word_count = len(content.split())
            if word_count < 1500:
                recommendations.append(f"Expand content to at least 1500 words (currently {word_count})")
            recommendations.append("Add subheadings (H2, H3) to structure content")
        
        if keyword_score < 80:
            recommendations.append("Increase keyword usage naturally throughout content")
        
        # Check for internal/external links
        if 'http' not in content and '[' not in content:
            recommendations.append("Add relevant internal and external links")
        
        # Check for images
        if '![' not in content and '<img' not in content:
            recommendations.append("Add images with alt text for better engagement")
        
        return recommendations

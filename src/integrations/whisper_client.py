"""
Whisper Audio Transcription Client.

Provides audio transcription and analysis capabilities using OpenAI's Whisper API.
Enables multi-modal processing for podcasts, voice notes, customer calls, etc.
"""

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO
import httpx

from src.utils.structured_logging import get_logger
from src.utils.circuit_breaker import CircuitBreaker

logger = get_logger("whisper_client")

# Whisper circuit breaker
whisper_circuit = CircuitBreaker(
    "whisper",
    failure_threshold=3,
    timeout=60.0,
    success_threshold=2
)


class TranscriptionLanguage(str, Enum):
    """Supported transcription languages."""
    AUTO = "auto"
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"


class OutputFormat(str, Enum):
    """Output formats for transcription."""
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    VERBOSE_JSON = "verbose_json"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    language: str
    duration: float
    segments: List[TranscriptionSegment]
    word_count: int
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.word_count:
            self.word_count = len(self.text.split())


@dataclass
class AudioAnalysis:
    """Analysis results from audio content."""
    transcription: TranscriptionResult
    summary: Optional[str] = None
    key_topics: List[str] = None
    sentiment: Optional[str] = None
    action_items: List[str] = None
    speakers: List[str] = None
    
    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []
        if self.action_items is None:
            self.action_items = []
        if self.speakers is None:
            self.speakers = []


class WhisperClient:
    """
    OpenAI Whisper client for audio transcription.
    
    Features:
    - Audio file transcription
    - Multi-language support
    - Segment-level timestamps
    - Speaker diarization (when available)
    - Integration with LLM for analysis
    """
    
    API_URL = "https://api.openai.com/v1/audio"
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
    SUPPORTED_FORMATS = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._configured = bool(self.api_key)
        
        if not self._configured:
            logger.warning("Whisper not configured - OPENAI_API_KEY not set")
    
    @property
    def is_configured(self) -> bool:
        return self._configured
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate audio file before processing."""
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Maximum: {self.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
    
    @whisper_circuit.protect
    async def transcribe(
        self,
        audio_path: str,
        language: TranscriptionLanguage = TranscriptionLanguage.AUTO,
        output_format: OutputFormat = OutputFormat.VERBOSE_JSON,
        prompt: str = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language hint (auto-detect if not specified)
            output_format: Output format
            prompt: Optional prompt to guide transcription style
            
        Returns:
            TranscriptionResult with full transcription and segments
        """
        if not self._configured:
            raise RuntimeError("Whisper not configured - OPENAI_API_KEY required")
        
        file_path = Path(audio_path)
        self._validate_file(file_path)
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(file_path, "rb") as audio_file:
                    files = {"file": (file_path.name, audio_file, "audio/mpeg")}
                    data = {
                        "model": "whisper-1",
                        "response_format": output_format.value
                    }
                    
                    if language != TranscriptionLanguage.AUTO:
                        data["language"] = language.value
                    
                    if prompt:
                        data["prompt"] = prompt
                    
                    response = await client.post(
                        f"{self.API_URL}/transcriptions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    
                    result_data = response.json()
            
            # Parse segments
            segments = []
            if "segments" in result_data:
                for seg in result_data["segments"]:
                    segments.append(TranscriptionSegment(
                        id=seg.get("id", 0),
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", "").strip(),
                        confidence=seg.get("avg_logprob")
                    ))
            
            result = TranscriptionResult(
                text=result_data.get("text", "").strip(),
                language=result_data.get("language", "unknown"),
                duration=result_data.get("duration", 0),
                segments=segments,
                word_count=len(result_data.get("text", "").split()),
                metadata={
                    "file": str(file_path),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "Audio transcribed successfully",
                duration=result.duration,
                word_count=result.word_count,
                language=result.language
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Whisper API error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    @whisper_circuit.protect
    async def translate(
        self,
        audio_path: str,
        output_format: OutputFormat = OutputFormat.VERBOSE_JSON,
        prompt: str = None
    ) -> TranscriptionResult:
        """
        Translate audio to English.
        
        Args:
            audio_path: Path to the audio file
            output_format: Output format
            prompt: Optional prompt to guide translation style
            
        Returns:
            TranscriptionResult with English translation
        """
        if not self._configured:
            raise RuntimeError("Whisper not configured")
        
        file_path = Path(audio_path)
        self._validate_file(file_path)
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(file_path, "rb") as audio_file:
                    files = {"file": (file_path.name, audio_file, "audio/mpeg")}
                    data = {
                        "model": "whisper-1",
                        "response_format": output_format.value
                    }
                    
                    if prompt:
                        data["prompt"] = prompt
                    
                    response = await client.post(
                        f"{self.API_URL}/translations",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    
                    result_data = response.json()
            
            segments = []
            if "segments" in result_data:
                for seg in result_data["segments"]:
                    segments.append(TranscriptionSegment(
                        id=seg.get("id", 0),
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", "").strip()
                    ))
            
            return TranscriptionResult(
                text=result_data.get("text", "").strip(),
                language="en",  # Translation is always to English
                duration=result_data.get("duration", 0),
                segments=segments,
                word_count=len(result_data.get("text", "").split()),
                metadata={
                    "file": str(file_path),
                    "translated": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    async def analyze_audio(
        self,
        audio_path: str,
        llm_client = None,
        include_summary: bool = True,
        include_topics: bool = True,
        include_sentiment: bool = True,
        include_action_items: bool = False
    ) -> AudioAnalysis:
        """
        Transcribe and analyze audio content.
        
        Args:
            audio_path: Path to audio file
            llm_client: Optional LLM client for analysis
            include_summary: Generate summary
            include_topics: Extract key topics
            include_sentiment: Analyze sentiment
            include_action_items: Extract action items
            
        Returns:
            AudioAnalysis with transcription and insights
        """
        # First, transcribe
        transcription = await self.transcribe(audio_path)
        
        analysis = AudioAnalysis(transcription=transcription)
        
        # If no LLM client, return just transcription
        if not llm_client:
            return analysis
        
        # Build analysis prompt
        prompt_parts = [
            f"Analyze the following transcription:\n\n{transcription.text[:4000]}\n\n"
        ]
        
        if include_summary:
            prompt_parts.append("1. Provide a brief summary (2-3 sentences)")
        if include_topics:
            prompt_parts.append("2. List the key topics discussed (max 5)")
        if include_sentiment:
            prompt_parts.append("3. Overall sentiment (positive/negative/neutral)")
        if include_action_items:
            prompt_parts.append("4. Extract any action items or next steps")
        
        prompt_parts.append("\nFormat as JSON with keys: summary, topics, sentiment, action_items")
        
        try:
            # Use LLM for analysis
            response = await llm_client.generate(
                prompt="\n".join(prompt_parts),
                temperature=0.3
            )
            
            # Parse response (simplified - would need proper JSON extraction)
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                analysis.summary = data.get("summary")
                analysis.key_topics = data.get("topics", [])
                analysis.sentiment = data.get("sentiment")
                analysis.action_items = data.get("action_items", [])
                
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        return analysis


# Global client instance
whisper_client = WhisperClient()

"""
Text Analysis Plugin - Example of a Universal Plugin
Demonstrates the plugin interface and capabilities
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter


class TextAnalysisPlugin:
    """Universal plugin for text analysis capabilities"""
    
    def __init__(self):
        self.name = "text_analysis"
        self.version = "1.0.0"
        self.capabilities = [
            "text_analysis",
            "sentiment_analysis", 
            "keyword_extraction",
            "readability_analysis",
            "language_detection",
            "text_statistics"
        ]
        self.metadata = {
            "name": self.name,
            "version": self.version,
            "description": "Comprehensive text analysis capabilities",
            "author": "Universal System",
            "created_at": datetime.now().isoformat(),
            "capabilities": self.capabilities
        }
        self.config = {}
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_processing_time": 0.0,
            "error_count": 0
        }
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin with configuration"""
        self.config = config
        return True
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text analysis based on context"""
        start_time = datetime.now()
        
        try:
            # Extract text and analysis type from context
            text = context.get("text", "")
            analysis_type = context.get("analysis_type", "comprehensive")
            
            if not text:
                return {
                    "success": False,
                    "error": "No text provided for analysis"
                }
            
            # Perform analysis based on type
            if analysis_type == "sentiment":
                result = await self._analyze_sentiment(text)
            elif analysis_type == "keywords":
                result = await self._extract_keywords(text)
            elif analysis_type == "readability":
                result = await self._analyze_readability(text)
            elif analysis_type == "statistics":
                result = await self._calculate_statistics(text)
            elif analysis_type == "language":
                result = await self._detect_language(text)
            else:  # comprehensive
                result = await self._comprehensive_analysis(text)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, True)
            
            return {
                "success": True,
                "result": result,
                "processing_time": processing_time,
                "analysis_type": analysis_type
            }
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, False)
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text (simplified implementation)"""
        # Simple keyword-based sentiment analysis
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "enjoy", "happy", "pleased", "satisfied", "positive"
        ]
        negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "dislike", "sad",
            "angry", "frustrated", "disappointed", "negative", "poor", "worst"
        ]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.5
        elif positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / total_sentiment_words
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "details": {
                "positive_indicators": [word for word in words if word in positive_words],
                "negative_indicators": [word for word in words if word in negative_words]
            }
        }
    
    async def _extract_keywords(self, text: str, top_n: int = 10) -> Dict[str, Any]:
        """Extract keywords from text"""
        # Simple keyword extraction based on word frequency
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
            "they", "me", "him", "her", "us", "them", "my", "your", "his", "her",
            "its", "our", "their"
        }
        
        # Clean and tokenize text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Get top keywords
        top_keywords = word_freq.most_common(top_n)
        
        return {
            "keywords": [{"word": word, "frequency": freq} for word, freq in top_keywords],
            "total_words": len(words),
            "unique_words": len(set(words)),
            "filtered_words": len(filtered_words)
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability (simplified implementation)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        
        # Calculate basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simple readability score (0-100, higher is more readable)
        readability_score = max(0, min(100, 100 - (avg_sentence_length * 2) - (avg_word_length * 5)))
        
        # Determine reading level
        if readability_score >= 80:
            reading_level = "Very Easy"
        elif readability_score >= 70:
            reading_level = "Easy"
        elif readability_score >= 60:
            reading_level = "Fairly Easy"
        elif readability_score >= 50:
            reading_level = "Standard"
        elif readability_score >= 40:
            reading_level = "Fairly Difficult"
        elif readability_score >= 30:
            reading_level = "Difficult"
        else:
            reading_level = "Very Difficult"
        
        return {
            "readability_score": readability_score,
            "reading_level": reading_level,
            "average_sentence_length": avg_sentence_length,
            "average_word_length": avg_word_length,
            "total_sentences": len(sentences),
            "total_words": len(words)
        }
    
    async def _calculate_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics"""
        # Basic counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Character analysis
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Word analysis
        words = text.split()
        unique_words = len(set(word.lower() for word in words))
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "unique_words": unique_words,
            "average_word_length": avg_word_length,
            "character_breakdown": {
                "letters": letters,
                "digits": digits,
                "spaces": spaces,
                "punctuation": punctuation
            },
            "lexical_diversity": unique_words / word_count if word_count > 0 else 0
        }
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text (simplified implementation)"""
        # Very basic language detection based on common words
        language_indicators = {
            "english": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
            "spanish": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no"],
            "french": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
            "german": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
            "italian": ["il", "di", "che", "e", "la", "per", "in", "un", "è", "non"]
        }
        
        words = text.lower().split()
        language_scores = {}
        
        for language, indicators in language_indicators.items():
            score = sum(1 for word in words if word in indicators)
            language_scores[language] = score / len(words) if words else 0
        
        detected_language = max(language_scores, key=language_scores.get)
        confidence = language_scores[detected_language]
        
        return {
            "detected_language": detected_language,
            "confidence": confidence,
            "language_scores": language_scores
        }
    
    async def _comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis"""
        # Combine all analysis types
        sentiment = await self._analyze_sentiment(text)
        keywords = await self._extract_keywords(text)
        readability = await self._analyze_readability(text)
        statistics = await self._calculate_statistics(text)
        language = await self._detect_language(text)
        
        return {
            "sentiment_analysis": sentiment,
            "keyword_extraction": keywords,
            "readability_analysis": readability,
            "text_statistics": statistics,
            "language_detection": language,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics["total_analyses"] += 1
        
        if success:
            self.performance_metrics["successful_analyses"] += 1
        else:
            self.performance_metrics["error_count"] += 1
        
        # Update average processing time
        total_time = (self.performance_metrics["average_processing_time"] * 
                     (self.performance_metrics["total_analyses"] - 1) + processing_time)
        self.performance_metrics["average_processing_time"] = (
            total_time / self.performance_metrics["total_analyses"]
        )
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities"""
        return self.capabilities
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {
            **self.metadata,
            "performance_metrics": self.performance_metrics,
            "config": self.config
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics
    
    def get_usage_examples(self) -> List[Dict[str, Any]]:
        """Get usage examples for the plugin"""
        return [
            {
                "description": "Comprehensive text analysis",
                "context": {
                    "text": "This is a sample text for analysis.",
                    "analysis_type": "comprehensive"
                }
            },
            {
                "description": "Sentiment analysis only",
                "context": {
                    "text": "I love this product! It's amazing.",
                    "analysis_type": "sentiment"
                }
            },
            {
                "description": "Keyword extraction",
                "context": {
                    "text": "Machine learning and artificial intelligence are transforming technology.",
                    "analysis_type": "keywords"
                }
            },
            {
                "description": "Readability analysis",
                "context": {
                    "text": "The quick brown fox jumps over the lazy dog.",
                    "analysis_type": "readability"
                }
            }
        ]
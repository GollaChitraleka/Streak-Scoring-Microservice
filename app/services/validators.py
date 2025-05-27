from typing import Dict, Any, Tuple, Optional
import logging
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from collections import Counter
from app.core.config_loader import AppConfig

logger = logging.getLogger(__name__)

class ActionValidator:
    """Validates user actions based on configured rules with enhanced help post validation"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models = self._load_models()

        # Thresholds
        try:
            help_post_config = self.config.action_types["help_post"]
            validation_config = help_post_config.validation
            threshold_config = validation_config.threshold if hasattr(validation_config, 'threshold') else {}

            if hasattr(threshold_config, 'get'):
                self.word_count_threshold = threshold_config.get("min_word_count", 10)
                self.min_confidence = threshold_config.get("min_confidence", 0.5)
            else:
                self.word_count_threshold = getattr(threshold_config, "min_word_count", 10)
                self.min_confidence = getattr(threshold_config, "min_confidence", 0.5)
        except (KeyError, AttributeError) as e:
            logger.warning(f"Could not load config thresholds, using defaults: {e}")
            self.word_count_threshold = 10
            self.min_confidence = 0.5

        self.repetition_threshold = 0.3

    def _load_models(self):
        models = {}
        try:
            if self.config.action_types["help_post"].validation.require_ai:
                app_dir = Path(__file__).resolve().parent.parent
                model_path = app_dir / "models" / "help_post_classifier.pkl"
                vectorizer_path = app_dir / "models" / "tfidf_vectorizer.pkl"
                scaler_path = app_dir / "models" / "feature_scaler.pkl"

                if model_path.exists() and vectorizer_path.exists() and scaler_path.exists():
                    models["classifier"] = joblib.load(model_path)
                    models["vectorizer"] = joblib.load(vectorizer_path)
                    models["scaler"] = joblib.load(scaler_path)
                    logger.info("✓ Help post AI models loaded successfully")
                else:
                    logger.warning("⚠ Missing one or more AI model files")
        except Exception as e:
            logger.exception("✗ Failed to load models")
        return models

    def _detect_repetition(self, content: str) -> bool:
        """
        Detect if content contains excessive repetition (from original HelpPostValidator)
    
        Args:
            content: Text content to analyze
        
        Returns:
            True if content is repetitive, False otherwise
        """
        words = content.lower().split()
        if len(words) < 5:
            return False
    
        # Check for single character/emoji spam first
        if len(set(content.replace(' ', ''))) <= 2:  # Only 1-2 unique characters
            return True
        
        # Count word frequencies
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
    
        # Check for excessive word repetition
        max_word_freq = max(word_counts.values())
        repetition_ratio = max_word_freq / total_words
    
        if repetition_ratio > self.repetition_threshold:
            return True
    
        # Check for low vocabulary diversity (keyword stuffing)
        vocabulary_ratio = unique_words / total_words
        if vocabulary_ratio < 0.5:  # Less than 50% unique words
            return True
    
        # Check for repeated phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(' '.join(words[i:i+2]))
        for i in range(len(words) - 2):
            phrases.append(' '.join(words[i:i+3]))
        
        if phrases:
            phrase_counts = Counter(phrases)
            unique_phrases = len(phrase_counts)
            total_phrases = len(phrases)
        
            # FIXED: Calculate repetition as 1 - (unique phrases / total phrases)
            phrase_repetition_ratio = 1 - (unique_phrases / total_phrases)
        
            return phrase_repetition_ratio > self.repetition_threshold
    
        return False

    def _extract_features(self, text: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features exactly matching the training code (16 features)"""
        words = text.split()
        word_count = metadata.get("word_count", len(words))

        # Vague and informal phrases (same as training)
        vague_phrases = ['i dont know', 'idk', 'just google', 'not sure', 'maybe', 'i think', 'probably']
        informal_words = ['lol', 'omg', 'wtf', 'tbh', 'imo', 'btw']

        vague_count = sum(1 for phrase in vague_phrases if phrase in text.lower())
        informal_count = sum(1 for word in informal_words if word in text.lower())

        # Extract exactly the same 16 features as training
        features = [
            word_count,                                                                    # 0
            len(text),                                                                     # 1
            len([s for s in text.split('.') if s.strip()]),                              # 2
            int('```' in text),                                                           # 3
            int('`' in text and '```' not in text),                                      # 4
            sum(1 for kw in ['function', 'variable', 'loop', 'return'] if kw in text.lower()), # 5
            sum(len(w) for w in words) / max(word_count, 1),                             # 6
            sum(1 for c in text if c in '.,!?;:'),                                       # 7
            text.count('?'),                                                              # 8
            text.count('!'),                                                              # 9
            sum(1 for phrase in ['example', 'for instance', 'such as'] if phrase in text.lower()), # 10
            sum(1 for word in ['because', 'since', 'therefore'] if word in text.lower()), # 11
            int(any(lang in text.lower() for lang in ['python', 'java', 'html'])),      # 12
            int(metadata.get("contains_code", False)),                                    # 13
            vague_count,                                                                  # 14
            informal_count                                                                # 15
        ]

        return np.array(features)

    def _get_ai_reasoning(self, X_combined, confidence: float, is_valid: bool, content: str) -> str:
        """Get enhanced AI reasoning based on model feature weights and content analysis"""
        try:
            # Get model coefficients (feature weights)
            coefficients = self.models["classifier"].coef_[0]
            
            # Get feature names
            tfidf_features = self.models["vectorizer"].get_feature_names_out()
            numeric_features = [
                'word_count', 'char_count', 'sentence_count', 'has_code_block',
                'has_inline_code', 'code_keyword_count', 'avg_word_length',
                'punctuation_count', 'question_marks', 'exclamation_marks',
                'examples_mentioned', 'explanatory_words', 'mentions_language', 
                'contains_code', 'vague_count', 'informal_count'
            ]
            
            all_features = list(tfidf_features) + numeric_features
            
            # Calculate feature contributions for this specific input
            feature_values = X_combined.toarray()[0]
            contributions = feature_values * coefficients
            
            # Get top contributing features
            feature_contributions = list(zip(all_features, contributions))
            
            # Additional content analysis for better reasoning
            word_count = len(content.split())
            has_repetition = self._detect_repetition(content)
            
            if is_valid:
                # Sort by positive contributions (features that made it valid)
                top_features = sorted(feature_contributions, key=lambda x: x[1], reverse=True)[:7]
                positive_features = [f for f, contrib in top_features if contrib > 0]
                
                reason_parts = []
                
                # Analyze specific positive indicators
                if any('code' in f or 'algorithm' in f or 'function' in f for f, _ in top_features):
                    reason_parts.append("contains relevant technical content")
                if 'examples_mentioned' in [f for f, _ in top_features if f in numeric_features]:
                    reason_parts.append("provides helpful examples")
                if 'explanatory_words' in [f for f, _ in top_features if f in numeric_features]:
                    reason_parts.append("offers clear explanations")
                if word_count >= self.word_count_threshold:
                    reason_parts.append("meets minimum length requirements")
                if 'mentions_language' in [f for f, _ in top_features if f in numeric_features]:
                    reason_parts.append("discusses specific programming topics")
                
                if not reason_parts:
                    reason_parts.append("demonstrates overall helpfulness patterns")
                
                return f"✓ AI validation passed: {' and '.join(reason_parts)} (confidence: {confidence:.2f})"
                
            else:
                # Sort by negative contributions (features that made it invalid)
                top_features = sorted(feature_contributions, key=lambda x: x[1])[:7]
                
                reason_parts = []
                
                # Analyze specific negative indicators
                if has_repetition:
                    reason_parts.append("contains repetitive content")
                if word_count < self.word_count_threshold:
                    reason_parts.append(f"below minimum word count ({word_count} < {self.word_count_threshold})")
                if 'vague_count' in [f for f, _ in top_features if f in numeric_features]:
                    reason_parts.append("contains vague or unhelpful phrases")
                if 'informal_count' in [f for f, _ in top_features if f in numeric_features]:
                    reason_parts.append("uses overly informal language")
                
                if not reason_parts:
                    reason_parts.append("lacks indicators of helpful content")
                
                return f"✗ AI validation failed: {' and '.join(reason_parts)} (confidence: {confidence:.2f})"
                    
        except Exception as e:
            logger.exception("Error generating AI reasoning")
            status = " passed" if is_valid else " failed"
            return f"AI validation {status} (confidence: {confidence:.2f})"

    def validate_login(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate login action"""
        return True, " Login validation passed"

    def validate_quiz(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate quiz action with enhanced feedback"""
        if not self.config.action_types["quiz"].enabled:
            return False, " Quiz action type is disabled"

        threshold = self.config.action_types["quiz"].validation.threshold

        if threshold and threshold.min_score is not None:
            score = metadata.get("score", 0)
            if score < threshold.min_score:
                return False, f" Score below threshold ({score} < {threshold.min_score})"

        if threshold and threshold.max_time_sec is not None:
            time_taken = metadata.get("time_taken_sec", 0)
            if time_taken > threshold.max_time_sec:
                return False, f"✗ Time exceeded threshold ({time_taken}s > {threshold.max_time_sec}s)"

        return True, " Quiz validation passed"

    def validate_help_post(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Enhanced help post validation with comprehensive checks"""
        if not self.config.action_types["help_post"].enabled:
            return False, "✗ Help post action type is disabled"

        content = metadata.get("content", "")
        word_count = metadata.get("word_count", len(content.split()) if content else 0)

        if not content or not content.strip():
            return False, "✗ Empty content provided"

        if word_count < 3:
            return False, f"✗ Content too short ({word_count} words)"

        if self._detect_repetition(content):
            return False, "✗ Repetitive content detected"

        
        try:
            threshold = self.config.action_types["help_post"].validation.threshold
            if threshold:
                min_words = None
                if hasattr(threshold, 'get') and threshold.get("min_word_count") is not None:
                    min_words = threshold["min_word_count"]
                elif hasattr(threshold, 'min_word_count') and threshold.min_word_count is not None:
                    min_words = threshold.min_word_count

                if min_words is not None and word_count < min_words:
                    return False, f"✗ Word count below threshold ({word_count} < {min_words})"
        except (AttributeError, KeyError):
            logger.warning("Could not access threshold config, using default validation")

        if self.config.action_types["help_post"].validation.require_ai:
            if "classifier" in self.models and "vectorizer" in self.models:
                try:
                    X_text = self.models["vectorizer"].transform([content])
                    X_numeric = self._extract_features(content, metadata)

                    if "scaler" in self.models:
                        X_numeric_scaled = self.models["scaler"].transform(X_numeric.reshape(1, -1))
                    else:
                        X_numeric_scaled = X_numeric.reshape(1, -1)
                        logger.warning("Using unscaled features - may affect accuracy")

                    X_combined = hstack([X_text, X_numeric_scaled])
                    prediction = self.models["classifier"].predict(X_combined)[0]
                    probability = self.models["classifier"].predict_proba(X_combined)[0]

                    confidence = float(max(probability))
                    prob_valid = float(probability[1])
                    prob_invalid = float(probability[0])
                    is_valid = bool(prediction == 1)

                    # Use the higher confidence score for validation
                    if is_valid:
                        final_confidence = prob_valid
                    else:
                        final_confidence = prob_invalid

                    reason = self._get_ai_reasoning(X_combined, final_confidence, is_valid, content)

                    # Debug logging
                    logger.info(f"AI Prediction Debug: prediction={prediction}, prob_valid={prob_valid:.4f}, prob_invalid={prob_invalid:.4f}, confidence={final_confidence:.4f}, is_valid={is_valid}")

                    # The model's prediction is final - don't override with confidence threshold
                    if is_valid:
                        return True, reason
                    else:
                        return False, reason

                except Exception as e:
                    logger.exception("AI validation failed")
                    return False, f"✗ AI validation error: {str(e)}"
            else:
                logger.warning("AI validation required but models not loaded")
                if word_count >= self.word_count_threshold:
                    return True, f"⚠ Skipped AI validation (models not loaded) - passed basic checks"
                else:
                    return False, f"✗ Failed basic validation (no AI models available)"

        return True, "✓ Help post validation passed"
    
    def validate_action(self, action_type: str, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Main validation entry point with enhanced error handling"""
        try:
            if action_type not in self.config.action_types:
                return False, f"✗ Unsupported action type: {action_type}"

            if action_type == "login":
                return self.validate_login(metadata)
            elif action_type == "quiz":
                return self.validate_quiz(metadata)
            elif action_type == "help_post":
                return self.validate_help_post(metadata)
            else:
                return False, f"✗ No validator implemented for action type: {action_type}"
                
        except Exception as e:
            logger.exception(f"Validation error for {action_type}")
            return False, f"✗ Validation error: {str(e)}"

    def get_validator_info(self) -> Dict[str, Any]:
        """Get information about the validator configuration"""
        return {
            'word_count_threshold': self.word_count_threshold,
            'repetition_threshold': self.repetition_threshold,
            'min_confidence': self.min_confidence,
            'models_loaded': {
                'classifier': 'classifier' in self.models,
                'vectorizer': 'vectorizer' in self.models,
                'scaler': 'scaler' in self.models
            },
            'features': [
                'tfidf_features', 'word_count', 'char_count', 'sentence_count',
                'code_features', 'quality_indicators', 'repetition_detection',
                'educational_content', 'language_mentions', 'vague_content_detection',
                'informal_language_detection'
            ]
        }
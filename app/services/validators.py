import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import pandas as pd
import re
from datetime import datetime
from app.feature_estimator import FeatureExtractor
import pickle
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "FeatureExtractor":
            return FeatureExtractor
        return super().find_class(module, name)

def custom_load(file_path):
    with open(file_path, 'rb') as f:
        return CustomUnpickler(f).load()

class EnhancedStudentContentValidator:
    def preprocess_text(self, text):
        """Preprocess text for model input."""
        if pd.isna(text) or not text.strip():
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,!?;:()-]', ' ', text).strip()
        return text

    def detect_binary_search_misconception(self, content):
        """Detect binary search misconceptions."""
        content = content.lower()
        if 'binary' not in content or 'search' not in content:
            return False
        misconception_patterns = [
            r'search.*files.*folder.*one\s+by\s+one',
            r'check.*from.*beginning.*to.*end',
            r'go.*through.*each.*file',
            r'look.*at.*every.*item',
            r'start.*from.*beginning',
            r'search.*one\s+by\s+one',
            r'check.*every.*element',
            r'examine.*each.*one',
            r'linear.*search.*but.*call.*binary',
            r'sequential.*search.*binary'
        ]
        for pattern in misconception_patterns:
            if re.search(pattern, content):
                return True
        if 'binary search' in content:
            linear_indicators = [
                'one by one', 'beginning to end', 'check every', 'each file',
                'all files', 'every element', 'sequential', 'in order'
            ]
            linear_count = sum(1 for indicator in linear_indicators if indicator in content)
            if linear_count >= 2:
                return True
        return False

    def detect_sophisticated_spam(self, content, features):
        """Detect sophisticated spam patterns."""
        words = content.split()
        if len(words) < 20:
            return False
        word_counts = Counter(words)
        total_words = len(words)
        for word, count in word_counts.items():
            if len(word) > 3:
                word_ratio = count / total_words
                if word_ratio > 0.15:
                    remaining_words = [w for w in words if w != word]
                    if len(set(remaining_words)) < len(remaining_words) * 0.45:
                        return True
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        if len(valid_sentences) >= 3:
            similar_structure_count = 0
            for i in range(len(valid_sentences) - 1):
                sent1_words = set(valid_sentences[i].split())
                sent2_words = set(valid_sentences[i + 1].split())
                if len(sent1_words & sent2_words) > len(sent1_words | sent2_words) * 0.6:
                    similar_structure_count += 1
            if similar_structure_count >= len(valid_sentences) * 0.5:
                return True
        filler_patterns = [
            r'\b(binary\s+)+binary\b',
            r'\b(but\s+)+but\b',
            r'\b(and\s+)+and\b',
            r'\b(the\s+)+the\b',
            r'\b(\w+\s+)\1{2,}'
        ]
        for pattern in filler_patterns:
            if re.search(pattern, content):
                return True
        if features.get('spam_score', 0) > 1.5:
            repetition_ratio = features.get('repetition_ratio', 0)
            semantic_coherence = features.get('semantic_coherence', 0)
            if repetition_ratio > 0.1 and semantic_coherence < 2:
                return True
        return False

    def detect_valid_dictionary_analogy(self, content, features):
        """Detect valid dictionary analogies."""
        content = content.lower()
        if 'dictionary' not in content and 'flip' not in content:
            return False
        dictionary_patterns = [
            r'looking.*for.*word.*dictionary',
            r'flip.*middle',
            r'dictionary.*middle',
            r'imagine.*dictionary',
            r'like.*dictionary',
            r'word.*dictionary.*middle'
        ]
        analogy_score = 0
        for pattern in dictionary_patterns:
            if re.search(pattern, content):
                analogy_score += 1
        educational_indicators = [
            'imagine', 'like', 'similar', 'analogy', 'example',
            'works', 'helps', 'that\'s how', 'this is how'
        ]
        edu_score = sum(1 for indicator in educational_indicators if indicator in content)
        if analogy_score >= 1 and edu_score >= 1:
            repetition_ratio = features.get('repetition_ratio', 0)
            spam_score = features.get('spam_score', 0)
            if repetition_ratio < 0.2 and spam_score < 2.5:
                return True
        return False

    def apply_enhanced_rule_override(self, content: str, prediction: str, features: Dict[str, Any]) -> Tuple[str, str]:
        """Apply enhanced rule-based overrides."""
        original_pred = prediction
        reason = ""
        content_lower = content.lower()
        if features.get('is_plagiarized'):
            prediction = 'invalid'
            reason = f"Rule override: Plagiarized content - {features.get('plagiarism_reason', '')}"
        elif features.get('is_excessive_informal_slang') and not features.get('is_technical') and features.get('semantic_coherence', 0) < 5.0:
            prediction = 'invalid'
            reason = f"Rule override: Excessive informal slang - {features.get('informal_slang_reason', '')}"
        elif features.get('is_negative_sentiment') and features.get('word_count', 0) > 50:
            prediction = 'invalid'
            reason = f"Rule override: Negative sentiment rant (score={features.get('sentiment_score', 0):.2f})"
        elif self.detect_binary_search_misconception(content_lower):
            prediction = 'invalid'
            reason = "Rule override: Binary search misconception detected"
        elif self.detect_sophisticated_spam(content_lower, features):
            prediction = 'invalid'
            reason = "Rule override: Sophisticated spam pattern detected"
        elif self.detect_valid_dictionary_analogy(content_lower, features):
            prediction = 'valid'
            reason = "Rule override: Valid dictionary analogy detected"
        elif (features.get('is_meaningless') and
              features.get('spam_score', 0) > 2.0 and
              features.get('repetition_ratio', 0) > 0.1):
            prediction = 'invalid'
            reason = "Rule override: Strong spam indicators"
        elif (features.get('semantic_coherence', 0) < 1.5 and
              features.get('repetition_ratio', 0) > 0.15):
            prediction = 'invalid'
            reason = "Rule override: Low coherence with repetition"
        return prediction, reason

class ActionValidator:
    def __init__(self, config):
        """Initialize the ActionValidator with configuration."""
        self.config = config
        self.models = {}
        self.content_validator = EnhancedStudentContentValidator()
        self._load_models()

    def _load_models(self):
        """Load AI models from .pkl files specified in config."""
        try:
            for model_name, model_config in self.config.ai_models.items():
                # Load classifier
                model_path = Path("app") / "models" / model_config.model_file
                if not model_path.exists():
                    logger.error(f"Classifier file not found: {model_path}")
                    raise FileNotFoundError(f"Classifier file not found: {model_path}")
                classifier = joblib.load(model_path)
                logger.info(f"Successfully loaded classifier from: {model_path}")

                # Load vectorizer
                vectorizer_path = Path("app") / "models" / model_config.vectorizer_file
                if not vectorizer_path.exists():
                    logger.error(f"Vectorizer file not found: {vectorizer_path}")
                    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
                vectorizer = joblib.load(vectorizer_path)
                logger.info(f"Successfully loaded vectorizer from: {vectorizer_path}")

                # Load feature extractor
                feature_extractor_path = Path("app") / "models" / model_config.scaler_file
                if not feature_extractor_path.exists():
                    logger.error(f"Feature extractor file not found: {feature_extractor_path}")
                    raise FileNotFoundError(f"Feature extractor file not found: {feature_extractor_path}")
                feature_extractor = custom_load(feature_extractor_path)
                logger.info(f"Successfully loaded feature extractor from: {feature_extractor_path}")

                # Store models with threshold
                self.models[model_name] = {
                    'classifier': classifier,
                    'vectorizer': vectorizer,
                    'feature_extractor': feature_extractor,
                    'threshold': model_config.threshold
                }
            logger.info(f"Loaded models: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _get_ai_reasoning(self, content: str, confidence: float, is_valid: bool, prediction: str, features: Dict[str, Any], override_reason: str) -> str:
        """Generate detailed AI reasoning for validation decision."""
        try:
            reasoning_parts = [
                f"Confidence: {confidence:.3f} ({'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'})",
                f"Content length: {'Short' if len(content.split()) < 30 else 'Long' if len(content.split()) > 1000 else 'Appropriate'}"
            ]
            if override_reason:
                reasoning_parts.append(override_reason)
            if features.get('is_excessive_informal_slang'):
                reasoning_parts.append(f"Excessive informal slang: {features.get('informal_slang_reason', 'N/A')}")
            if features.get('is_negative_or_off_topic'):
                reasoning_parts.append("Negative sentiment or off-topic content detected")
            if features.get('is_plagiarized'):
                reasoning_parts.append(f"Plagiarized content: {features.get('plagiarism_reason', 'N/A')}")
            if features.get('is_meaningless'):
                reasoning_parts.append("Meaningless content detected")
            reasoning_parts.append(f"Repetition ratio: {features.get('repetition_ratio', 0):.3f}")
            reasoning_parts.append(f"Semantic coherence: {features.get('semantic_coherence', 0):.1f}")
            reasoning_parts.append(f"Sentiment score: {features.get('sentiment_score', 0):.2f}")
            reasoning_parts.append(f"Final prediction: {'Valid' if is_valid else 'Invalid'}")
            return "; ".join(reasoning_parts)
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {str(e)}")
            return f"Unable to generate reasoning: {str(e)}"

    def validate_help_post(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate help post action."""
        try:
            help_post_config = self.config.action_types.get('help_post')
            if not help_post_config or not help_post_config.enabled:
                return False, "Help post action type not enabled"
            required_fields = ['content']
            for field in required_fields:
                if field not in metadata:
                    return False, f"Missing required field: {field}"
            content = metadata.get('content', '')
            word_count = metadata.get('word_count', len(content.split()))
            if not content.strip():
                return False, "Content is empty"
            validation_config = help_post_config.validation
            min_word_count = validation_config.threshold.get('min_word_count', 30)
            max_word_count = validation_config.threshold.get('max_word_count', 1000)
            if word_count < min_word_count:
                return False, f"Word count too low: {word_count} (minimum: {min_word_count})"
            if word_count > max_word_count:
                return False, f"Word count too high: {word_count} (maximum: {max_word_count})"
            if validation_config.require_ai:
                return self._validate_with_ai('help_post', metadata)
            return True, "Help post validated successfully"
        except Exception as e:
            logger.error(f"Error validating help post: {str(e)}")
            return False, f"Help post validation error: {str(e)}"

    def _validate_with_ai(self, action_type: str, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate using AI models."""
        try:
            if action_type not in self.models:
                logger.warning(f"No AI model available for action type: {action_type}")
                return True, f"AI validation skipped - no model for {action_type}"
            model_info = self.models[action_type]
            classifier = model_info['classifier']
            vectorizer = model_info['vectorizer']
            feature_extractor = model_info['feature_extractor']
            content = metadata.get('content', '')
            if not content.strip():
                return False, "No content available for AI validation"
            # Preprocess and vectorize
            processed_content = self.content_validator.preprocess_text(content)
            tfidf_features = vectorizer.transform([processed_content])
            # Predict
            prediction = classifier.predict(tfidf_features)[0]
            probabilities = classifier.predict_proba(tfidf_features)[0]
            confidence = np.max(probabilities)
            threshold = self.config.action_types.get(action_type).validation.threshold.get('min_confidence', model_info['threshold'])
            # Extract features
            features = feature_extractor.transform([content])[0]
            # Apply rule overrides
            final_prediction, override_reason = self.content_validator.apply_enhanced_rule_override(content, prediction, features)
            is_valid = final_prediction == 'valid' and confidence >= threshold
            reasoning = self._get_ai_reasoning(content, confidence, is_valid, final_prediction, features, override_reason)
            return is_valid, f"AI validation {'passed' if is_valid else 'failed'}: {reasoning}"
        except Exception as e:
            logger.error(f"Error in AI validation: {str(e)}")
            return False, f"AI validation error: {str(e)}"

    def validate_action(self, action_type: str, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Main validation method that routes to specific validators."""
        try:
            if action_type not in self.config.action_types:
                return False, f"Unknown action type: {action_type}"
            if action_type == 'help_post':
                return self.validate_help_post(metadata)
            action_config = self.config.action_types[action_type]
            if not action_config.enabled:
                return False, f"Action type '{action_type}' not enabled"
            if action_config.validation.require_ai:
                return self._validate_with_ai(action_type, metadata)
            return True, f"Action type '{action_type}' validated successfully"
        except Exception as e:
            logger.error(f"Error validating action {action_type}: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def test_model_prediction(self, test_content: str = "This is a test content for validation") -> Dict[str, Any]:
        """Test method to debug model predictions."""
        try:
            if 'help_post' not in self.models:
                return {"error": "Help post model not available"}
            model_info = self.models['help_post']
            classifier = model_info['classifier']
            vectorizer = model_info['vectorizer']
            feature_extractor = model_info['feature_extractor']
            processed_content = self.content_validator.preprocess_text(test_content)
            tfidf_features = vectorizer.transform([processed_content])[0]
            prediction = classifier.predict(tfidf_features)[0]
            probabilities = classifier.predict_proba(tfidf_features)[0]
            confidence = np.max(probabilities)
            threshold = self.config.action_types.get('help_post').validation.threshold.get('min_confidence', model_info['threshold'])
            features = feature_extractor.transform([test_content])[0]
            # Apply rule overrides
            final_prediction, override_reason = self.content_validator.apply_enhanced_rule_override(test_content, prediction, features)
            is_valid = final_prediction == 'valid' and confidence >= threshold
            result = {
                "model_type": str(type(classifier)),
                "test_content": test_content,
                "prediction": final_prediction,
                "original_prediction": prediction,
                "override_reason": override_reason,
                "prediction_successful": True,
                "probabilities": probabilities.tolist(),
                "confidence": float(confidence),
                "is_valid": is_valid,
                "features": features,
                "reasoning": self._get_ai_reasoning(test_content, confidence, is_valid, final_prediction, features, override_reason)
            }
            return result
        except Exception as e:
            return {"error": str(e), "prediction_successful": False}

    def get_validator_info(self) -> Dict[str, Any]:
        """Get information about the validator and loaded models."""
        try:
            model_info = {}
            for model_name, model_data in self.models.items():
                model_info[model_name] = {
                    'threshold': model_data['threshold'],
                    'loaded': True,
                    'model_type': str(type(model_data['classifier'])),
                }
            action_types_info = {}
            for action_type, config in self.config.action_types.items():
                action_types_info += action_type
                {
                    'enabled': config.enabled,
                    'requires_ai': config.validation.require_ai,
                    'threshold': config.validation.threshold if config.validation.threshold else {},
                }
            return {
                'validator_version': '1.0',
                'models': model_info,
                'action_types': action_types_info,
                'total_valid_models': len(self.models),
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting validator info: {str(e)}")
            return {
                "error": str(e),
                'validator_version': '1.0',
                'timestamp': datetime.utcnow().isoformat(),
            }
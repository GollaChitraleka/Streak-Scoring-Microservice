import re
from collections import Counter
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Download VADER lexicon
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.error(f"Error downloading VADER lexicon: {e}")

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            feat = {}
            text = str(text).lower() if not pd.isna(text) else ""
            words = text.split()
            word_count = len(words)
            # Repetition
            repetition_ratio, _, spam_score = self.detect_word_repetition_enhanced(text)
            feat['repetition_ratio'] = repetition_ratio
            feat['spam_score'] = spam_score
            # Coherence
            feat['semantic_coherence'] = self.calculate_semantic_coherence_enhanced(text)
            # Informal slang
            is_informal, informal_reason = self.detect_excessive_informal_slang(text)
            feat['is_excessive_informal_slang'] = is_informal
            feat['informal_slang_reason'] = informal_reason
            # Negative sentiment
            is_negative, sentiment_score = self.detect_negative_sentiment(text)
            feat['is_negative_sentiment'] = is_negative
            feat['sentiment_score'] = sentiment_score
            # Plagiarism
            is_plagiarized, plagiarism_reason = self.detect_plagiarized_content(text)
            feat['is_plagiarized'] = is_plagiarized
            feat['plagiarism_reason'] = plagiarism_reason
            # Off-topic
            off_topic_indicators = ['social media', 'meme', 'cats', 'online', 'argument', 'drama']
            feat['is_negative_or_off_topic'] = any(i in text for i in off_topic_indicators) or is_negative
            # Additional features
            feat['word_count'] = word_count
            feat['is_meaningless'] = self.detect_meaningless_content_enhanced(text)[0]
            feat['is_programming'] = self.detect_programming_content(text)[0]
            feat['is_technical'] = self.detect_technical_explanation(text)[0]
            features.append(feat)
        return features

    def detect_word_repetition_enhanced(self, text):
        if pd.isna(text) or not str(text).strip():
            return 0, 0, 0
        text_str = str(text).lower()
        words = text_str.split()
        if len(words) < 5:
            return 0, 0, 0
        word_counts = Counter(words)
        total_words = len(words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        repetition_ratio = most_common_count / total_words if total_words > 0 else 0
        highly_repeated_words = sum(1 for count in word_counts.values() if count > max(3, total_words * 0.1))
        spam_score = 0
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                consecutive_repeats += 1
        for word, count in word_counts.items():
            if len(word) > 3 and count > 5:
                word_dominance = count / total_words
                if word_dominance > 0.2:
                    spam_score += word_dominance * 10
        spam_patterns = [
            r'\b(\w+)\s+\1\s+\1\b',
            r'\b(\w+)\s+(\w+)\s+\1\s+\w+\s+\1\b',
            r'\b(\w+)\s+(\w+)\s+\1\s+\2\s+\1\b'
        ]
        for pattern in spam_patterns:
            matches = re.findall(pattern, text_str)
            spam_score += len(matches) * 2
        meaningless_combinations = 0
        sentences = re.split(r'[.!?]+', text_str)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_words = sentence.split()
            if len(sent_words) < 3:
                continue
            sent_word_counts = Counter(sent_words)
            max_word_in_sentence = max(sent_word_counts.values()) if sent_word_counts else 0
            if max_word_in_sentence > len(sent_words) * 0.5:
                meaningless_combinations += 1
        spam_score += meaningless_combinations * 1.5
        return repetition_ratio, highly_repeated_words, spam_score

    def detect_excessive_informal_slang(self, text):
        if pd.isna(text) or not str(text).strip():
            return False, ""
        text_str = str(text).lower()
        words = text_str.split()
        if len(words) < 2:
            return False, ""
        slang_indicators = [
            'lol', 'lmao', 'bro', 'dude', 'man', 'yo', 'yep', 'nah', 'wtf', 'haha',
            'kinda', 'sorta', 'umm', 'idk', 'tbh', 'bruh', 'fr', 'smh', 'rofl', 'btw',
            'chill', 'lit', 'vibe', 'nope', 'yup', 'cuz', 'prolly', 'wassup', 'fam',
            'easy', 'anyways', 'lame'
        ]
        slang_count = sum(1 for word in words if word in slang_indicators)
        slang_ratio = slang_count / len(words) if len(words) > 0 else 0
        casual_fillers = ['like', 'basically', 'you know', 'i mean', 'stuff', 'things', 'actually']
        filler_count = sum(1 for filler in casual_fillers if filler in text_str)
        is_informal = slang_ratio > 0.1 or filler_count >= 3 or (slang_count >= 2 and len(words) <= 15)
        reason = f"Slang ratio={slang_ratio:.3f}, filler_count={filler_count}"
        logger.debug(f"Slang: text={text[:50]}..., slang_count={slang_count}, filler_count={filler_count}, ratio={slang_ratio:.3f}, is_informal={is_informal}")
        return is_informal, reason

    def detect_negative_sentiment(self, text):
        if pd.isna(text) or not str(text).strip():
            return False, 0.0
        text_str = str(text)
        sentiment_scores = self.sid.polarity_scores(text_str)
        negative_indicators = ['done', 'mess', 'annoying', 'over it', 'ugh', 'whatever', 'don\'t get it', 'frustrated', 'sucks']
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_str.lower())
        is_negative = (sentiment_scores['compound'] < -0.1 or sentiment_scores['neg'] > 0.15 or negative_count >= 3)
        logger.debug(f"Sentiment: text={text[:50]}..., scores={sentiment_scores}, negative_count={negative_count}, is_negative={is_negative}")
        return is_negative, sentiment_scores['compound']

    def detect_plagiarized_content(self, text):
        if pd.isna(text) or not str(text).strip():
            return False, ""
        text_str = str(text).lower()
        words = text_str.split()
        formal_indicators = [
            'is defined as', 'is a type of', 'is used to', 'is characterized by',
            'refers to', 'consists of', 'comprises', 'according to', 'in the context of',
            'can be described as', 'is implemented by', 'is typically', 'is known as'
        ]
        formal_score = sum(1 for indicator in formal_indicators if indicator in text_str)
        generic_terms = [
            'data structure', 'algorithm', 'method', 'technique', 'approach',
            'system', 'process', 'operation', 'functionality', 'implementation'
        ]
        term_count = sum(1 for term in generic_terms if term in text_str)
        passive_patterns = [
            r'is\s+\w+ed\s*by\b',
            r'are\s+\w+ed\s*by\b',
            r'was\s+\w+ed\s*by\b',
            r'were\s+\w+ed\s*by\b'
        ]
        passive_score = 0
        for pattern in passive_patterns:
            passive_score += 1 if re.search(pattern, text_str) else 0
        informal_score = sum(1 for indicator in ['like', 'so', 'imagine', 'think of', 'basically'] if indicator in text_str)
        is_plagiarized = (formal_score >= 2 or term_count >= 3 or passive_score >= 2) and informal_score <= 1
        if len(words) < 15 and formal_score >= 1:
            is_plagiarized = True
        reason = f"Formal score={formal_score}, term_count={term_count}, passive_score={passive_score}, informal_score={informal_score}" if is_plagiarized else ""
        return is_plagiarized, reason

    def detect_meaningless_content_enhanced(self, text):
        if pd.isna(text) or not str(text).strip():
            return True, "Empty content"
        text_str = text.lower()
        words = text_str.split()
        if len(words) < 5:
            return False, ""
        repetition_ratio, _, spam_score = self.detect_word_repetition_enhanced(text)
        if repetition_ratio > 0.18:
            return True, f"High word repetition ({repetition_ratio:.2%})"
        if spam_score > 2.0:
            return True, f"Spam patterns detected (score: {spam_score:.1f})"
        is_plagiarized, plagiarism_reason = self.detect_plagiarized_content(text)
        if is_plagiarized:
            return True, plagiarism_reason
        is_negative, sentiment_score = self.detect_negative_sentiment(text)
        if is_negative and len(words) >= 50:
            return True, f"Negative sentiment rant (score: {sentiment_score:.2f})"
        word_counts = Counter(words)
        if len(word_counts) > 0:
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            common_words = {
                'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'while', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
                'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                'when', 'where', 'why', 'how', 'can', 'will', 'do', 'does', 'have', 'has',
                'had', 'be', 'been', 'was', 'were', 'would', 'could', 'should', 'may', 'might'
            }
            if (most_common_word not in common_words and
                most_common_count > len(words) * 0.18 and
                len(most_common_word) > 3):
                return True, f"Content dominated by '{most_common_word}' ({most_common_count}/{len(words)})"
        if self.detect_binary_search_misconception(text_str)[0]:
            return True, "Contains binary search misconception"
        sentences = re.split(r'[.!?]+', text_str)
        meaningless_sentences = 0
        total_valid_sentences = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_words = sentence.split()
            if len(sent_words) < 3:
                continue
            total_valid_sentences += 1
            sent_word_counts = Counter(sent_words)
            if len(sent_word_counts) > 0:
                max_word_freq = max(sent_word_counts.values())
                if max_word_freq > len(sent_words) * 0.45:
                    meaningless_sentences += 1
                    continue
            unique_words_in_sentence = len(set(sent_words))
            if unique_words_in_sentence < len(sent_words) * 0.45:
                meaningless_sentences += 1
        if total_valid_sentences > 0 and meaningless_sentences / total_valid_sentences > 0.35:
            return True, f"High proportion of meaningless sentences ({meaningless_sentences}/{total_valid_sentences})"
        return False, ""

    def detect_programming_content(self, text):
        if pd.isna(text) or not str(text).strip():
            return False, 0
        text_str = str(text).lower()
        programming_keywords = {
            'python': ['def', 'import', 'class', 'return', 'if', 'else', 'for', 'while', 'try', 'except'],
            'javascript': ['function', 'var', 'let', 'const', 'return', 'if', 'else', 'for', 'while'],
            'java': ['public', 'private', 'class', 'static', 'void', 'int', 'string', 'return'],
            'c_cpp': ['int', 'char', 'float', 'double', 'void', 'return', 'printf', 'scanf', 'include'],
            'general': ['algorithm', 'function', 'variable', 'loop', 'array', 'string', 'push', 'pop', 'append']
        }
        code_patterns = [
            r'\b(def|function|class|public|private|static|void)\s+\w+\s*\(',
            r'\b(if|while|for)\s+\(',
            r'\{[^}]*\}',
            r'\[[^\]]*\]',
            r'[a-zA-Z_]\w*\s*\(',
            r'#include\s*',
            r'import\s+\w+',
            r'\w+\.\w+\(',
            r'\w+\s*=\s*'
        ]
        programming_score = 0
        for language, keywords in programming_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_str)
            programming_score += matches * 0.5
        for pattern in code_patterns:
            matches = len(re.findall(pattern, text_str))
            programming_score += matches * 1.0
        if re.search(r'^\s*(def|function|class|public|private)', text_str, re.MULTILINE):
            programming_score += 1.0
        return programming_score > 2.0, programming_score

    def detect_technical_explanation(self, text):
        if pd.isna(text) or not str(text).strip():
            return False, 0
        text_str = str(text).lower()
        explanation_indicators = [
            'is used', 'helps', 'allows', 'enables', 'works by',
            'algorithm', 'method', 'approach', 'technique', 'process',
            'step', 'first', 'then', 'next', 'finally', 'such as', 'like',
            'binary search', 'sorted array', 'divide', 'efficiency', 'complexity'
        ]
        technical_terms = [
            'array', 'list', 'string', 'integer', 'boolean', 'variable', 'function',
            'method', 'class', 'object', 'algorithm', 'data structure', 'complexity',
            'time complexity', 'space complexity', 'logarithmic', 'linear',
            'stack', 'queue', 'heap', 'tree'
        ]
        explanation_score = 0
        for indicator in explanation_indicators:
            if indicator in text_str:
                explanation_score += 1.0
        for term in technical_terms:
            if term in text_str:
                explanation_score += 0.5
        sentences = re.split(r'[.!?]+', text_str)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]
        if len(valid_sentences) >= 1:
            explanation_score += 1.0
        for phrase in ['help', 'understand', 'learn', 'explain', 'concept', 'programming']:
            if phrase in text_str:
                explanation_score += 0.5
        return explanation_score > 1.5, explanation_score

    def calculate_semantic_coherence_enhanced(self, text):
        if pd.isna(text) or not str(text).strip():
            return 0
        text_str = str(text).lower()
        repetition_ratio, _, spam_score = self.detect_word_repetition_enhanced(text)
        if spam_score > 2.0:
            return max(0, 1 - (spam_score * 0.4))
        if repetition_ratio > 0.2:
            return max(0, 2 - (repetition_ratio * 8))
        is_plagiarized, _ = self.detect_plagiarized_content(text)
        if is_plagiarized:
            return 0.5
        is_negative, sentiment_score = self.detect_negative_sentiment(text)
        if is_negative and len(text_str.split()) >= 50:
            return 0.5
        if self.detect_binary_search_misconception(text_str)[0]:
            return 0.5
        score = 0
        sentences = re.split(r'[.!?]+', text_str)
        valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
        if self.detect_programming_content(text)[0]:
            score += 1
        if not valid_sentences:
            return score
        flow_indicators = [
            'because', 'since', 'therefore', 'however', 'moreover', 'furthermore',
            'first', 'second', 'next', 'then', 'finally', 'in conclusion',
            'for example', 'such as', 'in other words', 'that is',
            'this means', 'this shows', 'this demonstrates', 'as a result',
            'like', 'imagine', 'similar to', 'basically', 'essentially',
            'algorithm', 'function', 'works by', 'used to', 'helps to',
            'dictionary', 'middle', 'flip', 'analogy'
        ]
        flow_score = sum(1 for indicator in flow_indicators if indicator in text_str)
        score += min(flow_score, 4)
        sentences_text = ' '.join(valid_sentences)
        sentences_words = sentences_text.split()
        if len(sentences_words) > 0:
            sentences_word_counts = Counter(sentences_words)
            max_sentence_word_freq = max(sentences_word_counts.values())
            if max_sentence_word_freq > len(sentences_words) * 0.15:
                score -= 1
        complete_sentences = 0
        spam_sentences = 0
        for sentence in valid_sentences:
            words = sentence.split()
            if len(words) >= 5:
                word_counts = Counter(words)
                max_word_count = max(word_counts.values()) if word_counts else 0
                if max_word_count > len(words) * 0.35:
                    spam_sentences += 1
                    continue
                meaningful_verbs = [
                    'is', 'are', 'was', 'were', 'have', 'has', 'had',
                    'works', 'helps', 'uses', 'makes', 'allows', 'can',
                    'does', 'performs', 'searches', 'finds', 'divides', 'splits',
                    'compares', 'checks', 'looks', 'flips', 'imagine'
                ]
                if any(verb in words for verb in meaningful_verbs):
                    complete_sentences += 1
        if len(valid_sentences) > 0:
            spam_ratio = spam_sentences / len(valid_sentences)
            if spam_ratio > 0.3:
                score -= 1
            structure_score = (complete_sentences / len(valid_sentences)) * 5
            score += structure_score
        return max(0, score)

    def detect_binary_search_misconception(self, content):
        if pd.isna(content) or not content.strip() or 'binary' not in content.lower() or 'search' not in content.lower():
            return False, ""
        content = content.lower()
        # Ensure the content describes a linear search process in the context of binary search
        misconception_patterns = [
            r'binary\s+search.*(one\s+by\s+one|each\s+(file|item|element)\s+by\s+\2|check\s+every\s+(file|item|element))',
            r'binary\s+search.*(from.*beginning.*to.*end|start.*beginning.*until.*end)',
            r'binary\s+search.*(go\s+through\s+each|look\s+at\s+every\s+(file|item|element))',
            r'binary\s+search.*(sequential|linear\s+search|in\s+order\s+from\s+start)',
            r'(describe|call|is).*binary\s+search.*(check\s+all|every\s+(file|item|element)\s+one\s+at\s+a\s+time)'
        ]
        for pattern in misconception_patterns:
            if re.search(pattern, content):
                return True, "Binary search misconception: Describes linear search as binary search"
        linear_indicators = [
            'one by one', 'check every file', 'check every item', 'check every element',
            'beginning to end', 'each file by file', 'each item by item', 'all files one at a time',
            'every element in order', 'sequential search'
        ]
        # Require at least two indicators and explicit mention of "binary search" within 20 words
        linear_count = sum(1 for indicator in linear_indicators if indicator in content)
        if linear_count >= 2:
            words = content.split()
            for i, word in enumerate(words):
                if word == 'binary' and i + 1 < len(words) and words[i + 1] == 'search':
                    context = ' '.join(words[max(0, i - 10):i + 10])
                    if any(indicator in context for indicator in linear_indicators):
                        return True, "Binary search misconception: Multiple linear search indicators near binary search"
        return False, ""
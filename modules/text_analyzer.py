"""
Enhanced text analysis module with compound phrase detection.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
import math
import numpy as np

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import spacy
from spacy.tokens import Doc
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Project imports
from config.config import (
    MAX_WORDS_WORDCLOUD,
    MIN_WORD_LENGTH,
    MAX_WORD_LENGTH,
    MIN_WORD_FREQUENCY,
    STOPWORDS_DE_PATH,
    STOPWORDS_EN_PATH,
    OUTPUT_DIR,
    COMMON_STOPWORDS,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    ERROR_MESSAGES
)
from modules.utils import setup_logging, safe_file_read, safe_file_write, get_timestamp

# Initialize logger
logger = setup_logging(__name__)

class TextAnalysisError(Exception):
    """Base class for text analysis errors."""
    pass

class LanguageError(TextAnalysisError):
    """Error for language-related issues."""
    pass

class TextAnalyzer:
    """Enhanced text analyzer with compound phrase detection."""
    
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        """
        Initialize text analyzer.
        
        Args:
            language: Language code ('en' or 'de')
        """
        # Initialize stats first to ensure it exists
        self.stats = {
            'processed_words': 0,
            'unique_words': 0,
            'phrases_found': 0,
            'average_word_length': 0,
            'warnings': []
        }
        
        if language not in SUPPORTED_LANGUAGES:
            raise LanguageError(
                ERROR_MESSAGES['language_not_supported'].format(lang=language)
            )
            
        self.language = language.lower()
        
        # Initialize NLTK resources first
        self._initialize_nltk()
        
        # Then load stopwords
        self.stopwords = self._load_stopwords()
        
        # Initialize language-specific tools
        self._initialize_language_tools()

    def _initialize_nltk(self):
        """Initialize required NLTK resources."""
        required_data = ['punkt', 'stopwords']
        if self.language == 'en':
            required_data.extend(['wordnet', 'averaged_perceptron_tagger'])
            
        nltk_dir = Path.home() / 'nltk_data'  # Use home directory for NLTK data
        nltk_dir.mkdir(exist_ok=True)
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                logger.info(f"Downloading NLTK data: {data}")
                try:
                    nltk.download(data, download_dir=str(nltk_dir), quiet=True)
                except Exception as e:
                    warning = f"Failed to download {data}: {str(e)}"
                    logger.warning(warning)
                    self.stats['warnings'].append(warning)
                    # Try alternative download location
                    try:
                        nltk.download(data, download_dir=str(Path(__file__).parent / 'nltk_data'), quiet=True)
                    except Exception as e2:
                        logger.error(f"All download attempts failed for {data}: {str(e2)}")

    def _initialize_language_tools(self):
        """Initialize language-specific processing tools."""
        if self.language == 'en':
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = SnowballStemmer('german')
        
        # Initialize collocation measures
        self.bigram_measures = BigramAssocMeasures()
        self.trigram_measures = TrigramAssocMeasures()

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from all sources."""
        stopwords_set = set()
        
        # Load NLTK stopwords
        try:
            stopwords_set.update(stopwords.words(
                'german' if self.language == 'de' else 'english'
            ))
        except Exception as e:
            warning = f"Could not load NLTK stopwords: {str(e)}"
            logger.warning(warning)
            self.stats['warnings'].append(warning)
        
        # Load custom stopwords
        stopwords_file = STOPWORDS_DE_PATH if self.language == 'de' else STOPWORDS_EN_PATH
        try:
            if Path(stopwords_file).exists():
                stopwords_set.update(
                    safe_file_read(stopwords_file).splitlines()
                )
        except Exception as e:
            warning = f"Could not load custom stopwords: {str(e)}"
            logger.warning(warning)
            self.stats['warnings'].append(warning)
        
        # Add common stopwords
        stopwords_set.update(COMMON_STOPWORDS.get(self.language, []))
        
        return stopwords_set

    def _normalize_word(self, word: str) -> Optional[str]:
        """
        Clean and normalize a word.
        
        Args:
            word: Word to normalize
            
        Returns:
            Optional[str]: Normalized word or None if invalid
        """
        word = word.lower().strip()
        
        # Remove non-alphanumeric (keeping umlauts for German)
        if self.language == 'de':
            word = ''.join(c for c in word if c.isalnum() or c in 'äöüß')
        else:
            word = ''.join(c for c in word if c.isalnum())
        
        # Validate length
        if not MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH:
            return None
            
        # Ensure at least one letter
        if not any(c.isalpha() for c in word):
            return None
            
        # Apply lemmatization/stemming
        try:
            if self.language == 'en':
                word = self.lemmatizer.lemmatize(word)
            else:
                word = self.lemmatizer.stem(word)
        except Exception as e:
            logger.debug(f"Lemmatization failed for word '{word}': {str(e)}")
            
        return word

    def _extract_phrases(self, sentences: List[List[str]]) -> List[Tuple[str, float]]:
        """
        Extract meaningful phrases using collocation detection.
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            List[Tuple[str, float]]: List of phrases with scores
        """
        try:
            # Extract bigrams
            bigram_finder = BigramCollocationFinder.from_documents(sentences)
            bigram_finder.apply_freq_filter(MIN_WORD_FREQUENCY)
            bigram_finder.apply_word_filter(
                lambda w: w.lower() in self.stopwords or not w.isalnum()
            )
            
            scored_bigrams = bigram_finder.score_ngrams(
                self.bigram_measures.likelihood_ratio
            )
            
            # Extract trigrams
            trigram_finder = TrigramCollocationFinder.from_documents(sentences)
            trigram_finder.apply_freq_filter(MIN_WORD_FREQUENCY)
            trigram_finder.apply_word_filter(
                lambda w: w.lower() in self.stopwords or not w.isalnum()
            )
            
            scored_trigrams = trigram_finder.score_ngrams(
                self.trigram_measures.likelihood_ratio
            )
            
            # Combine and normalize scores
            phrases = []
            
            for gram, score in scored_bigrams:
                phrase = ' '.join(gram)
                phrases.append((phrase, score))
                
            for gram, score in scored_trigrams:
                phrase = ' '.join(gram)
                phrases.append((phrase, score))
            
            # Normalize scores
            if phrases:
                max_score = max(score for _, score in phrases)
                phrases = [(phrase, score/max_score) for phrase, score in phrases]
            
            self.stats['phrases_found'] = len(phrases)
            return sorted(phrases, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Phrase extraction error: {str(e)}")
            return []

    def _calculate_tfidf(self, 
                        frequencies: Dict[str, int],
                        total_docs: int,
                        doc_frequencies: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores.
        
        Args:
            frequencies: Term frequencies
            total_docs: Total number of documents
            doc_frequencies: Document frequencies
            
        Returns:
            Dict[str, float]: TF-IDF scores
        """
        total_terms = sum(frequencies.values())
        scores = {}
        
        if total_terms == 0:
            return scores
            
        for term, freq in frequencies.items():
            tf = freq / total_terms
            df = doc_frequencies.get(term, 1)
            idf = math.log(total_docs / df + 1)
            scores[term] = tf * idf
            
        return scores

    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict: Analysis results
        """
        if not text:
            raise TextAnalysisError("Empty text provided for analysis")
            
        try:
            # Reset statistics for new analysis
            self.stats.update({
                'processed_words': 0,
                'unique_words': 0,
                'phrases_found': 0,
                'average_word_length': 0,
                'warnings': []
            })
            
            # Tokenize into sentences and words
            sentences = sent_tokenize(text)
            tokenized_sentences = [word_tokenize(sent) for sent in sentences]
            
            # Process words
            word_freq = Counter()
            valid_words = []
            total_length = 0
            
            for sentence in tokenized_sentences:
                for word in sentence:
                    normalized = self._normalize_word(word)
                    if normalized and normalized not in self.stopwords:
                        word_freq[normalized] += 1
                        valid_words.append(normalized)
                        total_length += len(normalized)
            
            # Calculate word statistics
            self.stats['processed_words'] = len(valid_words)
            self.stats['unique_words'] = len(word_freq)
            
            if valid_words:
                self.stats['average_word_length'] = total_length / len(valid_words)
            
            # Extract phrases
            phrases = self._extract_phrases(tokenized_sentences)
            
            # Calculate TF-IDF scores
            tfidf_scores = self._calculate_tfidf(
                word_freq,
                1,  # treating text as single document
                {word: 1 for word in word_freq}
            )
            
            # Combine results
            results = {
                'word_frequencies': dict(word_freq),
                'phrases': dict(phrases),
                'tfidf_scores': tfidf_scores,
                'statistics': self.stats.copy()  # Create a copy to avoid reference issues
            }
            
            # Save results
            timestamp = get_timestamp()
            results_path = OUTPUT_DIR / f'analysis_{self.language}_{timestamp}.json'
            safe_file_write(results, results_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Text analysis error: {str(e)}")
            raise TextAnalysisError(f"Analysis failed: {str(e)}")

    def generate_wordcloud(self, 
                          scores: Dict[str, float],
                          output_path: Optional[Path] = None) -> str:
        """
        Generate word cloud visualization.
        
        Args:
            scores: Word scores
            output_path: Optional custom output path
            
        Returns:
            str: Path to generated word cloud
        """
        if not scores:
            raise TextAnalysisError("No scores provided for word cloud generation")
            
        try:
            if output_path is None:
                timestamp = get_timestamp()
                output_path = OUTPUT_DIR / f'wordcloud_{self.language}_{timestamp}.png'
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=1600,
                height=800,
                background_color='white',
                max_words=MAX_WORDS_WORDCLOUD,
                min_font_size=10,
                prefer_horizontal=0.7,
                collocations=True
            ).generate_from_frequencies(scores)
            
            # Save visualization
            plt.figure(figsize=(20, 10), facecolor='none')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Word cloud generation error: {str(e)}")
            raise TextAnalysisError(f"Word cloud generation failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Example usage
        text = "Example text for analysis..."
        analyzer = TextAnalyzer(language='en')
        results = analyzer.analyze_text(text)
        
        # Generate word cloud
        wordcloud_path = analyzer.generate_wordcloud(
            {word: score for word, score in results['tfidf_scores'].items()}
        )
        
        print(f"Analysis complete. Word cloud saved to: {wordcloud_path}")
        
    except TextAnalysisError as e:
        print(f"Analysis Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
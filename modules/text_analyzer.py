"""
Enhanced text analysis module with compound phrase detection and offline support.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
import math
import numpy as np
import os

# SSL context modification moved to separate function
def configure_ssl_context():
    """Configure SSL context for downloads if needed."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# Rest of imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Project imports remain the same
from config.config import (
    MAX_WORDS_WORDCLOUD,
    MIN_WORD_LENGTH,
    MAX_WORD_LENGTH,
    MIN_WORD_FREQUENCY,
    STOPWORDS_DE_PATH,
    STOPWORDS_EN_PATH,
    OUTPUT_DIR,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    ERROR_MESSAGES,
    MAX_WORDS_FOR_AI,
    WORD_SCORE_WEIGHTS,
    MIN_TFIDF_THRESHOLD
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

class StopwordsHandler:
    """Handler for managing and combining stopwords from NLTK and custom files."""
    
    def __init__(self, language: str):
        self.language = language.lower()
        self.stopwords_set = set()
        self._load_stopwords()
        
    def _load_stopwords(self):
        """Load stopwords from NLTK and custom file."""
        # 1. Load NLTK stopwords
        try:
            self.stopwords_set.update(stopwords.words(
                'german' if self.language == 'de' else 'english'
            ))
            logger.info(f"Loaded {len(self.stopwords_set)} NLTK stopwords")
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords: {str(e)}")
        
        # 2. Load custom stopwords from config file
        stopwords_file = STOPWORDS_DE_PATH if self.language == 'de' else STOPWORDS_EN_PATH
        try:
            if Path(stopwords_file).exists():
                custom_words = set(word.strip() for word in 
                    safe_file_read(stopwords_file).splitlines() 
                    if word.strip())
                self.stopwords_set.update(custom_words)
                logger.info(f"Added {len(custom_words)} custom stopwords")
        except Exception as e:
            logger.warning(f"Could not load custom stopwords: {str(e)}")
        
    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.lower().strip() in self.stopwords_set
        
    def add_stopwords(self, words: List[str]):
        """Add new stopwords and save to custom file."""
        new_words = {word.lower().strip() for word in words if word.strip()}
        self.stopwords_set.update(new_words)
        self._save_custom_stopwords()
        
    def remove_stopwords(self, words: List[str]):
        """Remove words from stopwords and update custom file."""
        words = {word.lower().strip() for word in words if word.strip()}
        self.stopwords_set.difference_update(words)
        self._save_custom_stopwords()
        
    def _save_custom_stopwords(self):
        """Save current custom stopwords to file."""
        stopwords_file = STOPWORDS_DE_PATH if self.language == 'de' else STOPWORDS_EN_PATH
        try:
            # Get only custom words (those not in NLTK)
            nltk_stops = set(stopwords.words(
                'german' if self.language == 'de' else 'english'
            ))
            custom_stops = self.stopwords_set - nltk_stops
            
            # Save only if we have custom stopwords
            if custom_stops:
                content = '\n'.join(sorted(custom_stops))
                safe_file_write(content, stopwords_file)
                logger.info(f"Saved {len(custom_stops)} custom stopwords to {stopwords_file}")
        except Exception as e:
            logger.error(f"Failed to save custom stopwords: {str(e)}")


class TextAnalysisError(Exception):
    """Base class for text analysis errors."""
    pass

class TextAnalyzer:
    """Enhanced text analyzer with compound phrase detection and offline support."""
    
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        """
        Initialize text analyzer with improved offline support.
        
        Args:
            language: Language code ('en' or 'de')
        """
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
        
        # Define NLTK data paths
        self.nltk_paths = [
            Path.home() / 'nltk_data',  # Primary path in home directory
            Path(__file__).parent / 'nltk_data',  # Fallback path in module directory
            Path(nltk.data.find('.')).parent  # Default NLTK path
        ]
        
        # Initialize NLTK resources first
        self._initialize_nltk()
        
        # Initialize stopwords handler
        self.stopwords_handler = StopwordsHandler(language)
        self.stopwords = self.stopwords_handler.stopwords_set
        
        # Initialize language-specific tools
        self._initialize_language_tools()

    def _check_nltk_data(self, resource: str) -> bool:
        """
        Check if NLTK resource exists in any of the data directories.
        
        Args:
            resource: Name of the NLTK resource
            
        Returns:
            bool: True if resource exists, False otherwise
        """
        for path in self.nltk_paths:
            resource_path = path / resource
            if resource_path.exists():
                return True
        return False

    def _initialize_nltk(self):
        """Initialize required NLTK resources with offline support."""
        required_data = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
        }
        
        if self.language == 'en':
            required_data.update({
                'wordnet': 'corpora/wordnet',
                'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
            })

        # Create data directories if they don't exist
        for path in self.nltk_paths:
            path.mkdir(parents=True, exist_ok=True)
            
        # Add all paths to NLTK's search path
        for path in self.nltk_paths:
            nltk.data.path.append(str(path))

        missing_resources = []
        for resource, path in required_data.items():
            if not any(self._check_nltk_data(path) for path in self.nltk_paths):
                missing_resources.append(resource)

        if missing_resources:
            logger.info(f"Missing NLTK resources: {missing_resources}")
            try:
                # Configure SSL only if we need to download
                configure_ssl_context()
                
                for resource in missing_resources:
                    try:
                        logger.info(f"Attempting to download {resource}")
                        nltk.download(resource, download_dir=str(self.nltk_paths[0]), quiet=True)
                    except Exception as e:
                        warning = f"Failed to download {resource} to primary location: {str(e)}"
                        logger.warning(warning)
                        self.stats['warnings'].append(warning)
                        
                        # Try fallback location
                        try:
                            nltk.download(resource, download_dir=str(self.nltk_paths[1]), quiet=True)
                        except Exception as e2:
                            error_msg = f"Failed to download {resource} to all locations. Will attempt to continue."
                            logger.error(error_msg)
                            self.stats['warnings'].append(error_msg)
            except Exception as e:
                warning = f"Network error during NLTK downloads: {str(e)}. Will attempt to continue with available resources."
                logger.warning(warning)
                self.stats['warnings'].append(warning)

    def _initialize_language_tools(self):
        """Initialize language-specific processing tools."""
        if self.language == 'en':
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = SnowballStemmer('german')
        
        # Initialize collocation measures
        self.bigram_measures = BigramAssocMeasures()
        self.trigram_measures = TrigramAssocMeasures()

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

    def _combine_word_scores(self, frequencies: Dict[str, int], tfidf_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Combine word frequencies and TF-IDF scores into a single relevance score.

        Args:
            frequencies: Word frequency dictionary
            tfidf_scores: TF-IDF scores dictionary

        Returns:
            Dict[str, float]: Combined scores dictionary
        """
        if not frequencies or not tfidf_scores:
            logger.warning("Empty frequencies or TF-IDF scores in _combine_word_scores")
            return {}

        # Normalize frequency scores to 0-1 range
        max_freq = max(frequencies.values())
        if max_freq == 0:
            logger.warning("Maximum frequency is 0")
            return {}

        normalized_freq = {
            word: count/max_freq 
            for word, count in frequencies.items()
        }

        # Normalize TF-IDF scores to 0-1 range
        max_tfidf = max(tfidf_scores.values())
        if max_tfidf == 0:
            logger.warning("Maximum TF-IDF score is 0")
            return {}

        normalized_tfidf = {
            word: score/max_tfidf 
            for word, score in tfidf_scores.items()
        }

        # Combine scores using configured weights
        combined_scores = {}
        for word in frequencies:
            if word in tfidf_scores:
                freq_component = normalized_freq[word] * WORD_SCORE_WEIGHTS['frequency_weight']
                tfidf_component = normalized_tfidf[word] * WORD_SCORE_WEIGHTS['tfidf_weight']
                combined_scores[word] = freq_component + tfidf_component

        if not combined_scores:
            logger.warning("No combined scores generated")

        return combined_scores

    def _get_top_words(self, frequencies: Dict[str, int], tfidf_scores: Dict[str, float]) -> Dict:
        """
        Get the most relevant words based on combined frequency and TF-IDF scores.

        Args:
            frequencies: Word frequency dictionary
            tfidf_scores: TF-IDF scores dictionary

        Returns:
            Dict: Dictionary containing selected words with their metrics
        """
        logger.debug(f"Input frequencies count: {len(frequencies)}")
        logger.debug(f"Input TF-IDF scores count: {len(tfidf_scores)}")

        # Get combined relevance scores
        combined_scores = self._combine_word_scores(frequencies, tfidf_scores)

        if not combined_scores:
            logger.warning("No combined scores generated in _get_top_words")
            return {}

        # Create detailed word statistics
        word_stats = {}
        for word in combined_scores:
            if word in frequencies and word in tfidf_scores:
                word_stats[word] = {
                    'frequency': frequencies[word],
                    'tfidf': tfidf_scores[word],
                    'combined_score': combined_scores[word]
                }

        # Sort by combined score and take top words
        sorted_items = sorted(
            word_stats.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:MAX_WORDS_FOR_AI]

        sorted_words = dict(sorted_items)

        logger.debug(f"Output word stats count: {len(sorted_words)}")
        return sorted_words

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
            logger.warning("No terms found for TF-IDF calculation")
            return scores

        for term, freq in frequencies.items():
            tf = freq / total_terms
            df = doc_frequencies.get(term, 1)
            idf = math.log(total_docs / df + 1)
            scores[term] = tf * idf

        if not scores:
            logger.warning("No TF-IDF scores generated")
        else:
            logger.debug(f"Generated {len(scores)} TF-IDF scores")

        return scores

    def generate_wordcloud(self, 
                          scores: Dict[str, float],
                          output_path: Optional[Path] = None) -> str:
        """
        Generate word cloud visualization.
        """
        if not scores:
            logger.error("Empty scores dictionary provided to generate_wordcloud")
            raise TextAnalysisError("No scores provided for word cloud generation")
        
        logger.debug(f"Generating wordcloud with {len(scores)} words")
            
        try:
            if output_path is None:
                timestamp = get_timestamp()
                output_path = OUTPUT_DIR / f'wordcloud_{self.language}_{timestamp}.png'
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert scores to frequency format expected by WordCloud
            if isinstance(next(iter(scores.values())), dict):
                # If scores contain detailed statistics, use combined scores
                wordcloud_scores = {
                    word: stats['combined_score'] 
                    for word, stats in scores.items()
                    if isinstance(stats, dict) and 'combined_score' in stats
                }
            else:
                # If scores are already simple word->value mapping, use as is
                wordcloud_scores = scores
                
            if not wordcloud_scores:
                logger.error("No valid scores after conversion for wordcloud")
                raise TextAnalysisError("No valid scores for word cloud generation")
                
            logger.debug(f"Wordcloud scores count after conversion: {len(wordcloud_scores)}")
            
            # Create word cloud
            wordcloud = WordCloud(
                width=1600,
                height=800,
                background_color='white',
                max_words=MAX_WORDS_WORDCLOUD,
                min_font_size=10,
                prefer_horizontal=0.7,
                collocations=True
            ).generate_from_frequencies(wordcloud_scores)
            
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

    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis with enhanced word selection.

        Args:
            text: Text to analyze

        Returns:
            Dict: Analysis results including:
                - word_frequencies: Dict with word stats (frequency, TF-IDF, combined score)
                - phrases: Extracted meaningful phrases
                - statistics: Various text statistics
                - raw_frequencies: Original word frequencies (for reference)
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
                    if normalized and not self.stopwords_handler.is_stopword(normalized):
                        word_freq[normalized] += 1
                        valid_words.append(normalized)
                        total_length += len(normalized)

            # Sort word frequencies by count (descending) and then alphabetically
            sorted_frequencies = dict(
                sorted(
                    word_freq.items(),
                    key=lambda x: (-x[1], x[0])  # Sort by frequency desc, then word asc
                )
            )

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

            # Get most relevant words using combined scoring
            top_words = self._get_top_words(sorted_frequencies, tfidf_scores)

            # Generate word cloud using combined scores
            wordcloud_path = None
            try:
                wordcloud_path = self.generate_wordcloud(top_words)
            except Exception as e:
                logger.warning(f"Word cloud generation failed: {str(e)}")
                self.stats['warnings'].append(f"Word cloud generation failed: {str(e)}")

            # Combine results with both processed and raw data
            results = {
                'word_frequencies': top_words,  # Contains frequency, TF-IDF, and combined scores
                'phrases': dict(phrases),
                'statistics': self.stats.copy(),
                'raw_frequencies': sorted_frequencies,  # Keep original frequencies for reference
                'metadata': {
                    'language': self.language,
                    'timestamp': get_timestamp(),
                    'text_length': len(text),
                    'sentence_count': len(sentences),
                    'wordcloud_path': wordcloud_path
                }
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
            scores: Word scores (can be frequencies, TF-IDF, or combined scores)
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

            # Convert scores to frequency format expected by WordCloud
            if isinstance(next(iter(scores.values())), dict):
                # If scores contain detailed statistics, use combined scores
                wordcloud_scores = {
                    word: stats['combined_score'] 
                    for word, stats in scores.items()
                }
            else:
                # If scores are already simple word->value mapping, use as is
                wordcloud_scores = scores

            # Create word cloud
            wordcloud = WordCloud(
                width=1600,
                height=800,
                background_color='white',
                max_words=MAX_WORDS_WORDCLOUD,
                min_font_size=10,
                prefer_horizontal=0.7,
                collocations=True
            ).generate_from_frequencies(wordcloud_scores)

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
"""
Enhanced AI handler module with detailed input logging capabilities.
"""
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib
import inspect

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

# Project imports
from config.config import (
    AI_SETTINGS,
    CACHE_DIR,
    ERROR_MESSAGES,
    SYSTEM_PROMPTS,
    detect_language, 
    get_system_prompt, 
    CACHE_DURATION,
    MAX_WORDS_FOR_AI, 
    WORD_SCORE_WEIGHTS, 
    MIN_TFIDF_THRESHOLD
)
from modules.utils import (
    setup_logging,
    safe_file_read,
    safe_file_write,
    get_timestamp
)

# Initialize logger
logger = setup_logging(__name__)

# Add logging configuration
DETAILED_LOGGING = True  # Set to False to disable detailed AI input/output logging
LOG_DIR = Path('logs')
AI_LOG_DIR = LOG_DIR / 'ai_interactions'
AI_LOG_DIR.mkdir(parents=True, exist_ok=True)

class AIError(Exception):
    """Base class for AI-related errors."""
    pass

class AIConnectionError(AIError):
    """Error for AI service connection issues."""
    pass

class AIResponseError(AIError):
    """Error for invalid AI responses."""
    pass

class AILogger:
    """Handles detailed logging of AI interactions."""
    
    def __init__(self, enabled: bool = True):
        """Initialize AI logger.
        
        Args:
            enabled: Whether detailed logging is enabled
        """
        self.enabled = enabled
        self.logger = logging.getLogger('ai_interaction')
        self._setup_logger()
        
    def _setup_logger(self):
        """Configure the logger with appropriate handlers."""
        if not self.enabled:
            return
            
        self.logger.setLevel(logging.DEBUG)
        
        # Create timestamp-based log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = AI_LOG_DIR / f'ai_interaction_{timestamp}.log'
        
        # Add file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s\n'
            'Function: %(funcName)s\n'
            'Message:\n%(message)s\n'
            '-' * 80 + '\n'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
    def _format_data(self, data: Union[Dict, List, str]) -> str:
        """Format data for logging with proper indentation."""
        if isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)
        return str(data)
        
    def log_input_collection(self, metadata: Dict, content: Dict):
        """Log the input collection phase."""
        if not self.enabled:
            return
            
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame)
        
        log_message = f"""
Input Collection (from {caller_info.filename}, line {caller_info.lineno})
{'-' * 40}
Metadata:
{self._format_data(metadata)}

Content:
{self._format_data(content)}
"""
        self.logger.debug(log_message)
        
    def log_prompt_generation(self, prompt: str):
        """Log the generated prompt."""
        if not self.enabled:
            return
            
        log_message = f"""
Generated Prompt:
{'-' * 40}
{prompt}
"""
        self.logger.debug(log_message)
        
    def log_ai_request(self, messages: List[Dict]):
        """Log the actual request sent to the AI."""
        if not self.enabled:
            return
            
        log_message = f"""
AI Request Messages:
{'-' * 40}
{self._format_data(messages)}
"""
        self.logger.debug(log_message)
        
    def log_ai_response(self, response: str, parsed_result: Optional[Dict] = None):
        """Log the AI's response and parsed result."""
        if not self.enabled:
            return
            
        log_message = f"""
AI Raw Response:
{'-' * 40}
{response}

Parsed Result:
{'-' * 40}
{self._format_data(parsed_result) if parsed_result else 'No parsed result available'}
"""
        self.logger.debug(log_message)

class AIHandler:
    """Enhanced AI handler with detailed logging capabilities."""
    
    def __init__(self):
        """Initialize AI handler with configuration settings."""
        try:
            self.client = OpenAI(
                base_url=AI_SETTINGS['base_url'],
                api_key=AI_SETTINGS['api_key']
            )
            self.model = AI_SETTINGS['model']
            self.temperature = AI_SETTINGS['temperature']
            self.max_tokens = AI_SETTINGS['max_tokens']
            
            # Initialize AI logger
            self.ai_logger = AILogger(enabled=DETAILED_LOGGING)
            
            # Ensure cache directory exists
            self.cache_dir = CACHE_DIR / 'ai_responses'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize request tracking
            self.request_stats = {
                'total_requests': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': 0,
                'average_response_time': 0
            }

            # Prompts in the correct languages
            self.FORMAT_PROMPTS = {
                'en': {
                    'metadata_section': """=== METADATA ===
                    Title: {title}
                    Category: {category}
                    Application: {area_of_application}
                    Target Age: {target_age_group}
                    Denomination: {denomination}
                    Bible Passages: {bible_passage}""",

                                    'content_section': """=== CONTENT ===
                    {text}""",

                                    'key_info_section': """=== KEY INFORMATION ===
                    Most Relevant Words (Top {max_words}):
                    {word_frequencies}

                    Document Statistics:
                    Total Words: {processed_words}
                    Unique Words: {unique_words}
                    Sentences: {sentence_count}

                    Main Topics:
                    {suggested_topics}""",

                                    'output_section': """REQUIRED OUTPUT:

                    ANALYSIS:
                    [2-5 sentences identifying the specific document type and subject]

                    TAGS:
                    {{
                        "tags": [
                            {{"index": 1, "tag": "word"}},
                            {{"index": 2, "tag": "another"}},
                            {{"index": 3, "tag": "final"}}
                        ]
                    }}

                    CRITICAL TAG RULES:
                    - Each tag is ONE single word
                    - Include general terms and specific identifiers
                    - Split compound terms into separate tags
                    - {min_keywords} to {max_keywords} tags total
                    - Only lowercase letters
                    - No special characters""",

                    'retry_hint': "\n\nIMPORTANT: Each tag MUST be in the format: {{\"index\": number, \"tag\": \"word\"}}"
                },

                'de': {
                    'metadata_section': """=== METADATEN ===
                    Titel: {title}
                    Kategorie: {category}
                    Anwendungsbereich: {area_of_application}
                    Zielgruppe: {target_age_group}
                    Konfession: {denomination}
                    Bibelstellen: {bible_passage}""",

                    'content_section': """=== INHALT ===
                    {text}""",

                    'key_info_section': """=== SCHLÜSSELINFORMATIONEN ===
                    Relevanteste Wörter (Top {max_words}):
                    {word_frequencies}

                    Dokument-Statistiken:
                    Gesamtanzahl Wörter: {processed_words}
                    Eindeutige Wörter: {unique_words}
                    Sätze: {sentence_count}

                    Hauptthemen:
                    {suggested_topics}""",

                    'output_section': """ERFORDERLICHE AUSGABE:

                    ANALYSE:
                    [2-5 Sätze zur Identifizierung des spezifischen Dokumenttyps und Themas]

                    TAGS:
                    {{
                        "tags": [
                            {{"index": 1, "tag": "wort"}},
                            {{"index": 2, "tag": "weiteres"}},
                            {{"index": 3, "tag": "letztes"}}
                        ]
                    }}

                    KRITISCHE TAG-REGELN:
                    - Jeder Tag ist EIN einzelnes Wort
                    - Enthält allgemeine Begriffe und spezifische Bezeichner
                    - Zusammengesetzte Begriffe in separate Tags aufteilen
                    - {min_keywords} bis {max_keywords} Tags insgesamt
                    - Nur Kleinbuchstaben
                    - Keine Sonderzeichen""",

                    'retry_hint': "\n\nWICHTIG: Jeder Tag MUSS im Format sein: {{\"index\": nummer, \"tag\": \"wort\"}}"
                }
            }
            
        except Exception as e:
            logger.error(f"AI handler initialization failed: {str(e)}")
            raise AIError(f"Initialization failed: {str(e)}")

    def _get_cache_key(self, content: Union[str, Dict]) -> str:
        """
        Generate unique cache key.
        
        Args:
            content: Content to hash
            
        Returns:
            str: Cache key
        """
        try:
            # Convert content to string if it's a dictionary
            if isinstance(content, dict):
                content = json.dumps(content, sort_keys=True)
                
            # Create hash of content
            return hashlib.sha256(
                content.encode('utf-8')
            ).hexdigest()
            
        except Exception as e:
            logger.warning(f"Cache key generation failed: {str(e)}")
            # Fallback to timestamp if hashing fails
            return f"fallback_{get_timestamp()}"

    async def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve cached response if valid.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Optional[Dict]: Cached response or None
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                return None
                
            # Read cache file
            data = json.loads(safe_file_read(cache_file))
            cache_time = datetime.fromisoformat(data['timestamp'])
            
            # Check if cache is still valid
            if datetime.now() - cache_time < timedelta(days=CACHE_DURATION):
                self.request_stats['cache_hits'] += 1
                return data['result']
                
            # Remove expired cache
            cache_file.unlink()
            return None
            
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
            return None

    def _cache_response(self, cache_key: str, result: Dict):
        """
        Cache the response with timestamp.
        
        Args:
            cache_key: Cache key
            result: Response to cache
        """
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'result': result
            }
            cache_file = self.cache_dir / f"{cache_key}.json"
            safe_file_write(cache_data, cache_file)
            
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")

    def _format_stats(self, stats: Dict) -> str:
        """Format statistics for prompt."""
        return "\n".join(
            f"- {key}: {value}"
            for key, value in stats.items()
        )
    
    def _format_word_frequencies(self, word_data: Dict) -> str:
        """
        Format word frequency data for the prompt in a simplified format.
        Only shows word and relevance score, sorted by relevance.
    
        Args:
            word_data: Dictionary containing word statistics
    
        Returns:
            str: Simplified string of word information
        """
        try:
            formatted_entries = []
            
            for word, stats in word_data.items():
                # Get the relevance score (combined_score)
                if isinstance(stats, dict) and 'combined_score' in stats:
                    relevance = stats['combined_score']
                else:
                    continue
                    
                formatted_entries.append((word, relevance))
            
            # Sort by relevance score (descending)
            formatted_entries.sort(key=lambda x: x[1], reverse=True)
            
            # Format as simple comma-separated list
            return ', '.join(
                f"{word} ({relevance:.3f})"
                for word, relevance in formatted_entries
            )
            
        except Exception as e:
            logger.error(f"Error formatting word frequencies: {str(e)}")
            return "Error formatting word data"
    
    def _format_suggested_topics(self, topics: List[str]) -> str:
        """
        Formatiert die vorgeschlagenen Themen für den Prompt.

        Args:
            topics (List[str]): Eine Liste von vorgeschlagenen Themen.

        Returns:
            str: Ein formatierter String der Themen, jeweils vorangestellt mit einem Bindestrich.
                 Falls keine Themen vorhanden sind, wird eine entsprechende Nachricht zurückgegeben.
        """
        if not topics:
            return "Keine vorgeschlagenen Themen verfügbar." #'Diese Themen müssen 1:1 so bleiben! z.B. "manual" und "instruction" anstatt "instruction manual". Hier die Themen: ["manual", "A7iii", "Objektiv", "Kamera", "Sony", "Sony A7", "instruction"]' 

        # Entfernt doppelte Themen und sortiert sie alphabetisch
        unique_topics = sorted(set(topic.strip() for topic in topics if topic.strip()))

        # Formatiert jedes Thema mit einem Bindestrich und einem Leerzeichen
        formatted_topics = "\n".join(f"- {topic}" for topic in unique_topics)

        return formatted_topics


    def _format_prompt(self, metadata: Dict, content: Dict) -> str:
        """
        Format prompt with metadata and enhanced content analysis.
        Now includes improved error handling and proper JSON escaping.
        """
        try:
            # Log input collection
            self.ai_logger.log_input_collection(metadata, content)

            # Detect language from content
            text = content.get('text', '')
            language = detect_language(text)
            
            if language not in self.FORMAT_PROMPTS:
                logger.warning(f"Unsupported language {language}, falling back to English")
                language = 'en'

            # Get language-specific prompt templates
            prompts = self.FORMAT_PROMPTS[language]

            # Format sections with proper error handling
            try:
                metadata_section = prompts['metadata_section'].format(
                    title=metadata.get('title', 'N/A'),
                    category=metadata.get('category', 'N/A'),
                    area_of_application=metadata.get('area_of_application', 'N/A'),
                    target_age_group=metadata.get('target_age_group', 'N/A'),
                    denomination=metadata.get('denomination', 'N/A'),
                    bible_passage=metadata.get('bible_passage', 'N/A')
                )
            except KeyError as e:
                raise AIError(f"Missing metadata field: {str(e)}")

            try:
                content_section = prompts['content_section'].format(
                    text=content.get('text', '')
                )
            except KeyError as e:
                raise AIError(f"Missing content field: {str(e)}")

            # Get analysis results with proper error handling
            analysis = content.get('analysis', {})
            statistics = analysis.get('statistics', {})
            word_frequencies = analysis.get('word_frequencies', {})

            try:
                key_info_section = prompts['key_info_section'].format(
                    max_words=MAX_WORDS_FOR_AI,
                    word_frequencies=self._format_word_frequencies(word_frequencies),
                    processed_words=statistics.get('processed_words', 'N/A'),
                    unique_words=statistics.get('unique_words', 'N/A'),
                    sentence_count=content.get('metadata', {}).get('sentence_count', 'N/A'),
                    suggested_topics=self._format_suggested_topics(content.get('suggested_topics', []))
                )
            except KeyError as e:
                raise AIError(f"Missing analysis field: {str(e)}")

            try:
                output_section = prompts['output_section'].format(
                    min_keywords=AI_SETTINGS['min_keywords'],
                    max_keywords=AI_SETTINGS['max_keywords']
                )
            except KeyError as e:
                raise AIError(f"Missing AI settings: {str(e)}")

            # Combine all sections with proper spacing
            prompt = f"{metadata_section.strip()}\n\n{content_section.strip()}\n\n{key_info_section.strip()}\n\n{output_section.strip()}"

            # Log generated prompt
            self.ai_logger.log_prompt_generation(prompt)

            return prompt.strip()

        except Exception as e:
            logger.error(f"Prompt formatting error: {str(e)}")
            raise AIError(f"Failed to format prompt: {str(e)}")

    def _validate_response(self, response: List[Dict]) -> bool:
        """
        Validate AI response format and content with stricter validation.
    
        Args:
            response: Parsed response to validate
    
        Returns:
            bool: True if valid
        """
        try:
            if not isinstance(response, list):
                logger.error("Response is not a list")
                return False
    
            if not (AI_SETTINGS['min_keywords'] <= 
                   len(response) <= 
                   AI_SETTINGS['max_keywords']):
                logger.error(f"Invalid number of keywords: {len(response)}")
                return False
    
            required_fields = {'tag', 'index'}
            used_indices = set()
    
            for item in response:
                # Check type
                if not isinstance(item, dict):
                    logger.error(f"Invalid item type: {type(item)}")
                    return False
    
                # Check required fields
                if not all(field in item for field in required_fields):
                    logger.error(f"Missing required fields in item: {item}")
                    return False
    
                # Validate index
                if not isinstance(item['index'], int) or item['index'] < 1:
                    logger.error(f"Invalid index in item: {item}")
                    return False
    
                # Check for duplicate indices
                if item['index'] in used_indices:
                    logger.error(f"Duplicate index found: {item['index']}")
                    return False
                used_indices.add(item['index'])
    
                # Validate tag
                if not isinstance(item['tag'], str) or not item['tag'].strip():
                    logger.error(f"Invalid tag in item: {item}")
                    return False
    
            return True
    
        except Exception as e:
            logger.error(f"Response validation error: {str(e)}")
            return False

    @retry(
        stop=stop_after_attempt(AI_SETTINGS['retry_attempts']),
        wait=wait_exponential(multiplier=AI_SETTINGS['retry_delay']),
        retry=retry_if_exception_type(Exception),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )

    async def _make_ai_request(self, messages: List[Dict]) -> str:
        """Make API request with retry logic."""
        start_time = time.time()
        self.request_stats['total_requests'] += 1
        
        # Log AI request
        self.ai_logger.log_ai_request(messages)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Log AI response
            self.ai_logger.log_ai_response(response)
            
            # Update average response time
            elapsed = time.time() - start_time
            prev_avg = self.request_stats['average_response_time']
            prev_count = self.request_stats['total_requests'] - 1
            self.request_stats['average_response_time'] = (
                (prev_avg * prev_count + elapsed) / 
                self.request_stats['total_requests']
            )
            
            return response
            
        except Exception as e:
            self.request_stats['errors'] += 1
            logger.error(f"AI request failed: {str(e)}")
            raise AIConnectionError(ERROR_MESSAGES['ai_connection_error'])

    @retry(
        stop=stop_after_attempt(AI_SETTINGS['retry_attempts']),
        wait=wait_exponential(multiplier=AI_SETTINGS['retry_delay']),
        retry=retry_if_exception_type(Exception),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )

    async def extract_keywords(self, metadata: Dict, content: Dict) -> Dict:
        """
        Extract tags from content using AI analysis with language detection.

        Args:
            metadata: Document metadata
            content: Content and analysis data

        Returns:
            Dict: Extraction results including tags and processing metadata
        """
        start_time = time.time()
        retry_count = 0
        max_retries = AI_SETTINGS['retry_attempts']

        # Detect language from content
        text = content.get('text', '')
        detected_language = detect_language(text)

        while retry_count < max_retries:
            try:
                # Generate cache key
                cache_key = self._get_cache_key({
                    'metadata': metadata,
                    'content': content
                })

                # Check cache
                cached = await self._get_cached_response(cache_key)
                if cached:
                    self.ai_logger.log_ai_response("Using cached response", cached)
                    return cached

                self.request_stats['cache_misses'] += 1

                # Format prompt and make request
                prompt = self._format_prompt(metadata, content)
                if retry_count > 0:
                    # Use language-specific retry hint
                    prompt += self.FORMAT_PROMPTS[detected_language]['retry_hint']

                # Get language-specific system prompt
                system_prompt = get_system_prompt(text)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                response = await self._make_ai_request(messages)

                # Extract JSON part from response
                try:
                    # Find JSON block
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1

                    if json_start == -1:
                        raise AIResponseError("No JSON block found")

                    json_str = response[json_start:json_end]
                    parsed_data = json.loads(json_str)

                    # Handle tag format
                    if 'tags' not in parsed_data:
                        raise AIResponseError("No tags found in response")

                    raw_tags = parsed_data['tags']

                    # Handle array of strings format
                    if isinstance(raw_tags, list) and all(isinstance(x, str) for x in raw_tags):
                        tags = [
                            {
                                'tag': tag.strip(),
                                'index': idx + 1
                            }
                            for idx, tag in enumerate(raw_tags)
                        ]
                    # Handle array of objects format
                    elif isinstance(raw_tags, list) and all(isinstance(x, dict) for x in raw_tags):
                        tags = raw_tags
                    else:
                        raise AIResponseError("Unrecognized tags format")

                    # Validate and clean tags
                    validated_tags = []
                    for tag in tags:
                        if isinstance(tag, dict) and 'tag' in tag:
                            validated_tag = {
                                'tag': str(tag.get('tag', '')).strip(),
                                'index': int(tag.get('index', len(validated_tags) + 1))
                            }
                            if validated_tag['tag']:  # Only add non-empty tags
                                validated_tags.append(validated_tag)

                    if not validated_tags:
                        raise AIResponseError("No valid tags found")

                    # Sort tags by index
                    validated_tags.sort(key=lambda x: x['index'])

                    # Prepare final result with metadata
                    final_result = {
                        'tags': validated_tags,
                        'processing_time': time.time() - start_time,
                        'cached': False,
                        'success': True,
                        'metadata': {
                            'timestamp': get_timestamp(),
                            'model': self.model,
                            'temperature': self.temperature,
                            'detected_language': detected_language  # Add detected language to metadata
                        }
                    }

                    # Log final parsed result
                    self.ai_logger.log_ai_response(response, final_result)

                    # Cache result
                    self._cache_response(cache_key, final_result)

                    return final_result

                except (json.JSONDecodeError, KeyError) as e:
                    raise AIResponseError(f"Invalid JSON response: {str(e)}")

            except Exception as e:
                logger.error(f"Attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(f"All {max_retries} attempts failed")
                    return {
                        'tags': [],
                        'processing_time': time.time() - start_time,
                        'error': str(e),
                        'success': False,
                        'metadata': {
                            'timestamp': get_timestamp(),
                            'error_type': type(e).__name__,
                            'detected_language': detected_language  # Add detected language to metadata
                        }
                    }

                await asyncio.sleep(1)

    def get_stats(self) -> Dict:
        """
        Get handler statistics.
        
        Returns:
            Dict: Usage statistics
        """
        return {
            **self.request_stats,
            'cache_size': len(list(self.cache_dir.glob('*.json')))
        }

async def main():
    """Example usage."""
    try:
        handler = AIHandler()
        
        # Example data
        metadata = {"title": "Test Document"}
        content = {
            "word_frequencies": {"test": 5, "example": 3},
            "statistics": {"total_words": 100}
        }
        
        # Extract keywords
        result = await handler.extract_keywords(metadata, content)
        
        # Print results and stats
        print("Extraction Result:")
        print(json.dumps(result, indent=2))
        print("\nHandler Stats:")
        print(json.dumps(handler.get_stats(), indent=2))
        
    except AIError as e:
        print(f"AI Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

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
    SYSTEM_PROMPT,
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

        Args:
            metadata: Document metadata
            content: Content and analysis data

        Returns:
            str: Formatted prompt
        """
        try:
            # Log input collection
            self.ai_logger.log_input_collection(metadata, content)

            # Get analysis results from content
            analysis = content.get('analysis', {})
            statistics = analysis.get('statistics', {})
            word_frequencies = analysis.get('word_frequencies', {})

            # Format prompt with detailed sections
            prompt = f"""
            # **Aufgabe:**
            Erstelle mindestens {AI_SETTINGS['min_keywords']} bis idealerweise **{AI_SETTINGS['max_keywords']}** aussagekräftige Tags im JSON-Format, die das Dokument präzise beschreiben. Diese Tags sollen das Wesen, Hauptthemen, Namen und -konzepte erfassen und können auch inferierte Begriffe enthalten, die nicht direkt vorkommen, aber es ideal beschreiben.

            ## **Ziel:**
            Erstellen Sie eine Liste von Tags, die für die Suche und das Verknüpfen ähnlicher Inhalte auf unserer Plattform verwendet werden können.

            ## **Vorgehensweise:**
            - Analysieren Sie die bereitgestellten Metadaten, den Inhalt und die Wortstatistiken sorgfältig
            - Beachten Sie besonders Wörter mit hoher TF-IDF-Relevanz
            - Identifizieren Sie Worte die den Kern des Dokuments darstellen
            - Schließen Sie relevante Begriffe ein, die das Thema umfassend darstellen, auch wenn sie nicht explizit im Text erwähnt werden
            - Vermeiden Sie allgemeine Füllwörter und nicht aussagekräftige Begriffe

            ## **ANTWORTEN SIE SO:**
            A. Ausgeschriebene Überlegung und Analyse, worum es sich bei dem Dokument vermutlich handelt und welche Schlagworte als Tags wichtig wären, um das Dokument bestmöglich und spezifisch zu beschreiben.
            B. (CRITICAL OUTPUT FORMAT FOR TAGS!) Die Tags ausschließlich als ein zusammenhängendes intaktes JSON mit fortlaufender Nummerierung, zum Beispiel:
            {{
              "tags": [
                {{"index": 1, "tag": "..."}},
                {{"index": 2, "tag": "..."}}
              ]
            }}
            C. Stoppen Sie nach dem fertigen JSON!

            ## **Eingaben:**

            1. **Metadaten:**
            - **Titel:** {metadata.get('title', 'N/A')}
            - **Beschreibung:** {metadata.get('description', 'N/A')}
            - **Kategorie:** {metadata.get('category', 'N/A')}
            - **Denomination:** {metadata.get('denomination', 'N/A')}
            - **Zielgruppe (Alter):** {metadata.get('target_age_group', 'N/A')}
            - **Anwendungsbereich:** {metadata.get('area_of_application', 'N/A')}
            - **Bibelstellen:** {metadata.get('bible_passage', 'N/A')}
            - **Thumbnail:** {metadata.get('thumbnail', 'N/A')}

            2. **Inhalt:**
            {content.get('text', '')}

            3. **Wortanalyse (Top {MAX_WORDS_FOR_AI} relevante Wörter mit Statistiken):**
            {self._format_word_frequencies(word_frequencies)}

            4. **Allgemeine Statistiken:**
            - Gesamtanzahl der Wörter: {statistics.get('processed_words', 'N/A')}
            - Einzigartige Wörter: {statistics.get('unique_words', 'N/A')}
            - Durchschnittliche Wortlänge: {statistics.get('average_word_length', 'N/A'):.2f}
            - Anzahl der Sätze: {content.get('metadata', {}).get('sentence_count', 'N/A')}

            5. **Vorgeschlagene Themen:**
            {self._format_suggested_topics(content.get('suggested_topics', []))}

            ## **Hinweise:**
            - Die Tags sollen relevant und prägnant sein
            - Die Tags sollen nach Möglichkeit Einzelwörter sein. Anstelle von Leerzeichen nutzen Sie seperate Tags.
            - Nur Worte aus den Sprachen, die im Dokument gesprochen werden, oder Namen
            - Beachten Sie besonders Wörter mit hoher kombinierter Relevanz und TF-IDF-Werten
            """

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
        Extract keywords with enhanced content analysis and stricter response parsing.
        Returns a dictionary with 'tags' instead of 'keywords' for consistency.

        Args:
            metadata: Document metadata
            content: Content and analysis data

        Returns:
            Dict: Extraction results including keywords and processing metadata
        """
        start_time = time.time()

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
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            response = await self._make_ai_request(messages)

            # Extract JSON part from response using stricter parsing
            try:
                # Find JSON block
                json_start = response.find('{\n  "tags":')
                json_end = response.rfind('}') + 1

                if json_start == -1:
                    raise AIResponseError("No JSON tags block found")

                json_str = response[json_start:json_end]

                # Parse JSON
                parsed_data = json.loads(json_str)

                if not isinstance(parsed_data, dict) or 'tags' not in parsed_data:
                    raise AIResponseError("Invalid JSON structure")

                result = parsed_data['tags']

            except (json.JSONDecodeError, KeyError) as e:
                raise AIResponseError(f"Invalid JSON response: {str(e)}")

            if not self._validate_response(result):
                raise AIResponseError("Response validation failed")

            # Sort tags by index
            result.sort(key=lambda x: x['index'])

            # Prepare final result with metadata
            final_result = {
                'tags': result,  # Changed from 'keywords' to 'tags' for consistency
                'processing_time': time.time() - start_time,
                'cached': False,
                'success': True,
                'metadata': {
                    'timestamp': get_timestamp(),
                    'model': self.model,
                    'temperature': self.temperature
                }
            }

            # Log final parsed result
            self.ai_logger.log_ai_response(response, final_result)

            # Cache result
            self._cache_response(cache_key, final_result)

            return final_result

        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return {
                'tags': [],  # Changed from 'keywords' to 'tags' for consistency
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'metadata': {
                    'timestamp': get_timestamp(),
                    'error_type': type(e).__name__
                }
            }

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

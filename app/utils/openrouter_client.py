import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from .logger import setup_logger
from constants.prompts import (
    CLASSIFICATION_SYSTEM_MESSAGE,
    SUMMARIZATION_SYSTEM_MESSAGE,
    CLASSIFICATION_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
    DEFAULT_CLASSIFICATION_TEMPERATURE,
    DEFAULT_SUMMARIZATION_TEMPERATURE,
    DEFAULT_CLASSIFICATION_MAX_TOKENS,
    DEFAULT_SUMMARIZATION_MAX_TOKENS,
    CHUNK_SEPARATOR
)

# Load environment variables from .env file
load_dotenv(find_dotenv())

logger = setup_logger(__name__)

class OpenRouterClient:
    """
    OpenRouter AI client utility for making API calls.
    Can be reused across different parts of the application.
    """
    
    def __init__(self):
        """Initialize OpenRouter client with API key from environment."""
        self.api_key = os.getenv("OPENROUTERAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTERAI_API_KEY not found in environment variables")

        # Clean up API key (remove quotes if present)
        self.api_key = self.api_key.strip('"\'')
        
        # OpenRouter uses OpenAI-compatible API
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Default model - you can change this based on your needs
        self.default_model = "openai/gpt-5"
        
        logger.info("OpenRouter client initialized successfully")
    
    async def classify_document(
        self, 
        text: str, 
        routing_rules: Dict[str, str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify a document using OpenRouter AI.
        
        Args:
            text: Document text to classify
            routing_rules: Dictionary of document types to routing destinations
            model: Model to use (defaults to self.default_model)
        
        Returns:
            Dictionary with classification results
        """
        try:
            model = model or self.default_model
            
            # Create the classification prompt
            categories = ", ".join(routing_rules.keys())
            prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                categories=categories,
                text=text
            )
            
            logger.info(f"Classifying document with model: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=DEFAULT_CLASSIFICATION_TEMPERATURE,
                max_tokens=DEFAULT_CLASSIFICATION_MAX_TOKENS
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            import json
            try:
                # Clean the response text - sometimes there might be extra whitespace or formatting
                cleaned_text = result_text.strip()
                
                # Try to extract JSON if it's embedded in markdown code blocks or other formatting
                if "```json" in cleaned_text:
                    start_idx = cleaned_text.find("```json") + 7
                    end_idx = cleaned_text.find("```", start_idx)
                    if end_idx > start_idx:
                        cleaned_text = cleaned_text[start_idx:end_idx].strip()
                elif "```" in cleaned_text:
                    # Handle cases where JSON is in code blocks without json specifier
                    start_idx = cleaned_text.find("```") + 3
                    end_idx = cleaned_text.rfind("```")
                    if end_idx > start_idx:
                        cleaned_text = cleaned_text[start_idx:end_idx].strip()
                
                result = json.loads(cleaned_text)
                
                # Validate required fields
                if not all(key in result for key in ["category", "confidence", "summary"]):
                    logger.warning(f"Missing required fields in AI response. Got keys: {list(result.keys())}")
                    raise ValueError("Missing required fields in AI response")
                
                # Ensure category exists in routing rules
                if result["category"] not in routing_rules:
                    logger.warning(f"AI returned unknown category: {result['category']}, defaulting to 'Other'")
                    result["category"] = "Other"
                
                # Validate confidence is a number
                if not isinstance(result["confidence"], (int, float)) or not (0 <= result["confidence"] <= 1):
                    logger.warning(f"Invalid confidence value: {result['confidence']}, setting to 0.5")
                    result["confidence"] = 0.5
                
                logger.info(f"Document classified as: {result['category']} (confidence: {result['confidence']})")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {result_text[:500]}...")
                logger.error(f"JSON decode error: {str(e)}")
                # Try to extract useful information even from malformed JSON
                category = "Other"
                confidence = 0.5
                summary = "Classification failed - manual review required"
                
                # Simple pattern matching to extract category if visible
                if '"category"' in result_text:
                    try:
                        import re
                        category_match = re.search(r'"category":\s*"([^"]+)"', result_text)
                        if category_match and category_match.group(1) in routing_rules:
                            category = category_match.group(1)
                            logger.info(f"Extracted category from malformed JSON: {category}")
                    except Exception:
                        pass
                
                return {
                    "category": category,
                    "confidence": confidence,
                    "summary": summary
                }
        
        except Exception as e:
            logger.error(f"Error during document classification: {str(e)}")
            # Fallback response
            return {
                "category": "Other",
                "confidence": 0.0,
                "summary": f"Classification error: {str(e)}"
            }
    
    async def summarize_chunks(self, chunks: list, model: Optional[str] = None) -> str:
        """
        Summarize multiple chunks of text to build context for classification.
        
        Args:
            chunks: List of text chunks
            model: Model to use (defaults to self.default_model)
        
        Returns:
            Combined summary of all chunks
        """
        try:
            model = model or self.default_model
            
            # Combine chunks with separators
            combined_text = CHUNK_SEPARATOR.join(chunks)
            
            logger.debug(f"Combined text length: {len(combined_text)}, first 200 chars: {combined_text[:200]!r}")
            
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
                combined_text=combined_text
            )
            
            logger.info(f"Summarizing {len(chunks)} chunks with model: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARIZATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=DEFAULT_SUMMARIZATION_TEMPERATURE,
                max_tokens=DEFAULT_SUMMARIZATION_MAX_TOKENS
            )
            
            raw_content = response.choices[0].message.content
            logger.debug(f"Raw API response content: {raw_content!r}")
            
            if raw_content is None:
                logger.error("API returned None content for summarization")
                return "Unable to summarize document - API returned empty response"
            
            summary = raw_content.strip()
            logger.debug(f"Cleaned summary: {summary[:500]!r}")
            
            if not summary:
                logger.error("API returned empty content after stripping")
                return "Unable to summarize document - empty response from AI model"
            
            logger.info(f"Successfully summarized document chunks, summary length: {len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error during chunk summarization: {str(e)}")
            return f"Summarization failed: {str(e)}"

# Global client instance - will be initialized lazily
openrouter_client = None

def get_openrouter_client() -> OpenRouterClient:
    """Get or create the OpenRouter client instance."""
    global openrouter_client
    if openrouter_client is None:
        openrouter_client = OpenRouterClient()
    return openrouter_client

import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from .logger import setup_logger

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
        # Try both possible environment variable names
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTERAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENROUTERAI_API_KEY not found in environment variables")
        
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
            prompt = f"""
Analyze the following document text and classify it into one of these categories: {categories}

Document text:
{text}

Based on the content, determine:
1. Which category this document belongs to
2. Your confidence level (0.0 to 1.0)
3. A brief summary of the document (2-3 sentences)

Respond in JSON format with the following structure:
{{
    "category": "exact category name from the list",
    "confidence": confidence_score,
    "summary": "brief summary of the document"
}}
"""
            
            logger.info(f"Classifying document with model: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a document classification expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=4000 # Increased to prevent JSON truncation
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
            combined_text = "\n\n--- CHUNK SEPARATOR ---\n\n".join(chunks)
            
            prompt = f"""
Analyze the following document chunks and provide a comprehensive summary that captures:
1. The document type and purpose
2. Key information and topics covered
3. Important details that would help classify this document

Document chunks:
{combined_text}

Provide a clear, concise summary in 3-4 sentences that would help classify this document.
"""
            
            logger.info(f"Summarizing {len(chunks)} chunks with model: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a document analysis expert. Provide clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Successfully summarized document chunks")
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

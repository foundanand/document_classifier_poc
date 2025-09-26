"""
AI Prompt templates for the document classifier application.

This module contains all prompt templates used for interacting with AI models
for document classification and summarization tasks.
"""

# System messages for different AI tasks
CLASSIFICATION_SYSTEM_MESSAGE = "You are a document classification expert. Always respond with valid JSON."

SUMMARIZATION_SYSTEM_MESSAGE = "You are a document analysis expert. Provide clear, concise summaries."

# Document classification prompt template
CLASSIFICATION_PROMPT_TEMPLATE = """Analyze the following document text and classify it into one of these categories: {categories}

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
}}"""

# Document summarization prompt template
SUMMARIZATION_PROMPT_TEMPLATE = """Analyze the following document chunks and provide a comprehensive summary that captures:
1. The document type and purpose
2. Key information and topics covered
3. Important details that would help classify this document

Document chunks:
{combined_text}

Provide a clear, concise summary in 3-4 sentences that would help classify this document."""

# Model configuration constants
DEFAULT_CLASSIFICATION_TEMPERATURE = 0.1
DEFAULT_SUMMARIZATION_TEMPERATURE = 0.1
DEFAULT_CLASSIFICATION_MAX_TOKENS = 4000
DEFAULT_SUMMARIZATION_MAX_TOKENS = 800

# Chunk separator for combining multiple document chunks
CHUNK_SEPARATOR = "\n\n--- CHUNK SEPARATOR ---\n\n"
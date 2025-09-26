from typing import Dict, Any
from fastapi import UploadFile

from ..utils.logger import setup_logger
from ..utils.pdf_processor import pdf_processor
from ..utils.openrouter_client import get_openrouter_client

logger = setup_logger(__name__)

# Define routing rules
ROUTING_RULES = {
    "Invoice": "Accounts Payable",
    "Purchase Order": "Procurement", 
    "Contract": "Legal",
    "Expense Report": "Finance",
    "Other": "Back Office (Review)"
}

class DocumentClassifier:
    """
    Service for classifying documents using AI and routing them appropriately.
    """
    
    def __init__(self):
        """Initialize the document classifier with routing rules."""
        self.routing_rules = ROUTING_RULES
        logger.info(f"Document classifier initialized with {len(self.routing_rules)} routing rules")
    
    async def classify_document(self, file: UploadFile) -> Dict[str, Any]:
        """
        Classify a document and determine its routing.
        
        Args:
            file: Uploaded file (PDF or text)
        
        Returns:
            Dictionary containing classification results
        """
        try:
            logger.info(f"Starting classification for file: {file.filename}")
            
            # Extract text from the uploaded file
            text = await pdf_processor.extract_text_from_upload(file)
            
            if not text.strip():
                logger.warning("No text extracted from document")
                return {
                    "filename": file.filename,
                    "label": "Other",
                    "confidence": 0.0,
                    "routing": self.routing_rules["Other"],
                    "summary": "No readable text found in document"
                }
            
            # Determine processing approach based on document size
            if pdf_processor.is_large_document(text):
                logger.info("Large document detected, using chunking approach")
                classification_result = await self._classify_large_document(text)
            else:
                logger.info("Small document detected, processing all at once")
                openrouter_client = get_openrouter_client()
                classification_result = await openrouter_client.classify_document(
                    text, self.routing_rules
                )
            
            # Build final response
            result = {
                "filename": file.filename,
                "label": classification_result["category"],
                "confidence": classification_result["confidence"],
                "routing": self.routing_rules[classification_result["category"]],
                "summary": classification_result["summary"]
            }
            
            logger.info(f"Classification completed: {classification_result['category']} "
                       f"(confidence: {classification_result['confidence']})")
            
            return result
        
        except Exception as e:
            logger.error(f"Error during document classification: {str(e)}")
            # Return error result
            return {
                "filename": file.filename,
                "label": "Other",
                "confidence": 0.0,
                "routing": self.routing_rules["Other"],
                "summary": f"Classification failed: {str(e)}"
            }
    
    async def _classify_large_document(self, text: str) -> Dict[str, Any]:
        """
        Classify a large document using chain-of-thought approach.
        
        Args:
            text: Full document text
        
        Returns:
            Classification result dictionary
        """
        try:
            logger.info("Processing large document with chain-of-thought approach")
            
            # Split document into chunks
            chunks = pdf_processor.chunk_text(text)
            
            # For very large documents, summarize chunks first
            if len(chunks) > 10:
                logger.info(f"Document has {len(chunks)} chunks, creating summary first")
                
                # Take representative chunks (first few, middle few, last few)
                representative_chunks = []
                chunk_count = len(chunks)
                
                # First 3 chunks
                representative_chunks.extend(chunks[:3])
                
                # Middle 2 chunks
                mid_start = max(3, chunk_count // 2 - 1)
                mid_end = min(chunk_count - 3, chunk_count // 2 + 1)
                representative_chunks.extend(chunks[mid_start:mid_end])
                
                # Last 3 chunks
                representative_chunks.extend(chunks[-3:])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_chunks = []
                for chunk in representative_chunks:
                    if chunk not in seen:
                        seen.add(chunk)
                        unique_chunks.append(chunk)
                
                logger.info(f"Using {len(unique_chunks)} representative chunks for analysis")
                
                # Get summary of representative chunks
                openrouter_client = get_openrouter_client()
                summary_text = await openrouter_client.summarize_chunks(unique_chunks)
                
                # Classify based on the summary
                classification_result = await openrouter_client.classify_document(
                    summary_text, self.routing_rules
                )
                
            else:
                # For moderately large documents, use first few chunks
                logger.info(f"Using first {min(5, len(chunks))} chunks for classification")
                analysis_chunks = chunks[:5]
                
                # Combine chunks for analysis
                combined_text = "\n\n".join(analysis_chunks)
                
                # If still too large, summarize first
                openrouter_client = get_openrouter_client()
                if len(combined_text) > 8000:
                    summary_text = await openrouter_client.summarize_chunks(analysis_chunks)
                    classification_result = await openrouter_client.classify_document(
                        summary_text, self.routing_rules
                    )
                else:
                    classification_result = await openrouter_client.classify_document(
                        combined_text, self.routing_rules
                    )
            
            logger.info("Large document classification completed successfully")
            return classification_result
            
        except Exception as e:
            logger.error(f"Error in large document classification: {str(e)}")
            return {
                "category": "Other",
                "confidence": 0.0,
                "summary": f"Large document classification failed: {str(e)}"
            }

# Global classifier instance
document_classifier = DocumentClassifier()
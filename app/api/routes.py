from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any

from ..utils.logger import setup_logger
from ..services.classifier import document_classifier

logger = setup_logger(__name__)

router = APIRouter()

@router.post("/classify", response_model=Dict[str, Any])
async def classify_document(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload and classify a document (.txt or .pdf).
    
    Args:
        file: Uploaded document file
    
    Returns:
        Classification results with routing information
    """
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Validate file type
        allowed_types = ["application/pdf", "text/plain"]
        allowed_extensions = [".pdf", ".txt"]
        
        file_valid = (
            file.content_type in allowed_types or
            any(file.filename.lower().endswith(ext) for ext in allowed_extensions)
        )
        
        if not file_valid:
            logger.warning(f"Invalid file type uploaded: {file.content_type}, filename: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF and text files are supported. Please upload a .pdf or .txt file."
            )
        
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if file.size and file.size > max_size:
            logger.warning(f"File too large: {file.size} bytes")
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum file size is 10MB."
            )
        
        logger.info(f"File validation passed for: {file.filename}")
        
        # Classify the document
        result = await document_classifier.classify_document(file)
        
        logger.info(f"Classification completed successfully for: {file.filename}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during classification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during document classification: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "document-classifier"}

@router.get("/routing-rules")
async def get_routing_rules():
    """Get available document types and their routing destinations."""
    return {
        "routing_rules": document_classifier.routing_rules,
        "description": "Available document categories and their routing destinations"
    }
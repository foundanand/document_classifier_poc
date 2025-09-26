import io
from typing import List, Union
import PyPDF2
from fastapi import UploadFile

from .logger import setup_logger

logger = setup_logger(__name__)

class PDFProcessor:
    """
    PDF processing utility for extracting text and chunking documents.
    """
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"PDF processor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF file content.
        
        Args:
            file_content: PDF file content as bytes
        
        Returns:
            Extracted text as string
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            total_pages = len(pdf_reader.pages)
            
            logger.info(f"Extracting text from PDF with {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    logger.debug(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    async def extract_text_from_upload(self, file: UploadFile) -> str:
        """
        Extract text from uploaded file (PDF or text).
        
        Args:
            file: FastAPI UploadFile object
        
        Returns:
            Extracted text as string
        """
        try:
            content = await file.read()
            
            if file.content_type == "application/pdf" or file.filename.lower().endswith('.pdf'):
                logger.info(f"Processing PDF file: {file.filename}")
                return self.extract_text_from_pdf(content)
            
            elif file.content_type == "text/plain" or file.filename.lower().endswith('.txt'):
                logger.info(f"Processing text file: {file.filename}")
                # Try to decode as UTF-8, fallback to other encodings if needed
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = content.decode('latin-1')
                        logger.info("Decoded text file using latin-1 encoding")
                    except UnicodeDecodeError:
                        text = content.decode('utf-8', errors='ignore')
                        logger.warning("Decoded text file with error ignoring")
                
                logger.info(f"Successfully extracted {len(text)} characters from text file")
                return text.strip()
            
            else:
                raise ValueError(f"Unsupported file type: {file.content_type}")
        
        except Exception as e:
            logger.error(f"Error extracting text from upload: {str(e)}")
            raise ValueError(f"Failed to process uploaded file: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            logger.info("Text is small enough, returning as single chunk")
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # If not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence endings in the last 200 characters
                last_part = text[max(0, end - 200):end]
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                
                best_break = -1
                for ending in sentence_endings:
                    pos = last_part.rfind(ending)
                    if pos > best_break:
                        best_break = pos
                
                if best_break > -1:
                    # Adjust end to the sentence break
                    end = max(0, end - 200) + best_break + 2
                else:
                    # Look for word boundaries in the last 50 characters
                    last_part = text[max(0, end - 50):end]
                    space_pos = last_part.rfind(' ')
                    if space_pos > -1:
                        end = max(0, end - 50) + space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def is_large_document(self, text: str, threshold: int = 3000) -> bool:
        """
        Determine if a document is large enough to require chunking.
        
        Args:
            text: Document text
            threshold: Character threshold for considering a document large
        
        Returns:
            True if document is large, False otherwise
        """
        is_large = len(text) > threshold
        logger.info(f"Document size: {len(text)} characters, is_large: {is_large}")
        return is_large

# Global PDF processor instance
pdf_processor = PDFProcessor()
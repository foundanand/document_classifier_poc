from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .utils.logger import setup_logger

logger = setup_logger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title="Document Classifier",
        description="AI-powered document classification and routing service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1", tags=["classification"])
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("Starting Document Classifier service")
        logger.info("Service is ready to classify documents")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("Shutting down Document Classifier service")
    
    @app.get("/")
    async def root():
        """Root endpoint with basic info."""
        return {
            "message": "Document Classifier API",
            "version": "0.1.0",
            "docs_url": "/docs",
            "classify_endpoint": "/api/v1/classify",
            "health_check": "/api/v1/health"
        }
    
    return app

# Create the app instance
app = create_app()
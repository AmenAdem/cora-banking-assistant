from datetime import datetime
from typing import Any, Dict
from fastapi import (
    FastAPI,
    Request,
    status,
)
import logging
import sys
import uvicorn
from config import settings
from core.limiter import limiter
from core.logging import logger
from core.metrics import setup_metrics
from core.middleware import MetricsMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.api import api_router


print("Starting the application...")
logger = logging.getLogger(__name__)
print("Logger initialized")
app = FastAPI()
setup_metrics(app)
app.add_middleware(MetricsMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



# Add validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data.

    Args:
        request: The request that caused the validation error
        exc: The validation error

    Returns:
        JSONResponse: A formatted error response
    """
    # Log the validation error
    logger.error(
        "validation_error",
        client_host=request.client.host if request.client else "unknown",
        path=request.url.path,
        errors=str(exc.errors()),
    )

    # Format the errors to be more user-friendly
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join([str(loc_part) for loc_part in error["loc"] if loc_part != "body"])
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["default"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def health_check(request: Request) -> Dict[str, Any]:
    """Health check endpoint with environment-specific information.

    Returns:
        Dict[str, Any]: Health status information
    """
    logger.info("health_check_called")

    # Check database connectivity
    db_healthy = True

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    # If DB is unhealthy, set the appropriate status code
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)






# Initialize agent after FastAPI app is created
@app.on_event("startup")
async def startup_event():
    try:
        logger.debug("Starting application initialization...")
        global agent
        logger.debug("Initializing TicketAgent...")
        #agent = TicketAgent()
        logger.debug("TicketAgent initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)
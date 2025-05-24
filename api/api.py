"""API v1 router configuration.

This module sets up the main API router and includes all sub-routers for different
endpoints like authentication and chatbot functionality.
"""

from fastapi import APIRouter

from api.route import router as route
from core.logging import logger

api_router = APIRouter()

# Include routers
api_router.include_router(route, prefix="/retrieving", tags=["retrieving"])


"""Health monitoring routes"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def health_status():
    """Get health status"""
    return {"message": "Health status endpoint - to be implemented"}

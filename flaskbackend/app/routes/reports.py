"""Health reports routes"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_reports():
    """Get health reports"""
    return {"message": "Reports endpoint - to be implemented"}

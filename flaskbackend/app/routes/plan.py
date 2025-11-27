"""Meal plan routes"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_meal_plans():
    """Get user meal plans"""
    return {"message": "Meal plans endpoint - to be implemented"}

@router.post("/")
async def create_meal_plan():
    """Create new meal plan"""
    return {"message": "Create meal plan endpoint - to be implemented"}

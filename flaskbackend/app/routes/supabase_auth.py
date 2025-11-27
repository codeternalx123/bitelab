"""Supabase authentication routes"""
from fastapi import APIRouter

router = APIRouter()

@router.post("/supabase/auth/signup")
async def supabase_signup():
    """Supabase signup"""
    return {"message": "Supabase signup endpoint - to be implemented"}

@router.post("/supabase/auth/login")
async def supabase_login():
    """Supabase login"""
    return {"message": "Supabase login endpoint - to be implemented"}

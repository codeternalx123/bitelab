"""M-Pesa payment routes"""
from fastapi import APIRouter

router = APIRouter()

@router.post("/mpesa/stkpush")
async def mpesa_stk_push():
    """Initiate M-Pesa STK push"""
    return {"message": "M-Pesa STK push endpoint - to be implemented"}

@router.post("/mpesa/callback")
async def mpesa_callback():
    """M-Pesa payment callback"""
    return {"message": "M-Pesa callback endpoint - to be implemented"}

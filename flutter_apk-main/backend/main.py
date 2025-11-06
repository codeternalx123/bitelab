from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
from datetime import datetime
import base64
import json

app = FastAPI()

# Pydantic models
class MpesaPaymentRequest(BaseModel):
    phone_number: str
    amount: float
    account_reference: str
    description: str
    callback_url: Optional[str] = None

class MpesaPaymentResponse(BaseModel):
    merchant_request_id: str
    checkout_request_id: str
    response_code: str
    response_description: str
    customer_message: str

class MpesaPaymentResult(BaseModel):
    transaction_id: str
    result_code: str
    result_description: str
    mpesa_receipt_number: Optional[str] = None
    transaction_date: Optional[datetime] = None
    amount: Optional[float] = None
    phone_number: Optional[str] = None

# API endpoints
@app.post("/api/v1/payments/mpesa/initiate", response_model=MpesaPaymentResponse)
async def initiate_mpesa_payment(request: MpesaPaymentRequest):
    try:
        # Here you would implement the actual M-Pesa API integration
        # This is a placeholder response
        return {
            "merchant_request_id": "12345-67890-1",
            "checkout_request_id": "ws_CO_123456789",
            "response_code": "0",
            "response_description": "Success. Request accepted for processing",
            "customer_message": "Success. Request accepted for processing"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/payments/mpesa/status/{checkout_request_id}", response_model=MpesaPaymentResult)
async def check_payment_status(checkout_request_id: str):
    try:
        # Here you would implement the actual status check
        # This is a placeholder response
        return {
            "transaction_id": "1234567890",
            "result_code": "0",
            "result_description": "The service request has been accepted successfully",
            "mpesa_receipt_number": "PGH123456",
            "transaction_date": datetime.now(),
            "amount": 100.00,
            "phone_number": "+254712345678"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/payments/mpesa/callback")
async def mpesa_callback(payload: dict):
    try:
        # Process the callback from M-Pesa
        # Store the transaction details in your database
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
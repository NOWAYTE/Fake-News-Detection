from __future__ import annotations

import os
from typing import Optional, Any, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import modal

load_dotenv()

app = FastAPI(title="Fake News Detection API")

# --- CORS setup ---
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modal app reference ---
_modal_app = None
_modal_text_func = None
_modal_url_func = None

def get_modal_functions():
    global _modal_text_func, _modal_url_func
    if _modal_text_func and _modal_url_func:
        return _modal_text_func, _modal_url_func

    app_name = os.getenv("MODAL_APP_NAME")
    text_func_name = os.getenv("MODAL_TEXT_FUNCTION_NAME", "infer_from_text")
    url_func_name = os.getenv("MODAL_URL_FUNCTION_NAME", "infer_from_url")
    if not app_name:
        raise RuntimeError("Missing MODAL_APP_NAME in .env")

    _modal_text_func = modal.Function.from_name(app_name, text_func_name)
    _modal_url_func = modal.Function.from_name(app_name, url_func_name)
    return _modal_text_func, _modal_url_func

# --- Request / Response models ---
class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[HttpUrl] = None

class Scores(BaseModel):
    real: float
    fake: float

class AnalyzeResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    tweet_url: Optional[str] = None
    tweet_id: Optional[str] = None
    created_at: Optional[str] = None

# --- Utilities ---
def normalize_response(data: Dict[str, Any]) -> AnalyzeResponse:
    # Extract basic fields
    text = data.get("text", "")
    prediction = data.get("prediction", "unknown")
    confidence = data.get("confidence", 0.5)
    
    # Extract tweet metadata if available
    tweet_url = data.get("tweet_url")
    tweet_id = data.get("tweet_id")
    created_at = data.get("created_at")
    
    # Ensure confidence is float
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.5
    
    return AnalyzeResponse(
        text=text,
        prediction=prediction,
        confidence=confidence,
        tweet_url=tweet_url,
        tweet_id=tweet_id,
        created_at=created_at
    )

# --- Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    text_func, url_func = get_modal_functions()

    # Decide which Modal function to call
    if body.url:
        func = url_func
        input_value = str(body.url)
    elif body.text:
        func = text_func
        input_value = body.text
    else:
        raise HTTPException(status_code=400, detail="Request must include 'text' or 'url'")

    try:
        # Call Modal function (use .remote() for async, .call() for sync)
        result = func.remote(input_value)
        
        if not isinstance(result, dict):
            raise ValueError("Modal function returned non-dict payload")
        
        return normalize_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Fake News Detection API is running"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Fake News Detection API", "version": "1.0"}

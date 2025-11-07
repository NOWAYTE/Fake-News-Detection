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

# --- Modal lazy loader ---
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

    _modal_text_func = modal.Function.lookup(app_name, text_func_name)
    _modal_url_func = modal.Function.lookup(app_name, url_func_name)
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
    author: Optional[str] = None
    created_at: Optional[str] = None
    scores: Scores

# --- Utilities ---
def normalize_scores(obj: Dict[str, Any]) -> Scores:
    real = obj.get("real")
    fake = obj.get("fake")
    if real is None and fake is None:
        p = float(obj.get("prob", obj.get("p", 0.5)))
        fake = p
        real = 1 - p
    elif real is None:
        fake = float(fake)
        real = 1 - fake
    elif fake is None:
        real = float(real)
        fake = 1 - real
    real_f = max(0.0, min(1.0, float(real)))
    fake_f = max(0.0, min(1.0, float(fake)))
    return Scores(real=real_f, fake=fake_f)

def normalize_response(data: Dict[str, Any]) -> AnalyzeResponse:
    text = str(data.get("text") or data.get("content") or "").strip()
    if not text:
        raise ValueError("Modal response missing 'text' or 'content'")

    author = data.get("author") or data.get("handle")
    created_at = data.get("created_at") or data.get("timestamp")
    if isinstance(created_at, (int, float)):
        try:
            created_at = datetime.utcfromtimestamp(float(created_at)).isoformat() + "Z"
        except Exception:
            created_at = None

    scores_raw = data.get("scores") or data.get("score") or {}
    if not isinstance(scores_raw, dict):
        scores_raw = {}
    scores = normalize_scores(scores_raw)

    return AnalyzeResponse(text=text, author=author, created_at=created_at, scores=scores)

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
        # Synchronous call to Modal function
        result = func.call(input_value)
        if not isinstance(result, dict):
            raise ValueError("Modal function returned non-dict payload")
        return normalize_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

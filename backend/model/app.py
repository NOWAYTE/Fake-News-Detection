import os
from typing import Dict

import modal
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Modal app name
app = modal.App("fake-news-inference")

# Build a lightweight image with needed deps (CPU for simplicity)
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "torch==2.3.1",
            "transformers==4.44.2",
            "tweepy==4.14.0",
            "python-dotenv==1.0.1",
            "numpy==1.26.4",
        ]
    )
)

# Mount the local model directory into the container at /model
# Local path: backend/model/model -> remote: /model
model_mount = modal.Mount.from_local_dir(
    local_path=os.path.join(os.path.dirname(__file__), "model"),
    remote_path="/model",
)

# Paths inside the container
MODEL_PATH = "/model/distilbert_model"
TOKENIZER_PATH = "/model/tokenizer"

# Lazy-loaded globals within the container
model = None
tokenizer = None

def ensure_loaded():
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()


def softmax_probs(logits: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # Assume binary: index 0 = Fake, index 1 = Real
    return {"fake": float(probs[0]), "real": float(probs[1])}


@app.function(image=image, mounts=[model_mount], timeout=120)
def infer_from_text(text: str):
    ensure_loaded()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax_probs(logits)
    # Return normalized payload for backend/UI
    return {
        "text": text,
        "scores": {"real": round(probs["real"], 4), "fake": round(probs["fake"], 4)},
    }


# Optional: fetch tweet text by URL using Tweepy v2 Client (Bearer Token)
@app.function(image=image, mounts=[model_mount], timeout=120)
def infer_from_url(url: str):
    import tweepy  # imported inside function to keep cold start lean

    # Read bearer token; support both BEARER_TOKEN and (legacy) BEARER_TOKE
    bearer = os.getenv("BEARER_TOKEN") or os.getenv("BEARER_TOKE")
    if not bearer:
        raise RuntimeError("Missing BEARER_TOKEN environment variable for Tweepy Client")

    # Extract tweet ID from URL
    tweet_id = url.rstrip("/").split("/")[-1]

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
    resp = client.get_tweet(id=tweet_id, tweet_fields=["created_at"])  # text included by default
    if not resp or not resp.data:
        raise RuntimeError("Failed to fetch tweet")

    text = getattr(resp.data, "text", None)
    if not text:
        raise RuntimeError("Tweet has no text field")

    return infer_from_text.remote(text)

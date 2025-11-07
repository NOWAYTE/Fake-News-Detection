import os
from typing import Dict
import modal

# Modal app name
app = modal.App("fake-news-inference")

# Build a lightweight image with needed deps (CPU for simplicity)
image = (
    modal.Image
    .from_registry("nvidia/cuda:12.1.1-devel-ubuntu20.04", add_python=True)
    .add_local_dir(
        local_path=os.path.join(os.path.dirname(__file__), "model"),
        remote_path="/model",
    )
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


secrets = [modal.Secret.from_name("x-creds")]

# Paths inside the container
MODEL_PATH = "/model/distilbert_model"
TOKENIZER_PATH = "/model/tokenizer"

# Lazy-loaded globals within the container
model = None
tokenizer = None
def ensure_loaded():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch, os

    global model, tokenizer

    if model is None or tokenizer is None:
        print(">>> Loading tokenizer from", TOKENIZER_PATH)
        print(">>> Loading model from", MODEL_PATH)
        print(">>> Files in /model:", os.listdir("/model"))
        print(">>> Files in /model/model:", os.listdir("/model/model") if os.path.exists("/model/model") else "none")

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print(">>> Tokenizer loaded:", tokenizer.__class__.__name__)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print(">>> Model loaded:", model.__class__.__name__)

        model.eval()
        print(">>> Model set to eval mode")


def softmax_probs(logits) -> Dict[str, float]:
    import torch
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    # Assume binary: index 0 = Fake, index 1 = Real
    return {"fake": float(probs[0]), "real": float(probs[1])}


@app.function(image=image, secrets=secrets, timeout=120)
def infer_from_text(text: str):
    import torch
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
    return {
        "text": text,
        "scores": {"real": round(probs["real"], 4), "fake": round(probs["fake"], 4)},
    }


# Optional: fetch tweet text by URL using Tweepy v2 Client (Bearer Token)
@app.function(image=image, secrets=secrets, timeout=120)
def infer_from_url(url: str):
    import tweepy  # imported inside function to keep cold start lean

    # Read bearer token; support both BEARER_TOKEN and (legacy) BEARER_TOKE
    bearer = os.getenv("BEARER_TOKEN") or os.getenv("BEARER_TOKEN")
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

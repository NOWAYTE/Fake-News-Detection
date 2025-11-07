import os
import modal

# Modal app
app = modal.App("fake-news-detector")

# Create a volume for model storage
model_volume = modal.Volume.from_name("model-store", create_if_missing=True)

# Build image
image = (
    modal.Image
    .from_registry("python:3.11-slim", add_python=False)
    .pip_install([
        "torch==2.3.1",
        "transformers==4.44.2",
        "tweepy==4.14.0",
        "numpy==1.26.4",
    ])
    .add_local_dir(
        local_path=os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/root",
    )
)

# Paths
VOLUME_PATH = "/model_volume"
MODEL_PATH = f"{VOLUME_PATH}/distilbert_model"
TOKENIZER_PATH = f"{VOLUME_PATH}/tokenizer"
LOCAL_MODEL_PATH = "/root/model/model/distilbert_model"
LOCAL_TOKENIZER_PATH = "/root/model/model/tokenizer"

# Import inside container context
with image.imports():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

# Define functions first, then decorate them
def load_model():
    """Load model from volume"""
    import os
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = None
    tokenizer = None

    if model is None or tokenizer is None:
        print("🔄 Loading model from volume...")

        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found in volume at {MODEL_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print("✅ Model loaded from volume!")

    return model, tokenizer

# Define the function separately
def _infer_from_text(text: str):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        model, tokenizer = load_model()

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
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        # Get prediction and confidence
        real_prob = probs[1]  # Assuming index 1 is "real"
        fake_prob = probs[0]  # Assuming index 0 is "fake"

        prediction = "real" if real_prob > fake_prob else "fake"
        confidence = real_prob if prediction == "real" else fake_prob

        result = {
            "text": text,
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }

        print("🎯 RESULT:", result)
        return result

    except Exception as e:
        error_result = {"error": str(e), "text": text}
        print("❌ ERROR:", error_result)
        return error_result

def _infer_from_url(tweet_url: str):  # Fixed: Added this function definition
    try:
        import tweepy
        import re

        bearer_token = os.getenv("BEARER_TOKEN")
        if not bearer_token:
            raise RuntimeError("BEARER_TOKEN not found")

        tweet_id_match = re.search(r'/status/(\d+)', tweet_url)
        if not tweet_id_match:
            raise ValueError(f"Invalid tweet URL: {tweet_url}")

        tweet_id = tweet_id_match.group(1)

        client = tweepy.Client(bearer_token=bearer_token)
        tweet = client.get_tweet(tweet_id, tweet_fields=["text", "created_at"])

        if not tweet or not tweet.data:
            raise RuntimeError("Failed to fetch tweet")

        # Get the tweet text
        tweet_text = tweet.data.text
        print(f"📱 Fetched tweet: {tweet_text}")

        # Call infer_from_text using .remote()
        result = infer_from_text.remote(tweet_text)

        # Add tweet metadata to the result
        result["tweet_url"] = tweet_url
        result["tweet_id"] = tweet_id
        if hasattr(tweet.data, 'created_at'):
            result["created_at"] = tweet.data.created_at.isoformat()

        print("🎯 TWEET RESULT:", result)
        return result

    except Exception as e:
        error_result = {"error": str(e), "url": tweet_url}
        print("❌ TWEET ERROR:", error_result)
        return error_result

def _setup_volume():
    """Copy local model files to the volume (run this once)"""
    import shutil
    import os

    print("📦 Setting up model volume...")

    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print("✅ Volume already populated")
        return {"status": "already_setup"}

    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"📁 Copying model from {LOCAL_MODEL_PATH} to {MODEL_PATH}")
        shutil.copytree(LOCAL_MODEL_PATH, MODEL_PATH)
    else:
        raise RuntimeError(f"Local model not found at {LOCAL_MODEL_PATH}")

    if os.path.exists(LOCAL_TOKENIZER_PATH):
        print(f"📁 Copying tokenizer from {LOCAL_TOKENIZER_PATH} to {TOKENIZER_PATH}")
        shutil.copytree(LOCAL_TOKENIZER_PATH, TOKENIZER_PATH)
    else:
        raise RuntimeError(f"Local tokenizer not found at {LOCAL_TOKENIZER_PATH}")

    print("✅ Volume setup complete!")
    return {"status": "volume_ready"}

def _health_check():
    return {"status": "healthy", "message": "App is running"}

# Now apply the decorators
infer_from_text = app.function(
    image=image,
    volumes={VOLUME_PATH: model_volume},
    secrets=[modal.Secret.from_name("x-creds")],
    scaledown_window=300,
    timeout=120,
)(_infer_from_text)

infer_from_url = app.function(
    image=image,
    volumes={VOLUME_PATH: model_volume},
    secrets=[modal.Secret.from_name("x-creds")],
    scaledown_window=300,
    timeout=120,
)(_infer_from_url)

setup_volume = app.function(
    image=image,
    volumes={VOLUME_PATH: model_volume},
)(_setup_volume)

health_check = app.function(image=image)(_health_check)

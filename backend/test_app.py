import modal

app = modal.App("test-fake-news")

image = modal.Image.debian_slim().pip_install(["transformers", "torch"])

@app.function(image=image)
def infer_from_text(text: str):
    return {"text": text, "result": "test successful"}

@app.function(image=image) 
def infer_from_url(url: str):
    return {"url": url, "result": "url test successful"}

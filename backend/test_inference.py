import modal

# Import your app
from model.app import app

def test_inference():
    # This will run remotely and show the output
    result = app.functions["infer_from_text"].remote("This is a test message about politics")
    print("Result:", result)

if __name__ == "__main__":
    test_inference()

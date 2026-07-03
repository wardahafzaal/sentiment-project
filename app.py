from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 200

app = FastAPI(title="Sentiment Classification API")

print("Loading model...")
model = tf.keras.models.load_model("model_v2.keras")

print("Loading word index...")
word_index = imdb.get_word_index()


def encode_review(text: str):
    words = text.lower().split()
    encoded = [1]  # 1 = start-of-sequence token, matches IMDB dataset convention
    for word in words:
        idx = word_index.get(word)
        if idx is not None and idx < VOCAB_SIZE:
            encoded.append(idx + 3)  # IMDB dataset reserves indices 0-3 for special tokens
        else:
            encoded.append(2)  # 2 = unknown word token
    return pad_sequences([encoded], maxlen=MAX_LEN)


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Sentiment Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: ReviewRequest):
    encoded = encode_review(request.text)
    prediction = model.predict(encoded)[0][0]
    sentiment = "positive" if prediction >= 0.5 else "negative"
    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": float(prediction)
    }

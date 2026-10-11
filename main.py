
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Batch Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()


class SentimentRequest(BaseModel):
    sentences: list[str]


class SentimentResult(BaseModel):
    sentence: str
    sentiment: str


class SentimentResponse(BaseModel):
    results: list[SentimentResult]


def analyze_sentiment(sentence: str) -> str:
    scores = analyzer.polarity_scores(sentence)
    compound = scores["compound"]

    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
        return "sad"
    return "neutral"


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(request: SentimentRequest):
    return {
        "results": [
            {
                "sentence": sentence,
                "sentiment": analyze_sentiment(sentence),
            }
            for sentence in request.sentences
        ]
    }

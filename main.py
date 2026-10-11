
from fastapi import FastAPI
from pydantic import BaseModel, Field
import re

app = FastAPI(title="Batch Sentiment Analysis API")


class SentimentRequest(BaseModel):
    sentences: list[str] = Field(...)


class SentimentResult(BaseModel):
    sentence: str
    sentiment: str


class SentimentResponse(BaseModel):
    results: list[SentimentResult]


POSITIVE_WORDS = {
    "love", "loved", "lovely", "great", "excellent", "amazing",
    "awesome", "good", "happy", "wonderful", "fantastic",
    "best", "enjoy", "enjoyed", "like", "liked", "perfect",
    "beautiful", "brilliant", "delighted", "pleased", "success",
    "successful", "helpful", "recommend", "positive", "excited",
    "impressive", "satisfied", "thankful", "grateful", "win",
    "winning", "superb", "outstanding", "fun", "hopeful",
    "comfortable", "easy", "improved", "improvement"
}

NEGATIVE_WORDS = {
    "hate", "hated", "terrible", "awful", "bad", "horrible",
    "sad", "angry", "upset", "disappointed", "disappointing",
    "worst", "poor", "fail", "failed", "failure", "problem",
    "problems", "difficult", "hard", "annoying", "annoyed",
    "boring", "useless", "broken", "wrong", "painful",
    "unhappy", "unfortunately", "regret", "frustrated",
    "frustrating", "fear", "scared", "worry", "worried",
    "stress", "stressed", "expensive", "slow", "ugly",
    "negative", "disgusting", "sucks", "dislike", "disliked",
    "hurt", "hurts", "cry", "crying", "lonely", "miserable"
}

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "hardly",
    "barely", "isn't", "wasn't", "aren't", "weren't",
    "don't", "doesn't", "didn't", "can't", "couldn't",
    "won't", "wouldn't", "shouldn't", "cannot"
}

INTENSIFIERS = {
    "very", "really", "extremely", "absolutely", "incredibly",
    "so", "too", "super"
}


def analyze_sentiment(sentence: str) -> str:
    text = sentence.lower().replace("’", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    positive_score = 0
    negative_score = 0
    negate_next = False

    # Handle common multiword expressions.
    if any(phrase in text for phrase in (
        "looking forward to", "can't wait", "cannot wait",
        "well done", "thank you", "thanks for"
    )):
        positive_score += 2

    if any(phrase in text for phrase in (
        "not bad", "not terrible", "not disappointed"
    )):
        positive_score += 2

    if any(phrase in text for phrase in (
        "not good", "not great", "not happy", "not impressed",
        "not satisfied", "not worth", "no good", "not working"
    )):
        negative_score += 2

    for i, word in enumerate(words):
        if word in NEGATIONS:
            negate_next = True
            continue

        # Detect sentiment words, including simple inflections.
        positive = word in POSITIVE_WORDS
        negative = word in NEGATIVE_WORDS

        if not positive and word.endswith("ly"):
            positive = word[:-2] in POSITIVE_WORDS
            negative = word[:-2] in NEGATIVE_WORDS

        if not positive and not negative:
            if word.endswith("ing"):
                positive = word[:-3] in POSITIVE_WORDS
                negative = word[:-3] in NEGATIVE_WORDS
            elif word.endswith("ed"):
                positive = word[:-2] in POSITIVE_WORDS
                negative = word[:-2] in NEGATIVE_WORDS

        # Intensifiers increase the following sentiment word's weight.
        weight = 2 if i > 0 and words[i - 1] in INTENSIFIERS else 1

        # A negation reverses the next sentiment word.
        if negate_next:
            positive, negative = negative, positive
            negate_next = False

        if positive:
            positive_score += weight
        if negative:
            negative_score += weight

    # Account for explicit punctuation and common emoticons.
    if re.search(r"[:;]-?\)", text) or "😊" in sentence or "❤️" in sentence:
        positive_score += 1
    if re.search(r"[:;]-?\(", text) or "😢" in sentence or "😭" in sentence:
        negative_score += 1

    if positive_score > negative_score:
        return "happy"
    if negative_score > positive_score:
        return "sad"
    return "neutral"


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(request: SentimentRequest):
    return {
        "results": [
            {
                "sentence": sentence,
                "sentiment": analyze_sentiment(sentence)
            }
            for sentence in request.sentences
        ]
    }

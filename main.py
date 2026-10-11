from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI(title="Batch Sentiment Analysis API")

# Allow requests from the assignment evaluator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SentimentRequest(BaseModel):
    sentences: list[str]


class SentimentResult(BaseModel):
    sentence: str
    sentiment: str


class SentimentResponse(BaseModel):
    results: list[SentimentResult]


POSITIVE_WORDS = {
    "love", "loved", "lovely", "great", "excellent", "amazing",
    "awesome", "good", "happy", "wonderful", "fantastic", "best",
    "enjoy", "enjoyed", "like", "liked", "perfect", "beautiful",
    "brilliant", "delighted", "pleased", "success", "successful",
    "helpful", "recommend", "positive", "excited", "impressive",
    "satisfied", "thankful", "grateful", "win", "winning", "superb",
    "outstanding", "fun", "hopeful", "comfortable", "easy",
    "improved", "improvement", "glad", "nice", "pleasant", "peaceful",
    "delicious", "favorite", "favourite", "joy", "joyful", "proud",
    "kind", "generous", "calm", "incredible", "enjoyable"
}

NEGATIVE_WORDS = {
    "hate", "hated", "terrible", "awful", "bad", "horrible",
    "sad", "angry", "upset", "disappointed", "disappointing",
    "worst", "poor", "fail", "failed", "failure", "problem",
    "problems", "difficult", "annoying", "annoyed", "boring",
    "useless", "broken", "wrong", "painful", "unhappy",
    "unfortunately", "regret", "frustrated", "frustrating",
    "fear", "scared", "worry", "worried", "stress", "stressed",
    "expensive", "slow", "ugly", "negative", "disgusting",
    "sucks", "dislike", "disliked", "hurt", "hurts", "cry",
    "crying", "lonely", "miserable", "horrendous", "annoyance",
    "disaster", "disastrous", "horrible", "dreadful", "weak",
    "worried", "anxious", "nervous", "irritated", "irritating",
    "unpleasant", "uncomfortable", "reject", "rejected",
    "loss", "lose", "lost", "damage", "damaged", "regretful"
}

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "hardly", "barely",
    "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't",
    "didn't", "can't", "couldn't", "won't", "wouldn't",
    "shouldn't", "cannot", "ain't"
}

INTENSIFIERS = {
    "very", "really", "extremely", "absolutely", "incredibly",
    "so", "too", "super", "quite", "particularly"
}

POSITIVE_PHRASES = [
    "looking forward to", "can't wait", "cannot wait",
    "well done", "thank you", "thanks for", "good job",
    "happy with", "love it", "works perfectly", "highly recommend"
]

NEGATIVE_PHRASES = [
    "not good", "not great", "not happy", "not impressed",
    "not satisfied", "not worth", "no good", "not working",
    "fed up", "waste of time", "worst experience", "very disappointed",
    "doesn't work", "does not work", "not worth it"
]

POSITIVE_EMOJIS = ["😊", "😄", "😁", "😍", "🥰", "❤️", "❤", "👍", "🎉"]
NEGATIVE_EMOJIS = ["😢", "😭", "☹", "🙁", "😞", "😡", "😠", "👎", "💔"]


def analyze_sentiment(sentence: str) -> str:
    text = sentence.lower().replace("’", "'")

    # Count sentiment-bearing words.
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    positive_score = 0
    negative_score = 0

    # Handle common phrases first.
    for phrase in POSITIVE_PHRASES:
        if phrase in text:
            positive_score += 2

    for phrase in NEGATIVE_PHRASES:
        if phrase in text:
            negative_score += 2

    # Avoid treating "not bad" as strongly negative.
    if "not bad" in text or "not terrible" in text:
        positive_score += 2

    # Handle "no problem" and similar expressions.
    if "no problem" in text or "no worries" in text:
        positive_score += 2

    # Score individual words, accounting for nearby negation.
    for i, word in enumerate(words):
        if word in NEGATIONS:
            continue

        positive = word in POSITIVE_WORDS
        negative = word in NEGATIVE_WORDS

        # Handle simple suffix variations.
        if not positive and not negative:
            candidates = []
            if word.endswith("ing"):
                candidates.append(word[:-3])
            if word.endswith("ed"):
                candidates.append(word[:-2])
            if word.endswith("ly"):
                candidates.append(word[:-2])
            if word.endswith("s") and len(word) > 3:
                candidates.append(word[:-1])

            for candidate in candidates:
                if candidate in POSITIVE_WORDS:
                    positive = True
                    break
                if candidate in NEGATIVE_WORDS:
                    negative = True
                    break

        # Look for negation in the preceding three words.
        previous_words = words[max(0, i - 3):i]
        negated = any(w in NEGATIONS for w in previous_words)

        # Ignore the word "not" in the phrase "not only".
        if "not" in previous_words and i > 0:
            prior_text = " ".join(words[max(0, i - 3):i + 1])
            if "not only" in prior_text:
                negated = False

        weight = 1
        if i > 0 and words[i - 1] in INTENSIFIERS:
            weight = 2

        if negated:
            positive, negative = negative, positive

        if positive:
            positive_score += weight
        if negative:
            negative_score += weight

    # Add emoji signals.
    for emoji in POSITIVE_EMOJIS:
        if emoji in sentence:
            positive_score += 1

    for emoji in NEGATIVE_EMOJIS:
        if emoji in sentence:
            negative_score += 1

    # Account for emphatic punctuation only when a sentiment is present.
    if "!" in sentence:
        if positive_score > negative_score:
            positive_score += 1
        elif negative_score > positive_score:
            negative_score += 1

    if positive_score > negative_score:
        return "happy"

    if negative_score > positive_score:
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
    return SentimentResponse(
        results=[
            SentimentResult(
                sentence=sentence,
                sentiment=analyze_sentiment(sentence)
            )
            for sentence in request.sentences
        ]
    )


\
from dataclasses import dataclass
from typing import Dict
from utils.llm_utils import build_sentiment_reply_chain, parse_sentiment_reply

@dataclass
class FeedbackResponse:
    sentiment: str
    reply: str

class FeedbackResponseAgent:
    """
    Agent 1: Accepts a single review string, returns sentiment + automated reply.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.chain = build_sentiment_reply_chain(model=model)

    def generate_reply(self, feedback_text: str) -> FeedbackResponse:
        raw = self.chain.invoke({"feedback": feedback_text})
        sentiment, reply = parse_sentiment_reply(raw)
        # Normalize any edge cases like "mixed" => "neutral"
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        return FeedbackResponse(sentiment=sentiment, reply=reply)

    def to_dict(self, response: FeedbackResponse) -> Dict[str, str]:
        return {"sentiment": response.sentiment, "reply": response.reply}

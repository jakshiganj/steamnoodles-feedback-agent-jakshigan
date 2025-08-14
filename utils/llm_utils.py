\
import os
from typing import Literal, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# Unified model schema for structured outputs
class SentimentReply(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the customer feedback"
    )
    reply: str = Field(description="A short, polite, context-aware reply")


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """
    Returns a ChatOpenAI LLM instance for LangChain.
    Set OPENAI_API_KEY in your environment or .env file.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def build_sentiment_reply_chain(model: str = "gpt-4o-mini"):
    """
    Builds a lightweight chain that takes feedback text and returns JSON with
    {sentiment, reply}, where sentiment ∈ {positive, negative, neutral}.
    """
    system = (
        "You are SteamNoodles' customer care assistant. "
        "Given a single restaurant review, do two things:\n"
        "1) Decide sentiment as one of: positive, negative, neutral.\n"
        "2) Produce a short, warm, professional reply that acknowledges specifics and sets the right tone.\n"
        "Keep replies ~1-2 sentences. Don't fabricate details."
    )
    human = (
        "Review:\n\n"
        "{feedback}\n\n"
        "Respond ONLY as compact JSON with keys 'sentiment' and 'reply'. "
        "Example: {{\"sentiment\":\"positive\",\"reply\":\"Thank you ...\"}}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    llm = get_llm(model=model, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    return chain


def parse_sentiment_reply(json_like: str) -> Tuple[str, str]:
    """
    Forgiving JSON parser – extracts 'sentiment' and 'reply'.
    Falls back to neutral + generic reply if parsing fails.
    """
    import json, re

    # Try strict JSON first
    try:
        data = json.loads(json_like)
        sentiment = str(data.get("sentiment", "neutral")).lower().strip()
        reply = str(data.get("reply", "")).strip()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        if not reply:
            reply = "Thank you for your feedback. We appreciate your time and will use your input to improve."
        return sentiment, reply
    except Exception:
        pass

    # Try to salvage with regex
    try:
        s_match = re.search(r'"sentiment"\s*:\s*"(?P<s>[^"]+)"', json_like, re.I)
        r_match = re.search(r'"reply"\s*:\s*"(?P<r>.*?)"\s*}?$', json_like, re.I | re.S)
        sentiment = (s_match.group("s").lower().strip() if s_match else "neutral")
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        reply = (r_match.group("r").strip() if r_match else "")
        if not reply:
            reply = "Thank you for your feedback. We appreciate your time and will use your input to improve."
        return sentiment, reply
    except Exception:
        return "neutral", "Thank you for your feedback. We appreciate your time and will use your input to improve."

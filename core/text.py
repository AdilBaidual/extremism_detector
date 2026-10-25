import re

URL_OR_MENTION = re.compile(r"https?://\S+|www\.\S+|@\w+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def clean_text(text: str) -> str:
    return " ".join(URL_OR_MENTION.sub(" ", text).split())


def split_sentences(text: str) -> list[str]:
    parts = (part.strip() for part in SENTENCE_BOUNDARY.split(text))
    return [part for part in parts if part]

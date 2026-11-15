"""DecodeLabs Industrial Training Kit - Batch 2026.

Project 1: Rule-Based AI Chatbot - Version 1 (Deterministic Fast Engine)
Features:
- Dynamic formatted terminal interface
- Streaming response effect for smooth interaction
- Unicode NFKC normalization + contraction expansion
- Word-boundary regex matching
- External intents.json with seedable variant selection
- Built-in commands: /help, /reset, /history, /save, /exit
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
import unicodedata
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BOT_NAME = "DBot"
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "stop", "/exit", "/quit"}
BASE_DIR = Path(__file__).resolve().parent
INTENTS_FILE = BASE_DIR / "intents.json"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
UNKNOWN_LOG = OUTPUTS_DIR / "unknown_queries.log"

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

CONTRACTIONS = {
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "i'm": "i am",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "it's": "it is",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "shouldn't": "should not",
}


def stream_print(text: str, delay: float = 0.01, end: str = "\n") -> None:
    """Stream text output dynamically."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def normalize_text(raw_input: Optional[str]) -> str:
    """Normalize text with Unicode NFKC, contraction expansion, and clean punctuation."""
    if raw_input is None:
        return ""
    text = unicodedata.normalize("NFKC", str(raw_input)).lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    for old, new in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    text = re.sub(r"[^a-z0-9\s'/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sanitize(raw_input: str) -> str:
    """Sanitize user input for consistent pipeline handling."""
    return normalize_text(raw_input)


def word_boundary_match(text: str, phrase: str) -> bool:
    """Check if whole phrase matches text using strict word boundaries."""
    if not text or not phrase:
        return False
    norm_phrase = normalize_text(phrase)
    if not norm_phrase:
        return False
    return bool(re.search(rf"\b{re.escape(norm_phrase)}\b", text))


def load_intents(filepath: Path = INTENTS_FILE) -> List[Dict[str, Any]]:
    """Load intents from JSON file with safe built-in fallback."""
    if filepath.is_file():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("intents", [])
        except Exception:
            pass

    return [
        {
            "intent": "greeting",
            "keywords": ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "howdy"],
            "responses": [f"Hey there! I'm {BOT_NAME}, your AI assistant. How can I help you today?"],
        },
        {
            "intent": "farewell",
            "keywords": ["bye", "goodbye", "see you later", "take care"],
            "responses": ["Goodbye! Keep building great things. 🚀"],
        },
        {
            "intent": "identity",
            "keywords": ["who are you", "what are you", "your name"],
            "responses": [f"I'm {BOT_NAME} — a fast deterministic AI chatbot built at DecodeLabs."],
        },
        {
            "intent": "ai_definition",
            "keywords": ["what is ai", "artificial intelligence"],
            "responses": ["AI is the simulation of human intelligence by computer systems."],
        },
        {
            "intent": "ml_definition",
            "keywords": ["what is ml", "machine learning", "what is machine learning"],
            "responses": ["ML is an AI subset where systems automatically learn patterns from data."],
        },
        {
            "intent": "time",
            "keywords": ["what time is it", "what is the time", "current time", "time now"],
            "responses": ["TIME_DYNAMIC"],
        },
        {
            "intent": "thanks",
            "keywords": ["thank you", "thanks"],
            "responses": ["Happy to help! That's what I'm here for. 😊"],
        },
    ]


def log_unknown_query(query: str) -> None:
    """Append unhandled query to outputs/unknown_queries.log."""
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(UNKNOWN_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {query}\n")
    except Exception:
        pass


def find_intent(
    text: str,
    intents: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Find winning intent using score: (match_count, max_keyword_length, priority)."""
    if intents is None:
        intents = load_intents()

    best_intent = None
    best_match_phrase = ""
    best_score = (-1, -1, -999)

    for idx, entry in enumerate(intents):
        matched_phrases = [
            kw for kw in entry.get("keywords", []) if word_boundary_match(text, kw)
        ]
        if not matched_phrases:
            continue

        match_count = len(matched_phrases)
        max_len = max(len(kw) for kw in matched_phrases)
        score = (match_count, max_len, -idx)
        if score > best_score:
            best_score = score
            best_intent = entry
            best_match_phrase = max(matched_phrases, key=len)

    if best_intent:
        return best_intent, best_match_phrase
    return None


def get_response(
    clean_input: str,
    seed: Optional[int] = None,
    log_unknown: bool = True,
) -> str:
    """Return deterministic or seed-controlled response for v1 chatbot."""
    text = normalize_text(clean_input)
    if not text:
        return "Say something! I'm all ears."

    if text in EXIT_COMMANDS or re.search(r"\b(exit|quit|goodbye|bye)\b", text):
        return "Goodbye! Keep building. 🚀"

    if text in {"help", "/help", "commands"}:
        return (
            "Available topics:\n"
            "  • Greetings: hello, hi, hey\n"
            "  • Identity : who are you, what can you do\n"
            "  • AI/ML    : what is ai, what is ml, what is llm, deep learning\n"
            "  • Utilities: what time is it, tell me a joke, motivate me\n"
            "  • Commands : /help, /history, /save, /exit"
        )

    intents = load_intents()
    match_result = find_intent(text, intents)

    if match_result:
        intent_entry, _ = match_result
        responses = intent_entry.get("responses", [])
        if not responses:
            return "I understood your intent, but my response template is empty."

        if seed is not None:
            rng = random.Random(seed)
            reply = rng.choice(responses)
        else:
            reply = random.choice(responses)

        if reply == "TIME_DYNAMIC":
            now = datetime.datetime.now()
            return f"Current time is {now.strftime('%I:%M:%S %p')} on {now.strftime('%A, %d %B %Y')}."
        return reply

    # Fuzzy typo tolerance
    all_keywords = []
    keyword_to_intent = {}
    for entry in intents:
        for kw in entry.get("keywords", []):
            norm_kw = normalize_text(kw)
            all_keywords.append(norm_kw)
            keyword_to_intent[norm_kw] = entry

    fuzzy_hits = get_close_matches(text, all_keywords, n=1, cutoff=0.8)
    if fuzzy_hits:
        matched_kw = fuzzy_hits[0]
        entry = keyword_to_intent[matched_kw]
        responses = entry.get("responses", [])
        reply = responses[0] if responses else "Understood."
        if reply == "TIME_DYNAMIC":
            now = datetime.datetime.now()
            return f"Current time is {now.strftime('%I:%M %p, %d %B %Y')}."
        return reply

    if log_unknown:
        log_unknown_query(clean_input)

    return (
        "I don't understand that query yet. Type 'help' to see my capabilities, "
        "or run chatbot_v2.py for real-time neural responses!"
    )


def run_chatbot(demo: bool = False, save_output: bool = False) -> List[Tuple[str, str]]:
    """Run interactive or demo session for Chatbot v1."""
    transcript: List[Tuple[str, str]] = []
    print("\n" + "=" * 62)
    print(f"  {BOLD}{CYAN}💬 DBot v1 — Pure Rule-Based Engine | DecodeLabs 2026{RESET}")
    print(f"  {DIM}Zero latency, deterministic execution, 100% offline.{RESET}")
    print(f"  Type {YELLOW}'help'{RESET} for topics. Type {YELLOW}'exit'{RESET} to quit.")
    print("=" * 62 + "\n")

    if demo:
        demo_queries = [
            "hello",
            "what is machine learning",
            "what is ai",
            "helo",
            "what time is it",
            "tell me a joke",
            "what are the projects",
            "thank you",
            "exit",
        ]
        for query in demo_queries:
            resp = get_response(query, seed=42, log_unknown=False)
            transcript.append((query, resp))
            print(f"{BOLD}You  :{RESET} {query}")
            print(f"{CYAN}{BOT_NAME}:{RESET} {resp}\n")
            time.sleep(0.2)
    else:
        while True:
            try:
                raw_input = input(f"{BOLD}You  :{RESET} ")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CYAN}{BOT_NAME}:{RESET} Caught interrupt. Goodbye! 🚀\n")
                break

            clean = sanitize(raw_input)
            if not clean:
                print(f"{CYAN}{BOT_NAME}:{RESET} Say something! I'm all ears.\n")
                continue

            if clean in EXIT_COMMANDS:
                print(f"{CYAN}{BOT_NAME}:{RESET} Goodbye! Keep building great things. 🚀\n")
                transcript.append((raw_input, "Goodbye! Keep building. 🚀"))
                break

            resp = get_response(clean)
            transcript.append((raw_input, resp))
            sys.stdout.write(f"{CYAN}{BOT_NAME}:{RESET} ")
            stream_print(resp, delay=0.008)
            print()

    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUTS_DIR / "chatbot_v1_transcript.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for u, b in transcript:
                f.write(f"User: {u}\nDBot: {b}\n\n")
        print(f"[Transcript saved to: {out_file}]")

    return transcript


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="DecodeLabs Chatbot V1 (Pure Rule-Based)")
    parser.add_argument("--demo", action="store_true", help="Run non-interactive automated demo")
    parser.add_argument("--save-output", action="store_true", help="Save transcript to outputs/")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_chatbot(demo=args.demo, save_output=args.save_output)

"""DecodeLabs Industrial Training Kit - Batch 2026.

Project 1: Rule-Based & Hybrid AI Chatbot - Version 2 (Dynamic Hybrid AI Engine)
Real runtime conversational experience:
- Streaming character/token effect for representable modern AI feel
- Colorized / formatted terminal output
- Real-time Groq Cloud Llama 3.3 generation when online
- Seamless fallbacks: Intent -> Regex/Fuzzy -> Dynamic Generative Fallback
- Dynamic multi-turn memory & contextual awareness
- System commands: /help, /reset, /history, /mode, /save, /exit
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
from collections import deque
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
INTENTS_FILE = BASE_DIR / "intents.json"
OUTPUTS_DIR = REPO_ROOT / "outputs"
UNKNOWN_LOG = OUTPUTS_DIR / "unknown_queries.log"

try:
    sys.path.insert(0, str(REPO_ROOT))
    from common.config import GROQ_API_KEY as CFG_GROQ_KEY, GROQ_MODEL as CFG_GROQ_MODEL
except Exception:
    CFG_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
    CFG_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

BOT_NAME = "DBot"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "stop", "/exit", "/quit"}

# Terminal Styling Helpers (ANSI)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Rate limiter state
_RATE_LIMIT_TIMESTAMPS: Deque[float] = deque(maxlen=25)

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

GUARDRAIL_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\bselect\s+\*\s+from\b",
    r"\b(?:hack|exploit|bomb|kill|suicide|attack)\b",
    r"\b(?:idiot|moron|stupid\s+bot|curse|abuse)\b",
]

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def stream_print(text: str, delay: float = 0.012, end: str = "\n") -> None:
    """Stream text output dynamically to simulate interactive neural generation."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def normalize_text(raw_input: Optional[str]) -> str:
    """Normalize text: Unicode NFKC, lowercase, contractions, strip punctuation except apostrophes."""
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
    """Strict word-boundary matching to prevent substring false-positives."""
    if not text or not phrase:
        return False
    norm_phrase = normalize_text(phrase)
    if not norm_phrase:
        return False
    return bool(re.search(rf"\b{re.escape(norm_phrase)}\b", text))


def redact_pii(text: str) -> str:
    """Redact email addresses and phone numbers before sending to external services."""
    redacted = EMAIL_REGEX.sub("[EMAIL REDACTED]", text)
    redacted = PHONE_REGEX.sub("[PHONE REDACTED]", redacted)
    return redacted


def check_guardrails(text: str) -> Optional[str]:
    """Check against abusive or prohibited keywords."""
    lowered = text.lower()
    for pattern in GUARDRAIL_PATTERNS:
        if re.search(pattern, lowered):
            return "I am programmed to be helpful, professional, and safe. I cannot assist with harmful, abusive, or destructive requests."
    return None


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
            "responses": [f"Hey! I'm {BOT_NAME}. What's on your mind today?"],
            "follow_up": "I can help explain AI/ML concepts, tell jokes, or guide you through our projects.",
        },
        {
            "intent": "farewell",
            "keywords": ["bye", "goodbye", "see you later", "take care"],
            "responses": ["Goodbye! Keep building great things. 🚀"],
            "follow_up": "Come back anytime for more questions.",
        },
        {
            "intent": "identity",
            "keywords": ["who are you", "what are you", "your name"],
            "responses": [f"I'm {BOT_NAME}, an intelligent assistant built at DecodeLabs in Lucknow."],
            "follow_up": "I run with high-speed deterministic rules and real-time Groq Llama 3.3 generation.",
        },
        {
            "intent": "ai_definition",
            "keywords": ["what is ai", "artificial intelligence"],
            "responses": ["AI (Artificial Intelligence) is the simulation of human cognitive intelligence by computing systems."],
            "follow_up": "Key AI domains include Machine Learning, Deep Learning, NLP, and Computer Vision.",
        },
        {
            "intent": "ml_definition",
            "keywords": ["what is ml", "machine learning", "what is machine learning"],
            "responses": ["Machine learning is an AI subfield where systems learn patterns from data and improve their performance over time."],
            "follow_up": "Common ML algorithms include KNN, Decision Trees, SVM, and Neural Networks.",
        },
        {
            "intent": "llm_definition",
            "keywords": ["what is llm", "large language model"],
            "responses": ["Large language models are deep neural networks trained on vast text to understand, reason, and generate natural language."],
            "follow_up": "Modern LLMs like Llama 3.3, GPT-4, and Claude utilize Transformer multi-head attention mechanisms.",
        },
        {
            "intent": "time",
            "keywords": ["what time is it", "what is the time", "current time", "time now"],
            "responses": ["TIME_DYNAMIC"],
            "follow_up": "Time management and continuous experimentation are key to engineering excellence.",
        },
        {
            "intent": "thanks",
            "keywords": ["thank you", "thanks"],
            "responses": ["Happy to help! Keep coding and exploring!"],
            "follow_up": "Let me know what else you'd like to work on.",
        },
    ]


def log_unknown_query(query: str) -> None:
    """Log unrecognized query to outputs/unknown_queries.log."""
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(UNKNOWN_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {query}\n")
    except Exception:
        pass


def init_groq():
    """Initialize Groq client if valid API key is present."""
    api_key = os.getenv("GROQ_API_KEY", CFG_GROQ_KEY).strip()
    if not api_key or api_key.startswith("your_") or api_key == "YOUR_GROQ_API_KEY_HERE":
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key, timeout=10.0)
    except Exception:
        return None


def can_make_groq_call() -> bool:
    """Check local throttle: max 25 calls in any 60-second window."""
    now = time.time()
    while _RATE_LIMIT_TIMESTAMPS and now - _RATE_LIMIT_TIMESTAMPS[0] > 60:
        _RATE_LIMIT_TIMESTAMPS.popleft()
    if len(_RATE_LIMIT_TIMESTAMPS) >= 25:
        return False
    _RATE_LIMIT_TIMESTAMPS.append(now)
    return True


def ask_groq(
    client,
    conversation_history: List[Dict[str, str]],
    user_input: str,
    max_retries: int = 3,
) -> str:
    """Query Groq Llama 3.3 with backoff, history capping, and PII redaction."""
    if client is None:
        return "I am currently running in offline rule mode. You can ask me about AI, ML, NLP, or DecodeLabs topics!"

    if not can_make_groq_call():
        return "Local request throttle reached (max 25 req/min). Please try again in a moment."

    safe_prompt = redact_pii(user_input[:600])
    system_prompt = (
        f"You are {BOT_NAME}, an advanced and engaging AI assistant engineered by DecodeLabs "
        f"for their AI Engineering Internship Batch 2026 in Greater Lucknow, India. "
        f"Answer clearly, dynamically, and informatively. Explain complex ideas simply with technical depth when appropriate."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in (conversation_history or [])[-8:]:
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": safe_prompt})

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                max_tokens=350,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "500" in err or "503" in err:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return "The AI engine is temporarily busy. Please ask again in a few seconds."
            if "401" in err or "403" in err:
                return "Groq authentication failed. Running in offline rule-based mode."
            return f"Service notice: {err[:80]}."

    return "Could not reach the neural engine. Please check network connection."


def find_best_intent(
    text: str,
    intents: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Score-based intent matcher prioritizing multi-word and specific phrases."""
    if intents is None:
        intents = load_intents()

    best_intent = None
    best_match_phrase = ""
    best_score = (-1, -1, -999)

    for idx, entry in enumerate(intents):
        matches = [
            kw for kw in entry.get("keywords", []) if word_boundary_match(text, kw)
        ]
        if not matches:
            continue

        match_count = len(matches)
        max_len = max(len(kw) for kw in matches)
        score = (match_count, max_len, -idx)
        if score > best_score:
            best_score = score
            best_intent = entry
            best_match_phrase = max(matches, key=len)

    if best_intent:
        return best_intent, best_match_phrase
    return None


class ChatbotState:
    """Maintain multi-turn session state for hybrid chatbot."""

    def __init__(self, mode: str = "hybrid") -> None:
        self.mode = mode  # "hybrid" or "offline"
        self.last_intent: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self.last_intent = None
        self.history.clear()


def get_response(
    clean_input: str,
    groq_client=None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    demo_mode: bool = False,
    state: Optional[ChatbotState] = None,
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    """Process user message and return tuple: (response_text, source_label)."""
    raw_str = "" if clean_input is None else str(clean_input)

    # 1. Length check & guard
    if len(raw_str) > 500:
        raw_str = raw_str[:500]
        length_notice = " [Note: Input truncated to 500 characters]"
    else:
        length_notice = ""

    norm = normalize_text(raw_str)

    # Empty input
    if not norm:
        return "I didn't catch that. Please type a question or type 'help'!", "[Fallback]"

    # Exit check
    if norm in EXIT_COMMANDS or re.search(r"\b(exit|quit|goodbye|bye)\b", norm):
        return "Goodbye! Keep building great things. 🚀", "[L2-Intent]"

    # Guardrails check
    guard_refusal = check_guardrails(norm)
    if guard_refusal:
        return guard_refusal, "[Fallback]"

    # Context follow-up queries (e.g. "tell me more", "why", "elaborate")
    followup_triggers = {"tell me more", "why", "more", "explain more", "elaborate", "can you elaborate"}
    if state and state.last_intent and norm in followup_triggers:
        followup_text = state.last_intent.get("follow_up")
        if followup_text:
            return followup_text, "[L2-Intent]"

    intents = load_intents()

    # Time zone vs current clock differentiation
    if "time zone" in norm or "timezone" in norm:
        if groq_client and (state is None or state.mode == "hybrid") and not demo_mode:
            return ask_groq(groq_client, conversation_history or [], raw_str) + length_notice, "[L4-Groq]"
        return "India operates on Indian Standard Time (IST), UTC+05:30.", "[L3-Regex]"

    # L2: Intent Matching
    match_result = find_best_intent(norm, intents)
    if match_result:
        intent_entry, _ = match_result
        if state is not None:
            state.last_intent = intent_entry

        responses = intent_entry.get("responses", [])
        if not responses:
            return "Intent recognized, but template is missing.", "[L2-Intent]"

        if seed is not None:
            rng = random.Random(seed)
            reply = rng.choice(responses)
        else:
            reply = random.choice(responses)

        if reply == "TIME_DYNAMIC":
            now = datetime.datetime.now()
            reply = f"The current system time is {now.strftime('%I:%M:%S %p')} on {now.strftime('%A, %d %B %Y')}."
        return reply + length_notice, "[L2-Intent]"

    # L3: Fuzzy typo recovery
    all_keywords = []
    kw_to_intent = {}
    for entry in intents:
        for kw in entry.get("keywords", []):
            k_norm = normalize_text(kw)
            all_keywords.append(k_norm)
            kw_to_intent[k_norm] = entry

    fuzzy_hits = get_close_matches(norm, all_keywords, n=1, cutoff=0.8)
    if fuzzy_hits:
        entry = kw_to_intent[fuzzy_hits[0]]
        if state is not None:
            state.last_intent = entry
        responses = entry.get("responses", [])
        reply = responses[0] if responses else "Understood."
        if reply == "TIME_DYNAMIC":
            now = datetime.datetime.now()
            reply = f"Current time is {now.strftime('%I:%M %p, %d %B %Y')}."
        return reply + length_notice, "[L3-Regex]"

    # L4: Real Groq LLM Fallback (Dynamic Runtime Generation)
    current_mode = state.mode if state else ("hybrid" if groq_client else "offline")
    if current_mode == "hybrid" and groq_client is not None and not demo_mode:
        llm_reply = ask_groq(groq_client, conversation_history or [], raw_str)
        return llm_reply + length_notice, "[L4-Groq]"

    # Unmatched query logging
    log_unknown_query(raw_str)

    return (
        "I can explain AI, ML, NLP, Deep Learning, and DecodeLabs projects. "
        "Try asking: 'what is machine learning', 'tell me a joke', 'what time is it', or type '/help'."
    ) + length_notice, "[Fallback]"


def run_chatbot(demo: bool = False, save_output: bool = False, mode: str = "hybrid") -> None:
    """Run interactive or demo session for Chatbot v2 in a representable, dynamic manner."""
    state = ChatbotState(mode=mode)
    groq_client = init_groq() if mode == "hybrid" else None

    # Representable Terminal Banner
    print("\n" + "=" * 65)
    print(f"  {BOLD}{CYAN}🤖 DBot v2 — Dynamic Industrial AI Chatbot{RESET}")
    print(f"  {DIM}DecodeLabs Industrial Training Kit | Batch 2026{RESET}")
    print(f"  Status : {GREEN}ONLINE{RESET} | Engine: {MAGENTA}{'Groq Llama 3.3 (Connected)' if groq_client else 'Rule-Based Engine (Offline)'}{RESET}")
    print(f"  Commands: {YELLOW}/help{RESET}, {YELLOW}/reset{RESET}, {YELLOW}/history{RESET}, {YELLOW}/mode{RESET}, {YELLOW}/save{RESET}, {YELLOW}/exit{RESET}")
    print("=" * 65 + "\n")

    transcript: List[Tuple[str, str, str]] = []

    if demo:
        test_queries = [
            "hello",
            "what is machine learning",
            "tell me more",
            "wat is ai",
            "what is the time",
            "what is the difference between supervised and unsupervised learning?",
            "/history",
            "exit",
        ]
        for query in test_queries:
            if query == "/history":
                print(f"{BOLD}You  :{RESET} {query}")
                print(f"{CYAN}{BOT_NAME}:{RESET} History contains {len(state.history)} message turns.\n")
                continue

            resp, source = get_response(
                query,
                groq_client=groq_client,
                conversation_history=state.history,
                demo_mode=False if groq_client else True,
                state=state,
                seed=42,
            )
            state.history.append({"role": "user", "content": query})
            state.history.append({"role": "assistant", "content": resp})
            transcript.append((query, resp, source))

            print(f"{BOLD}You  :{RESET} {query}")
            print(f"{CYAN}{BOT_NAME}:{RESET} {resp}")
            print(f"       {DIM}↳ Source: {source}{RESET}\n")
            time.sleep(0.3)
    else:
        # Welcome greeting
        welcome_msg = (
            f"Hello! I am {BOT_NAME}, your AI career and technical assistant at DecodeLabs. "
            f"Ask me any question about AI, Machine Learning, Python, or our project suite!"
        )
        sys.stdout.write(f"{CYAN}{BOT_NAME}:{RESET} ")
        stream_print(welcome_msg, delay=0.01)
        print()

        while True:
            try:
                raw_input = input(f"{BOLD}You  :{RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CYAN}{BOT_NAME}:{RESET} Caught interrupt. Goodbye! 🚀\n")
                break

            if not raw_input:
                print(f"{CYAN}{BOT_NAME}:{RESET} I'm listening! Please type a question.\n")
                continue

            # Command dispatcher
            if raw_input.startswith("/"):
                cmd = raw_input.lower().strip()
                if cmd in {"/exit", "/quit"}:
                    print(f"{CYAN}{BOT_NAME}:{RESET} Goodbye! Keep building great things. 🚀\n")
                    break
                elif cmd == "/reset":
                    state.reset()
                    print(f"{CYAN}{BOT_NAME}:{RESET} Conversation history and context memory reset.\n")
                    continue
                elif cmd == "/history":
                    print(f"\n{YELLOW}--- Recent Conversation History ({len(state.history)} messages) ---{RESET}")
                    for h in state.history[-8:]:
                        role_tag = f"{BOLD}You{RESET}" if h["role"] == "user" else f"{CYAN}{BOT_NAME}{RESET}"
                        print(f"  {role_tag}: {h['content']}")
                    print(f"{YELLOW}------------------------------------------------------------{RESET}\n")
                    continue
                elif cmd.startswith("/mode"):
                    parts = cmd.split()
                    if len(parts) > 1 and parts[1] in {"offline", "hybrid"}:
                        state.mode = parts[1]
                        groq_client = init_groq() if state.mode == "hybrid" else None
                        status_str = "Hybrid (Groq Llama 3.3)" if groq_client else "Offline (Rule-Based)"
                        print(f"{CYAN}{BOT_NAME}:{RESET} Mode switched to: {status_str}\n")
                    else:
                        print(f"{CYAN}{BOT_NAME}:{RESET} Usage: /mode offline | /mode hybrid\n")
                    continue
                elif cmd == "/help":
                    print(
                        f"\n{BOLD}Available Commands & Capabilities:{RESET}\n"
                        f"  {YELLOW}/help{RESET}            - Display this help manual\n"
                        f"  {YELLOW}/history{RESET}         - View active conversation context\n"
                        f"  {YELLOW}/reset{RESET}           - Reset context memory\n"
                        f"  {YELLOW}/mode <mode>{RESET}    - Switch mode ('offline' or 'hybrid')\n"
                        f"  {YELLOW}/save{RESET}            - Save transcript to outputs/\n"
                        f"  {YELLOW}/exit{RESET}            - End session\n"
                        f"\n{DIM}Topics you can explore: Machine Learning, Deep Learning, NLP, Time, Jokes, Projects.{RESET}\n"
                    )
                    continue
                elif cmd == "/save":
                    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                    target = OUTPUTS_DIR / "chatbot_v2_transcript.txt"
                    with open(target, "w", encoding="utf-8") as f:
                        for u, b, s in transcript:
                            f.write(f"User: {u}\nDBot: {b}\nSource: {s}\n\n")
                    print(f"{CYAN}{BOT_NAME}:{RESET} Transcript successfully saved to {target}\n")
                    continue

            resp, source = get_response(
                raw_input,
                groq_client=groq_client,
                conversation_history=state.history,
                demo_mode=False,
                state=state,
            )
            state.history.append({"role": "user", "content": raw_input})
            state.history.append({"role": "assistant", "content": resp})
            transcript.append((raw_input, resp, source))

            # Dynamic streaming print for response
            sys.stdout.write(f"{CYAN}{BOT_NAME}:{RESET} ")
            stream_print(resp, delay=0.008)
            print(f"       {DIM}↳ Source: {source}{RESET}\n")

            if normalize_text(raw_input) in EXIT_COMMANDS:
                break

    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUTS_DIR / "chatbot_v2_transcript.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for u, b, s in transcript:
                f.write(f"User: {u}\nDBot: {b}\nSource: {s}\n\n")
        print(f"[Transcript saved to: {out_file}]")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="DecodeLabs Chatbot V2 (Dynamic Hybrid AI Engine)")
    parser.add_argument("--demo", action="store_true", help="Run automated demo session")
    parser.add_argument("--save-output", action="store_true", help="Save transcript to outputs/")
    parser.add_argument("--mode", choices=["hybrid", "offline"], default="hybrid", help="Chatbot execution mode")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_chatbot(demo=args.demo, save_output=args.save_output, mode=args.mode)

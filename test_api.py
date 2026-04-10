"""
test_api.py — Test all VoiceBot API endpoints using Python requests.
Run AFTER the server is started (python main.py).

Usage:
    python test_api.py
"""

import requests
import json
import os

BASE = "http://localhost:8000"

SEP = "─" * 55

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ── 1. Health Check ───────────────────────────────────────────────────────
section("1. Health Check")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# ── 2. Predict Intent ─────────────────────────────────────────────────────
section("2. Intent Classification (Text)")

test_queries = [
    "Where is my order?",
    "I want to cancel my order immediately.",
    "Please refund my money.",
    "I cannot log into my account.",
    "My product arrived broken.",
    "Why is my shipment delayed?",
    "How do I track my package?",
    "I need to return this item.",
    "My payment failed at checkout.",
    "I want to upgrade my subscription plan.",
    "Please change my delivery address.",
]

for query in test_queries:
    r = requests.post(f"{BASE}/predict-intent", data={"text": query})
    d = r.json()
    print(f"  Query      : {query}")
    print(f"  Intent     : {d['intent']} ({d['confidence']*100:.1f}% confidence)")
    print()

# ── 3. Generate Response ──────────────────────────────────────────────────
section("3. Response Generation")

r = requests.post(f"{BASE}/generate-response",
                  data={"intent": "refund_request", "confidence": 0.95})
d = r.json()
print(f"  Intent   : {d['intent']}")
print(f"  Response : {d['response_text']}")

# ── 4. Synthesize (TTS) ───────────────────────────────────────────────────
section("4. Text-to-Speech Synthesis")

test_text = ("Thank you for reaching out! I can help you track your package. "
             "Please share your order number and I'll get the update right away.")

r = requests.post(f"{BASE}/synthesize", data={"text": test_text})
if r.status_code == 200:
    with open("test_tts_output.mp3", "wb") as f:
        f.write(r.content)
    print(f"  TTS audio saved → test_tts_output.mp3")
    print(f"  File size: {len(r.content) / 1024:.1f} KB")
else:
    print(f"  ERROR: {r.text}")

# ── 5. Full Pipeline (Text → Mock) ────────────────────────────────────────
section("5. Full Pipeline Summary")
print("  To test /voicebot, provide a real WAV file:")
print()
print('  curl -X POST "http://localhost:8000/voicebot" \\')
print('       -F "audio=@your_audio.wav" \\')
print('       -o response.mp3')
print()
print("  Or use the Swagger UI at: http://localhost:8000/docs")
print()
print(f"{SEP}")
print("  All tests complete!")
print(SEP)

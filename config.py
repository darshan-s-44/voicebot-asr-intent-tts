"""
config.py — Central configuration for the VoiceBot system.
All model paths, thresholds, and settings live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(BASE_DIR, "saved_models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
AUDIO_TMP_DIR   = os.path.join(BASE_DIR, "tmp_audio")

for d in [MODEL_DIR, LOG_DIR, AUDIO_TMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── ASR (Whisper) ──────────────────────────────────────────────────────────
ASR_MODEL_SIZE  = "base"          # tiny | base | small | medium | large
ASR_LANGUAGE    = "en"
ASR_DEVICE      = "cpu"           # "cuda" if GPU available

# ── Intent Classification ──────────────────────────────────────────────────
INTENT_BASE_MODEL       = "distilbert-base-uncased"
INTENT_MODEL_PATH       = os.path.join(MODEL_DIR, "intent_classifier")
INTENT_CONFIDENCE_THRESHOLD = 0.40   # below this → "unknown" intent

# 12 customer-support intents
INTENT_LABELS = [
    "order_status",
    "cancel_order",
    "refund_request",
    "subscription_issue",
    "payment_problem",
    "shipping_delay",
    "product_defect",
    "account_login",
    "change_address",
    "track_package",
    "return_item",
    "general_inquiry",
]

# ── TTS (gTTS) ────────────────────────────────────────────────────────────
TTS_LANGUAGE    = "en"
TTS_SLOW        = False   # True → slower speech

# ── API ───────────────────────────────────────────────────────────────────
API_HOST        = "0.0.0.0"
API_PORT        = 8000
MAX_AUDIO_SIZE_MB = 25

"""
generate_test_audio.py — Generate sample WAV test files using gTTS.
These simulate user voice input for testing the /voicebot endpoint.

Usage:
    python generate_test_audio.py
"""

import os
from gtts import gTTS

os.makedirs("test_audio", exist_ok=True)

SAMPLES = [
    ("sample_order_status.mp3",    "Where is my order? I placed it 3 days ago."),
    ("sample_cancel_order.mp3",    "I want to cancel my order immediately."),
    ("sample_refund.mp3",          "I need a refund for my damaged product."),
    ("sample_track_package.mp3",   "Can you help me track my package?"),
    ("sample_login.mp3",           "I cannot log into my account. I forgot my password."),
    ("sample_payment.mp3",         "My payment failed during checkout. Please help."),
    ("sample_shipping_delay.mp3",  "My delivery is very late. Can you check the status?"),
    ("sample_return.mp3",          "I want to return this item. It arrived damaged."),
    ("sample_subscription.mp3",    "My subscription is not working after renewal."),
    ("sample_address.mp3",         "I need to change my delivery address."),
]

print("Generating test audio files...\n")

for filename, text in SAMPLES:
    path = os.path.join("test_audio", filename)
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(path)
    size = os.path.getsize(path) / 1024
    print(f"  ✔ {filename:40s} ({size:.1f} KB)")

print(f"\nGenerated {len(SAMPLES)} test audio files in ./test_audio/")
print("\nTest with:")
print('  curl -X POST "http://localhost:8000/voicebot" \\')
print('       -F "audio=@test_audio/sample_order_status.mp3" \\')
print('       -o bot_response.mp3 -D -')

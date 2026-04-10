"""
VoiceBot API - FastAPI server with Whisper ASR, DistilBERT intent, gTTS TTS
"""

import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import whisper
from gtts import gTTS

# -------------------------------
# Configuration
# -------------------------------
ASR_MODEL_SIZE = "base"
ASR_DEVICE = "cpu"
INTENT_MODEL_PATH = "models/intent_model"   # <-- your trained model
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("voicebot")

# -------------------------------
# Load Models
# -------------------------------
logger.info("Loading Whisper ASR model...")
asr_model = whisper.load_model(ASR_MODEL_SIZE, device=ASR_DEVICE)
logger.info("Whisper loaded.")

# Load your fine-tuned intent classifier
logger.info(f"Loading intent model from {INTENT_MODEL_PATH}...")
intent_tokenizer = DistilBertTokenizer.from_pretrained(INTENT_MODEL_PATH)
intent_model = DistilBertForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
intent_model.eval()
logger.info("Intent model loaded.")

# Intent label mapping (must match training order)
INTENT_LABELS = [
    "order_status", "cancel_order", "refund", "payment_issue",
    "shipping_delay", "wrong_item", "change_address", "track_order",
    "product_info", "return_policy", "complaint", "greeting"
]

# Response mapping
RESPONSES = {
    "order_status": "Your order is on its way. It should arrive within two days.",
    "cancel_order": "Your order has been cancelled. A confirmation email has been sent.",
    "refund": "Refund will be processed in 5-7 business days.",
    "payment_issue": "Please check your payment method or contact your bank.",
    "shipping_delay": "We are sorry for the delay. Your order will arrive by Friday.",
    "wrong_item": "We will send you a replacement immediately.",
    "change_address": "Your shipping address has been updated.",
    "track_order": "You can track your order using the link sent to your email.",
    "product_info": "Please visit our website for detailed product information.",
    "return_policy": "You can return items within 30 days of delivery.",
    "complaint": "We have registered your complaint and will get back to you soon.",
    "greeting": "Hello! How can I help you today?"
}
DEFAULT_RESPONSE = "I'm sorry, I didn't understand. Can you please rephrase?"

# -------------------------------
# Helper functions
# -------------------------------
def predict_intent(text: str):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        outputs = intent_model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
    intent = INTENT_LABELS[pred_id]
    confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()
    return intent, confidence

def text_to_speech(text: str, output_path: str):
    tts = gTTS(text)
    tts.save(output_path)
    return output_path

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="VoiceBot", description="Speech-to-intent voice assistant")

@app.post("/voicebot")
async def voicebot(audio: UploadFile = File(...)):
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Transcribe with Whisper
    try:
        result = asr_model.transcribe(tmp_path)
        transcript = result["text"].strip()
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    if not transcript:
        transcript = ""

    # Predict intent
    intent, confidence = predict_intent(transcript)

    # Generate response text
    response_text = RESPONSES.get(intent, DEFAULT_RESPONSE)

    # Synthesize speech
    output_audio = os.path.join(OUTPUT_DIR, "bot_response.mp3")
    text_to_speech(response_text, output_audio)

    return {
        "transcription": transcript,
        "intent": intent,
        "confidence": confidence,
        "response": response_text,
        "audio_file": output_audio
    }

@app.get("/health")
def health():
    return {"status": "ok", "intent_model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
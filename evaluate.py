"""
evaluate.py — Evaluate ASR (Whisper) Word Error Rate on a small test set.
You supply a CSV with columns: audio_path, transcript

Usage:
    python evaluate.py --csv test_audio/eval.csv

CSV format:
    audio_path,transcript
    test_audio/sample1.wav,where is my order
    test_audio/sample2.wav,i want to cancel my order
"""

import argparse
import csv
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("evaluate")


def evaluate_wer(csv_path: str):
    from models.asr import ASRModel, compute_wer

    logger.info(f"Loading evaluation data from: {csv_path}")
    references, hypotheses = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = row["audio_path"].strip()
            ref_text   = row["transcript"].strip().lower()

            if not os.path.exists(audio_path):
                logger.warning(f"File not found, skipping: {audio_path}")
                continue

            asr = ASRModel()
            result = asr.transcribe(audio_path)
            hyp_text = result["text"].lower()

            references.append(ref_text)
            hypotheses.append(hyp_text)
            logger.info(f"REF: '{ref_text}' | HYP: '{hyp_text}'")

    if not references:
        logger.error("No valid audio files found for evaluation.")
        return

    wer_score = compute_wer(references, hypotheses)
    print("\n" + "="*45)
    print("     ASR EVALUATION — WORD ERROR RATE (WER)")
    print("="*45)
    print(f"  Files evaluated : {len(references)}")
    print(f"  WER             : {wer_score:.4f} ({wer_score*100:.2f}%)")
    print("="*45 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    args = parser.parse_args()
    evaluate_wer(args.csv)

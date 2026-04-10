# VoiceBot - Speech to Intent Voice Assistant

A voice bot that takes an audio file transcribes it with Whisper classifies intent with DistilBERT generates a response and returns spoken audio using gTTS and FastAPI

## Features

- Speech recognition with OpenAI Whisper
- Intent classification with fine tuned DistilBERT
- Rule based response generation
- Text to speech with gTTS
- REST API with FastAPI and Swagger UI

## Architecture

Audio Input -> Whisper ASR -> DistilBERT Intent -> Response Engine -> gTTS TTS -> Audio Output

## Installation

1. Clone the repository
2. Create a virtual environment

3. Install dependencies

4. Install FFmpeg required for Whisper
- Windows download from ffmpeg.org and add to PATH
- Mac brew install ffmpeg
- Linux sudo apt install ffmpeg

## Training the Intent Classifier

Run this command once before starting the API server


This trains DistilBERT on 12 intent classes and saves the model to models/intent_model

## Generating Test Audio Files

Run this command to create sample MP3 files for testing

## Starting the API Server

The server runs at http://localhost 8000

## Testing with Swagger UI

Open your browser to http://localhost 8000 docs

Click on POST voicebot then Try it out then upload an audio file from the test_audio folder then click Execute

You will receive a JSON response containing transcription intent confidence response text and an audio file path

## Testing with curl
curl -X POST "http://localhost:8000/voicebot" -F "audio=@test_audio/sample_order_status.mp3" -o bot_response.mp3 -D -


## Evaluating Word Error Rate

Create a CSV file with audio paths and reference transcripts then run

## Project Structure
voicebot/
├── main.py FastAPI server
├── train_intent.py Intent classifier training
├── generate_test_audio.py Test audio generator
├── evaluate.py WER evaluation
├── test_api.py API test script
├── requirements.txt Dependencies
├── config.py Configuration settings
├── models/
│ └── intent_model Trained DistilBERT model
├── data/
│ └── intent_data.py Training utterances
├── test_audio Sample audio files
├── outputs Generated response audio
└── confusion_matrix.png Training evaluation output


## Dependencies

- fastapi
- uvicorn
- openai whisper
- transformers
- torch
- gtts
- ffmpeg python
- scikit learn
- matplotlib
- seaborn
- datasets

## Results

- Intent classifier accuracy 33 percent with 12 training examples
- Confusion matrix saved as confusion_matrix.png
- Average latency 2 to 10 seconds on CPU

## License

MIT
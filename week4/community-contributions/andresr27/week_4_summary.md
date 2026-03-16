# Week 4: Personal Tutor Chatbot with Unified TTS Implementation

## Overview
This project implements a personal tutor chatbot that can answer questions related to code, LLMs, and other technical topics. The chatbot supports both OpenRouter (cloud) and Ollama (local) platforms, with a unified text-to-speech (TTS) system using Hugging Face's Transformers library.

## Key Features

### 1. Platform Support
- **OpenRouter (Cloud)**: Uses GPT-5-mini for chat responses
- **Ollama (Local)**: Uses Llama3.2 for chat responses
- **Unified TTS**: Both platforms use the same Hugging Face TTS system

### 2. Audio Capabilities
- **Speech Recognition**: Uses OpenAI's Whisper-tiny model for transcribing audio input
- **Text-to-Speech**: Uses Microsoft's SpeechT5 model from Hugging Face for generating audio responses
- **Audio Processing**: Includes resampling, filtering, and quality enhancements

### 3. User Interface
- Gradio-based web interface
- Real-time chat display
- Audio input/output support
- System logs display
- Platform switching capability

## Technical Implementation

### TTS System
The TTS functionality uses Hugging Face's Transformers library with the following components:
- **Model**: `microsoft/speecht5_tts` for text-to-speech conversion
- **Vocoder**: `microsoft/speecht5_hifigan` for audio waveform generation
- **Processor**: `SpeechT5Processor` for text preprocessing

### Key Functions
1. `load_tts_model()`: Loads and initializes the TTS models with authentication
2. `text_to_speech_hf()`: Converts text to audio with quality enhancements
3. `get_tts_audio()`: Wrapper function with fallback mechanisms
4. `process_text_input()`: Main processing function that integrates chat and TTS

### Audio Quality Enhancements
The implementation includes several audio processing techniques:
- Resampling to target sample rates
- Low-pass filtering to reduce robotic sounds
- Volume normalization
- Subtle reverb for natural sound
- Speed adjustment for better clarity

## Setup Requirements

### Environment Variables
- `HF_TOKEN`: Hugging Face authentication token
- `OPENROUTER_API_KEY`: OpenRouter API key (for cloud platform)
- `OPENROUTER_API_URL`: OpenRouter API URL

### Dependencies
Install the following Python packages (either by adding them to your own `requirements.txt` or installing them manually with `pip`):
- Core: gradio, openai, torch, transformers
- Audio processing: soundfile, scipy, librosa
- Utilities: structlog, python-dotenv

## Usage Instructions

1. **Install dependencies**: `pip install gradio openai torch transformers soundfile scipy librosa structlog python-dotenv`
2. **Set environment variables**: Configure HF_TOKEN and platform-specific keys
3. **Run the application**: `python week_4.py`
4. **Access the interface**: Open the Gradio interface in your browser

## Challenges and Solutions

### Challenge 1: Platform-Specific Audio Models
**Solution**: Implemented a unified TTS system using Hugging Face that works across both platforms, eliminating the need for platform-specific audio models.

### Challenge 2: Audio Quality
**Solution**: Added multiple audio processing steps including filtering, resampling, and reverb to improve naturalness.

### Challenge 3: Model Loading and Caching
**Solution**: Implemented proper caching with authentication and fallback mechanisms for offline scenarios.

### Challenge 4: Real-time Performance
**Solution**: Used efficient resampling and caching to minimize latency in audio generation.

## Testing
The system has been tested with:
- Both OpenRouter and Ollama platforms
- Various text inputs and audio recordings
- Different network conditions
- Error scenarios (missing tokens, network issues)

## Future Improvements
1. Support for more TTS models and voices
2. Better error handling and user feedback
3. Caching of frequently used audio responses
4. Multi-language support
5. Improved audio quality with advanced processing

## Conclusion
The implementation successfully creates a unified TTS system for the personal tutor chatbot, providing consistent audio responses across different platforms while maintaining high audio quality and reliability.

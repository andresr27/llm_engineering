# Week 3 Project: Enhanced Multimodal Chat Interface – Implementation Summary

## Assignment Overview
The Week 3 homework assignment involved creating a personal tutor tool that serves as a guide during the course. The tool needed to:
- Interact with both GPT models and open-source Llama models running locally
- Enable user interaction through text and audio inputs
- Structure responses in Markdown for clearer presentation
- Reinforce concepts learned through practical coding related to AI and model interactions

## Implementation summary

### 1. **Fixed Conversation History Management**
- **Problem**: In the previous implementation, conversation context wasn't properly maintained across different input types
- **Solution**: Used the GPT audio model's `modalities=["text", "audio"]` parameter to ensure synchronized text and audio outputs
- **Actual Implementation**: The `process_text_with_audio_response()` function requests both text transcripts and audio data simultaneously
- **Result**: Better conversation continuity when switching between text and audio inputs

### 2. **Simplified Speech Recognition**
- **Previous Approach**: Manual audio processing with custom resampling and format handling
- **New Approach**: Used Hugging Face's `pipeline("automatic-speech-recognition", model="openai/whisper-tiny")`
- **Benefits**: 
  - Automatic audio resampling to 16kHz (required by Whisper)
  - Cleaner code with fewer manual audio processing steps
  - Built-in error handling from the transformers library

### 3. **Added Local Deployment Option**
- **Cloud Option**: OpenRouter with GPT models for text and audio responses
- **Local Option**: Ollama with Llama for text generation + Piper for text-to-speech
- **Implementation**: 
  - Platform selector in the UI to switch between OpenRouter and Ollama
  - `text_to_speech_piper()` function for local audio generation
  - Fallback to Python-based audio resampling when ffmpeg is not available
- **Note**: Local TTS with Piper is slower than cloud-based solutions but works offline

### 4. **Enhanced User Interface**
- **Platform Selection**: Dropdown to choose between OpenRouter (cloud) and Ollama (local)
- **System Logs**: Real-time logging display using structlog
- **Chatbot Height**: Increased to 500px for better visibility
- **Button Layout**: Wider buttons for improved usability
- **Log Refresh**: Configurable refresh period for system logs

### 5. **Markdown Response Formatting**
- **Implementation**: Used a separate LLM call to format responses in Markdown
- **Structure**: Responses follow a Q&A forum format with sections for question summary, answer summary, detailed response, and takeaways
- **Note**: This adds latency but improves readability of responses

## Technical Implementation Details

### Platform Configuration System
- `get_platform_config()` function dynamically loads model configurations based on selected platform
- Automatic client reconfiguration when switching platforms
- Speech recognition model reloaded when STT model changes

### Audio Processing Pipeline
1. **Speech-to-Text**: Hugging Face Whisper pipeline for audio transcription
2. **Text Generation**: Platform-specific LLM (GPT-5-mini or Llama)
3. **Text-to-Speech**: 
   - Cloud: GPT audio model with streaming audio
   - Local: Piper TTS with ffmpeg or Python-based resampling
4. **Markdown Formatting**: Additional LLM call to structure responses

### Error Handling
- Structured logging with structlog for better debugging
- Graceful fallback to text-only responses when audio generation fails
- Input validation for both text and audio inputs

## What Was Not Implemented 

### 1. **Image Model Integration**
- The `openai/gpt-5-image-mini` model mentioned in some comments was not implemented
- The tool remains focused on text and audio modalities only

### 2. **Advanced Conversation Features**
- No conversation persistence across sessions
- No multiple conversation threads or search functionality
- Each session starts with a fresh conversation history

### 3. **Performance Optimizations**
- No response caching implemented
- Local TTS remains slow (~2-3 seconds for short responses)
- No async processing for non-blocking UI updates

### 4. **Comprehensive Testing**
- No unit tests or integration tests were written
- Limited error recovery beyond basic logging

## Requirements and Dependencies
The project requires several Python packages (see `requirements.txt`):
- Core: gradio, openai, python-dotenv, numpy, scipy, torch
- Hugging Face: transformers, accelerate
- Audio: soundfile
- Logging: structlog
- System: ffmpeg (for optimal audio processing), piper-tts (installed via uv)

## Running the Application
1. Install dependencies: `pip install -r requirements.txt`
2. Install system dependencies: ffmpeg (for audio processing)
3. Install Piper TTS: `uv add piper-tts`
4. Set up environment variables in `.env` file
5. Run: `python week_3.py`

## Conclusion
The Week 3 implementation successfully addresses the core assignment requirements:
1. **Interaction with both cloud and local models**: OpenRouter (GPT) and Ollama (Llama + Piper)
2. **Multimodal input processing**: Text and audio inputs with unified conversation history
3. **Markdown response formatting**: Structured responses for better readability
4. **Practical AI coding experience**: Implementation of speech recognition, text generation, and text-to-speech pipelines

The tool demonstrates key concepts in multimodal AI interfaces while providing a functional personal tutor application. While there are opportunities for further optimization and feature enhancement, the current implementation meets the assignment requirements and provides a solid foundation for future development.

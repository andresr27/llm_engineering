# Homework Assignment
For Week 3, the homework assignment typically involves creating a personal tutor tool similar to what was outlined in Week 1 and Week 2. 

## Objective
Build a tool that serves as your personal guide during the course. This tool should be able to answer questions regarding 
code, LLMs, or any other relevant topics you encounter.

## Implementation Steps Required for this Assignment:
Here are the key points about this assignment:

- Set up your coding environment to interact with both GPT models and the open-source Llama model running locally. 
- Fill in the code to enable user interaction where a user can input a question for your tutor tool to answer. Done.
- Structure the responses in Markdown for clearer presentation.
- Expected Experience: This assignment is designed to help reinforce concepts learned, through practical coding related to AI and model interactions.

Additional Resources: Check the community contributions in the course folders for examples of solutions provided by other students, which can help guide your implementation.

## Week 3 Improvements Summary

### 1. **Fixed Common History Problem**
- **Problem**: In Week 2, audio and text inputs were processed independently without shared context, leading to disjointed conversations.
- **Solution**: Properly utilized the `modalities` parameter of the GPT audio model to maintain a unified conversation history.
- **Implementation**: The `process_text_with_audio_response()` function now processes both text and audio modalities simultaneously, ensuring that:
  - Audio responses include synchronized transcripts
  - All interactions (text and audio) are stored in a single chat history
  - Context is preserved across different input types

### 2. **Simplified Speech Recognition with Hugging Face Pipelines**
- **Problem**: Complex audio processing code requiring manual handling of audio formats and sampling rates.
- **Solution**: Used Hugging Face's `pipeline()` API for automatic speech recognition (ASR).
- **Implementation**: 
  - Replaced custom audio processing with `pipeline("automatic-speech-recognition", model="openai/whisper-tiny")`
  - Automatic handling of audio resampling to 16kHz (Whisper's expected input)
  - Simplified error handling and improved maintainability
  - Reduced code complexity by ~40 lines

### 3. **Local TTS with Ollama and Piper**
- **Problem**: Dependency on cloud services for text-to-speech (TTS) functionality.
- **Solution**: Implemented local TTS using Piper for Ollama platform responses.
- **Implementation**:
  - Added `text_to_speech_piper()` function for local audio generation
  - Automatic fallback to Python-based resampling when ffmpeg is unavailable
  - Support for 16kHz mono PCM audio output (compatible with Gradio)
  - **Trade-off**: Local TTS is significantly slower (~2-3x) than cloud-based solutions but offers privacy and offline capability

## Implemented Features

### 1. **Multimodal Input Processing**
- **Text Input**: Users can type messages in a text box
- **Audio Input**: Users can record audio via microphone
- **Unified Processing**: Both input types are handled by a single `process_input()` function with shared history

### 2. **AI Model Integration**
- **Chat Model**: Uses `openai/gpt-5-mini` for text-only conversations
- **Audio Model**: Uses `openai/gpt-audio-mini` for multimodal (text+audio) conversations with proper modality handling
- **Streaming Audio**: Audio responses are streamed and played automatically
- **Local Option**: Ollama with Piper TTS for local deployment

### 3. **Conversation Management**
- **Chat History**: Maintains unified conversation context across turns and input types
- **System Prompt**: Includes a technical expert persona with specific guidelines
- **Role-based Messages**: Properly formats user, assistant, and system messages

### 4. **Audio Processing**
- **Speech Recognition**: Hugging Face pipeline with Whisper-tiny for accurate transcription
- **Audio Encoding/Decoding**: Proper handling of base64 audio data from API responses
- **Real-time Playback**: Uses Gradio's streaming audio with autoplay
- **Local TTS**: Piper-based text-to-speech for Ollama responses

### 5. **User Interface**
- **Gradio Interface**: Clean, functional web interface with platform selector
- **Dual Input Methods**: Separate buttons for text and audio submission
- **Chat Display**: Shows conversation history in a chat window with Markdown rendering
- **System Logs**: Real-time logging with structlog integration
- **Platform Switching**: Toggle between OpenRouter (cloud) and Ollama (local)

### 6. **Error Handling**
- **Input Validation**: Checks for empty text/audio inputs
- **Fallback Messages**: Provides default responses when audio model doesn't generate text
- **API Configuration**: Validates API keys and endpoints on startup
- **Structured Logging**: Comprehensive logging with structlog for debugging

## Suggested Improvements for Future Work

### 1. **Performance Optimization for Local Deployment**
- **Current Challenge**: Local TTS with Piper is slow (~2-3 seconds for short responses)
- **Target**: Reduce response time to under 100ms for local deployment
- **Optimization Strategies**:
  - **Pre-load Piper model**: Currently loads on each call; could cache in memory
  - **Audio response caching**: Cache common responses to avoid repeated TTS generation
  - **Async processing**: Use asynchronous operations for non-blocking UI updates
  - **Alternative TTS engines**: Consider faster options like Coqui AI or Edge TTS
  - **Profiling**: Use `cProfile` or `line_profiler` to identify specific bottlenecks

### 2. **Enhanced Markdown Processing**
- **Current Approach**: Separate LLM call to generate Markdown from transcripts
- **Optimization Opportunity**: Local tokenization-based summarization
- **Implementation Options**:
  - Use Hugging Face `transformers.AutoTokenizer` and `AutoModelForSeq2SeqLM` for extractive summarization
  - Implement chunking strategies for long transcripts
  - Cache summarized versions of common responses
- **Benefits**: Reduced latency, lower API costs, and offline capability

### 3. **Improved TTS Integration**
- **Current Limitation**: Relies on Piper (command-line tool) with subprocess calls
- **Future Direction**: Native Python TTS integration
- **Potential Solutions**:
  - Hugging Face `transformers` TTS models (e.g., SpeechT5, VITS)
  - `espnet` or `coqui-tts` for more voice options
  - OnnxRuntime for optimized inference
- **Benefits**: Better error handling, more voice/language support, and reduced memory usage through quantization

### 4. **Robust Error Recovery System**
- **Current State**: Basic error logging without automatic recovery
- **Enhanced Approach**: Comprehensive error handling with retry logic
- **Implementation**:
  - Use `tenacity` library for exponential backoff retry decorators
  - Implement circuit breaker pattern for API failures
  - Automatic fallback to alternative platforms when primary fails
  - Graceful degradation (e.g., text-only mode when audio fails)

### 5. **Code Quality and Testing**
- **Modularization**: Refactor monolithic functions into smaller, focused units
- **Type Safety**: Add comprehensive type hints for better IDE support and error detection
- **Testing Framework**:
  - Unit tests for core functions using `pytest`
  - Integration tests for API interactions
  - UI tests for Gradio interface components
- **Documentation**: Enhanced docstrings, usage examples, and architecture diagrams

### 6. **Advanced Features for Educational Use**
- **Conversation Persistence**: Save and load conversation histories
- **Multiple Tutor Personas**: Different expert roles (coding tutor, language coach, etc.)
- **Progress Tracking**: Monitor learning progress over time
- **Interactive Examples**: Code execution within safe sandbox environments
- **Multilingual Support**: Translation and TTS in multiple languages

### 7. **Deployment and Scalability**
- **Containerization**: Docker support for easy deployment
- **Configuration Management**: Environment-based configuration system
- **Monitoring**: Performance metrics and usage analytics
- **Scalability**: Support for multiple concurrent users

## Performance Considerations

### Local vs Cloud Trade-offs:
- **Cloud (OpenRouter)**: Faster (~200-500ms) but requires internet and API costs
- **Local (Ollama+Piper)**: Slower (~2000-3000ms) but private and offline-capable
- **Hybrid Approach**: Use cloud for primary responses with local fallback

### Memory Optimization:
- Current implementation loads Whisper model on startup (~1GB RAM)
- Consider model quantization or using smaller Whisper variants
- Implement lazy loading for TTS models

## Conclusion
The Week 3 implementation successfully addresses the core requirements while introducing important improvements:
1. Fixed conversation history management through proper modality usage
2. Simplified audio processing with Hugging Face pipelines
3. Added local deployment option with Ollama and Piper TTS

The tool now provides a robust foundation for a personal tutor application with support for both cloud and local deployment options. Future work should focus on performance optimization, especially for local deployment, and enhanced user experience features.


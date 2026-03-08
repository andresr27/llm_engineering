# Change Log

## Week 5: RAG System Refinement

### Changes Made:
1. **Improved Document Chunking**
   - Replaced simple chunking with a recursive character splitting strategy in `models.py`.
   - Increased default chunk size to 600 and overlap to 100 for better context retention.
2. **Enhanced Logging**
   - Updated `app.py` to provide clearer logs during the RAG retrieval process.

### Dependencies Added:
- pytest>=7.4.0
- pytest-asyncio>=0.23.0
- chromadb>=0.6.3

## Week 5: RAG System Implementation
**Goal:** Build an expert knowledge worker using Retrieval Augmented Generation to answer course-specific questions.
- **ChromaDB Integration:** Set up persistent vector storage for course documentation.
- **Recursive Character Splitting:** Implemented advanced document chunking (size 500, overlap 50) for better context.
- **Context Retrieval:** Developed logic to augment LLM prompts with retrieved document snippets.
- **Evaluation Framework:** Created a RAG evaluation system to measure retrieval success and context precision.
- **TTS Refinement:** Replaced heavy HF models with `edge-tts` for faster, more reliable audio responses.
- **Testing Suite:** Added `pytest` framework and `run_tests.py` for system validation.

### Dependencies Added:
- chromadb>=0.6.3
- edge-tts>=6.1.12
- pydub>=0.25.1
- pytest>=7.4.0

## Week 4: Unified TTS & Performance
**Goal:** Create a consistent TTS experience across all platforms using Hugging Face models.
- Implemented unified TTS using `microsoft/speecht5_tts`.
- Added audio quality enhancements (filtering, normalization).
- Introduced structured logging with `structlog`.

## Week 3: Enhanced Multimodal & Local Models
**Goal:** Extend the assistant to support local models (Ollama/Llama) and improve response formatting.
- Added platform switching between OpenRouter and Ollama.
- Integrated Whisper-tiny for local STT and Piper for local TTS.
- Implemented Markdown formatting for structured technical responses.

## Week 2: Multimodal Chat Interface
**Goal:** Build a chat interface capable of processing both text and audio inputs using OpenAI's multimodal models.
- Implemented `process_input` to handle text and audio.
- Integrated `gpt-5-mini` and `gpt-audio-mini`.
- Added real-time audio playback in Gradio.

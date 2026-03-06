# Change Log

## Week 5: TTS Testing Implementation

### Changes Made:

1. **Added pytest Framework**
   - Added pytest to requirements.txt
   - Created comprehensive TTS testing suite

2. **Test Files Created**
   - `test_tts.py`: Main test file with TTS functionality tests
   - `run_tests.py`: Test runner script for easy execution
   - `TESTING.md`: Documentation for running and understanding tests

3. **Test Features**
   - Main test: "Explain non-linearity in the context of AI engineering"
   - Tests text response generation and audio playback
   - Validates audio format, sample rate, and content
   - Includes fallback mechanism tests
   - Platform switching tests

4. **Key Test Cases**
   - `test_tts_non_linearity_explanation`: Main integration test
   - `test_tts_simple_fallback`: Tests fallback audio generation
   - `test_tts_model_loading`: Verifies Hugging Face model loading
   - `test_platform_switching`: Tests platform configuration changes

### Technical Details:
- Uses Hugging Face SpeechT5 for TTS
- Tests both Ollama and OpenRouter platforms
- Validates audio output format (16kHz, int16 numpy arrays)
- Includes error handling and fallback mechanisms

### Testing Commands:
```bash
# Run all tests
pytest test_tts.py -v

# Run specific test
pytest test_tts.py::test_tts_non_linearity_explanation -v

# Use test runner
python run_tests.py
```

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
```

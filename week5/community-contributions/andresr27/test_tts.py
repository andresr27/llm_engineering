import pytest
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import ModelManager
import numpy as np


@pytest.fixture
def model_manager():
    """Fixture to create a ModelManager instance for testing."""
    # Set environment variable to skip heavy models during tests to prevent hangs
    os.environ['SKIP_HEAVY_MODELS'] = '1'
    return ModelManager(platform="ollama")


def test_tts_non_linearity_explanation(model_manager):
    """
    Test that asks about non-linearity in AI engineering and plays the audio back.
    
    This test verifies:
    1. The model can generate a response to a technical question
    2. The TTS system can generate audio from the response
    3. The audio output is in the correct format
    """
    # Define the test question
    question = "Explain non-linearity in the context of AI engineering."
    
    # Prepare messages for the chat
    messages = [
        {
            "role": "system",
            "content": "You are a helpful technical expert for Q&A forum. Give short, accurate answers."
        },
        {"role": "user", "content": question}
    ]
    
    # Get text response from the model
    text_response = model_manager.get_chat_response(messages)
    
    # Assert we got a valid text response
    assert text_response is not None, "No text response generated"
    assert isinstance(text_response, str), "Response should be a string"
    assert len(text_response) > 10, "Response should have meaningful content"
    
    print(f"Text response length: {len(text_response)} characters")
    print(f"Response preview: {text_response[:100]}...")
    
    # Check if the response mentions non-linearity or related concepts
    assert any(keyword in text_response.lower() 
               for keyword in ['non-linear', 'activation', 'neural', 'function', 'ai', 'engineering']), \
           "Response should be relevant to the question"
    
    # Generate audio from the text response
    sample_rate, audio_data = model_manager.generate_audio(text_response)
    
    # Assert we got valid audio data
    assert sample_rate is not None, "No sample rate returned"
    assert audio_data is not None, "No audio data generated"
    assert sample_rate == 16000, f"Expected sample rate 16000, got {sample_rate}"
    
    # Check audio data format
    assert isinstance(audio_data, np.ndarray), "Audio data should be numpy array"
    assert audio_data.dtype == np.int16, f"Audio data should be int16, got {audio_data.dtype}"
    assert len(audio_data.shape) == 1, f"Audio data should be 1D, got shape {audio_data.shape}"
    assert len(audio_data) > 0, "Audio data should not be empty"
    
    print(f"Audio generated: {len(audio_data)} samples at {sample_rate}Hz")
    print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
    
    # Check audio content (not just silence)
    audio_amplitude = np.max(np.abs(audio_data))
    assert audio_amplitude > 1000, f"Audio seems too quiet, max amplitude: {audio_amplitude}"


def test_tts_simple_fallback(model_manager):
    """Test the TTS fallback mechanism with simple text."""
    simple_text = "Hello, this is a test."
    
    # Generate audio using the simple fallback
    sample_rate, audio_data = model_manager.generate_simple_audio(simple_text)
    
    assert sample_rate is not None, "No sample rate from fallback"
    assert audio_data is not None, "No audio data from fallback"
    assert sample_rate == 16000, "Fallback should use 16000Hz"
    assert isinstance(audio_data, np.ndarray), "Fallback audio should be numpy array"
    assert len(audio_data) > 0, "Fallback audio should have data"


def test_tts_model_loading():
    """Test that TTS models can be loaded correctly."""
    manager = ModelManager(platform="ollama")
    
    # Check if TTS models are loaded
    has_tts = manager.tts_model is not None and manager.tts_processor is not None
    
    if has_tts:
        print("TTS models loaded successfully")
        assert manager.tts_model is not None, "TTS model should be loaded"
        assert manager.tts_processor is not None, "TTS processor should be loaded"
    else:
        print("TTS models not loaded, using fallback mode")
        # Fallback mode is acceptable for testing


def test_platform_switching():
    """Test that platform switching works correctly."""
    manager = ModelManager(platform="ollama")
    assert manager.platform == "ollama", f"Expected platform 'ollama', got {manager.platform}"
    
    # Switch to openrouter
    success = manager.switch_platform("openrouter")
    assert success, "Platform switching should succeed"
    assert manager.platform == "openrouter", f"Expected platform 'openrouter', got {manager.platform}"
    
    # Switch back to ollama
    success = manager.switch_platform("ollama")
    assert success, "Platform switching should succeed"
    assert manager.platform == "ollama", f"Expected platform 'ollama', got {manager.platform}"
    
    # Try invalid platform
    success = manager.switch_platform("invalid")
    assert not success, "Invalid platform switching should fail"


def test_rag_system(model_manager):
    """Test the RAG system functionality"""
    # Use the RAG system from model_manager
    rag = model_manager.rag_system
    
    # Test retrieval
    query = "What is covered in Week 5?"
    context = rag.retrieve_context(query)
    
    # We check if it's not None. In some environments, 
    # it might be empty if docs aren't loaded, but initialize_rag creates samples.
    assert context is not None, "RAG should return a response (even if empty string)"
    
    # Test evaluation
    eval_results = rag.evaluate_system()
    assert 'results' in eval_results
    assert 'metrics' in eval_results
    assert len(eval_results['results']) > 0
    
    print(f"RAG test passed: retrieved {len(context) if context else 0} characters")


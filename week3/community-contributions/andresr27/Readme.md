## Installation Instructions

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd /path/to/project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (for optimal audio processing):
   ```bash
   # On Ubuntu/Debian
   sudo apt install ffmpeg
   
   # On macOS
   brew install ffmpeg
   ```

4. **Install Piper TTS** (for local Ollama deployment):
   ```bash
   uv add piper-tts
   ```

5. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_API_URL=https://openrouter.ai/api/v1
   ```

6. **Run the application**:
   ```bash
   python week_3.py
   ```

## Dependencies Overview

The project uses several key libraries:
- **Gradio**: For building the web interface
- **OpenAI Python**: For API interactions with GPT models
- **Transformers**: For Hugging Face pipelines (speech recognition)
- **PyTorch**: For model inference
- **Structlog**: For structured logging
- **SciPy/NumPy**: For audio signal processing

See `requirements.txt` for the complete list of dependencies.

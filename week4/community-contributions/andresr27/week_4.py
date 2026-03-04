import os
import base64
import torch
import scipy.signal
import numpy as np
import gradio as gr
import subprocess
import shutil
import structlog
import logging
import tempfile
import soundfile as sf
import time
from functools import wraps
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline, AutoProcessor, AutoModelForTextToWaveform
import huggingface_hub
from datasets import load_dataset
from datasets import load_dataset

# Set up structlog first
# Custom storage for UI log messages
class UILogStore:
    def __init__(self, max_messages=50):
        self.max_messages = max_messages
        self.messages = []
    
    def add_message(self, level, timestamp, message):
        # Format timing information if present
        if "duration_seconds" in str(message):
            try:
                # Extract duration from JSON if present
                import json
                msg_dict = json.loads(str(message))
                if "duration_seconds" in msg_dict:
                    duration = msg_dict["duration_seconds"]
                    message = f"{msg_dict.get('function', 'Unknown')} took {duration}s"
            except:
                pass  # Keep original message if parsing fails
                
        log_entry = f"{timestamp} - {level.upper()} - {message}"
        self.messages.append(log_entry)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_messages(self, last_n=20):
        return self.messages[-last_n:] if self.messages else []

ui_log_store = UILogStore(max_messages=50)

# Custom processor to capture logs for UI
def ui_capture_processor(logger, method_name, event_dict):
    # Add to UI log store
    ui_log_store.add_message(
        level=method_name,
        timestamp=event_dict.get("timestamp", ""),
        message=event_dict.get("event", "")
    )
    return event_dict

# Configure structlog with better formatting
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        ui_capture_processor,
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(indent=None)  # Changed from ConsoleRenderer
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger()

# Timing decorator
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            "pipeline_timing",
            function=func.__name__,
            duration_seconds=f"{duration:.2f}"
        )
        return result
    return wrapper

# Constants for model names
default_platform = "openrouter"  # Default platform

# Define platform-specific configurations
def get_platform_config(platform_name):
    """Get configuration for the specified platform"""
    config = {
        "models": {},
        "client_params": {}
    }
    
    # Load environment variables
    load_dotenv(override=True)
    
    if platform_name == "openrouter":
        config["models"]["CHAT_MODEL"] = "openai/gpt-5-mini"
        config["models"]["AUDIO_MODEL"] = None  # We'll use Hugging Face TTS instead
        config["models"]["STT_MODEL"] = "openai/whisper-tiny"
        
        # OpenRouter API configuration
        openai_api_key = os.getenv('OPENROUTER_API_KEY')
        openai_api_url = os.getenv('OPENROUTER_API_URL')
        
        if not (openai_api_key and openai_api_url):
            # Use print for initial configuration before logger is fully ready
            # But logger should be ready now
            logger.warn("OpenRouter API Key not set")
            
        config["client_params"]["base_url"] = openai_api_url
        config["client_params"]["api_key"] = openai_api_key
        
    elif platform_name == "ollama":
        config["models"]["CHAT_MODEL"] = "llama3.2"
        config["models"]["AUDIO_MODEL"] = None  # We'll use Hugging Face TTS
        config["models"]["STT_MODEL"] = "openai/whisper-tiny"
        
        # Ollama configuration
        config["client_params"]["base_url"] = "http://localhost:11434/v1"
        config["client_params"]["api_key"] = "ollama"
        
    else:
        logger.error("Invalid platform specified", platform_name=platform_name)
        return get_platform_config(default_platform)
        
    return config

# Initialize with default platform
platform_config = get_platform_config(default_platform)
CHAT_MODEL = platform_config["models"]["CHAT_MODEL"]
AUDIO_MODEL = platform_config["models"]["AUDIO_MODEL"]
STT_MODEL = platform_config["models"]["STT_MODEL"]

# Create OpenAI client with default platform
client = OpenAI(**platform_config["client_params"])

# Function to update the log box
def update_logs():
    messages = ui_log_store.get_messages(last_n=20)
    if messages:
        return "\n".join(messages)
    return "No logs yet. Start interacting with the system to see logs."

# Function to load speech recognition model
def load_speech_recognition_model(model_name):
    logger.info("Loading speech recognition model", model_name=model_name)
    try:
        device = 0 if torch.cuda.is_available() else -1
        recognizer = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            language="en",
            device=device
        )
        logger.info("Speech recognition model loaded successfully")
        return recognizer
    except Exception as e:
        logger.error("Error loading speech recognition model", error=str(e))
        return None

# Load initial speech recognition model
speech_recognizer = load_speech_recognition_model(STT_MODEL)


def generate_markdown_from_transcript(transcription):
    system_message_markdown = """
    You are a helpful technical expert for Q&A forum expanding the content of the answer.
    Give short, courteous response, no more than 200 words.
    Return a text output structure in markdown with the following format:

    ## Q&A Forum
    - **Question:** (Specify the question summarized)
    - **Answer summary:** (Specify the anwser summary)

    ## Detailed response
    - Use bullet points for each key discussion point.

    ## Takeaways
    - Use bullet points for each key takeaway.

    Ensure no code blocks are used in the output.
    """

    user_prompt_markdown = f"""
    Below is an extract of the conversation we had with an audio model.
    Please write a short explanation in markdown without code blocks, including:
    - question summary
    - answer summary
    - detailed response
    - takeaways

    Transcription:
    {transcription}
    """

    messages = [
        {"role": "system", "content": system_message_markdown},
        {"role": "user", "content": user_prompt_markdown}
    ]

    # TODO: We should use tokenizer as shown in day 5 but due to time concerns I'm calling the chat model to produce the Markdown

    # Get text response
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )

    markdown = response.choices[0].message.content

    return markdown

# Functions to process inputs and generate responses using HF pipelines
# TODO: Optimize if possible. gpt-audio-mini seems to be doing a better job at speech recognition than TTS, I get foreign langs sometimes.
# TODO: How to get a transcript summary into the chatbox instead of just the transcript
# TODO: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
def process_audio_input(text_input, audio_input):
    """Process audio input using speech recognition and return updated history and audio response."""
    if audio_input is None:
        logger.warn("No audio input provided")
        return text_input
    
    # Check if speech recognizer is available
    if speech_recognizer is None:
        logger.error("Speech recognizer not available")
        return None

    # Transcribe audio
    sr, y = audio_input

    # Convert to float32 for the model
    audio_data = y.astype(np.float32) / 32768.0  # Assuming int16 input

    # Resample to 16000 Hz if necessary (Whisper expects 16000 Hz)
    target_sr = 16000
    if sr != target_sr:
        number_of_samples = int(len(audio_data) * target_sr / sr)
        audio_data = scipy.signal.resample(audio_data, number_of_samples)
        sr = target_sr

    # Transcription
    try:
        result = speech_recognizer({"sampling_rate": sr, "raw": audio_data})
        transcribed_text = result["text"]
    except Exception as e:
        logger.error("Error during speech recognition", error=str(e))
        transcribed_text = "[Could not transcribe audio]"
    
    # Combine with text input if provided
    if text_input and isinstance(text_input, str) and text_input.strip():
        final_user_message = text_input + " " + transcribed_text
    else:
        final_user_message = transcribed_text
    
    return final_user_message


def text_to_speech_piper(text: str, model_path: str):
    """
    Generates 16-bit PCM mono audio bytes at 16000Hz using Piper.
    Uses ffmpeg if available, otherwise resamples using scipy.signal.resample.
    """
    # uv installs the 'piper' binary into the venv path automatically
    piper_path = shutil.which("piper")
    if not piper_path:
        logger.error("Piper not found", hint="Did you run 'uv add piper-tts'?")
        return None

    piper_cmd = [piper_path, "--model", model_path, "--output-raw"]

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path:
        # Use ffmpeg for resampling
        ffmpeg_cmd = [
            ffmpeg_path, "-f", "s16le", "-ar", "22050", "-ac", "1", "-i", "pipe:0",
            "-ar", "16000", "-f", "s16le", "-ac", "1", "pipe:1"
        ]
        try:
            p1 = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p1.stdout.close()

            audio_bytes, _ = p2.communicate(input=text.encode('utf-8'))

            if len(audio_bytes) % 2 != 0:
                audio_bytes += b'\x00'

            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            return (16000, samples)
        except Exception as e:
            logger.error("TTS Error with ffmpeg", error=str(e))
            # Fall through to Python resampling method
    else:
        logger.warn("ffmpeg not found, using Python-based resampling")
        logger.info("Install ffmpeg for better performance", 
                   linux_cmd="sudo apt install ffmpeg", 
                   macos_cmd="brew install ffmpeg")
    
    # Python-based resampling fallback
    try:
        # Run piper to get raw audio at 22050 Hz
        p1 = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio_bytes, _ = p1.communicate(input=text.encode('utf-8'))
        
        if len(audio_bytes) % 2 != 0:
            audio_bytes += b'\x00'
            
        # Convert to numpy array
        samples_22050 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Resample from 22050 Hz to 16000 Hz
        # Calculate number of samples for target rate
        num_samples_16000 = int(len(samples_22050) * 16000 / 22050)
        samples_16000 = scipy.signal.resample(samples_22050, num_samples_16000)
        
        # Convert back to int16
        samples_16000_int16 = samples_16000.astype(np.int16)
        
        return (16000, samples_16000_int16)
        
    except Exception as e:
        logger.error("TTS Error in Python fallback", error=str(e))
        return None


# Add near the top with other global variables
tts_model = None
tts_processor = None
tts_vocoder = None

# Set up Hugging Face authentication
def setup_hf_auth():
    """Set up Hugging Face authentication using HF_TOKEN environment variable"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is not set. Please set it to use Hugging Face models.")
        logger.info("You can get a token from: https://huggingface.co/settings/tokens")
        return False
        
    try:
        # First, try to set the token in environment
        os.environ['HF_TOKEN'] = hf_token
        
        # Try to login
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        
        # Verify token is working by trying a simple operation
        try:
            # Try to access a public model info to verify token
            from huggingface_hub import model_info
            # Use a small public model for verification
            _ = model_info("microsoft/speecht5_tts", token=hf_token)
            logger.info("Hugging Face authentication successful")
            return True
        except Exception as verify_error:
            logger.warning(f"Token verification had an issue: {verify_error}")
            # Still return True if login succeeded, as some operations might work
            logger.info("Hugging Face login succeeded, but verification had issues")
            return True
    except Exception as e:
        logger.error(f"Hugging Face authentication failed: {str(e)}")
        logger.info("Please check your HF_TOKEN and internet connection")
        return False

# Create cache directory if it doesn't exist
def setup_cache_directory():
    """Create cache directory for Hugging Face models and datasets"""
    cache_dir = "./cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Created cache directory: {cache_dir}")
    else:
        logger.info(f"Cache directory already exists: {cache_dir}")

# Call these functions early
# Note: setup_hf_auth() is now called within load_tts_model()
setup_cache_directory()

# Function to load TTS models
def load_tts_model():
    """Load TTS models and return them"""
    global tts_model, tts_processor, tts_vocoder
    
    logger.info("Loading TTS models")
    try:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        import torch
        
        # Verify HF auth first
        if not setup_hf_auth():
            logger.error("Cannot load TTS models without valid Hugging Face authentication")
            return False
            
        # Create and verify cache directory
        cache_dir = os.path.abspath("./cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        if not os.access(cache_dir, os.W_OK):
            logger.error(f"Cache directory {cache_dir} is not writable")
            return False
            
        # Set environment variables before loading models
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # Load models with explicit token
        hf_token = os.getenv("HF_TOKEN")
        model_name = "microsoft/speecht5_tts"
        vocoder_name = "microsoft/speecht5_hifigan"
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        logger.info(f"Loading TTS processor from {model_name}")
        tts_processor = SpeechT5Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=hf_token,  # Changed from use_auth_token
            trust_remote_code=True
        )
        
        logger.info(f"Loading TTS model from {model_name}")
        tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=hf_token,  # Changed from use_auth_token
            trust_remote_code=True
        )
        
        logger.info(f"Loading TTS vocoder from {vocoder_name}")
        tts_vocoder = SpeechT5HifiGan.from_pretrained(
            vocoder_name,
            cache_dir=cache_dir,
            token=hf_token,  # Changed from use_auth_token
            trust_remote_code=True
        )
        
        # Move models to device after loading
        tts_model = tts_model.to(device)
        tts_vocoder = tts_vocoder.to(device)
        
        logger.info(f"TTS models loaded successfully and moved to {device}")
        return True
                
    except Exception as e:
        logger.error("Error loading TTS models", error=str(e))
        tts_model = None
        tts_processor = None
        tts_vocoder = None
        return False

# Hugging Face TTS function
@measure_time
def text_to_speech_hf(text):
    """Generate audio from text using Hugging Face TTS"""
    global tts_model, tts_processor, tts_vocoder
    
    if text is None:
        logger.error("Text is None in text_to_speech_hf")
        return None, None
    
    try:
        # Check if models are loaded
        if tts_model is None or tts_processor is None or tts_vocoder is None:
            logger.error("TTS models not loaded properly")
            # Try to load them again
            if not load_tts_model():
                logger.error("Failed to load TTS models even after retry")
                return None, None
        
        # Ensure text is a string and add punctuation if missing
        text = str(text)
        if not text.strip().endswith(('.', '!', '?')):
            text = text.strip() + '.'
        # Limit text length to prevent issues
        if len(text) > 500:
            text = text[:500] + "..."
        
        # Add speech parameters for better control
        inputs = tts_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        speaker_embeddings = torch.zeros((1, 512)).to(device)  # Consistent voice
        
        # Generate speech with controlled parameters
        with torch.no_grad():
            speech = tts_model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=tts_vocoder
            )
        
        speech = speech.cpu().numpy()
        
        # Adjust speed and quality
        target_sample_rate = 22050  # Standard sample rate
        
        # Increase slowdown factor (from 1.1 to 1.3 for 30% slower)
        original_length = len(speech)
        target_length = int(original_length * 1.3)  # Changed from 1.1 to 1.3
        
        # Use more sophisticated resampling
        speech_resampled = scipy.signal.resample_poly(speech, target_length, original_length)
        
        # Apply stronger smoothing for more natural sound
        window_length = 11  # Increased from 5 to 11
        window = np.hanning(window_length)
        speech_smoothed = np.convolve(speech_resampled, window/window.sum(), mode='same')
        
        # Add a subtle low-pass filter to reduce robotic high frequencies
        cutoff_freq = 0.9  # Cutoff frequency as a fraction of Nyquist frequency
        b, a = scipy.signal.butter(4, cutoff_freq, btype='low')
        speech_filtered = scipy.signal.filtfilt(b, a, speech_smoothed)
        
        # Normalize volume after filtering
        speech_normalized = speech_filtered / np.max(np.abs(speech_filtered))
        
        # Add a tiny bit of reverb for more natural sound
        reverb_delay = int(0.03 * target_sample_rate)  # 30ms delay
        reverb = np.zeros_like(speech_normalized)
        reverb[reverb_delay:] = speech_normalized[:-reverb_delay] * 0.3
        speech_with_reverb = speech_normalized + reverb
        
        # Final normalization
        speech_final = speech_with_reverb / np.max(np.abs(speech_with_reverb))
        
        # Convert to 16-bit PCM with proper scaling
        speech_int16 = (speech_final * 32767).astype(np.int16)
        
        logger.info("TTS generation complete", 
                   text_length=len(text),
                   audio_length=f"{len(speech_int16)/target_sample_rate:.2f}s")
        
        return target_sample_rate, speech_int16
        
    except Exception as e:
        logger.error("TTS Error", error=str(e))
        # Fallback to a simple beep
        try:
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            frequency = 440.0
            audio_array = 0.3 * np.sin(2 * np.pi * frequency * t)
            audio_data = (audio_array * 32767).astype(np.int16)
            logger.warn("Using fallback audio generation", text_length=len(text))
            return sample_rate, audio_data
        except Exception as fallback_error:
            logger.error("Fallback TTS also failed", error=str(fallback_error))
            return None, None

# Initialize TTS models
logger.info("Attempting to load TTS models...")
tts_loaded = load_tts_model()
if not tts_loaded:
    logger.error("Failed to load TTS models - TTS functionality will be limited. Please check HF_TOKEN and internet connection.")
    # We'll use fallback methods when needed
else:
    logger.info("TTS models loaded successfully")

# Function to get TTS audio - wrapper that handles model loading state
def get_tts_audio(text):
    """Get audio from text using Hugging Face TTS, with fallback if needed"""
    if tts_loaded and tts_model is not None and tts_processor is not None and tts_vocoder is not None:
        result = text_to_speech_hf(text)
        if result is not None:
            sample_rate, audio_data = result
            if audio_data is not None:
                return sample_rate, audio_data
    
    # Fallback: generate a simple beep sound
    logger.warn("Using fallback audio generation")
    try:
        import numpy as np
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440.0
        audio_array = 0.3 * np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_array * 32767).astype(np.int16)
        return sample_rate, audio_data
    except Exception as e:
        logger.error("Fallback audio generation failed", error=str(e))
        return None, None

# Example Usage with Ollama output:
# audio = text_to_speech_piper("Hello world", "en_US-amy-medium.onnx")

@measure_time
def process_text_only_with_llama(messages):
    # Get text response
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    text_data = response.choices[0].message.content
    return text_data

@measure_time
def process_text_with_audio_response(messages, history):
    """This function is kept for compatibility but is no longer used.
    We now use Hugging Face TTS for both platforms."""
    logger.warn("process_text_with_audio_response is deprecated. Using Hugging Face TTS instead.")
    # Get text response using the chat model
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )
    text_data = response.choices[0].message.content
    
    # Generate audio using the wrapper function
    tts_result = get_tts_audio(text_data)
    audio_output = None
    if tts_result is not None:
        sample_rate, audio_data = tts_result
        if audio_data is not None:
            audio_output = (sample_rate, audio_data)
        
    return text_data, audio_output

@measure_time
def process_text_input(text_input, history, platform="ollama"):
    """Process text input and return updated history and audio response."""
    global client, CHAT_MODEL, AUDIO_MODEL, STT_MODEL
    
    # Update client and models based on selected platform
    platform_config = get_platform_config(platform)
    client = OpenAI(**platform_config["client_params"])
    CHAT_MODEL = platform_config["models"]["CHAT_MODEL"]
    AUDIO_MODEL = platform_config["models"]["AUDIO_MODEL"]
    STT_MODEL = platform_config["models"]["STT_MODEL"]

    # Messages
    system_message_audio = """
    You are a helpful technical expert for Q&A forum.
    Give short, courteous answers, no more than 20 sentences.
    Always be accurate. If you don't know the answer, say so.
    """
    messages = [{"role": "system", "content": system_message_audio}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    # Add user message
    messages.append({"role": "user", "content": text_input})

    try:
        # Get text response from the appropriate model
        if platform == "openrouter":
            # For OpenRouter, we need to use the chat model to get text response
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages
            )
            text_output = response.choices[0].message.content
        elif platform == "ollama":
            text_output = process_text_only_with_llama(messages)
        else:
            logger.error("Invalid platform specified", valid_platforms=["openrouter", "ollama"])
            text_output = "Error: Invalid platform specified."
        
        # Ensure text_output is valid
        if text_output is None:
            text_output = "No response generated."
        
        # Generate audio using Hugging Face TTS for both platforms
        tts_result = get_tts_audio(text_output)
        audio_output = None
        if tts_result is not None:
            sample_rate, audio_data = tts_result
            if audio_data is not None:
                audio_output = (sample_rate, audio_data)
            else:
                logger.warn("TTS returned None audio data")
        else:
            logger.warn("TTS failed, audio will not be available")

        # Generate markdown from transcript and update history
        markdown_content = generate_markdown_from_transcript(text_output)
        updated_history = history + [
            {"role": "user", "content": text_input},
            {"role": "assistant", "content": markdown_content}
        ]
        
        return updated_history, audio_output
    except Exception as e:
        logger.error("Error in process_text_input", error=str(e))
        error_message = f"Sorry, an error occurred: {str(e)}"
        updated_history = history + [
            {"role": "user", "content": text_input},
            {"role": "assistant", "content": error_message}
        ]
        return updated_history, None


def process_input(text_input, audio_input, history, platform="ollama"):
    """Process input by delegating to appropriate function based on input type."""
    global speech_recognizer, STT_MODEL
    
    try:
        # Update speech recognizer if platform changed
        platform_config = get_platform_config(platform)
        new_stt_model = platform_config["models"]["STT_MODEL"]
        
        if new_stt_model != STT_MODEL:
            STT_MODEL = new_stt_model
            speech_recognizer = load_speech_recognition_model(STT_MODEL)
        
        # Check if there's audio input
        if audio_input is not None:
            transcript = process_audio_input(text_input, audio_input)
            if transcript is None:
                logger.error("Failed to process audio input")
                return history, None
            text_input = transcript  # Update text input with transcribed text

        # Process final text input
        if text_input and isinstance(text_input, str) and text_input.strip():
            updated_history, audio_output = process_text_input(text_input, history, platform)
            return updated_history, audio_output
        else:
            logger.warn("No valid input provided")
            return history, None
    except Exception as e:
        logger.error("Error in process_input", error=str(e))
        error_message = f"Sorry, an error occurred: {str(e)}"
        updated_history = history + [{"role": "assistant", "content": error_message}]
        return updated_history, None


def clear_audio():
    return None

with gr.Blocks() as ui:
    gr.Markdown("# AI Tutor Chat")
    
    with gr.Row(equal_height=True):
        # Left sidebar for controls and logs
        with gr.Column(scale=2, min_width=300):  # Changed from 1.5 to 2
            gr.Markdown("## Controls")
            platform_selector = gr.Dropdown(
                choices=["ollama", "openrouter"],
                value=default_platform,
                label="Platform"
            )
            
            # Move logs into the same column below controls
            gr.Markdown("## System Logs")
            log_box = gr.Textbox(
                label="",
                lines=35,
                max_lines=40,
                interactive=False
            )
        
        # Center column for chat
        with gr.Column(scale=3, min_width=400):  # Changed from 2.5 to 3
            chatbot = gr.Chatbot(
                height=600,
                show_label=False
            )
            
            with gr.Row():
                message = gr.Textbox(
                    label="Type your message",
                    placeholder="Type your message here...",
                    scale=4
                )
                submit_text = gr.Button("Send", scale=1)
            
            with gr.Row():
                mic_in = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record Audio",
                    scale=4
                )
                submit_audio = gr.Button("Send Audio", scale=1)
            
            audio_output = gr.Audio(
                label="AI Response",
                autoplay=True,
                streaming=True
            )

    # Add platform change event handler
    def handle_platform_change(platform):
        logger.info("Platform changed", new_platform=platform)
        # Don't return anything
    
    platform_selector.change(
        fn=handle_platform_change,
        inputs=[platform_selector],
        outputs=[]
    ).then(
        fn=update_logs,
        inputs=[],
        outputs=[log_box]
    )
    
    # Update logs on load
    ui.load(
        fn=update_logs,
        inputs=[],
        outputs=[log_box]
    )

    submit_audio.click(
        process_input, inputs=[message, mic_in, chatbot, platform_selector], outputs=[chatbot, audio_output]
    ).then(lambda: "", outputs=message).then(
        update_logs, inputs=[], outputs=[log_box]
    )
    
    submit_text.click(
        clear_audio, outputs=mic_in
    ).then(
        process_input, inputs=[message, mic_in, chatbot, platform_selector], outputs=[chatbot, audio_output]
    ).then(lambda: "", outputs=message).then(
        update_logs, inputs=[], outputs=[log_box]
    )

ui.launch(inbrowser=True)

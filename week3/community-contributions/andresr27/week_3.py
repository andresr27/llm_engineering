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
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline

# Set up structlog first
# Custom storage for UI log messages
class UILogStore:
    def __init__(self, max_messages=50):
        self.max_messages = max_messages
        self.messages = []
    
    def add_message(self, level, timestamp, message):
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

# Configure structlog with processors including our custom one BEFORE the renderer
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        ui_capture_processor,  # Our custom processor before renderer
        structlog.dev.ConsoleRenderer()  # Renderer should be last
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger()

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
        config["models"]["AUDIO_MODEL"] = "openai/gpt-audio-mini"
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
        config["models"]["AUDIO_MODEL"] = None  # No audio model for Ollama
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


# Example Usage with Ollama output:
# audio = text_to_speech_piper("Hello world", "en_US-amy-medium.onnx")

def process_text_only_with_llama(messages):
    # Get text response
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    text_data = response.choices[0].message.content
    return text_data

def process_text_with_audio_response(messages, history):
    """Process text input using OpenAI's audio model and return text and audio response."""
    text_data = ""
    audio_data_base64 = ""
    
    if not AUDIO_MODEL:
        logger.warn("Audio model not available for this platform")
        return "Audio model not available for this platform.", None
    
    try:
        # The history parameter ensures we maintain context across conversation turns
        # This is crucial for coherent audio responses that reference previous exchanges
        # The messages parameter already contains the full conversation history
        # including system message, previous turns, and current user input
        with client.chat.completions.create(
                model=AUDIO_MODEL,
                modalities=["text", "audio"],  # Request both text and audio outputs
                audio={"voice": "alloy", "format": "pcm16"},
                stream=True,
                messages=messages) as stream:
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'audio') and delta.audio:
                        if 'data' in delta.audio:
                            audio_data_base64 += delta.audio['data']
                        if 'transcript' in delta.audio:
                            text_data += delta.audio['transcript']

        if not audio_data_base64:
            logger.warn("No audio data received")
            return text_data, None

        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data_base64)
        
        # Ensure even number of bytes for int16 samples
        if len(audio_bytes) % 2 != 0:
            audio_bytes += b'\x00'
            
        # Convert to numpy array of int16 samples
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Return sample rate and audio data
        audio_output = (24000, samples)
        
        if not text_data:
            logger.warn("No text data received")
            
        return text_data, audio_output
        
    except Exception as e:
        logger.error("Error processing audio response", error=str(e))
        return text_data or "Error processing response", None

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
    Give short, courteous answers, no more than 20 sentence.
    Always be accurate. If you don't know the answer, say so.
    """
    messages = [{"role": "system", "content": system_message_audio}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    # Add user message
    messages.append({"role": "user", "content": text_input})

    try:
        if platform == "openrouter":
            text_output, audio_output = process_text_with_audio_response(messages, history)
        elif platform == "ollama":
            text_output = process_text_only_with_llama(messages)
            audio_output = text_to_speech_piper(text_output, "en_US-amy-medium.onnx")
            if audio_output is None:
                logger.warn("TTS failed, audio will not be available")
        else:
            logger.error("Invalid platform specified", valid_platforms=["openrouter", "ollama"])
            text_output = "Error: Invalid platform specified."
            audio_output = None

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
    with gr.Row():
        # Left column for controls and logs
        with gr.Column(scale=1):
            platform_selector = gr.Dropdown(
                choices=["ollama", "openrouter"],
                value=default_platform,
                label="Platform"
            )
            refresh_period = gr.Number(
                value=10,
                label="Refresh period (seconds)",
                minimum=1,
                maximum=60,
                step=1
            )
            log_box = gr.Textbox(
                label="System Logs", 
                lines=15,
                max_lines=20,
                interactive=False
            )
        
        # Right column for chatbot
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)  # Increased from 400 for better visibility

    # Audio output row
    audio_output = gr.Audio(streaming=True, autoplay=True, label="AI Voice Response")

    # Input row
    with gr.Row():
        message = gr.Textbox(
            label="Type your message", 
            placeholder="Or use the mic below...",
            scale=3
        )
        submit_text = gr.Button("Send text", scale=2)
    
    # Audio input row
    with gr.Row():
        mic_in = gr.Audio(
            sources=["microphone"], 
            type="numpy", 
            label="Record Audio",
            scale=3
        )
        submit_audio = gr.Button("Send audio", scale=2)
    
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
    
    # Update logs when refresh_period changes
    refresh_period.change(
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

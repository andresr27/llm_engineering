import gradio as gr
import structlog
import logging
import time
from functools import wraps
from models import ModelManager

# Setup logging
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

def log_processor(logger, method_name, event_dict):
    ui_log_store.add_message(
        level=method_name,
        timestamp=event_dict.get("timestamp", ""),
        message=event_dict.get("event", "")
    )
    return event_dict

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-d %H:%M:%S"),
        structlog.processors.add_log_level,
        log_processor,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Timing decorator
def track_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

# Initialize model manager
model_manager = ModelManager()

def update_log_display():
    """Update the log box in UI"""
    messages = ui_log_store.get_messages(last_n=20)
    return "\n".join(messages) if messages else "No logs yet."

@track_time
def process_with_rag(text_input, audio_input, chat_history, platform, use_rag):
    """Process input with optional RAG"""
    # Update platform if changed
    if platform != model_manager.platform:
        model_manager.switch_platform(platform)
        logger.info(f"Switched to platform: {platform}")
    
    final_text = text_input.strip() if text_input else ""
    
    # Process audio if provided
    if audio_input is not None:
        transcribed = model_manager.transcribe_audio(audio_input)
        if transcribed:
            final_text = f"{final_text} {transcribed}".strip()
            logger.info(f"Transcribed audio: {transcribed[:50]}...")
    
    if not final_text:
        logger.warn("No input provided")
        return chat_history, None
    
    try:
        # Prepare chat messages
        messages = [
            {
                "role": "system",
                "content": "You are a friendly Children's Book Reader who loves telling stories."
            }
        ]
        
        # Add conversation history
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": final_text})
        
        # Get response with or without RAG
        if use_rag:
            logger.info("Initiating RAG-augmented generation")
            response_text = model_manager.get_rag_response(messages, use_rag=True)
        else:
            logger.info("Initiating standard generation")
            response_text = model_manager.get_chat_response(messages)
        
        # Generate markdown summary
        markdown_summary = model_manager.generate_markdown_summary(response_text)
        
        # Generate audio using Hugging Face TTS
        sample_rate, audio_data = model_manager.generate_audio(response_text)
        audio_output = (sample_rate, audio_data) if audio_data is not None else None
        
        # Update chat history using the correct messages format
        updated_history = chat_history + [
            {"role": "user", "content": final_text},
            {"role": "assistant", "content": markdown_summary}
        ]
        
        logger.info(f"Generated response: {len(response_text)} chars")
        return updated_history, audio_output
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        updated_history = chat_history + [[final_text, error_msg]]
        return updated_history, None

def evaluate_rag_system():
    """Evaluate the RAG system and return results"""
    try:
        if not hasattr(model_manager, 'rag_system') or model_manager.rag_system is None:
            return "# RAG Evaluation\n\nRAG system not initialized"
        
        results = model_manager.rag_system.evaluate_system()
        
        # Format evaluation report
        report = "# RAG System Evaluation Report\n\n"
        report += "## Performance Metrics\n"
        report += f"- **Total Questions Tested**: {results['metrics']['total_questions']}\n"
        report += f"- **Retrieval Success Rate**: {results['metrics']['retrieval_success_rate']}\n"
        report += f"- **Average Context Length**: {results['metrics']['average_context_length']}\n"
        report += f"- **Context Precision**: {results['metrics']['context_precision']}\n\n"
        
        report += "## Question-by-Question Analysis\n"
        for i, result in enumerate(results['results']):
            status = "[PASS]" if result['context_retrieved'] else "[FAIL]"
            report += f"### {i+1}. {status} {result['question']}\n"
            if result['context_retrieved']:
                report += f"- **Retrieved**: Yes ({result['context_length']} characters)\n"
                if result['context_preview']:
                    report += f"- **Context Preview**: {result['context_preview']}\n"
            else:
                report += "- **Retrieved**: No context found\n"
            report += "\n"
        
        report += "## Recommendations for Improvement\n"
        report += "1. **Document Coverage**: Add more documents to the `docs` folder for better coverage\n"
        report += "2. **Chunk Optimization**: Adjust chunk size (currently 500) and overlap (50) parameters\n"
        report += "3. **Query Expansion**: Consider adding synonyms to user queries\n"
        report += "4. **Metadata Enhancement**: Add more detailed metadata to document chunks\n"
        report += "5. **Regular Testing**: Run this evaluation weekly to track improvements\n"
        
        report += "\n## Next Steps\n"
        report += "- Implement more sophisticated evaluation metrics (e.g., ROUGE, BLEU)\n"
        report += "- Add user feedback collection mechanism\n"
        report += "- Consider fine-tuning the embedding model on insurance domain data\n"
        
        logger.info("RAG evaluation completed successfully")
        return report
        
    except Exception as e:
        error_msg = f"# RAG Evaluation Failed\n\n## Error Details\n```\n{str(e)}\n```\n\n## Troubleshooting\n1. Check if ChromaDB is running\n2. Verify the docs folder exists\n3. Ensure all dependencies are installed"
        logger.error(f"RAG evaluation error: {e}")
        return error_msg

def handle_platform_change(platform):
    """Handle platform selector change"""
    logger.info(f"Platform changed to: {platform}")
    model_manager.switch_platform(platform)
    
    config = model_manager.platform_config[platform]
    chat_model = config["chat_model"]
    stt_model = config["stt_model"]
    tts_model = "gpt-4o-audio-preview" if platform == "openai" else "microsoft/speecht5_tts"
    
    return chat_model, stt_model, tts_model

def clear_audio_input():
    """Clear audio input"""
    return None

# Build UI
with gr.Blocks(title="Magic Story Reader", theme=gr.themes.Soft(), css="button.primary { background-color: lightblue !important; } button.secondary { background-color: lightblue !important; }") as ui:
    gr.Markdown("# 📖 Magic Story Reader")
    gr.Markdown("Ask questions about your favorite stories and characters!")
    
    with gr.Row():
        # Left sidebar
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Settings")
            platform_select = gr.Dropdown(
                choices=["openrouter", "ollama", "openai"],
                value="openrouter",
                label="Select Platform",
                info="OpenRouter for cloud, Ollama for local"
            )
            
            # RAG Controls
            with gr.Group():
                gr.Markdown("### Story Library (RAG)")
                
                with gr.Row():
                    chat_model_display = gr.Textbox(
                        label="Chat Model",
                        value=model_manager.platform_config[model_manager.platform]["chat_model"],
                        interactive=False
                    )
                with gr.Row():
                    stt_model_display = gr.Textbox(
                        label="STT Model",
                        value=model_manager.platform_config[model_manager.platform]["stt_model"],
                        interactive=False
                    )
                with gr.Row():
                    tts_model_display = gr.Textbox(
                        label="TTS Model",
                        value="gpt-4o-audio-preview" if model_manager.platform == "openai" else "microsoft/speecht5_tts",
                        interactive=False
                    )
                
                with gr.Row():
                    use_rag = gr.Checkbox(
                        label="Enable Story Knowledge",
                        value=True,
                        info="Use the book library to answer questions about stories"
                    )
                    
                    evaluate_rag_btn = gr.Button(
                        "Evaluate RAG System",
                        variant="secondary",
                        size="sm"
                    )
                
                # Evaluation output
                rag_eval_output = gr.Markdown(
                    label="RAG Evaluation Results",
                    value="Click 'Evaluate RAG System' to see evaluation metrics",
                    elem_id="rag_eval_output"
                )
            
            gr.Markdown("### System Logs")
            log_display = gr.Textbox(
                label="",
                lines=20,
                max_lines=25,
                interactive=False,
                show_label=False
            )
        
        # Main chat area
        with gr.Column(scale=2, min_width=500):
            chatbot = gr.Chatbot(
                height=750,
                show_label=False
            )
            
            with gr.Row():
                text_input = gr.Textbox(
                    label="Message",
                    placeholder="Type your question here...",
                    scale=4,
                    container=False
                )
                send_text = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Or record audio",
                    scale=4
                )
                send_audio = gr.Button("Send Audio", variant="secondary", scale=1)
            
            audio_output = gr.Audio(
                label="AI Response",
                autoplay=True,
                interactive=False
            )
    
    # Event handlers
    platform_select.change(
        fn=handle_platform_change,
        inputs=[platform_select],
        outputs=[chat_model_display, stt_model_display, tts_model_display]
    ).then(
        fn=update_log_display,
        inputs=[],
        outputs=[log_display]
    )
    
    # Text submission
    send_text.click(
        fn=clear_audio_input,
        outputs=[audio_input]
    ).then(
        fn=process_with_rag,
        inputs=[text_input, audio_input, chatbot, platform_select, use_rag],
        outputs=[chatbot, audio_output]
    ).then(
        fn=lambda: "",
        outputs=[text_input]
    ).then(
        fn=update_log_display,
        inputs=[],
        outputs=[log_display]
    )
    
    # Audio submission
    send_audio.click(
        fn=process_with_rag,
        inputs=[text_input, audio_input, chatbot, platform_select, use_rag],
        outputs=[chatbot, audio_output]
    ).then(
        fn=lambda: "",
        outputs=[text_input]
    ).then(
        fn=update_log_display,
        inputs=[],
        outputs=[log_display]
    )
    
    # RAG Evaluation button
    evaluate_rag_btn.click(
        fn=evaluate_rag_system,
        outputs=[rag_eval_output]
    ).then(
        fn=update_log_display,
        inputs=[],
        outputs=[log_display]
    )
    
    # Initialize logs
    ui.load(
        fn=update_log_display,
        inputs=[],
        outputs=[log_display]
    )

if __name__ == "__main__":
    ui.launch(
        inbrowser=True,
        share=False,
        server_name="0.0.0.0"
    )

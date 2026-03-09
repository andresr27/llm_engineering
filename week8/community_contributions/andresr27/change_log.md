# Change Log

## Week 8: Capstone Project - RAG UI & Completion
**Goal:** Finalize the "The Price Is Right" project with a RAG-enabled user interface.
- **UI Development:** Created a Gradio-based chat interface in `capstone.py`.
- **RAG Integration:** Connected ChromaDB vector storage to provide context-aware responses.
- **Hardware Optimization:** Resolved "NotImplementedError: Unsloth cannot find any torch accelerator" by reinstalling Nvidia drivers and verifying CUDA 12.x compatibility.
- **Inference:** Optimized inference using Unsloth's 4-bit quantization to fit within local GPU memory constraints.

## Week 7: Capstone Project - Fine-Tuning & Optimization
**Goal:** Fine-tune the Llama 3.2 model for price estimation tasks using efficient PEFT techniques.
- **Fine-Tuning Setup:** Integrated `unsloth` for 2x faster 4-bit quantization fine-tuning.
- **Dataset Preparation:** Processed and pushed prompt-completion pairs to Hugging Face Hub.
- **LoRA Implementation:** Configured Low-Rank Adaptation (LoRA) with target modules for efficient parameter updates.
- **Training Pipeline:** Executed SFTTrainer with optimized hyperparameters (AdamW 8-bit, linear scheduler).
- **Experiment Tracking:** Integrated `wandb` (Weights & Biases) for real-time monitoring of training loss and GPU metrics.
- **Issue Resolved:** Successfully integrated `unsloth` for optimized 4-bit training on CUDA-compatible hardware.
- **Ongoing Issue:** Encountering `torch.OutOfMemoryError` on local GPU (3.81 GiB capacity) during inference. Attempted mitigations include `expandable_segments:True`, `max_split_size_mb:32`, and `torch.cuda.empty_cache()`, but the 3B model remains near the hardware limit.
- **Model Deployment:** Pushed the fine-tuned LoRA adapters to Hugging Face Hub.

## Week 6: Capstone Project - The Price Is Right
**Goal:** Consolidate the price prediction model and implement advanced neural network features.
- **Model Consolidation:** Migrated code from Jupyter notebooks into `capstone.py`.
- **Architecture Upgrade:** Implemented an 8-layer neural network with `BatchNorm1d` and `Dropout` for better generalization.
- **Optimization:** Switched to `AdamW` optimizer and added `CosineAnnealingLR` scheduler.
- **Evaluation:** Established baseline performance using human predictions and initial neural network runs.

## Week 5: RAG System Implementation
**Goal:** Build an expert knowledge worker using Retrieval Augmented Generation to answer course-specific questions.
- **ChromaDB Integration:** Set up persistent vector storage for course documentation.
- **Recursive Character Splitting:** Implemented advanced document chunking (size 500, overlap 50) for better context.
- **Context Retrieval:** Developed logic to augment LLM prompts with retrieved document snippets.
- **Evaluation Framework:** Created a RAG evaluation system to measure retrieval success and context precision.
- **TTS Refinement:** Replaced heavy HF models with `edge-tts` for faster, more reliable audio responses.
- **Testing Suite:** Added `pytest` framework and `run_tests.py` for system validation.

### Dependencies Added:
- chromadb>=0.5.10
- edge-tts>=6.1.12
- pydub>=0.25.1
- pytest>=8.3.0

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
```

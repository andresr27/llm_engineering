import os
import torch
import numpy as np
import scipy.signal
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline
import chromadb
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress httpx logs to avoid 404 warnings
logging.getLogger("httpx").setLevel(logging.WARNING)

class RAGSystem:
    """Retrieval Augmented Generation system for Children's Book Reader"""
    
    def __init__(self, docs_folder="books", collection_name="childrens_books"):
        self.docs_folder = docs_folder
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.initialize_rag()
        
    def initialize_rag(self):
        """Initialize the RAG system by loading documents and preparing the collection"""
        db_path = "./chroma_db"
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            
            if self.collection.count() == 0:
                logger.info("Collection empty, loading documents...")
                if not os.path.exists(self.docs_folder) or not os.listdir(self.docs_folder):
                    self.create_sample_docs()
                self.load_documents()
            
            logger.info(f"RAG system initialized with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
    
    def create_sample_docs(self):
        """Create sample documents for Children's Book Reader if book folder doesn't exist"""
        os.makedirs(self.docs_folder, exist_ok=True)
        
        # Sample book information
        sample_docs = {
            "the_magic_forest.txt": """The Magic Forest is a place where trees whisper secrets and flowers glow in the dark. 
            Leo the Lion lives here. He is a brave lion with a golden mane who loves to help his friends. 
            One day, Leo found a lost blue bird named Pip. Pip had lost his way home to the Great Oak Tree.""",

            "ocean_adventures.txt": """Deep in the sparkling blue ocean lives Marina the Mermaid. 
            She has a tail that shimmers like pearls. Her best friend is Barnaby the Bubble-blowing Whale. 
            Together they explore coral reefs and look for sunken pirate treasures hidden in the sand.""",

            "peter_rabbit.txt": """Once upon a time there were four little Rabbits, and their names were— Flopsy, Mopsy, Cotton-tail, and Peter. They lived with their Mother in a sand-bank, underneath the root of a very big fir-tree. 'Now, my dears,' said old Mrs. Rabbit one morning, 'you may go into the fields or down the lane, but don't go into Mr. McGregor's garden: your Father had an accident there; he was put in a pie by Mrs. McGregor.'"""
        }
        
        # Write sample documents to files
        for filename, content in sample_docs.items():
            filepath = os.path.join(self.docs_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Created sample documents in {self.docs_folder}")
    
    def load_documents(self):
        """Load and process documents from docs folder"""
        if not os.path.exists(self.docs_folder):
            logger.warning(f"Docs folder '{self.docs_folder}' not found")
            return 0
        
        documents = []
        
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(('.txt', '.md', '.pdf')):
                file_path = os.path.join(self.docs_folder, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Split document into chunks (simple approach)
                        chunks = self.split_document(content, chunk_size=500, overlap=50)
                        
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'id': f"{filename}_{i}",
                                'text': chunk,
                                'metadata': {
                                    'source': filename,
                                    'chunk': i,
                                    'total_chunks': len(chunks)
                                }
                            })
                            
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")
        
        # Add to vector database
        if documents:
            self.add_to_collection(documents)
            logger.info(f"Loaded {len(documents)} document chunks")
        
        return len(documents)
    
    def split_document(self, text, chunk_size=500, overlap=50):
        """Split document into overlapping chunks using recursive character splitting logic"""
        if not text: return []
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_recursive(current_text, separators):
            if len(current_text) <= chunk_size or not separators:
                return [current_text]
            sep = separators[0]
            parts = current_text.split(sep)
            chunks, current_chunk = [], ""
            for part in parts:
                if len(current_chunk) + len(part) + (len(sep) if current_chunk else 0) <= chunk_size:
                    current_chunk += (sep if current_chunk else "") + part
                else:
                    if current_chunk: chunks.append(current_chunk)
                    if len(part) > chunk_size: chunks.extend(split_recursive(part, separators[1:]))
                    else: current_chunk = part
            if current_chunk: chunks.append(current_chunk)
            return chunks

        raw_chunks = split_recursive(text.strip(), separators)
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and overlap > 0:
                chunk = raw_chunks[i-1][-overlap:] + " " + chunk
            chunks.append(chunk)
        return chunks
    
    def add_to_collection(self, documents):
        """Add documents to ChromaDB collection using default embeddings"""
        if not documents:
            return
        
        # Prepare data for insertion
        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        try:
            # Add without custom embeddings and let ChromaDB generate them using its default light model
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to collection")
                
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
    
    def retrieve_context(self, query, n_results=3):
        """Retrieve relevant context for a query"""
        try:
            # Ensure we have documents in the collection before querying
            if self.collection.count() == 0:
                logger.warning("Query attempted on empty collection")
                return None

            # Query the collection directly - ChromaDB will handle embeddings
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format the retrieved context
            context_parts = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                for i, doc in enumerate(documents):
                    if i < len(metadatas) and metadatas[i]:
                        metadata = metadatas[i]
                        source = metadata.get('source', 'unknown')
                        chunk = metadata.get('chunk', 0)
                        total_chunks = metadata.get('total_chunks', 1)
                    else:
                        source = "unknown"
                        chunk = 0
                        total_chunks = 1
                    
                    if i < len(distances):
                        distance = distances[i]
                        relevance = 1 - distance  # Convert distance to similarity
                        relevance_str = f"Relevance: {relevance:.2f}"
                    else:
                        relevance_str = ""
                    
                    context_parts.append(
                        f"[Source: {source}, Chunk {chunk+1}/{total_chunks}, {relevance_str}]\n"
                        f"{doc}\n"
                    )
            
            return "\n".join(context_parts) if context_parts else None
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            # Try alternative query method
            try:
                # Simple query without specifying include
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                if results['documents'] and len(results['documents']) > 0:
                    documents = results['documents'][0]
                    return "\n\n".join(documents)
            except Exception as e2:
                logger.error(f"Alternative query also failed: {e2}")
            
            return None
    
    def evaluate_system(self):
        """Evaluate the RAG system with book questions"""
        test_questions = [
            "Who is Leo the Lion?",
            "Where does Marina the Mermaid live?",
            "What color is Pip the bird?",
            "Who is Barnaby?"
        ]
        
        results = []
        for question in test_questions:
            context = self.retrieve_context(question, n_results=2)
            results.append({
                'question': question,
                'context_retrieved': bool(context),
                'context_length': len(context) if context else 0,
                'context_preview': context[:200] + "..." if context and len(context) > 200 else context
            })
        
        # Calculate metrics
        total = len(results)
        retrieved = sum(1 for r in results if r['context_retrieved'])
        avg_length = sum(r['context_length'] for r in results) / total if total > 0 else 0
        
        # Calculate precision score (simple version)
        precision_score = 0
        if retrieved > 0:
            # For each retrieved context, check if it contains relevant keywords
            relevant_keywords = ['llm', 'engineer', 'rag', 'retrieval', 'course', 'week']
            for result in results:
                if result['context_retrieved'] and result['context_preview']:
                    context_lower = result['context_preview'].lower()
                    keyword_matches = sum(1 for keyword in relevant_keywords if keyword in context_lower)
                    if keyword_matches >= 2:  # At least 2 relevant keywords
                        precision_score += 1
            precision_score = (precision_score / retrieved) * 100
        
        return {
            'results': results,
            'metrics': {
                'total_questions': total,
                'retrieval_success_rate': f"{(retrieved/total)*100:.1f}%" if total > 0 else "0%",
                'average_context_length': f"{avg_length:.0f} characters",
                'context_precision': f"{precision_score:.1f}%"
            }
        }

class ModelManager:
    def __init__(self, platform="openrouter"):
        self.platform = platform
        self.tts_model = None
        self.tts_processor = None
        self.tts_vocoder = None
        self.load_environment()
        self.setup_platform_config()
        self.initialize_client()
        self.load_models()
        # Initialize RAG system
        self.rag_system = RAGSystem()
        
    def load_environment(self):
        """Load environment variables"""
        load_dotenv(override=True)
        
    def setup_platform_config(self):
        """Set up platform-specific configurations"""
        self.platform_config = {
            "openrouter": {
                "chat_model": "openai/gpt-5-mini",
                "stt_model": "openai/whisper-tiny",
                "base_url": os.getenv('OPENROUTER_API_URL'),
                "api_key": os.getenv('OPENROUTER_API_KEY')
            },
            "ollama": {
                "chat_model": "llama3.2",
                "stt_model": "openai/whisper-tiny",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama"
            },
            "openai": {
                "chat_model": "gpt-4o-audio-preview",
                "stt_model": "openai/whisper-tiny",
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv('OPENAI_API_KEY')
            }
        }
        
    def initialize_client(self):
        """Initialize OpenAI client for the current platform"""
        config = self.platform_config[self.platform]
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"]
        )
        
    def load_models(self):
        """Load all required models"""
        self.load_speech_recognizer()
        self.load_tts_model()
        
    def load_speech_recognizer(self):
        """Load speech recognition model"""
        config = self.platform_config[self.platform]
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.speech_recognizer = pipeline(
                "automatic-speech-recognition",
                model=config["stt_model"],
                language="en",
                device=device
            )
        except Exception as e:
            print(f"Error loading speech recognition model: {e}")
            self.speech_recognizer = None
            
    def load_tts_model(self):
        """Load TTS model with edge-tts as primary and pyttsx3 as fallback"""
        try:
            import edge_tts
            self.tts_model = 'edge-tts'
            logger.info("Loaded edge-tts as primary TTS engine")
            return
        except ImportError:
            logger.warning("edge-tts not installed, trying fallbacks")

        try:
            import pyttsx3
            self.pytts_engine = pyttsx3.init()
            self.pytts_engine.setProperty('rate', 150)
            self.tts_model = 'pyttsx3'
            logger.info("Loaded pyttsx3 as fallback TTS engine")
        except Exception as e:
            logger.error(f"All TTS engines failed to load: {e}")
            self.tts_model = None
            
    def transcribe_audio(self, audio_input):
        """Transcribe audio input to text"""
        if not self.speech_recognizer or audio_input is None:
            return None
            
        sr, y = audio_input
        audio_data = y.astype(np.float32) / 32768.0
        
        # Resample to 16000 Hz if necessary
        target_sr = 16000
        if sr != target_sr:
            number_of_samples = int(len(audio_data) * target_sr / sr)
            audio_data = scipy.signal.resample(audio_data, number_of_samples)
            sr = target_sr
            
        try:
            result = self.speech_recognizer({"sampling_rate": sr, "raw": audio_data})
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
            
    def get_chat_response(self, messages):
        """Get chat response from LLM"""
        config = self.platform_config[self.platform]
        
        try:
            response = self.client.chat.completions.create(
                model=config["chat_model"],
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Chat API error: {e}")
            return f"Error: {str(e)}"
            
    def generate_markdown_summary(self, text):
        """Generate markdown summary of the story or book being read"""
        system_message = """
        You are a magical Children's Book Storyteller.
        Create a charming and engaging summary of the story part just read.
        Use warm, enchanting language suitable for children.
        Format response as markdown with:
        
        # ✨ Story Time Summary
        - **The Adventure:** [brief summary of what happened]
        - **Magic Moments:** [key highlights]
        
        ## 📖 Story Details
        - [enchanting bullet points about characters and setting]
        
        ## 🌟 Lessons & Wonders
        - [simple takeaways for a child]
        
        No code blocks or technical jargon in output.
        """
        
        user_prompt = f"Create a markdown summary of this conversation:\n\n{text}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.get_chat_response(messages)
        
    def generate_audio(self, text):
        """Generate audio using available TTS method"""
        if not text:
            return self.generate_simple_audio(text)
        
        text = str(text).strip()
        if len(text) > 1000:
            text = text[:1000] + "..."

        if self.tts_model == 'edge-tts':
            try:
                import asyncio
                import edge_tts
                import tempfile
                import wave
                import io
                from pydub import AudioSegment

                async def _generate():
                    communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp_path = tmp.name
                    await communicate.save(tmp_path)
                    return tmp_path

                mp3_path = asyncio.run(_generate())
                
                # Convert mp3 to wav for Gradio/Numpy processing
                audio = AudioSegment.from_mp3(mp3_path)
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                
                with wave.open(wav_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_np = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
                
                os.unlink(mp3_path)
                return sample_rate, audio_np
            except Exception as e:
                logger.error(f"edge-tts generation failed: {e}")

        if self.tts_model == 'pyttsx3':
            try:
                import tempfile
                import wave
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_name = tmp.name
                self.pytts_engine.save_to_file(text, tmp_name)
                self.pytts_engine.runAndWait()
                with wave.open(tmp_name, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_np = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
                os.unlink(tmp_name)
                return sample_rate, audio_np
            except Exception as e:
                logger.error(f"pyttsx3 generation failed: {e}")

        return self.generate_simple_audio(text)
        
    def generate_simple_audio(self, text):
        """Generate simple audio response (fallback)"""
        if not text:
            return None, None
            
        sample_rate = 24000
        duration = min(len(text.split()) * 0.5, 5.0)  # Max 5 seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a simple tone
        frequency = 220.0  # A lower frequency for better sound
        audio_array = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some variation
        audio_array += 0.1 * np.sin(2 * np.pi * 440.0 * t)
        
        # Convert to int16
        audio_data = (audio_array * 32767).astype(np.int16)
        
        return sample_rate, audio_data
        
    def get_rag_response(self, messages, use_rag=True):
        """Get chat response with RAG augmentation"""
        if not use_rag or not self.rag_system:
            return self.get_chat_response(messages)
        
        try:
            # Extract the last user message
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if not user_messages:
                return self.get_chat_response(messages)
            
            last_query = user_messages[-1].get('content', '')
            if not last_query:
                return self.get_chat_response(messages)
            
            # Retrieve relevant context
            context = self.rag_system.retrieve_context(last_query)
            
            if context:
                # Create augmented system prompt
                rag_prompt = f"""You are a friendly and magical Children's Book Storyteller. 
Use the following stories to answer the child's question in an engaging and warm way:

{context}

Instructions:
1. Base your answer on the stories provided.
2. Use simple, enchanting language suitable for children.
3. Be warm, encouraging, and creative.
4. If you're talking about a character from the book, make them sound exciting!

Now answer the child's question:"""
                
                # Replace or augment the system message
                augmented_messages = []
                for msg in messages:
                    if msg.get('role') == 'system':
                        # Replace system message with RAG prompt
                        augmented_messages.append({
                            'role': 'system',
                            'content': rag_prompt
                        })
                    else:
                        augmented_messages.append(msg)
                
                logger.info(f"RAG context retrieved: {len(context)} chars")
                return self.get_chat_response(augmented_messages)
            else:
                logger.info("No RAG context found, using standard response")
                return self.get_chat_response(messages)
                
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return self.get_chat_response(messages)
        
    def switch_platform(self, new_platform):
        """Switch to a different platform"""
        if new_platform in self.platform_config:
            self.platform = new_platform
            self.initialize_client()
            self.load_speech_recognizer()
            return True
        return False

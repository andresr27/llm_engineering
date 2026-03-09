# AI bootcamp notes

## Course 1: AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents
Udemy:
https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/learn/lecture/52932247#overview




### WEEK 1

### Recap

1. Every call to an LLM is stateless
2. We pass in the entire conversation so far in the input prompt, every time
3. This gives the illusion that the LLM has memory - it apparently keeps the context of the conversation
4. But this is a trick; it's a by-product of providing the entire conversation, every time
5. An LLM just predicts the most likely next tokens in the sequence; if that sequence contains "My name is Ed" and later "What's my name?" then it will predict.. Ed!


### WEEK 2

Model Comparison, inference vs refernece reasoning


### WEEK 3


In week three of the course, the key takeaways focused on tokenizers and their critical role in language models (LLMs). Here’s a breakdown of the main points:

Definition of Tokenizers: Tokenizers convert natural language into tokens, which are smaller text chunks. These tokens can represent parts of words, entire words, or pairs of words.

Token IDs: Each token is assigned a unique numerical identifier (token ID) used by the model for processing input.

Tokenizer's Dictionary: The lecture discussed the tokenizer's dictionary, which includes all possible tokens and their corresponding IDs, along with special tokens that guide the model’s understanding.

Importance of Special Tokens: The consistent use of special tokens in training data allows the model to effectively learn and recognize patterns.

Varied Tokenization Strategies: Different LLMs may use different tokenization strategies, affecting both the number of tokens generated and the quality of the model's output.

Distinction Between Tokens and Vectors: The lecture clarified that tokens are different from vectors, which are derived from deeper layers of the neural network post-tokenization.

Practical Application: The week concluded with a practical session in Google Colab, allowing learners to work with different tokenizers, setting the foundation for more advanced topics ahead.

This week provided essential insight into how LLMs process text, laying a groundwork for future learning. If you have any specific questions or need further clarification on any of these points, feel free to ask!



#### difference between transformers and pipelines:

Transformers:

Transformers are a type of deep learning architecture specifically designed to handle sequential data, particularly in the context of Natural Language Processing (NLP). They utilize mechanisms like self-attention to process input data, allowing the model to weigh the importance of different words in a sentence relative to each other.
They are typically complex structures made up of layers, which include encoders and decoders, enabling the model to understand and generate language based on context.
Pipelines:

Pipelines are a higher-level abstraction that provides an easier interface for utilizing various machine learning models, including transformers. They allow users to perform specific tasks (like text generation, translation, or summarization) without needing to understand the internal workings of the models.
Using pipelines, you can quickly set up and execute a pre-trained model with minimal code to accomplish specific NLP tasks. For instance, Hugging Face’s libraries provide a variety of predefined pipelines that can leverage transformer models for different applications.
In summary, transformers are the underlying architecture that enables advanced language processing, while pipelines offer a simplified way to access and use these powerful models in practical applications. If you have more specific aspects you’d like to explore or any questions, feel free to ask!

#### Day 3

On Day 3 of Week 3, the lecture titled "Tokenizers in Action: Encoding and Decoding with Llama 3.1" focused on the practical application of tokenizers, particularly using the Llama 3.1 model from Meta. Here are the key points:

Accessibility: The instructor emphasized that the lab can be run on personal computers and does not require advanced GPU resources, making it accessible for all learners.

Acceptance of Terms: There was a discussion on the importance of accepting the terms of service for using the Llama model, including steps to gain access through Hugging Face. Users were advised to provide an educational purpose to expedite the approval process.

Creating a Tokenizer: The technical part involved creating a tokenizer using the Hugging Face library with the AutoTokenizer class and the from_pretrained method to retrieve the pre-trained Llama 3.1 model.

Encoding Process: The lecture explained the encoding process where text is converted into token IDs. The instructor demonstrated with a practical example, explaining how characters, words, and tokens are related, mentioning that approximately 0.75 of a word corresponds to a token.

Decoding Process: Participants learned about the decoding process, converting token IDs back into text, with attention to special tokens that signal the start of text for the model.

Batch Decoding: The session concluded with a demonstration of batch decoding, showcasing how individual tokens are represented and the importance of token structure relative to word fragments.

This lecture provided a foundational understanding of tokenization and its critical role in working with language models. If you have any more questions or need clarification on specific points, let me know!


#### Day 4

On Day 4 of Week 3, the focus was on understanding and experimenting with different types of tokenizers, particularly in relation to coding languages. Here are the key highlights:

Experimentation with Tokenizers: Participants were encouraged to engage in hands-on experimentation with various tokenizers. This practical approach aimed to help learners understand how different tokenizers operate and their effectiveness in handling both text and programming languages.

Identifying Unique Tokens: Learners were tasked with finding unique tokens formed from certain rare words. This activity helped illustrate how tokenization can vary in complexity and how specific words may be represented as single tokens in some tokenizer implementations.

Efficiency in Code Tokenization: The session highlighted how specialized tokenizers can be more efficient when dealing with coding languages due to their ability to better understand syntactical structures and conventions inherent to programming.

Preparation for Advanced Topics: The day concluded with a review of key concepts in preparation for upcoming lectures that would delve deeper into more complex ideas related to models and their functionalities.

Efficient tokenization is crucial for optimizing the performance of language models, particularly when dealing with complex inputs like programming languages or specialized text. Here are some key aspects of efficient tokenization:

Mapping Text to Tokens: Efficient tokenization involves transforming words or fragments of words into numerical representations, referred to as token IDs. This mapping allows language models to process data, as they require numerical inputs for computations. The tokenizer's job is to turn natural language into these numbers that the model can understand.

Fragmentation of Words: Instead of mapping entire words, efficient tokenizers often break text down into smaller parts or fragments (tokens). This "chunking" helps maintain the model's vocabulary within a manageable size, reducing issues like running out of vocabulary or accurately representing unique terms. For instance, instead of a full word being a token, smaller segments may be used efficiently, improving both speed and accuracy.

Handling Syntax in Programming: Specialized tokenizers for coding languages are engineered to recognize and efficiently process syntactical structures, improving parsing accuracy and model performance during tasks such as code generation or comprehension.

Cost and Resource Management: While the primary role of tokenizers is not to reduce costs, an efficient tokenization process streamlines the inputs for the model, ensuring that memory and processing power are effectively utilized. This efficiency enables better performance even on lower-resource machines.

Applicability in Different Contexts: Efficient tokenization is essential across various applications, guiding the development of systems like chatbots, code interpreters, or any LLM-based applications, where understanding and processing the input correctly is critical for generating desired outputs.

#### Tokenization efficiency

Efficient tokenization is crucial for optimizing the performance of language models, particularly when dealing with complex inputs like programming languages or specialized text. Here are some key aspects of efficient tokenization:

Mapping Text to Tokens: Efficient tokenization involves transforming words or fragments of words into numerical representations, referred to as token IDs. This mapping allows language models to process data, as they require numerical inputs for computations. The tokenizer's job is to turn natural language into these numbers that the model can understand.

Fragmentation of Words: Instead of mapping entire words, efficient tokenizers often break text down into smaller parts or fragments (tokens). This "chunking" helps maintain the model's vocabulary within a manageable size, reducing issues like running out of vocabulary or accurately representing unique terms. For instance, instead of a full word being a token, smaller segments may be used efficiently, improving both speed and accuracy.

Handling Syntax in Programming: Specialized tokenizers for coding languages are engineered to recognize and efficiently process syntactical structures, improving parsing accuracy and model performance during tasks such as code generation or comprehension.

Cost and Resource Management: While the primary role of tokenizers is not to reduce costs, an efficient tokenization process streamlines the inputs for the model, ensuring that memory and processing power are effectively utilized. This efficiency enables better performance even on lower-resource machines.

Applicability in Different Contexts: Efficient tokenization is essential across various applications, guiding the development of systems like chatbots, code interpreters, or any LLM-based applications, where understanding and processing the input correctly is critical for generating desired outputs.

Overall, efficient tokenization ensures that input data is not only accurately represented for model consumption but is also processed in a way that optimizes performance across diverse applications. If you need further insights or specifics on this topic, let me know!



#### Day 5

In Week 3, Day 5 of the course, the lecture focused on the architecture of the LLaMA model, specifically its 16 decoder layers. It started with an overview of the embedding layer, which transforms input data into a 2048-dimensional space.

Key points covered include:

Decoder Layers Components: Each decoder layer consists of three main parts: the attention layer, the multi-layer perceptron (MLP), and layer normalization.

Attention Layer: This component is crucial for determining which information from previous layers should be emphasized. It utilizes four parameters: query, key, value, and output, which are learned over time to enhance performance.

MLP Functionality: The MLP expands the dimensionality from 2048 to 8000 and then reduces it back to 2048. It includes a gating mechanism to filter important information.

Importance of Non-Linearity: The lecture discussed the significance of non-linear activation functions, specifically the Selu (sigmoid linear unit), to ensure meaningful contributions from all parameters in the network. This non-linearity enables more complex data representations.

Implications of Non-Linearity: Incorporating non-linear elements allows the LLaMA architecture to fully utilize its parameters, improving its ability to generate plausible next tokens compared to oversimplified linear combinations.

Overall, this session provided a deep understanding of LLaMA's decoder layers and the vital roles of attention and non-linearity in enhancing the model's intelligence. If you have any specific questions or need further clarification on this topic, feel free to ask!

#### NON-LINEARITY

Non-linearity is a crucial concept in the architecture of neural networks, particularly in the context of the LLaMA model. It primarily relates to how activation functions are utilized within the model to allow for more complex data representations.

Activation Functions: Non-linearity is introduced through activation functions, which are mathematical operations applied to the output of neurons in a network. A common choice is the Selu (Scaled Exponential Linear Unit), which helps maintain the mean and variance of the inputs, effectively normalizing the outputs across layers.

Benefits of Non-linearity:

Rich Representations: By allowing the model to learn non-linear relationships, it can capture intricate patterns in the data that linear combinations would miss. This is essential for tasks like language generation, where the relationships between words and context are often complex and multi-dimensional.
Full Utilization of Parameters: With non-linear activation, all the parameters in the network can contribute meaningfully to the final output. In contrast, linear combinations may lead to oversimplified models that cannot leverage the full capacity of the network.
Complexity Over Simplification: Non-linearity ensures that the network can combine inputs in various ways, allowing it to generate plausible next tokens—an essential feature for language models. Without non-linear transformations, the model may fail to account for the richness of language and the subtleties of context.

In summary, non-linearity enables the LLaMA model to enhance its performance by incorporating activation functions that introduce complexity and allow for a richer interaction between layers. This capability is vital for improving the quality of the generated text and making the model more intelligent overall. If you need further clarification or have specific questions about non-linearity, feel free to ask!

#### Homework Assignment

For Week 3, the homework assignment typically involves creating a personal tutor tool similar to what was outlined in Week 1 and Week 2. Here are the key points about this assignment:


**Objective**: Build a tool that serves as your personal guide during the course. This tool should be able to answer questions regarding code, LLMs, or any other relevant topics you encounter.

** Pre requirements**:
- Familiarity with Python and the Hugging Face library for working with LLMs.
- Access to both GPT models (via API) and the Llama model running locally on your machine.

** Current Progress**:


**Implementation Steps Required**:
- Set up your coding environment to interact with both GPT models and the open-source Llama model running locally.
- Fill in the code to enable user interaction where a user can input a question for your tutor tool to answer.
- Structure the responses in Markdown for clearer presentation.
- Expected Experience: This assignment is designed to help reinforce concepts learned, through practical coding related to AI and model interactions.

Additional Resources: Check the community contributions in the course folders for examples of solutions provided by other students, which can help guide your implementation.

If you need specific details or further clarifications about the homework, feel free to ask!

#### Comments on the solution

Option 1: Open-Source Hugging Face Pipelines (e.g., Whisper)
Pros:

Cost-Effective: Generally free to use, especially when running on a Colab GPU. You're not paying per minute or per API call.
Data Privacy & Control: Your audio data is processed locally on your Colab instance (or your machine), giving you more control over sensitive information and reducing reliance on third-party servers for processing.
No API Limits: You're not subject to external API rate limits or file size restrictions (though your Colab runtime's memory/disk limits still apply).
Customization: For advanced users, open-source models can often be fine-tuned on specific datasets to improve accuracy for niche vocabulary or accents.
Cons:

Resource Intensive: Requires a GPU and sufficient RAM to run efficiently. This can sometimes be a bottleneck in free Colab tiers or on local machines.
Setup Complexity: Can involve more initial setup, including installing libraries, downloading model weights, and managing dependencies.
Performance Variability: While very good, the quality and speed might sometimes vary compared to highly optimized commercial services, especially for very challenging audio.
Maintenance: You are responsible for managing the model and ensuring its environment is set up correctly.
Option 2: OpenAI API (or other commercial APIs via OpenRouter)
Pros:

Ease of Use: Typically involves simple API calls, requiring less code and setup once authenticated. You send the audio, and you get the text back.
High Performance & Accuracy: Commercial APIs often use very large, powerful, and continuously improved models that deliver high accuracy and fast transcription speeds.
Scalability: Designed for high-volume, scalable transcription without you needing to manage underlying hardware or infrastructure.
Managed Service: The API provider handles model updates, infrastructure, and resource allocation.
Cons:

Cost: Involves API costs, which can add up, especially with frequent or large audio files. You pay per minute or per token.
API Limits: Subject to rate limits and payload size limits (as you experienced with the 'Request Entity Too Large' error). For very large files, you might need to chunk the audio manually.

Data Privacy Concerns: Your audio data is sent to a third-party server for processing, which might be a concern for highly sensitive or proprietary information.
Dependency on External Service: Reliability and uptime depend entirely on the API provider.
Less Customization: You have less control over the underlying model's behavior or direct ability to fine-tune it.


### WEEK 4

Here’s a summary of the week along with key aspects to remember for correct advancement in the course:

Week Summary
AI Evaluation: Understanding different AI leaderboards such as GLUE, SuperGLUE, and others is crucial for evaluating model performance.
Prompting Techniques: Develop effective prompting strategies and learn to format outputs using Markdown to improve clarity in responses.
Hands-On Experience: Engage in practical exercises, building solutions and understanding key concepts related to neural networks.
Commercial Applications: Focus on real-world implementations such as summarization techniques that can be applied in various business contexts.
Key Aspects to Remember
Layered Learning: Each week is designed to build on previous weeks' knowledge. It's essential to grasp foundational skills before moving forward.

Practical Application: Emphasize applying skills in practical scenarios. Consider how to utilize AI techniques for impactful business solutions, such as summarization of documents or developing specific applications related to your job.

Engagement in Exercises: Actively participate in exercises and apply the learned techniques. This real-world practice is vital for retaining knowledge and skill development.

Continued Interaction: Stay connected with the course community, share your progress, and engage with instructors if you face difficulties. Collaboration and communication can enhance the learning experience.

Seek Feedback: As you create projects or apply your learning, seek feedback to identify strengths and areas for improvement. Utilize resources available throughout the course.

Reflect on these key aspects, and consider how each contributes to building expertise in AI engineering. This will assist you in gaining the most from the course as you advance through the material. If you have any specific questions or need further clarification, feel free to ask!

#### day 1

Here’s a concise, bullet-point summary of "Week 4 Day 1" using bold text for key concepts:

Introduction to AI Evaluation:

Explores the importance of evaluating language models (LLMs) effectively.
AI Leaderboards:

Discusses five key leaderboards used for model evaluation:
GLUE: General Language Understanding Evaluation.
SuperGLUE: A more challenging benchmark for LLMs.
SQuAD: Stanford Question Answering Dataset, focused on reading comprehension.
HellaSwag: Evaluates models' commonsense reasoning abilities.
LAMBADA: Assesses understanding of contexts using a fill-in-the-blank format.
Evaluating Model Performance:

Highlights the significance of comparative evaluations in determining model strengths and weaknesses.
Importance of understanding context windows and token limitations in model performance.
Cost of API Usage:

Overview of the costs associated with using various LLM APIs.
Encourages consideration of efficiency and resource management when deploying models.
Closing Thoughts:

Emphasizes the foundational knowledge necessary to engage with models effectively.
Encourages students to reflect on what makes different LLMs the "best" for specific tasks.
For any related homework or assignments, please refer directly to the course materials or announcements, as specifics were not mentioned in the context provided. If you have more questions or need further details, feel free to ask!

#### day 2

Here’s a concise, bullet-point summary of "Week 4 Day 2" using bold text for key concepts:

Effective Prompting:

Importance of crafting prompts to generate accurate model responses.
Defines "single-shot prompting": providing one example to guide response.
Formatting Responses:

Emphasizes the use of Markdown to structure model outputs.
Enhances readability and organization in generated content.
Practical Demonstration:

Demonstrated prompting the model for webpage summarization.
Highlights how prompting affects the quality of responses.
System vs. User Prompts:

The system prompt sets the conversation's tone and context.
The user prompt serves as the actual conversation starter; it’s critical for guiding the model's output.
Optimal Results:

Clarity and specificity in prompts lead to better performance.
Encourages experimentation with prompts to see different responses.
This summary captures the core elements discussed in the lecture, helping you grasp the key concepts and techniques relevant to effective prompting in language models. If you have more questions or need details on a specific aspect, feel free to ask!

Was this content relevant to you?

#### day 3

Here’s a concise, bullet-point summary of "Week 4 Day 3" using bold text for key concepts:

Frontier AI Models:

Overview of leading models in 2024: GPT-4, Claude, Gemini, and Llama.
Discussion on the key differences between these models.
Model Capabilities:

Comparison of capabilities and use cases for each AI model:
GPT-4 excels at content generation.
Claude shows strengths in business applications.
Gemini and Llama are explored for their unique features.
Applications:

Analysis of which model is best suited for tasks like coding, summarization, and general business needs.
Insights into context window sizes and computational requirements.
Open Source vs Proprietary:

Evaluates strengths and limitations of open-source models (e.g., Llama) compared to proprietary ones.
Critical insights into real-world applications of models like Claude 3.5 and its advanced capabilities.
Recent Developments:

Special focus on improvements in capabilities of models, including new benchmarks and performance in specific domains.
As for homework, please check the course materials or announcements for assignments related to this lecture, as specific homework details were not provided in the snippets. If you have any more questions or need clarification on a specific topic, feel free to ask!

#### day 4
Here’s a concise, bullet-point summary of "Week 4 Day 4" using bold text for key concepts:

Effective Prompting Techniques:

Focus on refining prompts to achieve precise outputs from the AI. Single vs multiple shot!
Encouraged repetitive phrasing in prompts to increase adherence to the task.
Markdown Formatting in Responses:

Demonstrated how to format outputs in Markdown for clarity and organization.
Highlights included using headings and bullet points to enhance readability.
Example Demonstration:

Showcased a real-world application of summarizing a webpage effectively.
Emphasized the integration of user prompts and system messages to guide the model accurately.
Quantization and Data Types:

Introduced bits and bytes quantization to improve data representation.
Explained the nf4 data type, enabling compact floating-point representation.
Practical Application:

Discussed creating minutes of meetings from transcripts using structured prompts.
Importance of including summaries, discussion points, and action items in the formatted output.
For any related homework or assignments, please refer to the course materials or announcements, as specific homework details were not mentioned in the content provided. If you have more questions or need clarification on a specific topic, feel free to ask!

#### day 5

Here’s a concise, bullet-point summary of "Week 4 Day 5" using bold text for key concepts:

Review of Week's Learning:

Reflection on key concepts learned throughout the week, emphasizing AI evaluation and model selection.
Overview of Topics Covered:

Covered the foundations of AI and generalization.
Discussed how to handle different datasets effectively for AI applications.
Reviewed techniques in fine-tuning models and understanding their performance through evaluation metrics.
Hands-On Activities:

Encouraged practical engagement with coding exercises, such as building JSON files and experimenting with model output.
Students should have gained experience running models in batch mode and performing basic neural network construction.
Commercial Applications:

Worked on real-world problems, specifically focusing on pricing strategies based on textual descriptions.
Developed a test harness to evaluate model outputs against predefined metrics.
Preparation for Upcoming Content:

Anticipation for the next week's focus on machine learning, neural networks, and integrating solutions to build comprehensive applications.
Mention of integrating RAG techniques and evaluations in the following sessions.
As for any homework assignments, please refer to the course materials or announcements for details since specific assignments were not highlighted. If you have further questions or need clarity on a particular topic, feel free to ask!

#### Homework Assignment

For homework this week, you are expected to build a tool that acts as your personal tutor. Here are the key points about the assignment:

Objective: Create a program that accepts user input to answer questions related to code, LLMs, and other topics encountered throughout the course.

Implementation:

Utilize both GPT and an open-source Llama model running on your computer.
Code a section to set up your environment for asking questions.
Enter a question related to your code and receive a response formatted in Markdown for clarity.
Expected Outcomes:

Experiment with receiving answers from both models.
Gain familiarity with the coding constructs used throughout the course.
Additional Resources: You can find community-contributed solutions for mini-projects in the course folders, which can aid your understanding and implementation of similar tasks.

If you need more specific details or further assistance with this assignment, feel free to ask!


### WEEK 5

In Day 1 of the course, several videos focus on the fundamentals of Retrieval Augmented Generation (RAG). Here’s a summary of the key content covered in those videos:

Overview of RAG: The instructor introduces RAG and explains its significance in enhancing the performance of language models by enriching prompts with relevant information.

Key Concepts: The big idea behind RAG is emphasized, focusing on how it differs from traditional methods of answering questions, such as simple SQL queries or text searches. It highlights the need for using language models to convert queries into vectors for better information retrieval.

Toy Version Implementation: A demonstration of implementing a toy version of RAG is provided, allowing students to gain a hands-on experience with the fundamental concepts discussed. This helps solidify understanding before moving on to advanced applications.

Practical Example: An example from an insurance tech startup illustrates how RAG can be applied to extract relevant information to answer expert questions from shared documents, showcasing the practical utility of the methodology.

User Interface with Gradio: The instructor mentions that students will learn to create a user interface using Gradio that can process questions and showcase the documents from which expertise is sourced, promoting interactive user experiences.

Foundation Recap: The importance of mastering foundational concepts is reiterated, ensuring that students have a solid grasp on RAG before diving into more complex implementations in the coming weeks.

This comprehensive approach ensures that students understand not only the theory behind RAG but also its practical implementations and potential applications in various fields.



#### DAY 1

In Day 1 of Week 5, the focus is on the fundamentals of Retrieval Augmented Generation (RAG) and its integration with various AI tools and models. Here’s a detailed summary highlighting specific terminologies and tools:

Introduction to RAG: The day begins with an overview of RAG and its significance in enhancing the capabilities of language models. It discusses how RAG combines traditional retrieval systems with generative models to improve response quality.

Key Tools and Frameworks:

LangChain: The concept of LangChain is introduced as a framework designed to facilitate the development of applications that leverage RAG efficiently. It provides tools for managing prompts and connecting different components seamlessly.
Hugging Face: The use of Hugging Face models is emphasized, showcasing its library as a resource for accessing pre-trained models that can be utilized in RAG systems.
Vector Databases: The importance of vector databases for storing and retrieving embeddings is discussed, focusing on how these databases enhance search efficiency by allowing semantic retrieval of documents based on user queries.
Data Handling and Processing: The lecture highlights techniques related to data sets and the evaluation of success in RAG implementations. It mentions the importance of context windows and token management, crucial for optimizing model input and output.

Coding Exercises: Practical coding exercises are incorporated, enabling students to apply RAG concepts by building a question-answering platform. The use of APIs in conjunction with frontier models sets a foundation for understanding more complex interactions.

Discussion on Future Applications: The day concludes with a look ahead at the integration of more advanced features and models, setting the stage for upcoming lessons on deploying these concepts in real-world scenarios.

Overall, Day 1 lays the groundwork for understanding RAG in the context of AI, providing essential knowledge on the terminology and tools that underpin its implementation.


#### Day 2
In Day 2 of Week 5, the course focuses on the evaluation aspects of Retrieval Augmented Generation (RAG) and explores various tools and methodologies to assess performance. Here’s a summary highlighting specific AI terminology and tools:

Importance of Evaluations: The day emphasizes the critical role of evaluations in determining the effectiveness of RAG systems. The instructor discusses how to gauge the performance of both the retrieval process and the quality of responses generated by the model. This includes assessing how accurately the system retrieves relevant context and whether the answers provided align with user expectations.

Quantitative Performance Metrics: Students learn about employing quantitative metrics to evaluate RAG systems. They are introduced to concepts such as benchmarks, which allow for assessing the models’ effectiveness in retrieving and answering queries. This systematic approach enhances the scientific rigor of evaluating AI systems.

Key Frameworks and Techniques:

LangChain: Building upon knowledge from Day 1, LangChain is utilized to structure RAG implementations effectively, allowing for streamlined management of different components, including question answering and retrieval.
Chroma: The functionality of Chroma in storing and managing vector embeddings is discussed, ensuring efficient retrieval across the RAG framework.
Experimenting with Chunkers and Encoders: The course encourages hands-on experimentation with different chunkers and encoders to find the most suitable configurations for specific tasks. This experimentation is framed around a scientific approach, supporting confidence in the results and decisions made.

Preview of Advanced Techniques: The instructor hints at the introduction of advanced RAG techniques in subsequent sessions, which may include query rewriting, document rewriting, and reranking strategies. This prepares students for more complex enhancements to their RAG systems.

Overall, Day 2 provides a comprehensive understanding of how to evaluate and improve the effectiveness of RAG implementations, reinforcing the essential tools and methodologies for achieving optimal performance in AI systems.

#### Day 3
In Day 3 of Week 5, the course delves deeper into the practical implementation of Retrieval Augmented Generation (RAG), focusing on vectors, LangChain, and the interaction of different components. Here’s a summary emphasizing specific AI terminology and tools:

Advanced Concepts in Vectors: The day begins with an exploration of vectors tailored for RAG applications. The focus is on how vectors are fundamental for semantic searches, allowing for more nuanced retrieval by representing user inquiries and documents in a high-dimensional space.

LangChain in Detail: The instructor revisits LangChain, shedding light on its capabilities and how it abstracts the complexity of integrating various AI components. Students learn to leverage LangChain to streamline the creation of embeddings, enabling easier manipulation of vectors derived from their knowledge bases.

Encoding and Chunking: Critical principles like chunking documents into smaller segments and converting these chunks into vectors using popular encoding models are covered. This step is essential for efficiently managing and retrieving relevant information, allowing the model to perform fuzzy searches based on user queries.

Using Chroma for Vector Storage: The practical aspect of storing and visualizing vectors in Chroma is introduced. This tool enables users to maintain and retrieve vector representations effectively, showcasing the relationships between different data points and enhancing understanding of the retrieval process.

Interactive Learning and Applications: Students engage in hands-on activities where they chunk documents, encode them, and store vectors. This promotes a deeper comprehension of the entire process and prepares them for real-world applications of these systems.

Comparison of AI Models: Additionally, there’s a segment comparing frontier AI models such as GPT-4, Claude, Gemini, and Llama, discussing their capabilities and limitations. This context helps students understand how different models perform across various tasks, including coding, summarization, and more specialized applications.

Overall, Day 3 equips students with the skill set to implement RAG effectively, fostering a strong foundation in working with vectors, leveraging LangChain, and utilizing modern AI models for practical applications.

### Temperature
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,  # Low temperature for consistent outputs
    api_key="your-api-key"
)


# Don't use temperature - use reasoning_effort instead
llm = ChatOpenAI(
    model="gpt-5-mini",
    # temperature=0.3  ← This will FAIL!
    model_kwargs={
        "reasoning_effort": "low",  # Control reasoning depth
        "verbosity": "concise"       # Control response length
    }
)
Key takeaway: The GPT-5 family focuses on deterministic reasoning, so randomness parameters like temperature and top_p have been replaced with reasoning control parameters.

#### Day 4

In Day 4 of Week 5, the course moves into a critical discussion on evaluations and the scientific assessment of Retrieval Augmented Generation (RAG) systems. Here’s a summary focusing on relevant AI terminology and tools:

Evaluations and Their Importance: The day emphasizes that evaluations are crucial for determining if the RAG systems built are functioning effectively. Students learn how to measure both the retrieval process’s accuracy and the quality of the generated answers. This approach helps ensure that the deployed system meets business objectives effectively.

Performance Metrics: The session discusses various metrics that can be used to assess performance. Key points include:

Establishing a benchmark using real-world questions helps gauge both retrieval accuracy and question-answering accuracy.
Insights into creating a "golden dataset" that can serve as a reference for evaluating the system's performance are shared.
Retrieval and Answering Mechanics: The course supports students in analyzing how well their systems retrieve relevant context and whether the answers align with users' inquiries. This ensures alignment with business needs and user satisfaction.

LangChain and Implementation: Building on previous knowledge, students further explore how LangChain can be utilized to structure evaluations systematically. This framework helps streamline the testing of various components within the RAG pipeline.

Challenges in Performance Assessment: Common challenges regarding evaluating RAG, particularly in unsupervised settings, are addressed. This includes strategies for obtaining meaningful assessments without labeled datasets and the importance of considering effective approaches to measure retrieval and answering efficiency.

Real-World Applications: The importance of identifying how RAG can serve various commercial applications is reiterated, reinforcing that evaluations can lead to immediate practical benefits for businesses leveraging expert knowledge systems.

Overall, Day 4 reinforces the understanding of how to robustly evaluate RAG systems, fostering a strong foundation for students to ensure their implementations are effective and aligned with expectations.


#### Day 5

In Day 5 of Week 5, the course culminates the discussions on Retrieval Augmented Generation (RAG) by focusing on practical applications, optimization techniques, and preparing students for advanced topics in further sessions. Here’s a summary that highlights key AI terminology and tools:

Recap of RAG Pipeline Creation: The instructor revisits the journey of building a RAG knowledge worker pipeline, underscoring the foundational skills learned. Students realize how to create a question-answering assistant specifically for a company scenario, demonstrating practical applications of RAG in business contexts.

Optimizing Retrieval and Response: The day emphasizes optimization techniques for enhancing the performance of RAG systems. Topics covered include the importance of experimentation with chunk sizes and overlaps, and how different configurations impact retrieval accuracy and quality of answers.

Hands-On Experience: Students are encouraged to experiment with various techniques, including chunking, encoding, and applying different strategies to improve RAG performance. This hands-on learning fosters a deeper understanding of the mechanics behind effective RAG implementations.

Introduction to Advanced Techniques: The instructor introduces a variety of optimization strategies that can take RAG further, such as reranking, query preprocessing, and other advanced methods. This sets the stage for exploring a "zoo" of techniques that enhance the reliability and efficiency of RAG systems.

Commercial Applications of RAG: A discussion on the commercial implications of the techniques covered is also highlighted, emphasizing how these systems can provide immediate value across different industries and business use cases.

Conclusion and Next Steps: By the end of the day, students find themselves well-equipped to explore advanced RAG techniques, laying the groundwork for continued learning and application in their future projects.

Overall, Day 5 encapsulates the course’s objectives in building, evaluating, and optimizing RAG systems, preparing students to drive significant understanding and expertise in AI applications.


### WEEK 6

Week 6 focuses on the landscape of open-source large language models (LLMs). Here’s a summary of key points:

Notable Models: The lecture highlights several important LLMs, including Meta's LLaMA series, Mistral, Quinn, Gemma, DeepSeek, and OpenAI's newly open-sourced GPT.

LLaMA Models: Meta's LLaMA models are emphasized, particularly LLaMA 3.2 for its smaller size, which allows for local use. LLaMA 4 is noted for its power.

Mistral: This French company's mixture of experts model optimizes responses by using multiple smaller models.

Quinn: Developed by Alibaba Cloud, this model is recognized for its capabilities, although it’s less well-known than LLaMA.

Gemma: An open-source alternative to Google’s Gemini, particularly notable for its smaller size of 270 million parameters.

DeepSeek: This model stands out for its efficiency, achieving high capabilities at a lower cost through a distillation process.

OpenAI's GPT: OpenAI’s move to open-source their GPT model is discussed, particularly its larger 120 billion parameter version.

Usage Contexts: The difference between using packaged products like ChatGPT, calling models via APIs, and running managed services from cloud providers is explained.

Practical Tools: The lecture introduces how to run open-source models directly, with tools like Olama for user-friendly local execution, and the Hugging Face Transformers library for a hands-on coding experience.

Inference: The concept of inference is introduced, highlighting how to generate outputs using these models.

If you have any specific questions about these topics, feel free to ask!

### Day 1

Day 1 of Week 6 centers on the landscape of open-source large language models (LLMs) and discusses both the technical challenges and tools associated with these models.

Technical Challenges:
Scalability: Open-source models like Meta's LLaMA series, especially LLaMA 3.2 and LLaMA 4, highlight the need for balancing model size versus capability. Smaller models are more accessible for local use but might not perform at the same level as larger models.

Optimization and Efficiency: The lecture introduces Mistral's mixture of experts model, which aims to use multiple smaller models to optimize performance based on specific queries. This raises challenges in effectively managing and integrating these multiple models to provide seamless user experiences.

Cost-Effectiveness: DeepSeek is noted for achieving high capabilities at a significantly lower cost than other models like OpenAI's. However, developing efficient models that maintain performance while reducing operational costs is a continual challenge.

Model Distillation: The process of creating smaller variants of powerful models through distillation involves challenges in ensuring that the smaller models retain sufficient capabilities and accuracy.

Tools Presented:
Olama: A user-friendly tool that allows for easy local execution of open-source models, which addresses accessibility challenges for users less familiar with coding or complex setups.

Hugging Face Transformers: This library enables a more hands-on coding experience for those looking to implement LLMs directly in their projects. It provides extensive documentation and resources but requires a level of technical expertise.

Inference Methods: The concept of inference, or the generation of outputs from inputs using these models, is essential for practical applications and poses its own challenges in implementation depending on the model's structure and capabilities.

These points capture the essence of the technical challenges and tools discussed on Day 1 of Week 6. If you need more details on any specific topic, let me know!

Was this content relevant to you?

In Day 1 of Week 6, the lecture dives into the landscape of open-source large language models (LLMs), with a focus on various notable models and their technical challenges.

Meta's LLaMA Series: The lecture emphasizes LLaMA 3.2 for its efficiency and smaller size suitable for local use, and LLaMA 4 for its higher performance. The technical challenge here lies in balancing model size with capability, ensuring smaller models can still deliver meaningful performance.

Mistral: This French company's mixture of experts model showcases a technical challenge of optimizing responses by leveraging multiple smaller models. The complexity of managing these experts effectively is a significant hurdle.

Quinn: Developed by Alibaba Cloud, Quinn faces challenges in gaining recognition despite its capabilities, highlighting the competitive nature of the LLM market.

Gemma: Positioned as an alternative to Google's Gemini, Gemma's smaller model (270 million parameters) presents a challenge in maintaining competitiveness with more substantial models while also focusing on accessible deployment.

DeepSeek: Notable for achieving high efficiency with lower costs compared to OpenAI’s offerings, DeepSeek's method of creating smaller variants using distillation poses challenges in ensuring the smaller models maintain the performance of their larger counterparts through effective training.

OpenAI’s GPT: The release of an open-sourced GPT model introduces hurdles related to its varied sizes and optimal use scenarios. The distinction between packaged products and API calls creates additional considerations for developers looking to harness its capabilities.

Running Open-Source Models: The challenges in deploying these models locally are addressed by introducing tools like Olama and the Hugging Face Transformers library. The usability of these tools plays a crucial role in enabling effective model inference and application.

Overall, this week's focus is on the technical intricacies of using and deploying these open-source models effectively, balancing performance, size, and implementation. Let me know if you need further details or have specific questions!

### Day 2
On Day 2 of Week 6, the main focus was on data preparation for machine learning, specifically for fine-tuning large language models (LLMs). Here’s a detailed summary highlighting the technical challenges, keywords, and specific model names discussed:

Technical Challenges:
Data Cleaning: Identifying and fixing anomalies in datasets is critical, but challenging, as errors can significantly affect model performance.

Dataset Construction: Participants faced difficulties in effectively assembling datasets from raw data, which is essential for training models that perform well in practical applications.

Integration of Techniques: Balancing traditional machine learning methods with advanced techniques can be confusing, as learners need to understand when to apply each approach strategically.

Keywords & Models:
Fine-Tuning: The process of adjusting pre-trained models to better capture features relevant to specific tasks, which was a central theme of the day.

Frontier Models: These include state-of-the-art models that are at the forefront of AI research. During the session, the emphasis was on how to utilize these models effectively.

LLaMA and LoRA/QLoRA: Mentioned as examples of frontier models that can be fine-tuned to enhance their ability to make predictions, particularly for price prediction tasks.

Transformers: This powerful neural network architecture, pivotal in NLP tasks, was highlighted, especially its underlying role in model advancements over the last few years.

The focus was on understanding how to handle data effectively to set the stage for training models that leverage the state-of-the-art techniques discussed, particularly in the context of fine-tuning frontier models.

If you have any more questions or need clarification on a specific point, feel free to ask!


### Day 3

On Day 3 of Week 6, the focus shifted towards hands-on model training, specifically emphasizing how to effectively train machine learning models and refine their performance. Here’s a detailed summary highlighting the technical challenges, keywords, and model names discussed:

Technical Challenges:
Data Understanding: A central challenge was comprehensively understanding datasets—not just in terms of structure but also regarding how to visualize and prepare them for training. Proper data curation is critical in ensuring the effectiveness of the models.

Building Neural Networks: Participants faced hurdles in constructing and training neural networks from scratch. This included understanding the architecture, selecting appropriate hyperparameters, and appreciating the nuances of operational mechanics.

Model Evaluation: Evaluating the performance of the models and identifying how well they generalize across different scenarios was highlighted as a challenge, particularly when comparing various architectures.

Keywords & Models:
Neural Network: Participants built a vanilla neural network, gaining practical experience in model training and performance evaluation.

Support Vector Regression: This model was previously discussed in the context of its performance, showing how it can help estimate outcomes based on input features.

Random Forest Regression: This ensemble technique combines multiple models by sampling features and data points to enhance prediction accuracy. It was featured prominently, as it is known for handling various dataset shapes effectively.

Frontier Models: The day involved insights into advanced frontier models, reinforcing how they can be improved through training and fine-tuning for better inference at runtime.

The activities on this day laid a strong foundation for understanding model capabilities in real-world applications. By addressing data integrity and model training, participants were better equipped to tackle complex machine learning problems moving forward.

If you have more questions or need clarification on specific parts of the training or models discussed, feel free to ask!


### Day 4

On Day 4 of Week 6, the focus was on advanced techniques for fine-tuning models and addressing data-related challenges in deep learning. Here’s a detailed summary emphasizing the technical challenges, keywords, and specific models mentioned:

Technical Challenges:
Data Curation: A critical challenge faced by participants was effectively curating datasets. The quality of the dataset has a major impact on the performance of the models during training. Understanding how to clean, visualize, and prepare data properly is vital but can be time-consuming.

Hyperparameter Selection: Participants were introduced to the concept of hyperparameters, specifically focusing on three key parameters: r, alpha, and target modules. Determining the optimal values for these hyperparameters is essential for improving model performance but can be complex and requires experimentation.

Quantization and Fine-Tuning Techniques: The day covered advanced fine-tuning techniques, specifically focusing on LoRA (Low-Rank Adaptation) and its application to reduce training costs and enhance performance. Understanding the nuances of quantization was also emphasized, as it plays a crucial role in making models efficient for deployment.

Keywords & Models:
LoRA: This technique allows models to be fine-tuned with fewer parameters, making the training process more efficient. It was discussed as a key method for enhancing the capabilities of large models without extensive resource use.

Quantization: A method introduced to reduce the precision of computations and model size, thus enabling faster inference while maintaining acceptable levels of performance.

Hyperparameters (r, alpha, target modules): These key terms were introduced, highlighting their importance in the fine-tuning process. Understanding these parameters would be crucial for participants as they engage in training models.

Frontier Models: As part of the discussion on fine-tuning, frontier models were again referenced, reinforcing their importance in the training context and the potential gains from proper adjustments.

The focus on data and advanced training techniques set the groundwork for deeper engagement with model performance metrics, preparing participants for more complex discussions in the following sessions.

If you have further questions or need clarification on any aspect, feel free to ask!

### Day 5

On Day 5 of Week 6, the focus was on implementing the strategies and techniques discussed throughout the week for model fine-tuning and evaluation. Here’s a detailed summary, highlighting the technical challenges, keywords, and models referenced:

Technical Challenges:
Model Selection for Fine-Tuning: Participants faced the challenge of selecting the appropriate open-source model for fine-tuning. Understanding the differences between various models and their suitability for specific tasks was crucial.

Evaluation of Model Performance: Evaluating the base model’s performance after initial training and identifying areas for improvement were emphasized. This involved assessing metrics and understanding how different configurations can impact results.

Transition from Theory to Practice: Bridging the gap between theoretical knowledge of model training and practical implementation posed a challenge. Making sense of the algorithms and how to apply them effectively in a real-world scenario required focused attention.

Keywords & Models:
Hyperparameters: Building on prior discussions, hyperparameters continued to play a significant role in the fine-tuning process, as participants had to adjust these settings to see their direct effects on model performance.

LoRA (Low-Rank Adaptation): LoRA was again a focal point in fine-tuning discussions, emphasizing its utility in enhancing model performance while minimizing computational costs.

Frontier Models: As in previous sessions, frontier models were highlighted for their advanced capabilities, demonstrating the cutting-edge technologies available for participants to leverage.

Open Source Models: The emphasis was placed on practical application with open-source models; participants fine-tuned these models based on their specific datasets and business problems.

The day was structured to allow for both theoretical grounding and practical capability, preparing participants for future sessions on model deployment and further refinements based on hands-on experience.

If you have any further questions or need additional detail on a specific topic, feel free to ask!

Was this content relevant to you?




### CAPSTONE PROJECT



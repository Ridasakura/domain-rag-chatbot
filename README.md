# Domain-Specific RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions from user-uploaded PDF documents (course notes, manuals, policies, or any domain-specific material). Answers are generated strictly from retrieved document content, with source document and page number shown for every answer.

## How it works

```
Upload PDF files
   |
Extract text from each page (pypdf, with page-level metadata)
   |
Split text into overlapping chunks (~800 characters, 120 overlap)
   |
Convert chunks into embeddings (Sentence Transformers: all-MiniLM-L6-v2)
   |
Store embeddings + metadata in a FAISS vector index
   |
User asks a question
   |
Retrieve the top 3-5 most relevant chunks
   |
Send retrieved context + question to the LLM (Groq / Llama 3.1)
   |
Display grounded answer with source document and page number
```

## Features

- Upload multiple PDFs at once
- Strict grounding: the model only answers from retrieved context and explicitly says when information isn't found
- Every answer shows its source document and page number
- Full chat history in the Streamlit session
- Clear Chat button to reset a conversation

## Tech stack

| Part | Tool |
|---|---|
| PDF extraction | pypdf |
| Text splitting | LangChain text splitters |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS |
| LLM | Groq (Llama 3.1 8B Instant) |
| Interface | Streamlit |

## Setup (local)

1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd domain_rag_chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

4. Run the app:
   ```
   streamlit run app.py
   ```

5. In the sidebar: paste your Groq API key, upload PDFs, click **Process Documents**, then ask questions in the chat box.

## Responsible AI & Security

- API keys are entered by the user at runtime and never hard-coded or committed to source control.
- The system prompt strictly instructs the model to answer only from retrieved context, refusing when the answer isn't present in the documents.
- Instructions embedded within uploaded documents that attempt to override the system prompt are explicitly ignored.
- Users should verify high-stakes information independently rather than relying solely on the chatbot's output.

## Project structure

```
domain_rag_chatbot/
|-- app.py              # Streamlit application
|-- requirements.txt    # Python dependencies
|-- README.md
```

## Author

Built as part of a student project on Retrieval-Augmented Generation (RAG) systems.

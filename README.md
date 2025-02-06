# Agentic-RAG


This project demonstrates an agentic Retrieval-Augmented Generation (RAG) system for answering technology-related questions.
It combines document retrieval from a pre-built Chroma vector store (created from PDF documents) with a two-step LLM reasoning workflow managed by Langgraph.
The system features a Gradio web interface that displays both the final answer and a detailed reasoning log.

## Setup

### 1. Clone the Repository

# Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/your_repo.git
cd your_repo
```

### 2. Create and Activate a Virtual Environment

# Create a virtual environment using `venv` and activate it:

```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Unix/MacOS:
source venv/bin/activate
```

### 3. Install Dependencies

# Install all required Python libraries:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

# Copy the sample environment file and update it with your model credentials:

```bash
cp .env.example .env
```

# Then, edit the `.env` file with your settings. For example:

```env
# Use HuggingFace for cloud-based inference:
USE_HUGGINGFACE=yes
HUGGINGFACE_API_TOKEN=your_huggingface_api_token_here

# Set your model IDs (compatible with HuggingFace or local inference)
DRAFT_MODEL_ID=your_draft_model_id_here
CONTROL_MODEL_ID=your_control_model_id_here
```

> **Note:** If you are using local inference, set `USE_HUGGINGFACE=no` and adjust the model parameters accordingly.

## Document Ingestion

1. **Place Your PDFs:**  
   # Put your PDF documents into the `data/` directory:

   ```bash
   mkdir data
   # Copy your PDFs into the data directory
   ```

2. **Build the Vector Store:**  
   # Run the following script to ingest PDFs, split them into chunks, generate embeddings, and create a persistent vector store in the `chroma_db/` directory:

   ```bash
   python build_vectorstore.py
   ```

## Usage

### Gradio Web Interface

# Launch the Gradio interface by running:

```bash
python gradio_app.py
```

# This will start a web server where you can type in your questions.
# The interface displays both the final answer and a reasoning log that details the agent’s decision process.

### Command-Line Interface

# Alternatively, you can run the system from the command line:

```bash
python main.py "Your question here"
```

## How It Works

- **Document Ingestion:**  
  # The `build_vectorstore.py` script:
  - Loads PDFs from the `data/` directory.
  - Splits documents into chunks (default: 1000 characters per chunk with 200 characters overlap).
  - Generates embeddings using the `sentence-transformers/all-mpnet-base-v2` model.
  - Creates and persists a Chroma vector store in the `chroma_db/` directory.

- **Agentic RAG System:**  
  # The system uses two LLMs:
  - **Draft Model:** Generates a draft answer based on the retrieved document context.
  - **Control Model:** Evaluates the draft answer and decides whether further refinement is needed.
  
  # The workflow is managed by Langgraph’s state graph, which logs detailed reasoning at each step.
  # This log is then displayed in the Gradio interface alongside the final answer.

- **Model Selection:**  
  - **HuggingFace Models:** Recommended for cloud-based inference. Requires an API token and provides scalable, production-grade performance.
  - **Local Inference:** Suitable for testing and development. Adjust the `.env` settings for local models if needed.

## Project Structure

```
project/
├── agents/
│   ├
│   └── technology_rag_agent.py   # Agent implementation with detailed reasoning logging
├── loaders/
│   |
│   └── pdf_loader.py            # PDF ingestion and vector store creation
├── models/
│   |
│   └── model_factory.py         # Model loading (HuggingFace or local inference)
├── data/                        # Place your PDF documents here
├── chroma_db/                   # Persisted vector store (generated by build_vectorstore.py)
├── build_vectorstore.py         # Script to build the vector store from PDFs
├── gradio_app.py                # Gradio interface for the agent with reasoning log
├── main.py                      # Command-line interface for the agent
└── .env                         # Environment configuration file
```

## Notes

- The vector store is persisted in the `chroma_db/` directory.
- Default text splitting: 1000 characters per chunk with 200 characters overlap.
- Embeddings are generated using the `sentence-transformers/all-mpnet-base-v2` model.
- The system supports both cloud-based inference (HuggingFace) and local inference.
- Detailed reasoning from each step is logged and displayed in the Gradio interface.



import os
import logging
from dotenv import load_dotenv
import gradio as gr

from models.model_factory import get_model
from loaders.pdf_loader import get_vector_store
from agents.technology_rag_agent import TechnologyRAGAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

draft_model_id = os.getenv("DRAFT_MODEL_ID")
control_model_id = os.getenv("CONTROL_MODEL_ID")
if not draft_model_id or not control_model_id:
    raise ValueError("DRAFT_MODEL_ID and CONTROL_MODEL_ID must be set in your .env file.")

draft_model = get_model(draft_model_id)
control_model = get_model(control_model_id)

db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vector_db = get_vector_store(db_dir)

agent = TechnologyRAGAgent(
    draft_model=draft_model,
    control_model=control_model,
    max_attempts=2,
    vector_db=vector_db
)

def answer_question(question: str) -> tuple:
    """Processes a user question through the RAG agent and returns the final answer and reasoning log."""
    
    if not question.strip():
        return "Please enter a valid question.", ""
    
    try:
        result = agent.run(question)  # Calls the TechnologyRAGAgent
        final_answer = result.get("final_answer", "No answer found.")
        reasoning_log = "\n".join(result.get("reasoning_log", []))  # Format reasoning log for display

        logger.info(f"Final Answer: {final_answer}")  # Debugging
        logger.info(f"Reasoning Log: {reasoning_log}")

        return final_answer, reasoning_log

    except Exception as e:
        logger.error("Error while processing the question: %s", e)
        return "An error occurred while processing your question.", ""


iface = gr.Interface(
    fn=answer_question,
    inputs=gr.components.Textbox(
        lines=4,
        placeholder="Type your question here...",
        label="Your Question"
    ),
    outputs=[
        gr.components.Textbox(label="Agent Answer",interactive=True),
        gr.components.Textbox(label="Reasoning Log", lines=10, interactive=True)
    ],

    title="Technology RAG Agent",
    description=(
        "Ask questions related to the pdf you put in data directory. This agent uses a vector store of documents "
        "to provide contextually relevant answers via a two-step reasoning workflow. "
        "The reasoning log shows the detailed decision process behind the final answer."
    )
)

if __name__ == "__main__":
    iface.launch()

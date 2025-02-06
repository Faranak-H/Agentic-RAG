import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()
from models.model_factory import get_model
from loaders.pdf_loader import get_vector_store
from agents.technology_rag_agent import TechnologyRAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    draft_model_id = os.getenv("DRAFT_MODEL_ID")
    control_model_id = os.getenv("CONTROL_MODEL_ID")
    if not draft_model_id or not control_model_id:
        raise ValueError("DRAFT_MODEL_ID and CONTROL_MODEL_ID must be set in your .env file.")

    draft_model = get_model(draft_model_id)
    control_model = get_model(control_model_id)
    
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    vector_db = get_vector_store(db_dir)
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = ("What are the main components of agents?")
    
    agent = TechnologyRAGAgent(
        draft_model=draft_model,
        control_model=control_model,
        max_attempts=2,
        vector_db=vector_db
    )
    result = agent.run(question)
    print("\n=== Final Answer ===\n")
    print(result["final_answer"])

if __name__ == "__main__":
    main()

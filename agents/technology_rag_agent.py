import os
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Extend the AgentState to include a reasoning log.
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    final_answer: str
    attempts: int
    retriever: Any
    reasoning_log: List[str]  # Stores reasoning logs.

class TechnologyRAGAgent:
    def __init__(self, draft_model, control_model, max_attempts: int = 2, vector_db=None):
        """
        Initialize the agent with:
          - draft_model: an LLM to generate draft answers.
          - control_model: an LLM to evaluate draft answers.
          - vector_db: a vector store instance with a similarity_search method.
        """
        self.draft_model = draft_model
        self.control_model = control_model
        self.max_attempts = max_attempts
        self.vector_db = vector_db
        self.workflow = self.build_workflow()
        logger.info("TechnologyRAGAgent initialized with max_attempts=%s", max_attempts)

    def build_workflow(self) -> Any:
        """Build and compile the state graph workflow."""
        workflow = StateGraph(AgentState)
        # Rename the node to "draft_node" to avoid clashing with the "draft_answer" state key.
        workflow.add_node("draft_node", self._draft_answer_step)
        # Similarly, rename the control node.
        workflow.add_node("control_node", self._control_decision_step)
        
        # Define flow: draft_node -> control_node.
        workflow.set_entry_point("draft_node")
        workflow.add_edge("draft_node", "control_node")
        
        # Conditional branch: either refine (loop back to draft_node) or accept (end).
        workflow.add_conditional_edges(
            "control_node",
            self._decide_next_step,
            {
                "refine": "draft_node",
                "accept": END
            }
        )
        logger.info("Workflow built and compiled.")
        return workflow.compile()

    def _draft_answer_step(self, state: AgentState) -> Dict:
        """Generate a draft answer based on the question and retrieved documents."""
        logger.info("Drafting answer for question: %s", state["question"])
        # Combine document contents into context.
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = (
            f"You are a technology domain expert. Answer the following question concisely and factually "
            f"based solely on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            f"Draft Answer:"
        )
        try:
            # invoke the answer
            draft_answer = self.draft_model.invoke(prompt)
            logger.info("Draft answer generated.")
        except Exception as e:
            logger.error("Error in draft_model.run: %s", e)
            draft_answer = "Unable to generate an answer due to an error."
        
        # Append reasoning information.
        log_msg = f"Draft Step: Used context (length {len(context)}). Generated draft answer: {draft_answer}"
        new_log = state.get("reasoning_log", []) + [log_msg]
        return {"draft_answer": draft_answer, "reasoning_log": new_log}
    
    def _control_decision_step(self, state: AgentState) -> Dict:
        """Evaluate the draft answer to decide whether to accept or refine it."""
        
        logger.info("Evaluating draft answer (attempt %s).", state["attempts"] + 1)

        # Construct the evaluation prompt
        prompt = (
            f"Evaluate the following draft answer. Respond with 'accept' if it fully answers the question. "
            f"Respond with 'refine' if the answer needs improvement.\n\n"
            f"Question: {state['question']}\n\n"
            f"Draft Answer: {state['draft_answer']}\n\n"
            f"Respond (only 'accept' or 'refine'):"
        )

        try:
            decision = self.control_model.invoke(prompt).strip().lower()
            logger.info(f"Control model raw output: {decision}")

            # Ensure only 'accept' or 'refine' are valid responses
            if decision not in ["accept", "refine"]:
                logger.warning(f"Unexpected control decision: {decision}. Defaulting to 'accept'.")
                decision = "accept"
        
        except Exception as e:
            logger.error("Error in control_model.invoke: %s", e)
            decision = "accept"

        # Handle looping to avoid endless refinement
        if state["attempts"] > 0 and state["draft_answer"] == state["final_answer"]:
            logger.warning("Loop detected: The draft answer hasn't changed. Accepting to avoid endless refinement.")
            decision = "accept"

        # Append reasoning information
        log_msg = (
            f"Control Step: For question '{state['question']}', the draft answer was '{state['draft_answer']}'. "
            f"Control decision received: '{decision}'."
        )
        new_log = state.get("reasoning_log", []) + [log_msg]

        return {"decision": decision, "reasoning_log": new_log}


    def _decide_next_step(self, state: AgentState) -> str:
        """
        Decide the next step based on the control decision and the number of attempts.
        If the decision is 'refine' and attempts remain, loop back; otherwise, accept.
        """
        if state["attempts"] >= self.max_attempts:
            log_msg = "Decide Step: Max attempts reached. Accepting current draft answer."
            new_log = state.get("reasoning_log", []) + [log_msg]
            state["reasoning_log"] = new_log
            state["final_answer"] = state["draft_answer"]
            return "accept"
        decision = state.get("decision", "accept")
        if decision == "refine":
            log_msg = f"Decide Step: Control decision 'refine' received. Incrementing attempts to {state['attempts'] + 1}."
            new_log = state.get("reasoning_log", []) + [log_msg]
            state["reasoning_log"] = new_log
            state["attempts"] += 1
            return "refine"
        else:
            log_msg = "Decide Step: Control decision 'accept' received. Accepting current draft answer."
            new_log = state.get("reasoning_log", []) + [log_msg]
            state["reasoning_log"] = new_log
            state["final_answer"] = state["draft_answer"]
            return "accept"

    def run(self, question: str) -> Dict:
        """
        Run the complete workflow:
          1. Retrieve relevant documents.
          2. Execute the state graph.
          3. Return the final accepted answer along with the reasoning log.
        """
        try:
            logger.info("Running TechnologyRAGAgent for question: %s", question)
            documents = self.vector_db.similarity_search(question, k=3)
            logger.info("Retrieved %s document(s) from vector store.", len(documents))
        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            raise RuntimeError("Document retrieval failed.") from e

        state: AgentState = {
            "question": question,
            "documents": documents,
            "draft_answer": "",
            "final_answer": "",
            "attempts": 0,
            "retriever": self.vector_db,
            "reasoning_log": []
        }
        try:
            final_state = self.workflow.invoke(state)
            logger.info("Workflow completed.")
        except Exception as e:
            logger.error("Error during workflow execution: %s", e)
            raise RuntimeError("Workflow execution failed.") from e

        return {"final_answer": final_state["final_answer"], "reasoning_log": final_state.get("reasoning_log", [])}


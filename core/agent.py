from core.vector_store import VectorStore
from core.llm_handler import LLMHandler
from utils.text_processor import clean_text
from core.embedding_handler import EmbeddingHandler
from core.agent_flow import banking_agent_flow, AgentState
from typing import Dict, Any, Tuple




class TicketAgent:
    def __init__(self):

        
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.embeding_handler = EmbeddingHandler()


    def process_query(self, title: str, description: str, category: str) -> Tuple[Dict[str, Any], str]:
        """Process a query using the agentic flow."""
        # Initialize the state
        initial_state = AgentState(
            query=f"{title} {description} {category}",
            query_type=None,
            is_valid=None,
            needs_clarification=False,
            reformulated_query=None,
            documents=None,
            ranked_documents=None,
            generated_response=None,
            confidence_score=None,
            escalation_required=False,
            final_response=None,
            agent_scratchpad=[]
        )
        
        # Run the agentic flow
        final_state = banking_agent_flow.invoke(initial_state)
        
        return final_state["final_response"]

    def update_tickets(self, tickets):
        """Updates the vector store with new tickets."""
        ticket_vectors = []
        for ticket in tickets:
            ticket_text = clean_text(f"{ticket.title} {ticket.description} {ticket.category}")
            vector = self.embeding_handler.get_embedding(ticket_text)
            ticket_vectors.append(vector)
        self.vector_store.add_tickets(tickets, ticket_vectors) 
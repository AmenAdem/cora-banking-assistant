from core.vector_store import VectorStore
from core.llm_handler import LLMHandler
from utils.text_processor import clean_text
from core.embedding_handler import EmbeddingHandler




class TicketAgent:
    def __init__(self):

        
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.embeding_handler = EmbeddingHandler()


    def process_query(self, title, description, category):
        query_text = clean_text(f"{title} {description} {category}")
        print("query text : "+query_text)
        vector = self.embeding_handler.get_embedding(query_text)
        search_results = self.vector_store.search_similar(vector)
        print(search_results)
        llm_response = self.llm_handler.generate_response(
            search_results,
            title,
            description
        )
        return search_results,llm_response

    def update_tickets(self, tickets):
        ticket_vectors = []
        for ticket in tickets:
            ticket_text = clean_text(f"{ticket.title} {ticket.description} {ticket.category}")
            vector = self.embeding_handler.get_embedding(ticket_text)
            ticket_vectors.append(vector)
        self.vector_store.add_tickets(tickets,ticket_vectors) 
from core.vector_store import VectorStore
from core.llm_handler import LLMHandler
from utils.text_processor import clean_text




class TicketAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

    def process_query(self, title, description, category):
        query_text = f"{clean_text(title)} {clean_text(description)} {clean_text(category)}"
        print("query text : "+query_text)
        search_results = self.vector_store.search_similar(query_text)
        print(search_results)
        llm_response = self.llm_handler.generate_response(
            search_results,
            title,
            description
        )
        return search_results,llm_response

    def update_tickets(self, tickets):
        self.vector_store.add_tickets(tickets) 
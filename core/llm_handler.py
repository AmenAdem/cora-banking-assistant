import requests
from config import settings
from prompts.templates import TICKET_RESOLUTION_TEMPLATE

class LLMHandler:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    def generate_response(self, search_results, query_title, query_description):
        context = self._format_context(search_results)
        prompt = TICKET_RESOLUTION_TEMPLATE.format(
            query_title=query_title,
            query_description=query_description,
            context=context
        )

        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _format_context(self, search_results):
        context = ""
        for result in search_results:
            context += f"Title: {result.payload['title']}\n"
            context += f"Category: {result.payload['category']}\n"
            context += f"Resolution: {result.payload['resolution']}\n"
            context += "-" * 50 + "\n"
        return context 
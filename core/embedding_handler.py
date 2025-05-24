import requests
from config import settings
class EmbeddingHandler:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_EMBEDDING_MODEL 

    def get_embedding(self, text):
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            raise RuntimeError(f"Error getting embedding: {str(e)}") 
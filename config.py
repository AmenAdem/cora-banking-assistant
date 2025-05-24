from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # This is a smaller model
    COLLECTION_NAME: str = "helpdesk_tickets"
    VECTOR_SIZE: int = 384  # Size for the smaller model

settings = Settings()
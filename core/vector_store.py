from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from config import settings
import random
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST, 
            port=settings.QDRANT_PORT
        )
        self._model = None
        self._initialize_collection()

    @property
    def model(self):
        if self._model is None:
            logger.info("Initializing sentence transformer model...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Model initialization complete")
        return self._model

    def _initialize_collection(self):
        if not self.client.collection_exists(settings.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config={
                    "size": settings.VECTOR_SIZE, 
                    "distance": "Cosine"
                }
            )

    def add_tickets(self, tickets):
        points = []
        for ticket in tickets:
            vector = self.model.encode(
                f"{ticket.title} {ticket.description} {ticket.category} {ticket.resolution}"
            ).tolist()
            point = PointStruct(
                id=random.randint(1, 100000),
                vector=vector,
                payload=ticket.dict()
            )
            points.append(point)
        self.client.upsert(
            collection_name=settings.COLLECTION_NAME, 
            points=points
        )

    def search_similar(self, query_text, top_k=3):
        query_vector = self.model.encode(query_text).tolist()
        return self.client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        ) 
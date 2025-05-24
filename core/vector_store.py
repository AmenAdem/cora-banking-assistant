from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from config import settings
import random
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST, 
            port=settings.QDRANT_PORT,
        )
        self._initialize_collection()

    def _initialize_collection(self):
        if not self.client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config={
                    "size": settings.QDRANT_VECTOR_SIZE, 
                    "distance": "Cosine"
                }
            )

    def add_tickets(self, tickets, vectors):
        points = []
        for ticket, vector in zip(tickets, vectors):
            point = PointStruct(
                id=random.randint(1, 100000),
                vector=vector,
                payload=ticket.dict()
            )
            points.append(point)
        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME, 
            points=points
        )

    def search_similar(self, query_vector, top_k=3):
        return self.client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        ) 
# from sentence_transformers import SentenceTransformer
# from app.config import settings
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def download_model():
#     logger.info(f"Pre-downloading model: {settings.EMBEDDING_MODEL}")
#     try:
#         model = SentenceTransformer(settings.EMBEDDING_MODEL)
#         logger.info("Model downloaded successfully")
#         return True
#     except Exception as e:
#         logger.error(f"Error downloading model: {str(e)}")
#         return False

# if __name__ == "__main__":
#     download_model() 
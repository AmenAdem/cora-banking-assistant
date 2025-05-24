##this a new agent implementation that will be used to make this agent read form pdf  files and chunks it and make embed it on vector database 
# 
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import nltk
import json 
import re
import random 

COLLECTION_NAME = "helpdesk_tickets"
DATA_PATH = "data"

model = SentenceTransformer("all-MiniLM-L6-v2") 
def main():

    # Check if the database should be cleared (using the --clear flag).
    #nltk.download('stopwords')
    #nltk.download('punkt_tab')
    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_db(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_db(chunks: list[Document]):
    # Load the existing database.

    db =  QdrantClient(
    url="https://72301f52-5834-40b2-8ecc-ca3631a148ee.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2ODk4Mjg5fQ.2skZStEOMLxrakiY1VWgkYsoDunXhzxWnaGz1gdmFwE")
    # if not db.collection_exists:
    db.create_collection(collection_name="test",vectors_config={"size": 384, "distance": "Cosine"} )
#         db.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config={"size": 384, "distance": "Cosine"}  # Match embedding model size
# )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    
    # Fetch existing points (IDs)
    existing_points = db.scroll("test", limit=10000, with_payload=False)
    existing_ids = {point.id for point in existing_points[0]}  # Extract existing IDs
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    # Filter new points
    new_points = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            print("id"+chunk.metadata["id"]) 
            # print("\n this is chunk \n"+chunk+"\n\n")
            point = PointStruct(
                    id=random.randint(1, 100000),
                    vector= model.encode(str(chunk)).tolist() , # Ensure chunk contains the embedding
                    payload=chunk.metadata,
                )
            db.upsert(collection_name="test",points=[point])
            
   
    
    # Insert new points if any
    # if new_points:
    #     print(f"👉 Adding new documents: {len(new_points)}")
    #     #db.upsert(collection_name=COLLECTION_NAME, points=new_points)
    # else:
    #     print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks




if __name__ == "__main__":
    main()




# STOPWORDS = set(stopwords.words("english"))
# def clean_text(text):
#     if not text:
#         return ""
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z\s]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     words = word_tokenize(text)
#     filtered_words = [word for word in words if word not in STOPWORDS]

#     return " ".join(filtered_words)



# qdrant = QdrantClient(host="localhost", port=6333)
# model = SentenceTransformer("all-MiniLM-L6-v2")  # Adjust model as needed
# COLLECTION_NAME = "helpdesk_tickets"

# if not qdrant.collection_exists:
#     qdrant.create_collection(collection_name=COLLECTION_NAME,vectors_config={"size": 384, "distance": "Cosine"} )
# qdrant.recreate_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config={"size": 384, "distance": "Cosine"}  # Match embedding model size
# )

# # Load tickets from external JSON file
# with open("data.json", "r", encoding="utf-8") as file:
#     tickets = json.load(file)
# points = []
# for ticket in tickets:
#     title = clean_text(ticket.get("title", ""))
#     description = clean_text(ticket.get("description", ""))
#     category = clean_text(ticket.get("category", ""))
#     resolution = clean_text(ticket.get("resolution", ""))
#     text_to_embed = f"{title} {description} {category} {resolution}"
#     vector = model.encode(text_to_embed).tolist()
#     point = PointStruct(
#         id=random.randint(1, 100000),
#         vector=vector,
#         payload={
#             "title": title,
#             "description": description,
#             "category": category,
#             "resolution": resolution
#         }
#     )
#     points.append(point)
# qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

# print("Resolved tickets inserted successfully from file!")

# def search_similar_tickets(query_title, query_description, query_category, top_k=3):
#     query_text = f"{query_title} {query_description} {query_category}"
#     query_vector = model.encode(query_text).tolist()
#     search_results = qdrant.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=top_k
#     )

#     print("\nSimilar Resolved Tickets:")
#     for result in search_results:
#         print(f"Score: {result.score:.4f}")
#         print(f"Title: {result.payload['title']}")
#         print(f"Category: {result.payload['category']}")
#         print(f"Resolution: {result.payload['resolution']}")
#         print("-" * 50)

# # Example: Searching for a similar issue
# search_similar_tickets(
#     query_title="Can't login to my account ",
#     query_description=" I tried to open my account but nothing is working",
#     query_category="Login Issues"
# )

# print("End of agent operation . ")
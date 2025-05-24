from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from llm_handler import LLMHandler
import nltk
import json 
import re
import random 




nltk.download('stopwords')
nltk.download('punkt_tab')

STOPWORDS = set(stopwords.words("english"))
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in STOPWORDS]

    return " ".join(filtered_words)



qdrant = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Adjust model as needed
COLLECTION_NAME = "helpdesk_tickets"

if not qdrant.collection_exists:
    qdrant.create_collection(collection_name=COLLECTION_NAME,vectors_config={"size": 384, "distance": "Cosine"} )
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"size": 384, "distance": "Cosine"}  # Match embedding model size
)

# Load tickets from external JSON file
with open("data.json", "r", encoding="utf-8") as file:
    tickets = json.load(file)
points = []
for ticket in tickets:
    title = clean_text(ticket.get("title", ""))
    description = clean_text(ticket.get("description", ""))
    category = clean_text(ticket.get("category", ""))
    resolution = clean_text(ticket.get("resolution", ""))
    text_to_embed = f"{title} {description} {category} {resolution}"
    vector = model.encode(text_to_embed).tolist()
    point = PointStruct(
        id=random.randint(1, 100000),
        vector=vector,
        payload={
            "title": title,
            "description": description,
            "category": category,
            "resolution": resolution
        }
    )
    points.append(point)
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print("Resolved tickets inserted successfully from file!")

def search_similar_tickets(query_title, query_description, query_category, top_k=3):
    query_text = f"{query_title} {query_description} {query_category}"
    query_vector = model.encode(query_text).tolist()
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    print("\nSimilar Resolved Tickets:")
    for result in search_results:
        print(f"Score: {result.score:.4f}")
        print(f"Title: {result.payload['title']}")
        print(f"Category: {result.payload['category']}")
        print(f"Resolution: {result.payload['resolution']}")
        print("-" * 50)
    
    # Initialize LLM handler and generate response
    print("Generating response from LLM")   
    llm_handler = LLMHandler()
    llm_response = llm_handler.generate_response(
        search_results, 
        query_title, 
        query_description
    )
    print("Response generated from LLM")
    print("\nGenerated Response:")
    print(llm_response)
    
    return search_results, llm_response

# Example usage
if __name__ == "__main__":
    search_results, response = search_similar_tickets(
        query_title="Can't login to my account ",
        query_description=" I tried to open my account but nothing is working",
        query_category="Login Issues"
    )
    print("Search results")
    print(search_results)
    print("Response")
    print(response)


print("End of agent operation . ")
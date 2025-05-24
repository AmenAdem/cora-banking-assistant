import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    STOPWORDS = set(stopwords.words('english')) | set(stopwords.words('french'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english')) | set(stopwords.words('french'))

def clean_text(text: str) -> str:
    """
    Clean and normalize text by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Removing special characters
    4. Removing stopwords (English and French)
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return ' '.join(filtered_words).strip() 
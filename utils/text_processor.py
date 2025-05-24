import re

def clean_text(text: str) -> str:
    """
    Clean and normalize text by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Removing special characters
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    
    return text.strip() 
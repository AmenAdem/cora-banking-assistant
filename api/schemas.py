from pydantic import BaseModel
from typing import List, Optional

class TicketQuery(BaseModel):
    title: str
    description: str
    category: str
    resolution: Optional[str] = None

class TicketResponse(BaseModel):
    #similar_tickets: any
    generated_response: str 


import json
from api.schemas import TicketQuery
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from config import settings
from core.limiter import limiter
from core.logging import logger
from core.agent import TicketAgent


router = APIRouter()
agent = TicketAgent()



@router.post("/query")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["default"][0])
async def query_ticket(query: TicketQuery,request: Request):
    try:
        response = agent.process_query(
            query.title,
            query.description,
            query.category
        )
        print(response)    
        # res =  TicketResponse(
        #     similar_tickets=results,
        #     generated_response=response
        # )
        return  response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/update-tickets")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["default"][0])
async def update_tickets(tickets: list[TicketQuery],request: Request):
    try:

        # TICKETS_JSON_PATH = "data/customer_support_tickets.json"
        # with open(TICKETS_JSON_PATH, "r", encoding="utf-8") as f:
        #     tickets_data = json.load(f)
        # tickets = [TicketQuery(**ticket) for ticket in tickets_data]
        # print(tickets)
        agent.update_tickets(tickets)
        return {"message": "Tickets updated successfully"}
    except Exception as e:
        logger.error(f"Error updating tickets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 



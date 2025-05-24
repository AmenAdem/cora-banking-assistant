

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
        results, response = agent.process_query(
            query.title,
            query.description,
            query.category
        )
        print(response)    
        # res =  TicketResponse(
        #     similar_tickets=results,
        #     generated_response=response
        # )
        return results, response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/update-tickets")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["default"][0])
async def update_tickets(tickets: list[TicketQuery],request: Request):
    try:
        print(request)
        agent.update_tickets(tickets)
        return {"message": "Tickets updated successfully"}
    except Exception as e:
        logger.error(f"Error updating tickets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 



from fastapi import FastAPI, HTTPException
from api.schemas import TicketQuery, TicketResponse
from core.agent import TicketAgent
import logging
import sys
import uvicorn

#Set up logging to show everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

print("Starting the application...")
logger = logging.getLogger(__name__)
print("Logger initialized")
app = FastAPI()
print("FastAPI app initialized")
@app.get("/")
def read_root():
    return {"Hello": "World"} 



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialize agent after FastAPI app is created
@app.on_event("startup")
async def startup_event():
    try:
        logger.debug("Starting application initialization...")
        global agent
        logger.debug("Initializing TicketAgent...")
        agent = TicketAgent()
        logger.debug("TicketAgent initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.post("/query")
async def query_ticket(query: TicketQuery):
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
    

@app.post("/update-tickets")
async def update_tickets(tickets: list[TicketQuery]):
    try:
        agent.update_tickets(tickets)
        return {"message": "Tickets updated successfully"}
    except Exception as e:
        logger.error(f"Error updating tickets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from core.vector_store import VectorStore
from core.langchain_llm_handler import OllamaLLMHandler
from core.embedding_handler import EmbeddingHandler
from prompts.templates import CLAIM_RESPONSE_GENERATION_SYSTEM_PROMPT, DISPUTE_RESPONSE_GENERATION_SYSTEM_PROMPT, DOCUMENT_RANKER_SYSTEM_PROMPT, QUERY_VALIDATOR_SYSTEM_PROMPT, RESPONSE_EVALUATION_SYSTEM_PROMPT, RESPONSE_EVALUATION_USER_PROMPT, SUPPORT_RESPONSE_GENERATION_SYSTEM_PROMPT
from utils.text_processor import clean_text
from config import settings
import logging
import json
from datetime import datetime
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# State definition
class AgentState(TypedDict):
    query: str
    query_type: Optional[Literal['support', 'claim', 'dispute']]
    is_valid: Optional[bool]
    needs_clarification: Optional[bool]
    reformulated_query: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    ranked_documents: Optional[List[Dict[str, Any]]]
    generated_response: Optional[str]
    confidence_score: Optional[float]
    escalation_required: Optional[bool]
    final_response: Optional[str]
    session_id: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    error: Optional[str]
    step_info: Optional[Dict[str, Any]]

class BankingAgentFlow:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_handler = OllamaLLMHandler()
        self.embedding_handler = EmbeddingHandler()
        logger.info("BankingAgentFlow initialized with all required components")

    def validate_query_node(self, state: AgentState) -> AgentState:
        """Node 1: Validates and classifies the user query."""
        logger.info(f"Step 1: Validating query: {state['query']}")
        
        try:
            # Add session tracking
            if not state.get("session_id"):
                state["session_id"] = str(uuid.uuid4())
                state["start_time"] = datetime.now().isoformat()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", QUERY_VALIDATOR_SYSTEM_PROMPT),
                ("user", "Query: {input}")
            ])
            
            chain = LLMChain(llm=self.llm_handler, prompt=prompt)
            result = chain.run(input=state["query"])
            
            # Parse the JSON response
            try:
                validation_data = json.loads(result.strip())
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                validation_data = {
                    "is_valid": "valid" in result.lower() and "bank" in result.lower(),
                    "query_type": "support" if "support" in result.lower() else "claim" if "claim" in result.lower() else "dispute",
                    "needs_clarification": "clarification" in result.lower(),
                    "reformulated_query": state["query"],
                    "reasoning": "Fallback parsing used"
                }
            
            # Update state
            state.update({
                "is_valid": validation_data.get("is_valid", True),
                "query_type": validation_data.get("query_type", "support"),
                "needs_clarification": validation_data.get("needs_clarification", False),
                "reformulated_query": validation_data.get("reformulated_query", state["query"]),
                "step_info": {
                    "step": "validation",
                    "reasoning": validation_data.get("reasoning", "Query validated")
                }
            })
            
            logger.info(f"Query validation completed: {json.dumps(validation_data, indent=2)}")
            return state
            
        except Exception as e:
            logger.error(f"Error in validate_query_node: {str(e.__cause__)}", exc_info=True)
            state["error"] = f"Validation error: {str(e)}"
            return state

    def retrieve_documents_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieves relevant documents using RAG."""
        logger.info("Step 2: Retrieving documents using RAG")
        
        try:
            # Use reformulated query if available, otherwise original query
            search_query = state.get("reformulated_query")
            if search_query == None:
                search_query = state.get("query")
        
            # Clean and embed the query
            query_text = clean_text(search_query)
            vector = self.embedding_handler.get_embedding(query_text)
            
            # Determine number of documents based on query type
            top_k = 10 if state.get("query_type") == "dispute" else 5
            
            # Retrieve documents
            results = self.vector_store.search_similar(vector, top_k=top_k)
            
            state["documents"] = results
            state["step_info"] = {
                "step": "retrieval",
                "documents_found": len(results),
                "search_query": search_query
            }
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve_documents_node: {str(e)}", exc_info=True)
            state["error"] = f"Document retrieval error: {str(e)}"
            return state

    def rank_documents_node(self, state: AgentState) -> AgentState:
        """Node 3: Ranks and filters documents for relevance."""
        logger.info("Step 3: Ranking documents for relevance")
        
        try:
            documents = state.get("documents", [])
            if not documents:
                logger.warning("No documents to rank")
                state["ranked_documents"] = []
                return state
            
            # Create context from documents for ranking
            doc_summaries = []
            for i, doc in enumerate(documents):
                doc_text = doc.payload.get('description')[:200]  # First 200 chars
                doc_summaries.append(f"Doc {i+1} score:{doc.score}: {doc_text}")
            
            context = "\n".join(doc_summaries)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", DOCUMENT_RANKER_SYSTEM_PROMPT),
                ("user", "Query: {query}\n\nDocuments:\n{context}")
            ])
            
            chain = LLMChain(llm=self.llm_handler, prompt=prompt)
            result = chain.run(query=state["query"], context=context)
            
            try:
                ranking_data = json.loads(result.strip())
                relevant_indices = ranking_data.get("relevant_docs", list(range(1, min(4, len(documents) + 1))))
            except json.JSONDecodeError:
                # Fallback: take top 3 documents
                relevant_indices = list(range(1, min(4, len(documents) + 1)))
            
            # Filter documents based on ranking (convert to 0-based indexing)
            ranked_documents = [documents[i-1] for i in relevant_indices if 0 <= i-1 < len(documents)]
            
            state["ranked_documents"] = ranked_documents
            state["step_info"] = {
                "step": "ranking",
                "total_docs": len(documents),
                "relevant_docs": len(ranked_documents),
                "selected_indices": relevant_indices
            }
            
            logger.info(f"Ranked documents: {len(ranked_documents)} out of {len(documents)} selected")
            return state
            
        except Exception as e:
            logger.error(f"Error in rank_documents_node: {str(e)}", exc_info=True)
            state["error"] = f"Document ranking error: {str(e)}"
            return state

    def generate_response_node(self, state: AgentState) -> AgentState:
        """Node 4: Generates response using LLM with ranked documents."""
        logger.info("Step 4: Generating response")
        
        try:
            documents = state.get("ranked_documents", [])
            query = state["query"]
            query_type = state.get("query_type", "support")
            
            if not documents:
                # No documents available
                fallback_response = "I apologize, but I couldn't find specific information to answer your query. Please contact our customer service for personalized assistance."
                state["generated_response"] = fallback_response
                state["step_info"] = {"step": "generation", "status": "fallback_used"}
                return state
            
            # Build context from documents
            context_parts = []
            for i, doc in enumerate(documents[:3]):  # Use top 3 documents
                doc_content = doc.payload.get("description", "");
                doc_title = doc.payload.get("title");
                doc_resolution = doc.payload.get("resolution")
                context_parts.append(f"Source {i+1}: title {doc_title}\n description:{doc_content}\n resolution:{doc_resolution}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt based on query type
            if query_type == "support":
                system_msg = SUPPORT_RESPONSE_GENERATION_SYSTEM_PROMPT
            elif query_type == "claim":
                system_msg = CLAIM_RESPONSE_GENERATION_SYSTEM_PROMPT
            else:  # dispute
                system_msg = DISPUTE_RESPONSE_GENERATION_SYSTEM_PROMPT
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg.format(context=context)),
                ("user", "{query}")
            ])
            
            chain = LLMChain(llm=self.llm_handler, prompt=prompt)
            response = chain.run(query=query)
            
            state["generated_response"] = response.strip()
            state["step_info"] = {
                "step": "generation",
                "context_sources": len(documents),
                "query_type": query_type
            }
            
            logger.info("Response generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_response_node: {str(e)}", exc_info=True)
            state["error"] = f"Response generation error: {str(e)}"
            return state

    def evaluate_response_node(self, state: AgentState) -> AgentState:
        """Node 5: Supervisor evaluation of response quality."""
        logger.info("Step 5: Evaluating response quality (Supervisor)")
        
        try:
            query = state["query"]
            response = state.get("generated_response", "")
            query_type = state.get("query_type", "support")
            
            if not response:
                state.update({
                    "confidence_score": 0.0,
                    "escalation_required": True,
                    "final_response": "I apologize, but I'm unable to provide a response at this time. Please contact our customer service team for assistance."
                })
                return state
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", RESPONSE_EVALUATION_SYSTEM_PROMPT),
                ("user", RESPONSE_EVALUATION_USER_PROMPT)
            ])
            
            chain = LLMChain(llm=self.llm_handler, prompt=prompt)
            evaluation = chain.run(
                query_type=query_type,
                query=query,
                response=response
            )
            
            try:
                eval_data = json.loads(evaluation.strip())
                confidence_score = float(eval_data.get("confidence_score", 0.5))
                escalation_required = eval_data.get("escalation_required", confidence_score < 0.75)
            except (json.JSONDecodeError, ValueError):
                # Fallback evaluation
                confidence_score = 0.6  # Default medium confidence
                escalation_required = True  # Default to escalation for safety
            
            # Final decision logic
            if escalation_required or confidence_score < 0.75:
                final_response = f"""I understand you're asking about {query.lower()}. While I have some information that might help, I'd like to connect you with one of our specialists who can provide you with the most accurate and personalized assistance.
                
Please contact our customer service team at [phone number] or visit your nearest branch. They'll be able to help you with the specific details of your situation.

Is there anything else I can help you with in the meantime?"""
            else:
                final_response = response
            
            state.update({
                "confidence_score": confidence_score,
                "escalation_required": escalation_required,
                "final_response": final_response,
                "end_time": datetime.now().isoformat(),
                "step_info": {
                    "step": "evaluation",
                    "confidence": confidence_score,
                    "escalated": escalation_required,
                    "feedback": eval_data.get("feedback", "Evaluation completed")
                }
            })
            
            logger.info(f"Response evaluation completed - Confidence: {confidence_score}, Escalation: {escalation_required}")
            return state
            
        except Exception as e:
            logger.error(f"Error in evaluate_response_node: {str(e)}", exc_info=True)
            state["error"] = f"Response evaluation error: {str(e)}"
            state["escalation_required"] = True  # Default to escalation on error
            return state

    def create_workflow(self) -> StateGraph:
        """Creates the complete workflow graph."""
        logger.info("Creating banking agent workflow")
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("validate_query", self.validate_query_node)
        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("rank_documents", self.rank_documents_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("evaluate_response", self.evaluate_response_node)
        
        # Define the flow edges
        workflow.add_edge("validate_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "rank_documents")
        workflow.add_edge("rank_documents", "generate_response")
        workflow.add_edge("generate_response", "evaluate_response")
        workflow.add_edge("evaluate_response", END)
        
        # Set entry point
        workflow.set_entry_point("validate_query")
        
        logger.info("Banking agent workflow created successfully")
        return workflow.compile()

# Factory function to create and return the agent flow
def create_banking_agent_flow():
    """Factory function to create the banking agent flow."""
    agent_flow = BankingAgentFlow()
    return agent_flow.create_workflow()

# Create the main workflow instance
banking_agent_flow = create_banking_agent_flow()

# Usage example function
def process_banking_query(query: str) -> Dict[str, Any]:
    """
    Process a banking query through the complete workflow.
    
    Args:
        query: The customer's banking query
        
    Returns:
        Dict containing the final response and processing details
    """
    try:
        # Initialize state
        initial_state = AgentState(query=query)
        
        # Run the workflow
        final_state = banking_agent_flow.invoke(initial_state)
        
        # Return structured response
        return {
            "success": True,
            "query": query,
            "final_response": final_state.get("final_response", ""),
            "escalation_required": final_state.get("escalation_required", False),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "query_type": final_state.get("query_type", "unknown"),
            "session_id": final_state.get("session_id", ""),
            "processing_time": final_state.get("end_time", "") and final_state.get("start_time", ""),
            "error": final_state.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error processing banking query: {str(e)}", exc_info=True)
        return {
            "success": False,
            "query": query,
            "final_response": "I apologize, but I'm experiencing technical difficulties. Please contact our customer service team for assistance.",
            "escalation_required": True,
            "error": str(e)
        }



# from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
# from langgraph.graph import StateGraph, END
# from langchain.tools import Tool
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import LLMChain
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain.agents.format_scratchpad import format_to_openai_functions
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain.tools.render import format_tool_to_openai_function
# from core.vector_store import VectorStore
# from core.langchain_llm_handler import OllamaLLMHandler
# from core.embedding_handler import EmbeddingHandler
# from utils.text_processor import clean_text
# from config import settings
# import logging
# import json
# from datetime import datetime
# import uuid

# # Configure logging
# logger = logging.getLogger(__name__)

# # State definition
# class AgentState(TypedDict):
#     query: str
#     query_type: Optional[Literal['support', 'claim', 'dispute']]
#     is_valid: Optional[bool]
#     needs_clarification: Optional[bool]
#     reformulated_query: Optional[str]
#     documents: Optional[List[Dict[str, Any]]]
#     ranked_documents: Optional[List[Dict[str, Any]]]
#     generated_response: Optional[str]
#     confidence_score: Optional[float]
#     escalation_required: Optional[bool]
#     final_response: Optional[str]
#     agent_scratchpad: Optional[List[Dict[str, Any]]]
#     session_id: Optional[str]
#     start_time: Optional[str]
#     end_time: Optional[str]
#     error: Optional[str]

# class BankingTools:
#     def __init__(self):
#         self.vector_store = VectorStore()
#         self.llm_handler = OllamaLLMHandler()
#         self.embedding_handler = EmbeddingHandler()
#         logger.info("BankingTools initialized with all required components")

#     def validate_query(self, query: str) -> Dict[str, Any]:
#         """Validates and classifies the user query."""
#         logger.info(f"Validating query: {query}")
#         try:
#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", "You are a banking query validator. Classify if the query is valid and determine its type."),
#                 ("user", "{query}")
#             ])
            
#             chain = LLMChain(llm=self.llm_handler, prompt=prompt)
#             result = chain.run(query=query)
            
#             validation_result = {
#                 "is_valid": "valid" in result.lower(),
#                 "query_type": "support" if "support" in result.lower() else "claim" if "claim" in result.lower() else "dispute",
#                 "reformulated_query": result
#             }
            
#             logger.info(f"Query validation result: {json.dumps(validation_result, indent=2)}")
#             return validation_result
#         except Exception as e:
#             logger.error(f"Error in validate_query: {str(e)}", exc_info=True)
#             raise

#     def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Retrieves relevant documents using vector similarity search."""
#         logger.info(f"Retrieving documents for query: {query} with top_k={top_k}")
#         try:
#             query_text = clean_text(query)
#             vector = self.embedding_handler.get_embedding(query_text)
#             results = self.vector_store.search_similar(vector, top_k=top_k)
#             logger.info(f"Retrieved {len(results)} documents")
#             return results
#         except Exception as e:
#             logger.error(f"Error in retrieve_documents: {str(e)}", exc_info=True)
#             raise

#     def rank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Ranks documents using similarity scores from vector search."""
#         logger.info(f"Ranking {len(documents)} documents for query: {query}")
#         try:
#             if not documents:
#                 logger.warning("No documents to rank")
#                 return []
            
#             # Documents are already ranked by similarity from vector search
#             # Just return them in the same order
#             return documents
#         except Exception as e:
#             logger.error(f"Error in rank_documents: {str(e)}", exc_info=True)
#             raise

#     def generate_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
#         """Generates response using LLM with context from documents."""
#         logger.info(f"Generating response for query: {query}")
#         try:
#             if not documents:
#                 logger.warning("No documents available for response generation")
#                 return "I apologize, but I couldn't find any relevant information to answer your query."
            
#             context = "\n".join([doc["payload"]["description"] for doc in documents])
#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", "You are a banking assistant. Use the following context to answer the user's question:\n{context}"),
#                 ("user", "{query}")
#             ])
            
#             chain = LLMChain(llm=self.llm_handler, prompt=prompt)
#             response = chain.run(context=context, query=query)
            
#             logger.info("Response generated successfully")
#             return response
#         except Exception as e:
#             logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
#             raise

#     def evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
#         """Evaluates response quality and confidence."""
#         logger.info("Evaluating response quality")
#         try:
#             eval_prompt = ChatPromptTemplate.from_messages([
#                 ("system", "Evaluate the response quality and confidence."),
#                 ("user", "Query: {query}\nResponse: {response}")
#             ])
            
#             chain = LLMChain(llm=self.llm_handler, prompt=eval_prompt)
#             evaluation = chain.run(query=query, response=response)
            
#             confidence_score = float(evaluation.split("confidence:")[1].strip()) if "confidence:" in evaluation else 0.0
#             evaluation_result = {
#                 "confidence_score": confidence_score,
#                 "escalation_required": confidence_score < 0.85
#             }
            
#             logger.info(f"Response evaluation result: {json.dumps(evaluation_result, indent=2)}")
#             return evaluation_result
#         except Exception as e:
#             logger.error(f"Error in evaluate_response: {str(e)}", exc_info=True)
#             raise

#     def get_tools(self) -> List[Tool]:
#         """Returns list of tools for the agent."""
#         logger.info("Getting tools for agent")
#         return [
#             Tool(
#                 name="validate_query",
#                 func=self.validate_query,
#                 description="Validates and classifies a banking query"
#             ),
#             Tool(
#                 name="retrieve_documents",
#                 func=self.retrieve_documents,
#                 description="Retrieves relevant documents from the knowledge base"
#             ),
#             Tool(
#                 name="rank_documents",
#                 func=self.rank_documents,
#                 description="Ranks documents by relevance using similarity scores"
#             ),
#             Tool(
#                 name="generate_response",
#                 func=self.generate_response,
#                 description="Generates a response using the LLM"
#             ),
#             Tool(
#                 name="evaluate_response",
#                 func=self.evaluate_response,
#                 description="Evaluates the quality and confidence of a response"
#             )
#         ]

# def process_agent_step(state: AgentState) -> AgentState:
#     """Process a single step in the agent workflow."""
#     try:
#         tools = BankingTools()
#         agent_tools = tools.get_tools()
        
#         # Create the agent
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a banking assistant agent. Use the available tools to help users with their banking queries.
#             Follow these steps:
#             1. Validate the query
#             2. Retrieve relevant documents
#             3. Rank the documents
#             4. Generate a response
#             5. Evaluate the response quality
#             If the response quality is low, escalate to a human agent."""),
#             ("user", "{query}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad")
#         ])
#         print("This is the prompt",prompt)
        
#         agent = create_openai_functions_agent(
#             llm=tools.llm_handler,
#             tools=agent_tools,
#             prompt=prompt
#         )
        
#         agent_executor = AgentExecutor(
#             agent=agent,
#             tools=agent_tools,
#             verbose=True,
#             handle_parsing_errors=True
#         )
        
#         # Execute the agent
#         result = agent_executor.invoke({
#             "query": state["query"],
#             "agent_scratchpad": state.get("agent_scratchpad", [])
#         })
        
#         # Update state with results
#         state["final_response"] = result.get("output", "")
#         state["agent_scratchpad"] = result.get("intermediate_steps", [])
        
#         return state
#     except Exception as e:
#         logger.error(f"Error in process_agent_step: {str(e)}", exc_info=True)
#         state["error"] = str(e)
#         return state

# # Node definitions
# def create_agent_flow() -> StateGraph:
#     logger.info("Creating agent flow")
    
#     # Create the graph
#     workflow = StateGraph(AgentState)
    
#     # Add nodes
#     workflow.add_node("agent", process_agent_step)
    
#     # Define edges
#     workflow.add_edge("agent", END)
    
#     # Set entry point
#     workflow.set_entry_point("agent")
    
#     logger.info("Agent flow created successfully")
#     return workflow.compile()

# # Create the graph instance
# banking_agent_flow = create_agent_flow() 
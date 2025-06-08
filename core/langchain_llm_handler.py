from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, Generation
from typing import Any, List, Optional, Dict
from pydantic import Field, validator
import logging
import requests
import json

logger = logging.getLogger(__name__)

class OllamaLLMHandler(LLM):
    """
    LangChain LLM Handler for Ollama - optimized for chain compatibility.
    """
    
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="qwen2")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    timeout: int = Field(default=12000)
    
    @validator('base_url')
    def validate_base_url(cls, v):
        return v.rstrip('/')
    
    def _call_ollama_api(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Make API call to Ollama server."""
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                }
            }
            
            if self.max_tokens:
                request_data["options"]["num_predict"] = self.max_tokens
            
            if stop:
                request_data["options"]["stop"] = stop
            
            logger.info("Call LLM with request:",request_data)
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["response"].strip()
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            raise RuntimeError(f"Failed to get response from Ollama: {str(e)}")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Required method for LangChain LLM base class."""
        return self._call_ollama_api(prompt, stop=stop, **kwargs)
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Required method for batch generation."""
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=response)])
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        """Required property for LLM identification."""
        return "ollama"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Required property for LLM parameters."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }

# from langchain.llms.base import LLM
# from langchain.callbacks.manager import CallbackManagerForLLMRun
# from langchain.llms import Ollama
# from langchain.schema import LLMResult, Generation
# from typing import Any, List, Optional, Dict
# from config import settings
# import logging
# from pydantic import Field, root_validator

# logger = logging.getLogger(__name__)

# class LangChainLLMHandler(LLM):
#     """LangChain LLM Handler that implements LLM for agent compatibility."""
    
#     model: Ollama = Field(default=None)
#     base_url: str = Field(default=settings.OLLAMA_BASE_URL)
#     model_name: str = Field(default=settings.OLLAMA_LANGUAGE_MODEL)
#     temperature: float = Field(default=settings.DEFAULT_LLM_TEMPERATURE)
    
#     @root_validator(pre=True)
#     def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
#         """Validate that the environment is properly set up."""
#         try:
#             values["model"] = Ollama(
#                 base_url=values.get("base_url", settings.OLLAMA_BASE_URL),
#                 model=values.get("model_name", settings.OLLAMA_LANGUAGE_MODEL),
#                 temperature=values.get("temperature", settings.DEFAULT_LLM_TEMPERATURE)
#             )
#             logger.info(f"Initialized LangChainLLMHandler with model: {values['model_name']}")
#             return values
#         except Exception as e:
#             logger.error(f"Error initializing Ollama model: {str(e)}", exc_info=True)
#             raise

#     def _generate(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> LLMResult:
#         """Generate text from the model."""
#         try:
#             # Remove callbacks from kwargs to prevent serialization issues
#             kwargs.pop('callbacks', None)
            
#             generations = []
#             for prompt in prompts:
#                 response = self.model.invoke(prompt, stop=stop, **kwargs)
#                 generations.append([Generation(text=response)])
#             return LLMResult(generations=generations)
#         except Exception as e:
#             logger.error(f"Error generating text: {str(e)}", exc_info=True)
#             raise

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """Call the LLM with the given prompt."""
#         try:
#             # Remove callbacks from kwargs to prevent serialization issues
#             kwargs.pop('callbacks', None)
            
#             response = self.model.invoke(prompt, stop=stop, **kwargs)
#             return response
#         except Exception as e:
#             logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
#             raise

#     @property
#     def _llm_type(self) -> str:
#         """Return the type of LLM."""
#         return "ollama"

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         """Get the identifying parameters."""
#         return {
#             "model_name": self.model_name,
#             "temperature": self.temperature,
#             "base_url": self.base_url
#         }

#     def predict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
#         """Predict text output for a single input."""
#         return self.model.invoke(text, stop=stop, **kwargs)

#     def predict_messages(self, messages: List[dict], *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
#         """This model doesn't support chat format, so we flatten messages."""
#         combined_prompt = "\n".join([msg.get("content", "") for msg in messages])
#         return self.predict(combined_prompt, stop=stop, **kwargs)

#     def generate_prompt(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any
#     ) -> LLMResult:
#         """Generate from prompts (used by LangChain chains)."""
#         return self._generate(prompts, stop=stop, **kwargs)

#     async def apredict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
#         raise NotImplementedError("Async support not implemented.")

#     async def apredict_messages(self, messages: List[dict], *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
#         raise NotImplementedError("Async support not implemented.")

#     async def agenerate_prompt(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any
#     ) -> LLMResult:
#         raise NotImplementedError("Async support not implemented.")

#     def __call__(self, input: str, **kwargs: Any) -> str:
#         return self.predict(input, **kwargs)

#     def invoke(
#         self,
#         input: str,
#         stop: Optional[List[str]] = None,
#         **kwargs: Any
#     ) -> str:
#         try:
#             return self.model.invoke(input, stop=stop, **kwargs)
#         except Exception as e:
#             logger.error(f"LLM invocation failed: {e}", exc_info=True)
#             raise
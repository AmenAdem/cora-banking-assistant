from typing import Any, Dict, List, Optional, Union
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
import logging
from config import settings
import json

logger = logging.getLogger(__name__)

class GenericLLMHandler(LLM, BaseModel):
    """Generic LLM handler that supports multiple LLM providers."""
    
    provider: str = Field(default="ollama", description="LLM provider (ollama, gemini, openai)")
    model_name: str = Field(default="llama2", description="Model name for the selected provider")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for the provider")
    
    _llm: Optional[Union[OllamaLLM, ChatGoogleGenerativeAI, ChatOpenAI]] = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_provider(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate provider-specific settings."""
        provider = values.get('provider', 'ollama')
        
        if provider == 'ollama':
            if not values.get('base_url'):
                values['base_url'] = settings.OLLAMA_BASE_URL
        elif provider == 'gemini':
            if not values.get('api_key'):
                values['api_key'] = settings.GEMINI_API_KEY
        elif provider == 'openai':
            if not values.get('api_key'):
                values['api_key'] = settings.OPENAI_API_KEY
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return values
    
    def _get_llm(self) -> Union[OllamaLLM, ChatGoogleGenerativeAI, ChatOpenAI]:
        """Initialize and return the appropriate LLM instance."""
        if self._llm is not None:
            return self._llm
            
        try:
            if self.provider == 'ollama':
                self._llm = OllamaLLM(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url=self.base_url
                )
            elif self.provider == 'gemini':
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    google_api_key=self.api_key,
                    max_output_tokens=self.max_tokens
                )
            elif self.provider == 'openai':
                self._llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            logger.info(f"Initialized {self.provider} LLM with model {self.model_name}")
            return self._llm
            
        except Exception as e:
            logger.error(f"Error initializing {self.provider} LLM: {str(e)}")
            raise
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM with the given prompt."""
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt, stop=stop, **kwargs)
            
            # Handle response extraction based on LLM provider
            if self.provider == "ollama":
                # OllamaLLM returns an object with .content (string)
                return str(getattr(response, "content", response))
            elif self.provider == "gemini":
                # ChatGoogleGenerativeAI returns a dict with 'candidates' or .content
                # Try to extract the most likely text
                if hasattr(response, "content"):
                    return str(response.content)
                elif isinstance(response, dict):
                    # Gemini API may return a dict with 'candidates'
                    candidates = response.get("candidates")
                    if candidates and isinstance(candidates, list):
                        # Try to get the first candidate's content
                        content = candidates[0].get("content")
                        if content:
                            return str(content)
                    # Fallback: try 'text' key
                    if "text" in response:
                        return str(response["text"])
                    # Fallback: return the whole dict as string
                    return str(response)
                else:
                    return str(response)
            elif self.provider == "openai":
                # ChatOpenAI returns an object with .content (string)
                return str(getattr(response, "content", response))
            else:
                # Unknown provider, just return stringified response
                return str(response)
        except Exception as e:
            logger.error(f"Error calling {self.provider} LLM: {str(e)}")
            raise
    
    def parse_json_response(self, response: str) -> Any:
        """Utility to clean and parse JSON from LLM responses, handling code blocks and language tags."""
        raw = response.strip()
        if raw.startswith("```"):
            # Remove triple backticks and optional language tag
            if raw.startswith("```json"):
                raw = raw[len("```json"):].strip()
            else:
                raw = raw[len("```"):].strip()
            # Remove trailing triple backticks if present
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        return json.loads(raw)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return f"generic_{self.provider}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Example usage:
# ollama_llm = GenericLLMHandler(provider="ollama", model_name="llama2")
# gemini_llm = GenericLLMHandler(provider="gemini", model_name="gemini-pro")
# openai_llm = GenericLLMHandler(provider="openai", model_name="gpt-3.5-turbo")



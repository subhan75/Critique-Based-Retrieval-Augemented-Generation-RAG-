"""
OpenAI API utilities with retry logic and cost tracking.
"""
import time
import os
from typing import Dict, Any, Optional, List
from functools import wraps
import openai
from dotenv import load_dotenv

from .logger import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class CostTracker:
    """Track OpenAI API costs."""
    
    # Pricing per 1M tokens (as of Nov 2025)
    PRICING = {
        'text-embedding-3-small': {
            'input': 0.02,  # per 1M tokens
        },
        'gpt-4o-mini': {
            'input': 0.15,   # per 1M tokens
            'output': 0.60,  # per 1M tokens
        }
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.calls = {
            'embedding': 0,
            'completion': 0,
        }
        self.tokens = {
            'embedding_input': 0,
            'completion_input': 0,
            'completion_output': 0,
        }
    
    def track_embedding(self, model: str, tokens: int):
        """Track embedding API call."""
        self.calls['embedding'] += 1
        self.tokens['embedding_input'] += tokens
        cost = (tokens / 1_000_000) * self.PRICING.get(model, {}).get('input', 0)
        self.total_cost += cost
        logger.debug(f"Embedding cost: ${cost:.6f} ({tokens} tokens)")
    
    def track_completion(self, model: str, input_tokens: int, output_tokens: int):
        """Track completion API call."""
        self.calls['completion'] += 1
        self.tokens['completion_input'] += input_tokens
        self.tokens['completion_output'] += output_tokens
        
        input_cost = (input_tokens / 1_000_000) * self.PRICING.get(model, {}).get('input', 0)
        output_cost = (output_tokens / 1_000_000) * self.PRICING.get(model, {}).get('output', 0)
        cost = input_cost + output_cost
        self.total_cost += cost
        logger.debug(f"Completion cost: ${cost:.6f} (in:{input_tokens}, out:{output_tokens})")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            'total_cost': self.total_cost,
            'total_calls': sum(self.calls.values()),
            'calls_breakdown': self.calls.copy(),
            'tokens_breakdown': self.tokens.copy(),
        }
    
    def reset(self):
        """Reset all counters."""
        self.total_cost = 0.0
        self.calls = {'embedding': 0, 'completion': 0}
        self.tokens = {
            'embedding_input': 0,
            'completion_input': 0,
            'completion_output': 0,
        }


# Global cost tracker
cost_tracker = CostTracker()


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}"
                    )
                    
                    time.sleep(min(delay, max_delay))
                    delay *= exponential_base
            
            raise last_exception
        
        return wrapper
    return decorator


def setup_openai_client(api_key: Optional[str] = None) -> openai.OpenAI:
    """
    Set up OpenAI client with API key.
    
    Args:
        api_key: OpenAI API key (if None, loads from environment)
        
    Returns:
        Configured OpenAI client
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set OPENAI_API_KEY environment variable or pass api_key parameter."
        )
    
    # Create client instance (new OpenAI SDK style)
    client = openai.OpenAI(api_key=api_key)
    logger.info("OpenAI client configured successfully")
    return client


@retry_with_exponential_backoff(max_retries=3)
def create_embedding(
    text: str | List[str],
    model: str = "text-embedding-3-small",
    client: Optional[openai.OpenAI] = None
) -> List[List[float]]:
    """
    Create embeddings using OpenAI API with retry logic.
    Latest OpenAI v1.54+ syntax.
    
    Args:
        text: Single text or list of texts to embed
        model: Embedding model name
        client: Optional OpenAI client (creates new one if None)
        
    Returns:
        List of embedding vectors
    """
    if isinstance(text, str):
        text = [text]
    
    # Create client if not provided
    if client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        client = openai.OpenAI(api_key=api_key)
    
    # Call embeddings API (v1.54+ syntax)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    
    # Track cost
    total_tokens = response.usage.total_tokens
    cost_tracker.track_embedding(model, total_tokens)
    
    # Extract embeddings from response
    embeddings = [item.embedding for item in response.data]
    return embeddings


@retry_with_exponential_backoff(max_retries=3)
def create_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    max_tokens: int = 256,
    response_format: Optional[Dict[str, str]] = None,
    client: Optional[openai.OpenAI] = None
) -> str:
    """
    Create chat completion using OpenAI API with retry logic.
    Latest OpenAI v1.54+ syntax.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        response_format: Optional response format (e.g., {"type": "json_object"})
        client: Optional OpenAI client (creates new one if None)
        
    Returns:
        Generated text response
    """
    # Create client if not provided
    if client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        client = openai.OpenAI(api_key=api_key)
    
    # Build kwargs for API call
    kwargs = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    
    # Add response format if specified
    if response_format:
        kwargs['response_format'] = response_format
    
    # Call chat completions API (v1.54+ syntax)
    response = client.chat.completions.create(**kwargs)
    
    # Track cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_tracker.track_completion(model, input_tokens, output_tokens)
    
    # Extract content from response
    return response.choices[0].message.content


def track_api_call(model: str, input_tokens: int, output_tokens: int):
    """
    Track an API call for cost calculation.
    
    Args:
        model: Model name (e.g., 'gpt-4o-mini')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    cost_tracker.track_completion(model, input_tokens, output_tokens)


def get_cost_summary() -> Dict[str, Any]:
    """Get current cost tracking summary."""
    summary = cost_tracker.get_summary()
    # Add more readable format
    summary['total_tokens'] = (
        summary['tokens_breakdown']['completion_input'] + 
        summary['tokens_breakdown']['completion_output'] +
        summary['tokens_breakdown']['embedding_input']
    )
    summary['input_tokens'] = (
        summary['tokens_breakdown']['completion_input'] +
        summary['tokens_breakdown']['embedding_input']
    )
    summary['output_tokens'] = summary['tokens_breakdown']['completion_output']
    return summary


def reset_cost_tracker():
    """Reset the cost tracker."""
    cost_tracker.reset()
    logger.info("Cost tracker reset")

"""
LLM setup with Claude Haiku 4.5.
"""
from langchain_anthropic import ChatAnthropic
from .config import CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE


def create_llm():
    """
    Create and return Claude LLM instance.
    
    Returns:
        ChatAnthropic instance configured with Claude Haiku 4.5
    """
    if not CLAUDE_API_KEY:
        raise ValueError(
            "CLAUDE_API_KEY environment variable is required. "
            "Set it in your .env file or environment."
        )
    
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=CLAUDE_API_KEY,
        temperature=CLAUDE_TEMPERATURE
    )
    
    return llm


# Global LLM instance (initialized on import)
llm = None


def get_llm():
    """
    Get or create the global LLM instance.
    
    Returns:
        ChatAnthropic instance
    """
    global llm
    if llm is None:
        llm = create_llm()
    return llm


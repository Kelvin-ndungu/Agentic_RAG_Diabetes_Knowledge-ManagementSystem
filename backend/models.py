"""
Pydantic models for structured outputs and API requests/responses.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import MessagesState


class QuerySafetyClassification(BaseModel):
    """
    Classification of query relevance and safety.
    First checks relevance, then safety only if relevant.
    """
    is_relevant: bool = Field(description="Whether the query is relevant to diabetes management and care")
    is_safe: Optional[bool] = Field(default=False, description="Whether the query is safe to answer (only assessed if relevant, False if not relevant)")
    risk_level: Literal["none", "low", "medium", "high"] = Field(default="none", description="Risk level (only assessed if relevant and unsafe)")
    reasoning: str = Field(description="Brief explanation for the classification decision")


class Source(BaseModel):
    """Source citation for generated response."""
    title: str = Field(description="Title of the source section")
    url: str = Field(description="URL path to the source")
    chunk_id: str = Field(description="Chunk ID from ChromaDB")


class FollowUpAnalysis(BaseModel):
    """Analysis of whether a message is a follow-up to previous conversation."""
    is_followup: bool = Field(description="Whether the message is a follow-up to previous conversation")
    reasoning: str = Field(description="Explanation for the decision")


class ClassifierOutput(BaseModel):
    """Single LLM call output for all classification logic"""
    # Query understanding
    intent: Optional[str] = Field(None, description="Contextually-aware rephrased query for retrieval (only for substantive queries)")
    
    # Classification
    query_type: Literal["greeting", "about_system", "substantive", "irrelevant", "unsafe"] = Field(
        description="Type of user query"
    )
    is_relevant: bool = Field(description="Is query about diabetes management/care")
    is_safe: bool = Field(description="Is safe to answer without personalized medical advice")
    
    # Direct response for non-substantive queries
    direct_response: Optional[str] = Field(None, description="Complete response for greetings/about_system/irrelevant/unsafe queries")
    
    # Routing
    should_generate: bool = Field(description="Whether to proceed to generator node")
    
    # User feedback (for streaming)
    status_message: str = Field(description="Status update for user (e.g., 'Understanding your query...')")


class GeneratorOutput(BaseModel):
    """Generator node structured output"""
    response: str = Field(description="Final answer with inline citations")
    has_sufficient_info: bool = Field(description="Whether sufficient chunks were found")
    sources_used: List[str] = Field(default_factory=list, description="List of source URLs used")


class ChatState(MessagesState):
    """
    Optimized state schema with structured outputs.
    Matches the notebook implementation for 2-LLM-call optimization.
    """
    # Classifier outputs
    classifier_output: Optional[ClassifierOutput]
    
    # Retrieval (programmatic)
    retrieved_chunks: List[Dict]
    
    # Generator outputs
    generator_output: Optional[GeneratorOutput]
    sources: List[Source]
    
    # Final response
    final_response: Optional[str]
    
    # Legacy fields (kept for backward compatibility during transition)
    classification: Optional[QuerySafetyClassification] = None
    is_followup: bool = False


# API Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(description="User's message/query")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")


class ClearChatRequest(BaseModel):
    """Request model for clearing chat."""
    session_id: str = Field(description="Session ID to clear")


class ChatResponse(BaseModel):
    """Response model for chat endpoint (non-streaming)."""
    content: str = Field(description="Assistant's response")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    session_id: str = Field(description="Session ID for conversation continuity")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="ok")


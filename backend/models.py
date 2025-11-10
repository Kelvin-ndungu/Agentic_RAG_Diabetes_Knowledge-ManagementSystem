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


class ChatState(MessagesState):
    """
    State schema using MessagesState for chat template support.
    Extends MessagesState with custom fields for classification, retrieval, and sources.
    """
    # Classification result
    classification: Optional[QuerySafetyClassification]
    
    # Retrieved chunks (replaced on new retrieval)
    retrieved_chunks: List[Dict]
    
    # Source citations for response
    sources: List[Source]
    
    # Whether message is a follow-up (determined by LLM)
    is_followup: bool


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


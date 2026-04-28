"""Pydantic models for API and workflow state."""
import re
import uuid
from typing import List, Dict, Optional, Literal

from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import MessagesState


class Source(BaseModel):
    """Source citation for generated response."""

    title: str = Field(description="Title of the source section")
    url: str = Field(description="URL path to the source")
    chunk_id: str = Field(description="Chunk ID from ChromaDB")


class ClassifierOutput(BaseModel):
    """Single LLM call output for routing and intent."""

    # Justification: simpler schema reduces brittleness and makes routing explicit.
    route: Literal["direct", "retrieve"] = Field(description="Routing decision for the query")
    safety: Literal["safe", "unsafe", "irrelevant"] = Field(description="Safety/relevance classification")
    intent: Optional[str] = Field(None, description="Rephrased query for retrieval (only when route=retrieve)")
    direct_response: Optional[str] = Field(None, description="Direct response for non-retrieval paths")
    status_message: str = Field(description="User-facing status update for streaming")


class GeneratorOutput(BaseModel):
    """Generator node structured output."""

    response: str = Field(description="Final answer with inline citations")
    has_sufficient_info: bool = Field(description="Whether sufficient chunks were found")
    sources_used: List[str] = Field(default_factory=list, description="List of source URLs used")


class ChatState(MessagesState):
    """
    Optimized state schema with structured outputs.
    """

    classifier_output: Optional[ClassifierOutput]
    retrieved_chunks: List[Dict]
    generator_output: Optional[GeneratorOutput]
    sources: List[Source]
    final_response: Optional[str]


class ChatRequest(BaseModel):
    """Request model for chat endpoint with input validation."""

    message: str = Field(description="User's message/query", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty")

        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty after trimming whitespace")

        if len(v) > 2000:
            raise ValueError("Message exceeds maximum length of 2000 characters")

        if len(v) < 1:
            raise ValueError("Message must be at least 1 character")

        if re.search(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", v):
            v = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", v)

        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None

        v = v.strip()
        if not v:
            return None

        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError("session_id must be a valid UUID format")


class ClearChatRequest(BaseModel):
    """Request model for clearing chat."""

    session_id: str = Field(description="Session ID to clear", min_length=1)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id cannot be empty")

        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError("session_id must be a valid UUID format")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="ok")

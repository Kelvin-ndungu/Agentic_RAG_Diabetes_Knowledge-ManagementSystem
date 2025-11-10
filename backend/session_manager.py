"""
Session manager for maintaining conversation state.
"""
import uuid
from typing import Dict, Optional
from backend.models import ChatState


class SessionManager:
    """
    Manages conversation sessions in memory.
    Each session maintains its own ChatState.
    """
    
    def __init__(self):
        """Initialize session manager with empty storage."""
        self.sessions: Dict[str, ChatState] = {}
    
    def get_session(self, session_id: Optional[str] = None) -> tuple[str, ChatState]:
        """
        Get or create a session.
        
        Args:
            session_id: Optional session ID. If None, creates a new session.
            
        Returns:
            Tuple of (session_id, ChatState)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            # Create new session with empty state
            self.sessions[session_id] = ChatState(
                messages=[],
                classification=None,
                retrieved_chunks=[],
                sources=[],
                is_followup=False
            )
        
        return session_id, self.sessions[session_id]
    
    def update_session(self, session_id: str, state: ChatState):
        """
        Update session state.
        
        Args:
            session_id: Session ID
            state: Updated ChatState
        """
        self.sessions[session_id] = state
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            True if session was cleared, False if session didn't exist
        """
        if session_id in self.sessions:
            # Reset to empty state but keep session
            self.sessions[session_id] = ChatState(
                messages=[],
                classification=None,
                retrieved_chunks=[],
                sources=[],
                is_followup=False
            )
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if session was deleted, False if session didn't exist
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


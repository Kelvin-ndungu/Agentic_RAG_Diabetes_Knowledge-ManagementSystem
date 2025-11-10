"""
Graph builder for LangGraph workflow.
"""
from langgraph.graph import StateGraph, END
from .models import ChatState
from .graph_nodes import (
    classify_query,
    route_classifier,
    not_relevant_response,
    unsafe_response,
    generator_node,
    set_chroma_reader
)


def initialize_state(state: ChatState) -> ChatState:
    """Initialize state with defaults if not present."""
    if "retrieved_chunks" not in state:
        state["retrieved_chunks"] = []
    if "sources" not in state:
        state["sources"] = []
    if "is_followup" not in state:
        state["is_followup"] = False
    return state


def build_graph(chroma_reader):
    """
    Build and compile the LangGraph workflow.
    
    Args:
        chroma_reader: ChromaDBReader instance to use for retrieval
        
    Returns:
        Compiled graph instance
    """
    # Set the chroma_reader in graph_nodes module
    set_chroma_reader(chroma_reader)
    
    # Build graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("not_relevant", not_relevant_response)
    workflow.add_node("unsafe", unsafe_response)
    workflow.add_node("generator", generator_node)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional routing from classifier
    workflow.add_conditional_edges(
        "classify",
        route_classifier,
        {
            "not_relevant": "not_relevant",
            "unsafe": "unsafe",
            "generator": "generator"
        }
    )
    
    # Add edges to END
    workflow.add_edge("not_relevant", END)
    workflow.add_edge("unsafe", END)
    workflow.add_edge("generator", END)
    
    # Compile graph
    graph = workflow.compile()
    
    return graph


"""
Graph builder for LangGraph workflow.
"""
from langgraph.graph import StateGraph, END
from .models import ChatState
from .graph_nodes import (
    classify_query_unified,
    retrieval_node,
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


def route_after_classifier(state: ChatState) -> str:
    """Route based on classifier decision"""
    classifier_output = state.get("classifier_output")
    if classifier_output and classifier_output.should_generate:
        return "retrieval"
    else:
        return END


def build_graph(chroma_reader):
    """
    Build and compile the optimized LangGraph workflow.
    
    Optimized workflow with only 2 LLM calls:
    - Classifier: Single LLM call for all classification logic
    - Generator: Single LLM call for answer generation
    
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
    workflow.add_node("classifier", classify_query_unified)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generator", generator_node)
    
    # Set entry point
    workflow.set_entry_point("classifier")
    
    # Add conditional routing after classifier
    workflow.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "retrieval": "retrieval",
            END: END
        }
    )
    
    # Linear path for substantive queries
    workflow.add_edge("retrieval", "generator")
    workflow.add_edge("generator", END)
    
    # Compile graph
    graph = workflow.compile()
    
    return graph


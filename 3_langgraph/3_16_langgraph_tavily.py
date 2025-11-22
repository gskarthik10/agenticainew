# Tavily search tool for agentic AI
# pip install  langchain-tavily  tavily-python

import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv(override=True)
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key)

class SearchState(TypedDict):
    query: str
    results: str

def internet_search(state: SearchState) -> SearchState:
    query = state["query"]
    response = tavily_client.search(
        query=query,
        max_results=5,
        topic="general",
        include_raw_content=False,
    )

    if response["results"]:
        texts = []
        for r in response["results"]:
            texts.append(r["content"][:200])
        state["results"] = "\n\n".join(texts)
    else:
        state["results"] = "No results found."

    return state 


def summarize_results(state: SearchState) -> SearchState:
    """Node 2: Summarize results simply (no LLMs, just truncation for demo)"""
    # Get the text stored in 'results' from the current state dictionary.
    # If 'results' does not exist, use an empty string "" as the default.
    text = state.get("results", "")

    # Split the text into separate lines wherever there is a newline character ("\n").
    # This creates a list where each element is one line of text.
    # Then, take only the first 5 lines to keep the summary short.
    lines = text.split("\n")[:5]

    # Create a simple summary string.
    # Start with the heading "Summary:" and then join the selected lines together.
    # Each line is separated by a newline character so that the output looks neat.
    summary = "Summary:\n" + "\n".join(lines)

    # Store this summary text back into the 'results' field of the state dictionary.
    state["results"] = summary

    # Return the updated state so that other nodes or steps in the graph can use it.
    return state

# --- Step 4: Build LangGraph ---
graph = StateGraph(SearchState)
graph.add_node("search", internet_search)
graph.add_node("summarize", summarize_results)
graph.set_entry_point("search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()


# --- Step 5: Run it ---
if __name__ == "__main__":
    query = input("Enter a topic to search: ")
    result = app.invoke({"query": query})
    print("\n" + result["results"])

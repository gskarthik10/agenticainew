# Dataset: https://www.kaggle.com/code/jayrdixit/amazon-product-dataset/input?select=amazon_products.csv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict
import pandas as pd
import gradio as gr
import os

# --- Step 1: Define state ---
class ProductState(TypedDict):
    query: str
    results: str

# --- Step 2: Load dataset and embeddings ---
csv_path = "c://code//agenticai//3_langgraph//amazon_products.csv"
df = pd.read_csv(csv_path)
# df = df.head(1000)  # Limit for demo

texts = df["title"].astype(str).tolist()
metadatas = df.to_dict(orient="records")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Step 3: Setup Chroma vector store ---
persist_dir = "c://code//agenticai//3_langgraph//chromadb"
collection_name = "products_collection"

# Import chromadb client to check existing collections
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path=persist_dir, settings=Settings())

existing_collections = [col.name for col in client.list_collections()]

if collection_name in existing_collections:
    print(f"Loading existing collection '{collection_name}'...")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
else:
    print(f"Creating new collection '{collection_name}'...")
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    vectordb.persist()

# --- Step 4: Define LangGraph nodes ---
def search_products(state: ProductState) -> ProductState:
    results = vectordb.similarity_search(state["query"], k=3)
    titles = [doc.metadata.get("title", "Unknown Product") for doc in results]
    state["results"] = "\n".join([f"• {title}" for title in titles])
    return state

def format_response(state: ProductState) -> ProductState:
    state["results"] = f"Found products:\n{state['results']}" if state["results"] else "No products found."
    return state

graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
runnable = graph.compile()

# --- Step 5: Gradio handlers ---
def search(query):
    return runnable.invoke({"query": query})["results"]

def chat_fn(message, history):
    return search(message)

# --- Step 6: Gradio UI ---
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Product Search (Chroma + HuggingFace)",
    examples=["wireless headphones", "gaming laptop", "DSLR camera"],
)

if __name__ == "__main__":
    demo.launch()

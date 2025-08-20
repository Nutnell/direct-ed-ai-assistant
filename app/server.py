# app/server.py

from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv

# Import our chain from the other file
from .chain import qa_chain

load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="DirectEd AI Assistant Server",
    version="1.0",
    description="A simple API server for the DirectEd AI assistant.",
)

# --- Add LangServe Routes ---
# This is the magic line that exposes our chain at the /chat endpoint
add_routes(
    app,
    qa_chain,
    path="/chat",
)

# A simple health check endpoint
@app.get("/")
def read_root():
    return {"status": "DirectEd AI Assistant is running"}


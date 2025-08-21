# app/server.py

from fastapi import FastAPI
from langserve import add_routes

from .chain import qa_chain

# FastAPI App Initialization
app = FastAPI(
    title="DirectEd AI Assistant Server",
    version="1.0",
    description="A simple API server for the DirectEd AI assistant.",
)

# Add LangServe Routes

# add the typed chain to the /chat endpoint
add_routes(
    app,
    qa_chain,
    path="/chat",
)

@app.get("/")
def read_root():
    return {"status": "DirectEd AI Assistant is running"}


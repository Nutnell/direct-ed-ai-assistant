# app/server.py

from fastapi import FastAPI
from langserve import add_routes

# Import orchestrated router chain
from app.chains.router import router_chain

# FastAPI App Initialization
app = FastAPI(
    title="DirectEd AI Assistant Server",
    version="1.0",
    description="A multi-functional API server for the DirectEd AI assistant.",
)

# LangServe Routes

# expose the unified router_chain at the /chat endpoint
add_routes(
    app,
    router_chain,
    path="/chat",
)

@app.get("/")
def read_root():
    return {"status": "DirectEd AI Assistant is running"}
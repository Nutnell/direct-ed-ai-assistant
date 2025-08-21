# app/server.py

from fastapi import FastAPI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

# memory-enabled router chain
from app.chains.router import router_chain_with_history

# FastAPI App Initialization
app = FastAPI(
    title="DirectEd AI Assistant Server",
    version="1.0",
    description="A multi-functional API server for the DirectEd AI assistant.",
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

# LangServe Routes
add_routes(
    app,
    router_chain_with_history,
    path="/chat",
)

@app.get("/")
def read_root():
    return {"status": "DirectEd AI Assistant is running"}
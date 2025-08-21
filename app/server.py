# app/server.py

import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.chains.router import router_chain_with_history

load_dotenv()

# API Key Security
API_KEY = os.getenv("BACKEND_SECRET_KEY")
API_KEY_NAME = "X-API-Key"

api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header_scheme)):
    """Dependency function to validate the API key."""
    if not api_key_header or api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )
    return api_key_header

# FastAPI App Initialization
app = FastAPI(
    title="DirectEd AI Assistant Server",
    version="1.0",
    description="A multi-functional API server for the DirectEd AI assistant.",
    dependencies=[Depends(get_api_key)],
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add LangServe Routes
add_routes(
    app,
    router_chain_with_history,
    path="/chat",
)

@app.get("/")
def read_root():
    return {"status": "DirectEd AI Assistant is running"}
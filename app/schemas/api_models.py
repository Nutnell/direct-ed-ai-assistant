# app/schemas/api_models.py

from pydantic import BaseModel, Field
from typing import List, Literal

# This defines the structure of the input for our API
class ChatInput(BaseModel):
    input: str = Field(
        ...,
        description="The user's question or the topic for content generation.",
        examples=["What is the difference between MLOps and LLMOps?"]
    )
    user_type: Literal["student", "instructor"] = Field(
        ...,
        description="The type of user making the request.",
        examples=["student"]
    )
    request_type: Literal["tutoring", "quiz_generation"] = Field(
        ...,
        description="The type of request, e.g., a tutoring question or a request to generate a quiz.",
        examples=["tutoring"]
    )

# This defines the structure of a single source document
class Source(BaseModel):
    source: str = Field(..., description="The URL of the source document.")
    name: str = Field(..., description="The name of the source (e.g., 'DirectEd Curriculum').")

# This defines the structure of the output from our API
class ChatOutput(BaseModel):
    answer: str = Field(..., description="The AI-generated answer or content.")
    sources: List[Source] = Field(..., description="A list of source documents used for the answer.")
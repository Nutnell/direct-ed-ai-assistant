# app/prompts/templates.py

from langchain_core.prompts import ChatPromptTemplate

# Prompt for the main RAG (tutoring) chain
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant for the DirectEd learning platform.
Answer the user's question based only on the following context.
Cite the source name and URL if possible.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


# Prompt for the quiz generation chain
QUIZ_GENERATOR_PROMPT_TEMPLATE = """
You are an expert quiz creator for a tech learning platform.
Your task is to create a 3-question multiple-choice quiz based on the provided context.
The questions should be challenging and relevant to the context.
Provide the question, four options (A, B, C, D), and the correct answer.

Format your response as follows:
1. [Question 1]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Correct Answer: [A, B, C, or D]

2. [Question 2]
   ...

Context:
{context}

Quiz Questions:
"""

quiz_generator_prompt = ChatPromptTemplate.from_template(QUIZ_GENERATOR_PROMPT_TEMPLATE)
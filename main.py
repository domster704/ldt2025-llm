import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from prompt import SYSTEM_PROMPT

app = FastAPI(title="LLM API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8005")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

llm = ChatOpenAI(
    model=MODEL_PATH if MODEL_PATH else "Qwen/Qwen3-8B",
    openai_api_key="EMPTY",  # type: ignore
    openai_api_base=f"{LLM_SERVER_URL}/v1/",  # type: ignore
    request_timeout=30,
    max_retries=5,
    temperature=TEMPERATURE,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.get("/health")
async def health_check():
    try:
        # Test LLM connectivity with a simple request
        test_prompt = ChatPromptTemplate([("user", "test")])
        test_result = await llm.ainvoke(test_prompt)
        llm_status = "healthy"
    except Exception as e:
        llm_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "service": "langchain-llama-api",
        "llm_status": llm_status,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):

    user_prompt = "Запрос: {input}"
    prompt = ChatPromptTemplate([("system", SYSTEM_PROMPT), ("user", user_prompt)])

    prompt = prompt.invoke({"input": request.message})

    try:
        result = await llm.ainvoke(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )

    response = result.content

    return ChatResponse(response=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

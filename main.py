import json
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from prompt import SYSTEM_PROMPT

app = FastAPI(title="LLM API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Validate LLM connection on startup."""
    try:
        test_prompt = ChatPromptTemplate([("user", "test")])
        await llm.ainvoke(test_prompt)
        print("✅ LLM connection validated successfully")
    except Exception as e:
        print(f"⚠️  Warning: LLM connection validation failed: {e}")
        print("Service will start but may not function properly")


# Environment variables validation
MODEL_PATH = os.getenv("MODEL_PATH", "")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8005")
TEMPERATURE_STR = os.getenv("TEMPERATURE", "0.0")

try:
    TEMPERATURE = float(TEMPERATURE_STR)
except ValueError:
    raise ValueError(
        f"Invalid TEMPERATURE value: {TEMPERATURE_STR}. Must be a valid float."
    )

if not LLM_SERVER_URL.startswith(("http://", "https://")):
    raise ValueError(
        f"Invalid LLM_SERVER_URL: {LLM_SERVER_URL}. Must start with http:// or https://"
    )

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
    model_kwargs={"response_format": {"type": "json_object"}},
)


class SummaryOutput(BaseModel):
    срок_беременности: str = Field(
        ..., description="Точный срок в неделях и днях, например: 38+3"
    )
    метод_определения_срока: str = Field(..., description="По ПМ или УЗИ I триместра")
    родовой_анамнез: str = Field(
        ..., description="Первородящая/повторнородящая, количество и исходы родов"
    )
    текущая_беременность_плоды: str = Field(
        ..., description="Один плод или многоплодная"
    )
    гестоз: str = Field(
        ...,
        description="Гестоз/преэклампсия: степень, АД, протеинурия или 'нет информации'",
    )
    сахарный_диабет: str = Field(
        ...,
        description="Гестационный/манифестный, уровень гликемии или 'нет информации'",
    )
    анемия: str = Field(..., description="Степень, Hb или 'нет информации'")
    инфекции: str = Field(
        ..., description="ОРВИ, COVID-19, температура и т.д. или 'нет информации'"
    )
    хронические_заболевания: str = Field(
        ..., description="Гипертония, пороки сердца и др. или 'нет информации'"
    )
    узи_предлежание: str = Field(
        ..., description="Предлежание плода: головное, тазовое и т.д."
    )
    узи_масса_плода: str = Field(
        ..., description="Крупный плод, ЗВУР, норма или 'нет информации'"
    )
    узи_околоплодные_воды: str = Field(..., description="Норма, маловодие, многоводие")
    узи_плацента: str = Field(
        ..., description="Степень зрелости плаценты: 0, I, II, III"
    )
    узи_допплер_маточные: str = Field(
        ..., description="PI, RI, S/D маточных артерий или 'нет информации'"
    )
    узи_допплер_пуповина: str = Field(
        ..., description="PI, RI, S/D артерии пуповины, реверсный/нулевой кровоток"
    )
    узи_допплер_сма: str = Field(..., description="PI СМА плода или 'нет информации'")
    узи_допплер_аорта: str = Field(
        ..., description="Признаки гипертензии плода или 'нет информации'"
    )
    шейка_матки_бишоп: str = Field(..., description="Степень зрелости по Бишопу")
    шейка_матки_раскрытие: str = Field(
        ..., description="Раскрытие в см на момент мониторинга"
    )
    период_родов_дородовой: str = Field(
        ..., description="Оценка состояния плода перед родами или 'нет информации'"
    )
    период_родов_i: str = Field(
        ..., description="I период: длительность, раскрытие или 'нет информации'"
    )
    период_родов_ii: str = Field(
        ..., description="II период: потуги или 'нет информации'"
    )
    функциональные_пробы_вас: str = Field(
        ..., description="Реакция на виброакустическую стимуляцию или 'нет информации'"
    )
    ктг_предыдущие: str = Field(
        ...,
        description="Результаты предыдущих КТГ по Савельевой/FIGO или 'нет информации'",
    )
    лаб_группа_крови: str = Field(
        ..., description="Группа крови, резус, анти-D антитела"
    )
    лаб_скрининги: str = Field(
        ..., description="Результаты УЗИ скринингов или 'нет информации'"
    )
    лаб_общий_анализ_крови: str = Field(
        ..., description="Hb, лейкоциты, тромбоциты или 'нет информации'"
    )
    лаб_биохимия: str = Field(
        ..., description="Глюкоза, печеночные пробы и т.д. или 'нет информации'"
    )
    лаб_коагулограмма: str = Field(
        ..., description="Фибриноген, D-димер и т.д. или 'нет информации'"
    )
    лаб_инфекции: str = Field(
        ..., description="ВИЧ, сифилис, гепатиты B/C или 'нет информации'"
    )
    лаб_torch: str = Field(
        ..., description="Антитела к краснухе, ЦМВ, герпесу или 'нет информации'"
    )
    лаб_анализ_мочи: str = Field(
        ..., description="Белок, лейкоциты, эритроциты или 'нет информации'"
    )


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: SummaryOutput


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

    user_prompt = f"На основе следующих данных составь анамнез строго по структуре JSON: {request.message}"
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                SYSTEM_PROMPT
                + '\n\n# Требования к выводу:\n- Отвечай ТОЛЬКО в формате JSON.\n- Используй ТОЛЬКО поля из указанной структуры.\n- Если информация отсутствует или нерелевантна — пиши "нет информации".\n- Ничего не придумывай. Не добавляй комментариев.',
            ),
            ("user", user_prompt),
        ]
    )

    prompt = prompt.invoke({"input": request.message})

    try:
        chain = prompt | llm
        result = await chain.ainvoke({})
        raw_content = result.content.strip()

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка парсинга JSON ответа от LLM: {str(e)}. Полученный ответ: {raw_content[:200]}...",
            )

        try:
            validated = SummaryOutput(**data)
            return ChatResponse(response=validated)
        except ValidationError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка валидации данных: {str(e)}. Полученные данные: {data}",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации структурированного ответа: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

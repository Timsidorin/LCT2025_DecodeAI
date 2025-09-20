# schemas/predict.py
from typing import List
from pydantic import BaseModel, Field


class ReviewInput(BaseModel):
    id: int = Field(..., description="Уникальный идентификатор отзыва")
    text: str = Field(..., description="Текст отзыва")


class PredictRequest(BaseModel):
    data: List[ReviewInput] = Field(..., description="Список отзывов для анализа")


class PredictionOutput(BaseModel):
    id: int = Field(..., description="ID отзыва")
    topics: List[str] = Field(..., description="Выявленные темы")
    sentiments: List[str] = Field(..., description="Тональность для каждой темы")


class PredictResponse(BaseModel):
    predictions: List[PredictionOutput] = Field(..., description="Результаты анализа")

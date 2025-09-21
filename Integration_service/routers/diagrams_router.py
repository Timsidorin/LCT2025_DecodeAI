from typing import List
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import UUID4
from starlette import status



diagrams_router = APIRouter(prefix="/training", tags=["Для диаграмм"])

@diagrams_router.post("/create_training", name="Создание тренинга")
async def diagrams(
):
   pass





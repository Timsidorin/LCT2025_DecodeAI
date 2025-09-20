FROM python:3.10-alpine

WORKDIR /ml_service

COPY ./requirements.txt /ml_service/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /ml_service/requirements.txt


COPY ./ml_service /ml_service/

CMD ["python", "main.py"]

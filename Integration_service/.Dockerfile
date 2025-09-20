FROM python:3.10-alpine

WORKDIR /Integration_service

COPY ./requirements.txt /Integration_service/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /Integration_service/requirements.txt


COPY ./Integration_service /Integration_service/

CMD ["python", "main.py"]

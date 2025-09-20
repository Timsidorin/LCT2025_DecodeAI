FROM python:3.10-alpine

WORKDIR /data_mining_service

COPY ./requirements.txt /data_mining_service/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /data_mining_service/requirements.txt

COPY ./data_mining_service /data_mining_service/

CMD ["python", "main.py"]

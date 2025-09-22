FROM python:3.10-alpine

WORKDIR /Manager_bot

COPY ./requirements.txt /Manager_bot/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /Manager_bot/requirements.txt


COPY ./Manager_bot /Manager_bot/

CMD ["python", "main.py"]

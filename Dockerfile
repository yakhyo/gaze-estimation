FROM python:3.12

COPY ./requirements.txt /app/requirements.txt
RUN pip install /app/requirements.txt

COPY . /app
WORKDIR /app
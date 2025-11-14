FROM python:3.10

WORKDIR /app

COPY fastapi_server /app/fastapi_server
COPY fastapi_server/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 10000

CMD ["uvicorn", "fastapi_server.main:app", "--host", "0.0.0.0", "--port", "10000"]

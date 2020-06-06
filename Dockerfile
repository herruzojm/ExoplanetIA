FROM python:3.7-slim
COPY /app/. /app
COPY /requirements.txt .
COPY /Procfile .
RUN pip install -r requirements.txt
WORKDIR /app
ENTRYPOINT ["python", "runsite.py"]
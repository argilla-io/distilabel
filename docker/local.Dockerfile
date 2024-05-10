FROM python:3.11-slim 

WORKDIR /app

RUN pip install "distilabel[argilla]"

EXPOSE 80

CMD ["distilabel"]

ARG PYTHON_VERSION="3.11-slim"

FROM python:${PYTHON_VERSION}

WORKDIR /app

RUN pip install "distilabel[argilla]"

EXPOSE 80

CMD ["distilabel"]

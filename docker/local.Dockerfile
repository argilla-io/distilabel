FROM python:3.11-slim AS base

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

FROM python:3.11-slim 

COPY . .

RUN pip install -e ".[argilla,hf-transformers,hf-inference-endpoints]"

EXPOSE 80

CMD ["distilabel"]

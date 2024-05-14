FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 AS base

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

COPY . .

RUN pip install -e ".[argilla,hf-transformers,hf-inference-endpoints,llama-cpp,vllm]"

EXPOSE 80

CMD ["distilabel"]
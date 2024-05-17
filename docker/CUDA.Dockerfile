# Use the specified base image
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 AS base

# Run system updates and clean up
RUN apt-get update && \
    apt-get install python3 python3-pip -y

RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHON=/usr/bin/python

ARG TORCH="2.2.0"

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch==${TORCH}

# Set the working directory to /
WORKDIR /app

FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

COPY . .

RUN pip install -e ".[argilla,hf-transformers,hf-inference-endpoints,llama-cpp,openai,vllm]"

EXPOSE 80

CMD ["distilabel"]
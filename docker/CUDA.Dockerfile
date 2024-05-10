# Use the specified base image
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 AS base

# Run system updates and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /
WORKDIR /app

FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN pip install "distilabel[argilla,hf-transformers,hf-inference-endpoints,llama-cpp,vllm]" openai

EXPOSE 80

CMD ["distilabel"]
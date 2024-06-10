# Arguments
ARG PYTORCH_VERSION=2.1.1
ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.2.0

# Using runpod base image and installing torch, cuda and ubuntu
FROM runpod/pytorch:${PYTORCH_VERSION}-py3.10-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# Run system updates and clean up
RUN apt-get update && \
    apt-get install python3 python3-pip -y

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHON=/usr/bin/python

# Install torch
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch==${TORCH_VERSION}

# Set the working directory to /
WORKDIR /

FROM runpod/pytorch:${PYTORCH_VERSION}-py3.10-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

COPY . .

# Installing distilabel with GPU-related dependencies
RUN pip install -e ".[argilla,hf-transformers,hf-inference-endpoints,llama-cpp,vllm]"

EXPOSE 80

CMD ["distilabel"]
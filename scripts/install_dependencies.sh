#!/bin/bash

set -e

python_version=$(python -c "import sys; print(sys.version_info[:2])")

python -m pip install uv

uv pip install --system -e ".[anthropic,argilla,cohere,groq,hf-inference-endpoints,hf-transformers,litellm,llama-cpp,ollama,openai,outlines,vertexai,mistralai,instructor,sentence-transformers,faiss-cpu,minhash,text-clustering]"

if [ "${python_version}" != "(3, 12)" ]; then
    uv pip install --system -e .[ray]
fi

./scripts/install_cpu_vllm.sh

uv pip install --system -e ".[dev,tests]"

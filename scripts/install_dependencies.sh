#!/bin/bash

set -e

python_version=$(python -c "import sys; print(sys.version_info[:2])")

python -m pip install uv

uv pip install --system -e ".[dev,tests,anthropic,argilla,cohere,groq,hf-inference-endpoints,hf-transformers,litellm,llama-cpp,ollama,openai,outlines,vertexai]"
if [ "${python_version}" != "(3, 8)" ]; then
	uv pip install --system -e .[mistralai,instructor]
fi

uv pip install --system git+https://github.com/argilla-io/LLM-Blender.git

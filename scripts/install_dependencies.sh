#!/bin/bash

python_version=$(python -c "import sys; print(sys.version_info[:2])")

pip install -e ".[dev,tests,anthropic,argilla,cohere,groq,hf-inference-endpoints,hf-transformers,litellm,llama-cpp,ollama,openai,outlines,vertexai]"
if [ "${python_version}" != "(3, 8)" ]; then
	pip install -e .[mistralai,instructor]
fi
if [ "${python_version}" != "(3, 12)" ]; then
	pip install -e .[vllm]
fi
pip install git+https://github.com/argilla-io/LLM-Blender.git

---
description: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
---

# distilabel

AI Feedback (AIF) framework to build datasets with and for LLMs:

- Integrations with the most popular libraries and APIs for LLMs: HF Transformers, OpenAI, vLLM, etc.
- Multiple tasks for Self-Instruct, Preference datasets and more.
- Dataset export to Argilla for easy data exploration and further annotation.

## Installation

```sh
pip install distilabel
```
Requires Python 3.8+

In addition, the following extras are available:

- `anthropic`: for using models available in [Anthropic API](https://www.anthropic.com/api) via the `AnthropicLLM` integration.
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).
- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `hf-inference-endpoints`: for using the [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `litellm`: for using [`LiteLLM`](https://github.com/BerriAI/litellm) to call any LLM using OpenAI format via the `LiteLLM` integration.
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) Python bindings for `llama.cpp` via the `LlamaCppLLM` integration.
- `mistralai`: for using models available in [Mistral AI API](https://mistral.ai/news/la-plateforme/) via the `MistralAILLM` integration.
- `ollama`: for using [Ollama](https://ollama.com/) and their available models via `OllamaLLM` integration.
- `openai`: for using [OpenAI API](https://openai.com/blog/openai-api) models via the `OpenAILLM` integration.
- `together`: for using [Together Inference](https://www.together.ai/products) via their Python client.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.
- `vertexai`: for using [Google Vertex AI](https://cloud.google.com/vertex-ai) proprietary models via the `VertexAILLM` integration.

## Quick example

ADD SHOWCASE EXAMPLE

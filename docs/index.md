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

- `anthropic`: for using [anthropic](https://github.com/anthropics/anthropic-sdk-python) models.
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).
- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `hf-inference-endpoints`: for using the [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `litellm`: for calling LLM APIs using OpenAI format via [litellm](https://github.com/BerriAI/litellm).
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) as Python bindings for `llama.cpp`.
- `mistralai`: for using [mistral AI](https://docs.mistral.ai/platform/endpoints/) models.
- `ollama`: for using [Ollama](https://github.com/ollama/ollama) and their available models via their Python client.
- `openai`: for using OpenAI API models via the `OpenAILLM` integration.
- `together`: for using [Together Inference](https://www.together.ai/products) via their Python client.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.
- `vertexai`: for using both [Google Vertex AI](https://cloud.google.com/vertex-ai/?&gad_source=1&hl=es) offerings: their proprietary models and endpoints via their Python client [`google-cloud-aiplatform`](https://github.com/googleapis/python-aiplatform).

## Quick example

ADD SHOWCASE EXAMPLE

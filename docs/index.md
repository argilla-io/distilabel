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

```python
--8<-- "docs/snippets/quick-example.py"
```

1. Create a `Task` for generating text given an instruction.
2. Create a `LLM` for generating text using the `Task` created in the first step. As the `LLM` will generate text, it will be a `generator`.
3. Create a pre-defined `Pipeline` using the `pipeline` function and the `generator` created in step 2. The `pipeline` function
will create a `labeller` LLM using `OpenAILLM` with the `UltraFeedback` task for instruction following assessment.

!!! note
    To run the script successfully, ensure you have assigned your OpenAI API key to the `OPENAI_API_KEY` environment variable.

For a more complete example, check out our awesome tutorials in the docs or the example below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/argilla-io/distilabel/blob/main/docs/tutorials/pipeline-notus-instructions-preferences-legal.ipynb) [![Open Source in Github](https://img.shields.io/badge/github-view%20source-black.svg)](https://github.com/argilla-io/distilabel/blob/main/docs/tutorials/pipeline-notus-instructions-preferences-legal.ipynb)

## Navigation

<div class="grid cards" markdown>

-   <p align="center"> [**Concept Guides**](./technical-reference/llms.md)</p>

    ---

    Understand the components and their interactions.

-   <p align="center"> [**API Reference**](./reference/distilabel/index.md)</p>

    ---

    Technical description of the classes and functions.

</div>

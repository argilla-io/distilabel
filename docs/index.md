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
- `cohere`: for using models available in [Cohere](https://cohere.ai/) via the `CohereLLM` integration.
- `hf-inference-endpoints`: for using the [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `litellm`: for using [`LiteLLM`](https://github.com/BerriAI/litellm) to call any LLM using OpenAI format via the `LiteLLM` integration.
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) Python bindings for `llama.cpp` via the `LlamaCppLLM` integration.
- `mistralai`: for using models available in [Mistral AI API](https://mistral.ai/news/la-plateforme/) via the `MistralAILLM` integration.
- `ollama`: for using [Ollama](https://ollama.com/) and their available models via `OllamaLLM` integration.
- `openai`: for using [OpenAI API](https://openai.com/blog/openai-api) models via the `OpenAILLM` integration, or the rest of the integrations based on OpenAI and relying on its client as `AnyscaleLLM`, `AzureOpenAILLM`, and `TogetherLLM`.
- `vertexai`: for using [Google Vertex AI](https://cloud.google.com/vertex-ai) proprietary models via the `VertexAILLM` integration.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.

## Quick example

```python
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset, TextGenerationToArgilla
from distilabel.steps.tasks import TextGeneration

with Pipeline("pipe-name", description="My first pipe") as pipeline:
    load_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    generate_with_openai = TextGeneration(
        name="generate_with_openai", llm=OpenAILLM(model="gpt-4-0125-preview")
    )

    to_argilla = TextGenerationToArgilla(
        name="to_argilla", dataset_name="text-generation-with-gpt4"
    )

    load_dataset.connect(generate_with_openai)
    generate_with_openai.connect(to_argilla)


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            "generate_with_openai": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                }
            },
            "to_argilla": {
                "api_url": "https://cloud.argilla.io",
                "api_key": "i.love.argilla",
            },
        },
    )
    distiset.push_to_hub(
        "distilabel-internal-testing/instruction-dataset-mini-with-generations"
    )
```

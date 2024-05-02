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
- `groq`: for using models available in [Groq](https://groq.com/) using [`groq`](https://github.com/groq/groq-python) Python client via the `GroqLLM` integration.
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

To run the following example you must install `distilabel` with both `openai` extra:

```sh
pip install "distilabel[openai]" --upgrade
```

Then run:

```python
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset
from distilabel.steps.tasks import TextGeneration

with Pipeline(
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:
    load_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    generate_with_openai = TextGeneration(
        name="generate_with_gpt35", llm=OpenAILLM(model="gpt-3.5-turbo")
    )

    load_dataset.connect(generate_with_openai)

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            "generate_with_gpt35": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                }
            },
        },
    )
```

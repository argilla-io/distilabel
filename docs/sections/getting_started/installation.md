---
description: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
hide:
  - toc
---

# Installation

!!! NOTE
    Since `distilabel` v1.0.0 was recently released, we refactored most of the stuff, so the installation below only applies to `distilabel` v1.0.0 and above.

You will need to have at least Python 3.9 or higher, up to Python 3.12, since support for the latter is still a work in progress.

To install the latest release of the package from PyPI you can use the following command:

```sh
pip install distilabel --upgrade
```

Alternatively, you may also want to install it from source i.e. the latest unreleased version, you can use the following command:

```sh
pip install "distilabel @ git+https://github.com/argilla-io/distilabel.git@develop" --upgrade
```

!!! NOTE
    We are installing from `develop` since that's the branch we use to collect all the features, bug fixes, and improvements that will be part of the next release. If you want to install from a specific branch, you can replace `develop` with the branch name.

## Extras

Additionally, as part of `distilabel` some extra dependencies are available, mainly to add support for some of the LLM integrations we support. Here's a list of the available extras:

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

## Recommendations / Notes

The [`mistralai`](https://github.com/mistralai/client-python) dependency requires Python 3.9 or higher, so if you're willing to use the `distilabel.llms.MistralLLM` implementation, you will need to have Python 3.9 or higher.

In some cases like [`transformers`](https://github.com/huggingface/transformers) and [`vllm`](https://github.com/vllm-project/vllm) the installation of [`flash-attn`](https://github.com/Dao-AILab/flash-attention) is recommended if you are using a GPU accelerator, since it will speed up the inference process, but the installation needs to be done separately, as it's not included in the `distilabel` dependencies.

```sh
pip install flash-attn --no-build-isolation
```

Also, if you are willing to use the [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) integration for running local LLMs, note that the installation process may get a bit trickier depending on which OS are you using, so we recommend you to read through their [Installation](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation) section in their docs.

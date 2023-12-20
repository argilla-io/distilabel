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

- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `hf-inference-endpoints`: for using the [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `openai`: for using OpenAI API models via the `OpenAILLM` integration.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) as Python bindings for `llama.cpp`.
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).

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

For a more complete example, check out our awesome notebook on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rO1-OlLFPBC0KPuXQOeMpZOeajiwNoMy?usp=sharing)

## Navigation

<div class="grid cards" markdown>

-   <p align="center"> [**Tutorials**](./learn/tutorials/)</p>

    ---

    End to end project lessons.

-   <p align="center"> [**User Guides**](./learn/user-guides/)</p>

    ---

    Practical guides to achieve specific tasks with `distilabel`.

-   <p align="center"> [**Concept Guides**](./technical-reference/llms.md)</p>

    ---

    Understand the components and their interactions.

-   <p align="center"> [**API Reference**](./reference/distilabel)</p>

    ---

    Technical description of the classes and functions.

</div>
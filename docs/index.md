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

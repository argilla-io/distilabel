<div align="center">
  <h1>⚗️ distilabel</h1>
  <p><em>AI Feedback (AIF) framework for building datasets with and for LLMs.</em></p>
</div>

> [!TIP]
> To discuss, get support, or give feedback [join Argilla's Slack Community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g) and you will be able to engage with our amazing community and also with the core developers of `argilla` and `distilabel`.

![overview](https://github.com/argilla-io/distilabel/assets/36760800/360110da-809d-4e24-a29b-1a1a8bc4f9b7)

## Features

- Integrations with the most popular libraries and APIs for LLMs: HF Transformers, OpenAI, vLLM, etc.
- Multiple tasks for Self-Instruct, Preference datasets and more.
- Dataset export to Argilla for easy data exploration and further annotation.

> [!WARNING]
> `distilabel` is currently under active development and we're iterating quickly, so take into account that we may introduce breaking changes in the releases during the upcoming weeks, and also the `README` might be outdated the best place to get started is the [documentation](http://distilabel.argilla.io/).

## Installation

```sh
pip install distilabel --upgrade
```

Requires Python 3.8+

In addition, the following extras are available:

- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `hf-inference-endpoints`: for using the [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `openai`: for using OpenAI API models via the `OpenAILLM` integration.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) as Python bindings for `llama.cpp`.
- `together`: for using [Together Inference](https://www.together.ai/products) via their Python client.
- `vertexai`: for using both [Google Vertex AI](https://cloud.google.com/vertex-ai/?&gad_source=1&hl=es) offerings: their proprietary models and endpoints via their Python client [`google-cloud-aiplatform`](https://github.com/googleapis/python-aiplatform).
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).

## Example

To run the following example you must install `distilabel` with both `openai` and `argilla` extras:

```sh
pip install "distilabel[openai,argilla]" --upgrade
```

Then run the following example:

```python
from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

# Create a `Task` for generating text given an instruction.
task = TextGenerationTask()

# Create a `LLM` for generating text using the `Task` created in
# the first step. As the `LLM` will generate text, it will be a `generator`.
generator = OpenAILLM(task=task, max_new_tokens=512)

# Create a pre-defined `Pipeline` using the `pipeline` function and the
# `generator` created in step 2. The `pipeline` function will create a
# `labeller` LLM using `OpenAILLM` with the `UltraFeedback` task for
# instruction following assessment.
pipeline = pipeline("preference", "instruction-following", generator=generator)

dataset = pipeline.generate(dataset)
```

Additionally, you can push the generated dataset to Argilla for further exploration and annotation:

```python
import argilla as rg

rg.init(api_url="<YOUR_ARGILLA_API_URL>", api_key="<YOUR_ARGILLA_API_KEY>")

# Convert the dataset to Argilla format
rg_dataset = dataset.to_argilla()

# Push the dataset to Argilla
rg_dataset.push_to_argilla(name="preference-dataset", workspace="admin")
```

## More examples

Find more examples of different use cases of `distilabel` under [`examples/`](./examples/).

Or check out the following Google Colab Notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rO1-OlLFPBC0KPuXQOeMpZOeajiwNoMy?usp=sharing)

## Badges

If you build something cool with `distilabel` consider adding one of these badges to your dataset or model card.

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

## Contribute

To directly contribute with `distilabel`, check our [good first issues](https://github.com/argilla-io/distilabel/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [open a new one](https://github.com/argilla-io/distilabel/issues/new/choose).

## References

* [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)
* [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](https://arxiv.org/abs/2310.17631)
* [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

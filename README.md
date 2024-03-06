<div align="center">
  <h1>⚗️ distilabel</h1>
</div>
<h3 align="center">Synthesize data for AI and add feedback on the fly!</h2>

<p align="center">
<a  href="https://pypi.org/project/distilabel/">
<img alt="CI" src="https://img.shields.io/pypi/v/distilabel.svg?style=flat-round&logo=pypi&logoColor=white">
</a>
<a href="https://pepy.tech/project/distilabel">
<img alt="CI" src="https://static.pepy.tech/personalized-badge/distilabel?period=month&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads/month">
</a>
</p>

<p align="center">
<a href="https://twitter.com/argilla_io">
<img src="https://img.shields.io/badge/twitter-black?logo=x"/>
</a>
<a href="https://www.linkedin.com/company/argilla-io">
<img src="https://img.shields.io/badge/linkedin-blue?logo=linkedin"/>
</a>
<a href="https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g">
<img src="https://img.shields.io/badge/slack-purple?logo=slack"/>
</a>
</p>

Distilabel is the **synthetic data and AI feedback framework for AI engineers** that require **high-quality outputs, full data ownership, and overall efficiency**.

If you just want to get started, we recommend you check the [documentation](http://distilabel.argilla.io/). Curious, and want to know more? Keep reading!
<!-- ![overview](https://github.com/argilla-io/distilabel/assets/36760800/360110da-809d-4e24-a29b-1a1a8bc4f9b7)  -->

## Why use Distilabel?

Whether you are working on **a predictive model** that computes semantic similarity or the next **generative model** that is going to beat the LLM benchmarks. Our framework ensures that the **hard data work pays off**.

### Improve your AI output quality through data quality

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time on **achieve and keep high-quality standards for your data**.

### Take control of your data and models

**Ownership of data for fine-tuning your own LLMs** is not easy but Distilabel can help you to get started. We integrate **AI feedback from any LLM provider out there** using one unified API.

### Improve efficiency by quickly iterating on the right research and LLMs

Synthesize and judge data with **latest research papers** while ensuring **flexibility, scalability and fault tolerance**. So you can focus on improving your data and training your models.

## 🏘️ Community

We are an open-source community-driven project and we love to hear from you. Here are some ways to get involved:

- [Community Meetup](https://lu.ma/embed-checkout/evt-IQtRiSuXZCIW6FB): listen in or present during one of our bi-weekly events.

- [Slack](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g): get direct support from the community.

- [Roadmap](https://github.com/orgs/argilla-io/projects/10/views/1): plans change but we love to discuss those with our community so feel encouraged to participate.

## What do people build with Distilabel?

Distilabel is a tool that can be used to **synthesize data and provide AI feedback**. Our community uses Distilabel to create amazing [datasets](https://huggingface.co/datasets?other=distilabel) and [models](https://huggingface.co/models?other=distilabel), and **we love contributions to open-source** ourselves too.

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of ~1 million AI preferences derived from teknium/OpenHermes-2.5. It shows how we can use Distilabel to **synthesize data at immense scale**.
- Our [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) and the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B),, show how we **improve model performance by filtering out 50%** of the original dataset through **AI feedback**.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) outlines how anyone can create a **dataset for a specific task** and **the latest research papers** to improve the quality of the dataset.

## 👨🏽‍💻 Installation

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
- `ollama`: for using [Ollama](https://github.com/ollama/ollama) and their available models via their Python client.
- `together`: for using [Together Inference](https://www.together.ai/products) via their Python client.
- `anyscale`: for using [Anyscale endpoints](https://www.anyscale.com/endpoints).
- `ollama`: for using [Ollama](https://ollama.ai/).
- `mistralai`: for using [Mistral AI](https://docs.mistral.ai/platform/endpoints/) via their Python client.
- `vertexai`: for using both [Google Vertex AI](https://cloud.google.com/vertex-ai/?&gad_source=1&hl=es) offerings: their proprietary models and endpoints via their Python client [`google-cloud-aiplatform`](https://github.com/googleapis/python-aiplatform).
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).

### Example

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

Find more examples of different use cases of `distilabel` under [`examples/`](./examples/).

## Badges

If you build something cool with `distilabel` consider adding one of these badges to your dataset or model card.

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

## Contribute

To directly contribute with `distilabel`, check our [good first issues](https://github.com/argilla-io/distilabel/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [open a new one](https://github.com/argilla-io/distilabel/issues/new/choose).


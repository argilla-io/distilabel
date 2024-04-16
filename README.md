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

Distilabel is the **framework for synthetic data and AI feedback for AI engineers** that require **high-quality outputs, full data ownership, and overall efficiency**.

If you just want to get started, we recommend you check the [documentation](http://distilabel.argilla.io/). Curious, and want to know more? Keep reading!
<!-- ![overview](https://github.com/argilla-io/distilabel/assets/36760800/360110da-809d-4e24-a29b-1a1a8bc4f9b7)  -->

## Why use Distilabel?

Whether you are working on **a predictive model** that computes semantic similarity or the next **generative model** that is going to beat the LLM benchmarks. Our framework ensures that the **hard data work pays off**. Distilabel is the missing piece that helps you **synthesize data** and provide **AI feedback**.

### Improve your AI output quality through data quality

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time on **achieveing and keeping high-quality standards for your data**.

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

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of ~1 million AI preferences derived from teknium/OpenHermes-2.5. It shows how we can use Distilabel to **synthesize data on an immense scale**.
- Our [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) and the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B),, show how we **improve model performance by filtering out 50%** of the original dataset through **AI feedback**.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) outlines how anyone can create a **dataset for a specific task** and **the latest research papers** to improve the quality of the dataset.

## 👨🏽‍💻 Installation

```sh
pip install distilabel --upgrade
```

Requires Python 3.8+

In addition, the following extras are available:

- `anthropic`: for using models available in [Anthropic API](https://www.anthropic.com/api) via the `AnthropicLLM` integration.
- `cohere`: for using models available in [Cohere](https://cohere.ai/) via the `CohereLLM` integration.
- `argilla`: for exporting the generated datasets to [Argilla](https://argilla.io/).
- `hf-inference-endpoints`: for using the [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) via the `InferenceEndpointsLLM` integration.
- `hf-transformers`: for using models available in [transformers](https://github.com/huggingface/transformers) package via the `TransformersLLM` integration.
- `litellm`: for using [`LiteLLM`](https://github.com/BerriAI/litellm) to call any LLM using OpenAI format via the `LiteLLM` integration.
- `llama-cpp`: for using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) Python bindings for `llama.cpp` via the `LlamaCppLLM` integration.
- `mistralai`: for using models available in [Mistral AI API](https://mistral.ai/news/la-plateforme/) via the `MistralAILLM` integration.
- `ollama`: for using [Ollama](https://ollama.com/) and their available models via `OllamaLLM` integration.
- `openai`: for using [OpenAI API](https://openai.com/blog/openai-api) models via the `OpenAILLM` integration, or the rest of the integrations based on OpenAI and relying on its client as `AnyscaleLLM`, `AzureOpenAILLM`, and `TogetherLLM`.
- `vertexai`: for using [Google Vertex AI](https://cloud.google.com/vertex-ai) proprietary models via the `VertexAILLM` integration.
- `vllm`: for using [vllm](https://github.com/vllm-project/vllm) serving engine via the `vLLM` integration.

### Example

To run the following example you must install `distilabel` with both `openai` and `argilla` extras:

```sh
pip install "distilabel[openai,argilla]" --upgrade
```

Then run the following example:

```python
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset, TextGenerationToArgilla
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
        name="generate_with_gpt3.5", llm=OpenAILLM(model="gpt-3.5-turbo")
    )

    to_argilla = TextGenerationToArgilla(name="to_argilla")

    load_dataset.connect(generate_with_openai)
    generate_with_openai.connect(to_argilla)


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
            "to_argilla": {
                "dataset_name": "text-generations-with-gpt35",
                "dataset_workspace": "admin",
            },
        },
    )
    distiset.push_to_hub(
        "distilabel-internal-testing/instruction-dataset-mini-with-generations"
    )
```

Distilabel integrates smoothly with Argilla and provides all the necessary configurations to make giving a final human touch as easy as possible.


## Badges

If you build something cool with `distilabel` consider adding one of these badges to your dataset or model card.

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

    [<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

[<img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-dark.png" alt="Built with Distilabel" width="200" height="32"/>](https://github.com/argilla-io/distilabel)

## Contribute

To directly contribute with `distilabel`, check our [good first issues](https://github.com/argilla-io/distilabel/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [open a new one](https://github.com/argilla-io/distilabel/issues/new/choose).


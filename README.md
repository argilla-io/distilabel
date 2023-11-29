<div align="center">
  <h1>⚗️ distilabel</h1>
  <p><em>AI Feedback (AIF) framework for building datasets and labellers with LLMs</em></p>
</div>

![overview](https://github.com/argilla-io/distilabel/assets/36760800/360110da-809d-4e24-a29b-1a1a8bc4f9b7)

> [!TIP]
> To discuss, get support, or give feedback [join Argilla's Slack Community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g) and you will be able to engage with our amazing community and also with the core developers of `argilla` and `distilabel`.

## What's `distilabel`?

`distilabel` is a framework for AI engineers to align LLMs using RLHF-related methods (e.g. reward models, DPO).

The initial focus is LLM fine-tuning and adaptation but we'll be extending it for predictive NLP use cases soon.

Main use cases are:

1. As an AI engineer I want to **build domain-specific instruction datasets** to fine-tune OSS LLMs with increased accuracy.
2. As an AI engineer I want to **build domain-specific and diverse preference datasets** to use RLHF-related methods and align LLMs (e.g, increase the ability to follow instructions or give truthful responses).

> [!WARNING]
> `distilabel` is currently under active development and we're iterating quickly, so take into account that we may introduce breaking changes in the releases during the upcoming weeks, and also the `README` might be outdated the best place to get started is the [documentation](http://distilabel.argilla.io/).

## Motivation

🔥 Recent projects like [Zephyr](https://huggingface.co/collections/HuggingFaceH4/zephyr-7b-6538c6d6d5ddd1cbb1744a66) and [Tulu](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101) have shown it's possible to **build powerful open-source models with DPO and AI Feedback** (AIF) datasets. 

👩‍🔬 There's a lot of exciting research in the AIF space, such as [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) (the dataset leveraged by Zephyr and Tulu), [JudgeLM](https://github.com/baaivision/JudgeLM), or [Prometheus](https://huggingface.co/kaist-ai/prometheus-13b-v1.0). 

🚀 However, going beyond research efforts and applying AIF at scale it's different. For enterprise and production use, we need framework that implements **key AIF methods on a robust, efficient and scalable way**. This framework should enable AI engineers to build custom datasets at scale for their own use cases. 

👩‍🎓 This, combined with humans-in-the-loop for improving dataset quality is the next big leap for OSS LLM models. 

⚗️ `distilabel` aims to bridge this gap.

## Key features

* 🤖 **Leverage OSS models and APIs**: 🤗 transformers, OpenAI, 🤗 Inference Endpoints, vLLM, llama.cpp, and more to come.

* 💻 **Scalable and extensible**: Scalable implementations of existing methods (e.g. UltraFeedback). Easily extensible to build and configure your own labellers.

* 🧑‍🦱 **Human-in-the-loop**: One line of code integration with Argilla to improve and correct datasets.

## Quickstart

### Installation

Install with `pip` (requires Python 3.8+):

```bash
pip install distilabel[openai,argilla]
```

### Try it out

After installing, you can immediately start experimenting with `distilabel`:

- **Explore Locally**: Follow the example below to build a preference dataset for DPO/RLHF.
- **Interactive Notebook**: Prefer an interactive experience? Try our Google Colab Notebook!

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rO1-OlLFPBC0KPuXQOeMpZOeajiwNoMy?usp=sharing)

### Example: Build a preference dataset for DPO/RLHF

```python
from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

# Load a dataset with instructions from the Hub
dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:5]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

# Use `OpenAILLM` (running `gpt-3.5-turbo`) to generate responses for given inputs
generator = OpenAILLM(
    task=TextGenerationTask(),
    max_new_tokens=512,
    # openai_api_key="sk-...",
)

pipeline = pipeline("preference", "instruction-following", generator=generator)

# Build a preference dataset comparing two responses focused on the instruction-following skill of the LLM
dataset = pipeline.generate(dataset)
```

The resulting dataset can already be used for preference tuning (a larger version of it). But beware these AIF dataset are imperfect. To get the most out of AIF, push to Argilla for human feedback:

```python
import argilla as rg

rg.init(
    api_key="<YOUR_ARGILLA_API_KEY>",
    api_url="<YOUR_ARGILLA_API_URL>"
)

rg_dataset = dataset.to_argilla()
rg_dataset.push_to_argilla(name="preference-dataset", workspace="admin")
```

https://github.com/argilla-io/distilabel/assets/1107111/be34c95c-8be4-46ef-9437-cbd2a7687e30

### More examples

Find more examples of different use cases of `distilabel` under [`examples/`](./examples/).

## Roadmap

- [ ] Add Critique Models and support for Prometheus OSS
- [ ] Add a generator with multiple models
- [ ] Train OSS labellers to replace OpenAI labellers
- [ ] Add labellers to evolve instructions generated with self-instruct
- [ ] Add labellers for predictive NLP tasks: text classification, information extraction, etc.
- [ ] Open an issue to suggest a feature!

## Contribute

To directly contribute with `distilabel`, check our [good first issues](https://github.com/argilla-io/distilabel/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [open a new one](https://github.com/argilla-io/distilabel/issues/new/choose).

## References

* [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)
* [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](https://arxiv.org/abs/2310.17631)
* [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

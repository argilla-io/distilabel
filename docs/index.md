---
description: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
hide:
  - navigation
---

<style>.md-typeset h1, .md-content__button { display: none;}</style>

<div align="center">
  <picture>
    <img alt="Distilabel Logo" src="./assets/distilabel-white.svg#only-dark" width="400">
    <img alt="Distilabel Logo" src="./assets/distilabel-black.svg#only-light" width="400">
  </picture>
</div>

<h3 align="center">Synthesize data for AI and add feedback on the fly!</h3>

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
  <a href="http://hf.co/join/discord">
  <img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
  </a>
</p>

Distilabel is the framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers.

If you just want to get started, we recommend you check the [documentation](http://distilabel.argilla.io/). Curious, and want to know more? Keep reading!

## Why use distilabel?

Distilabel can be used for generating synthetic data and AI feedback for a wide variety of projects including traditional predictive NLP (classification, extraction, etc.), or generative and large language model scenarios (instruction following, dialogue generation, judging etc.). Distilabel's programmatic approach allows you to build scalable pipelines data generation and AI feedback. The goal of distilabel is to accelerate your AI development by quickly generating high-quality, diverse datasets based on verified research methodologies for generating and judging with AI feedback.

### Improve your AI output quality through data quality

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time **achieving and keeping high-quality standards for your synthetic data**.

### Take control of your data and models

**Ownership of data for fine-tuning your own LLMs** is not easy but distilabel can help you to get started. We integrate **AI feedback from any LLM provider out there** using one unified API.

### Improve efficiency by quickly iterating on the right research and LLMs

Synthesize and judge data with **latest research papers** while ensuring **flexibility, scalability and fault tolerance**. So you can focus on improving your data and training your models.

## What do people build with distilabel?

The Argilla community uses distilabel to create amazing [datasets](https://huggingface.co/datasets?other=distilabel) and [models](https://huggingface.co/models?other=distilabel). Additonally, Argilla contributed to open-source too:

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of ~1 million AI preferences that have been generated using the [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) LLM. It is a great example on how you can use distilabel to scale and increase dataset development.
- [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) used to fine tune the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B). This dataset was built combining Argilla human curation with distilabel, leading to an improved version of the Intel Orca dataset and outperforming models fine tuned on the original dataset.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) is an example how anyone can create a synthetic dataset for a specific task, which after curation and evaluation can be used for fine-tuning custom LLMs.

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

<div class="grid cards" markdown>

-  __Get started in 5 minutes!__

    ---

    Install distilabel with `pip` and run your first `Pipeline` to generate and evaluate synthetic data.

    [:octicons-arrow-right-24: Quickstart](./sections/getting_started/quickstart.md)

-  __How-to guides__

    ---

    Get familiar with the basics of distilabel. Learn how to define `steps`, `tasks` and `llms` and run your `Pipeline`.

    [:octicons-arrow-right-24: Learn more](./sections/how_to_guides/index.md)

</div>

## Why use distilabel?

Distilabel can be used for generating synthetic data and AI feedback for a wide variety of projects including traditional predictive NLP (classification, extraction, etc.), or generative and large language model scenarios (instruction following, dialogue generation, judging etc.). Distilabel's programmatic approach allows you to build scalable pipelines for data generation and AI feedback. The goal of distilabel is to accelerate your AI development by quickly generating high-quality, diverse datasets based on verified research methodologies for generating and judging with AI feedback.

<p style="font-size:20px">Improve your AI output quality through data quality</p>

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time **achieving and keeping high-quality standards for your synthetic data**.

<p style="font-size:20px">Take control of your data and models</p>

**Ownership of data for fine-tuning your own LLMs** is not easy but distilabel can help you to get started. We integrate **AI feedback from any LLM provider out there** using one unified API.

<p style="font-size:20px">Improve efficiency by quickly iterating on the right data and models</p>

Synthesize and judge data with **latest research papers** while ensuring **flexibility, scalability and fault tolerance**. So you can focus on improving your data and training your models.

## What do people build with distilabel?

The Argilla community uses distilabel to create amazing [datasets](https://huggingface.co/datasets?other=distilabel) and [models](https://huggingface.co/models?other=distilabel).

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of ~1 million AI preferences derived from teknium/OpenHermes-2.5. It shows how we can use Distilabel to **synthesize data on an immense scale**.
- Our [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) and the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B), show how we **improve model performance by filtering out 50%** of the original dataset through **AI feedback**.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) outlines how anyone can create a **dataset for a specific task** and **the latest research papers** to improve the quality of the dataset.

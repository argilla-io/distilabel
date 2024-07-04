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

Distilabel is the **framework for synthetic data and AI feedback for AI engineers** that require **high-quality outputs, full data ownership, and overall efficiency**.

If you just want to get started, we recommend you check the [documentation](http://distilabel.argilla.io/). Curious, and want to know more? Keep reading!

## Why use Distilabel?

Whether you are working on **a predictive model** that computes semantic similarity or the next **generative model** that is going to beat the LLM benchmarks. Our framework ensures that the **hard data work pays off**. Distilabel is the missing piece that helps you **synthesize data** and provide **AI feedback**.

### Improve your AI output quality through data quality

Compute is expensive and output quality is important. We help you **focus on data quality**, which tackles the root cause of both of these problems at once. Distilabel helps you to synthesize and judge data to let you spend your valuable time on **achieveing and keeping high-quality standards for your data**.

### Take control of your data and models

**Ownership of data for fine-tuning your own LLMs** is not easy but Distilabel can help you to get started. We integrate **AI feedback from any LLM provider out there** using one unified API.

### Improve efficiency by quickly iterating on the right research and LLMs

Synthesize and judge data with **latest research papers** while ensuring **flexibility, scalability and fault tolerance**. So you can focus on improving your data and training your models.

## What do people build with Distilabel?

Distilabel is a tool that can be used to **synthesize data and provide AI feedback**. Our community uses Distilabel to create amazing [datasets](https://huggingface.co/datasets?other=distilabel) and [models](https://huggingface.co/models?other=distilabel), and **we love contributions to open-source** ourselves too.

- The [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) is a dataset of ~1 million AI preferences derived from teknium/OpenHermes-2.5. It shows how we can use Distilabel to **synthesize data on an immense scale**.
- Our [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) and the [improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B),, show how we **improve model performance by filtering out 50%** of the original dataset through **AI feedback**.
- The [haiku DPO data](https://github.com/davanstrien/haiku-dpo) outlines how anyone can create a **dataset for a specific task** and **the latest research papers** to improve the quality of the dataset.
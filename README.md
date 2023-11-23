 <div align="center">
   <h1>⚗️ distilabel</h1>
   <p>
     <em>AI Feedback framework for building datasets and labelers with LLMs</em>
   </p>
 </div>

## What's distilabel
distilabel is a framework for AI engineers to align LLM using RLHF-related methods (e.g., reward models, DPO).

The initial focus is LLM fine-tuning and adaptation but we'll be extending it for predictive NLP use cases soon.

Main use cases are:

1. As an AI engineer I want to **build domain-specific instruction datasets** to fine-tune OSS LLMs with increased accuracy.
2. As an AI engineer I want to **build domain-specific and diverse preference datasets** to use RLHF-related methods and align LLMs (e.g, increase the ability to follow instructions or give thruthful responses).

This readme might be outdated the best place to get started is the [documentation](http://distilabel.argilla.io/).

## Quickstart

Install with `pip` (requires Python 3.8+):
```sh
pip install distilabel[openai,argilla]
```

Build a preference dataset for DPO/RLHF:
```python
from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

# dataset with instructions
dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:5]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

# use gpt3.5 turbo for generating responses
task = TextGenerationTask() 

generator = OpenAILLM(
    task=task, 
    max_new_tokens=512
    #openai_api_key="sk-.."
)

# build preference dataset comparing two responses
# focusing on the instruction-following skill
pipe = pipeline("preference", "instruction-following", generator=generator)

dataset = pipe.generate(dataset, num_generations=2)
```

Push to Argilla for human feedback:

```python
import argilla as rg

rg.init(
    api_key="<YOUR_API_KEY>",
    api_url="<YOUR_ARGILLA_API_URL>"
)

rg_dataset = dataset.to_argilla()
rg_dataset.push_to_argilla(name="preference-dataset", workspace="admin")
```



https://github.com/argilla-io/distilabel/assets/1107111/be34c95c-8be4-46ef-9437-cbd2a7687e30



## Motivation
🔥 Recent projects like [Zephyr](https://huggingface.co/collections/HuggingFaceH4/zephyr-7b-6538c6d6d5ddd1cbb1744a66) and [Tulu](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101) have shown it's possible to **build powerful open-source models with DPO and AI Feedback** (AIF) datasets. 

👩‍🔬 There's a lot of exciting research in the AIF space, such as [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) (the dataset leveraged by Zephyr and Tulu), [JudgeLM](https://github.com/baaivision/JudgeLM), or [Prometheus](https://huggingface.co/kaist-ai/prometheus-13b-v1.0). 

🚀 However, going beyond research efforts and applying AIF at scale it's different. For enterprise and production use, we need framework that implements **key AIF methods on a robust, efficient and scalable way**. This framework should enable AI engineers to build custom datasets at scale for their own use cases. 

👩‍🎓 This, combined with humans-in-the-loop for improving dataset quality is the next big leap for OSS LLM models. 

⚗️ `distilabel` aims to bridge this gap.

## Key features

* 🤖 **Leverage OSS models and APIs**: HF Transformers, OpenAI, HF Inference Endpoints, vLLM, LlamaCPP, and more to come.

* 💻 **Scalable and extensible**: Scalable implementations of existing methods (e.g., UltraFeedback). Easily extensible to build and configure your own labelers.

* 🧑‍🦱 **Human-in-the-loop**: One line of code integration with Argilla to improve and correct datasets.

## Overview
![distilabel](https://github.com/argilla-io/distilabel/assets/1107111/b8e1aa40-0cd3-42df-8300-104cc59455e7)


## Roadmap

- Add Critique Models and support for Prometheus OSS
- Add a generator with multiple models
- Train OSS labelers to replace OpenAI labelers
- Add labelers to evolve instructions generated with self-instruct
- Add labelers for predictive NLP tasks: text classification, information extraction
- Open an issue to suggest a feature!

## How to generate instructions
If you don't have an instruction or prompts dataset you can generate one with our `self-instruct` inspired generator:

```python
import os
from distilabel.tasks import SelfInstructTask
from distilabel.pipeline import Pipeline
from distilabel.llm import OpenAILLM
from datasets import Dataset

math_topics = [
    "Algebraic Expressions",
    "Linear Equations",
    "Quadratic Equations",
    "Polynomial Functions",
    "Rational Expressions",
    "Exponential Functions",
    "Logarithmic Functions",
    "Sequences and Series",
    "Matrices",
    "Determinants",
    #...
]

dataset = Dataset.from_dict({
    "input": math_topics
})

# it will steer the generator
# to generate instructions for this specific app
instruction_task = SelfInstructTask(
    application_description= """
    An AI assistant adept at answering a wide array of math, logic, and reasoning puzzles, trivia, and general questions.
    """,
    num_instructions=10 # 10 instructions per input
)

# default model is: gpt3.5-turbo
# you can choose gpt-4 too
instruction_generator = OpenAILLM(
    task=instruction_task,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    num_threads=8,
    max_new_tokens=1024
)

pipeline = Pipeline(
    generator=instruction_generator
)

# will generate
distiset = pipeline.generate(
    dataset=dataset,
    # 10 instruction * 10 generations * 10 inputs = 1000 instructions
    num_generations=10, 
    batch_size=4
)
# Output:
# Number of generated instructions: 2044
# 1. Provide an explanation for solving a quadratic equation step by step.
# 2. What is the process for simplifying an algebraic expression with exponents?
# 3. Detail how to factorize a polynomial equation.
# ...
# 10. How can one determine if a given graph represents a linear or quadratic equation?
# 1. How can I simplify the algebraic expression (x^2 + 3x + 2)(2x - 1)?
# 2. Provide step-by-step instructions on how to solve the equation 4(x + 2) - 3 = 7(2x - 1).
# ...
```

---
description: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
hide:
  - toc
---

<a target="_blank" href="https://colab.research.google.com/drive/1DJFDZtOfnNYg7ZfmZPfICm750tuJLR9l">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Quickstart

Distilabel provides all the tools you need to your scalable and reliable pipelines for synthetic data generation and AI-feedback. Pipelines are used to generate data, evaluate models, manipulate data, or any other general task. They are made up of different components: Steps, Tasks and LLMs, which are chained together in a directed acyclic graph (DAG).

- **Steps**: These are the building blocks of your pipeline. Normal steps are used for basic executions like loading data, applying some transformations, or any other general task.
- **Tasks**: These are steps that rely on LLMs and prompts to perform generative tasks. For example, they can be used to generate data, evaluate models or manipulate data.
- **LLMs**: These are the models that will perform the task. They can be local or remote models, and open-source or commercial models.

Pipelines are designed to be scalable and reliable. They can be executed in a distributed manner, and they can be cached and recovered. This is useful when dealing with large datasets or when you want to ensure that your pipeline is reproducible.

Besides that, pipelines are designed to be modular and flexible. You can easily add new steps, tasks, or LLMs to your pipeline, and you can also easily modify or remove them. An example architecture of a pipeline to generate a dataset of preferences is the following:

## Installation

To install the latest release with `hf-inference-endpoints` extra of the package from PyPI you can use the following command:

```sh
pip install distilabel[hf-inference-endpoints] --upgrade
```

## Use a generic pipeline template

Distilabel comes with some built in templates for taks like Supervised Fine-Tuning. You can use these templates to generate data for your tasks. The templates are built using the `InstructionResponsePipeline` class, which uses the `InferenceEndpointsLLM` class to generate data based on the input data and the model.

### Generate Instructions and Responses

To use a generic pipeline for an ML task, you can use the `InstructionResponsePipeline` class. This class is a generic pipeline that can be used to generate data for supervised fine-tuning tasks. It uses the `InferenceEndpointsLLM` class to generate data based on the input data and the model.

```python
from distilabel.pipeline import InstructionResponsePipeline

pipeline = InstructionResponsePipeline()
dataset = pipeline.run()
```

The `InstructionResponsePipeline` class will use the `InferenceEndpointsLLM` class with the model `meta-llama/Meta-Llama-3.1-8B-Instruct` to generate data based on the system prompt. The output data will be a dataset with the columns `instruction` and `response`. The class uses a generic system prompt, but you can customize it by passing the `system_prompt` parameter to the class.

### Generate based on seed data

You can also use distilabel to generate data based on seed data. This is useful when you have an unstructured dataset that represents your domain and you want instruction response pairs for fine-tuning a model. You can use the `DatasetInstructionResponsePipeline` class with the `dataset` parameter to generate data based on the seed data.

```python
from datasets import Dataset
from distilabel.pipeline import DatasetInstructionResponsePipeline

pipeline = DatasetInstructionResponsePipeline(num_instructions=5) # define the number of instructions to generate per sample

distiset = pipeline.run(
    use_cache=False,
    dataset=Dataset.from_list(
        mapping=[
            {
                "input": "<document>",
            }
        ]
    ),
)

```



!!! note
    We're actively working on building more pipelines for different tasks. If you have any suggestions or requests, please let us know! We're currently working on pipelines for classification, Direct Preference Optimization, and Information Retrieval tasks.

## Define a Custom pipeline

In this guide we will walk you through the process of creating a simple pipeline that uses the [InferenceEndpointsLLM][distilabel.models.llms.InferenceEndpointsLLM] class to generate text. The [Pipeline][distilabel.pipeline.Pipeline] will process a dataset loaded directly using the Hugging Face `datasets` library and use the [InferenceEndpointsLLM][distilabel.models.llms.InferenceEndpointsLLM] class to generate text using the [TextGeneration][distilabel.steps.tasks.text_generation.TextGeneration] task.

> You can check the available models in the [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) and filter by `Inference status`.

```python
from datasets import load_dataset

from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline: # (1)
    TextGeneration( # (2)
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            generation_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        ),
    )

if __name__ == "__main__":
    dataset = load_dataset("distilabel-internal-testing/instructions", split="test") # (3)
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="distilabel-example") # (4)
```

1. We define a [Pipeline][distilabel.pipeline.Pipeline] using its context manager. Any [Step][distilabel.steps.base.Step] subclass defined within the context manager will be automatically added to the pipeline.
2. We define a [TextGeneration][distilabel.steps.tasks.text_generation.TextGeneration] task that uses the [InferenceEndpointsLLM][distilabel.models.llms.InferenceEndpointsLLM] class with the model Meta-Llama-3.1-8B-Instruct. The generation parameters are set directly in the LLM configuration with a temperature of 0.7 and maximum of 512 new tokens.
3. We load the dataset directly using the Hugging Face datasets library from the repository "distilabel-internal-testing/instructions" using the "test" split.
4. Optionally, we can push the generated [Distiset][distilabel.distiset.Distiset] to the Hugging Face Hub repository distilabel-example. This will allow you to share the generated dataset with others and use it in other pipelines.

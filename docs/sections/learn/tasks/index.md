# Tasks

This section presents the [`Task`][distilabel.steps.tasks.base.Task] class, a more concrete implementation of a [`Step`][distilabel.steps.base.Step] that includes extra formatting and the `LLM` to be directly used in a `Pipeline`.

## What is a Task in distilabel?

The [`Task`][distilabel.steps.tasks.base.Task] is a special type of `Step` that adds a both a `format_input` and `format_output` method to the interface, and it's directly related to the [`LLM`][distilabel.llms.base.LLM], using it as the engine to generate text for us. Let's see an example:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import TextGeneration

text_generation = TextGeneration(
    name="text-generation",
    llm=MistralLLM(
        model="mistral-tiny",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    input_batch_size=8,
    pipeline=Pipeline(name="sample-text-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
text_generation.load()
```

We have chosen the most direct type of task, [`TextGeneration`][distilabel.steps.tasks.text_generation.TextGeneration], and [`MistralLLM`][distilabel.llms.mistral.MistralLLM] with [`mistral-tiny`](https://docs.mistral.ai/platform/endpoints/), the API serving [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). Other than the `llm`, it's just another type of step, but thanks to the freedom the API let's you, they can be as complex as needed:

```python
result = next(text_generation.process([{"instruction": "What's synthetic generated text?"}]))
```

This simple task will add a `generation` entry in our task's response, let's take a look:

```python
print(result[0]["generation"])
# Synthetic generated text refers to text that is created artificially by a computer or machine learning model, rather than being written by a human. This text is often generated based on existing data or rules, and can be used for various purposes such as language translation, text summarization, chatbot responses, or even creative writing.

# One common type of synthetic generated text is known as "text-to-text" models, which generate new text based on a given input text. For example, a text-to-text model might be trained on a large dataset of English sentences, and then be able to generate a new sentence based on a given input sentence.

# Another type of synthetic generated text is known as "text-to-image" models, which generate descriptions or captions for images. These models can be useful for accessibility purposes, or for generating captions for social media or e-commerce websites.

# Synthetic generated text can be generated in a variety of ways, including using rule-based systems, statistical models, or deep learning models. The quality and accuracy of synthetic generated text can vary widely depending on the specific model and the data it was trained on.
```

### Defining your own Task

We will see how to create our own task in case we need some extra tuning other than the ones offered. We are going to recreate the generation task from [Magicoder: Source Code Is All You Need](https://arxiv.org/abs/2312.02120).

```python
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

oss_instruct_prompt = """Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in two distinct sections:
[Problem Description] and [Solution].
Code snippet for inspiration:

{code}

Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing
all the contextual information one needs to understand and solve the problem.
Assume common programming knowledge, but ensure that any specific context,
variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately
addresses the [Problem Description] you provided.
"""

class OSSInstruct(Task):
    _system_prompt: str = "You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions."
    _template: str = oss_instruct_prompt

    @property
    def inputs(self) -> List[str]:
        return ["code_snippet"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "system",
                "content": self._system_prompt,
                "role": "user",
                "content": self._template.format(code=input["code_snippet"])
            },
        ]

    @property
    def outputs(self) -> List[str]:
        return ["problem", "solution"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        problem, solution = output.split("[Solution]")
        return {
            "problem": problem.replace("[Problem Description]", "").strip(),
            "solution": solution.strip()
        }
```

Let's instantiate our custom task:

```python
import os

from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM

oss_instruct = OSSInstruct(
    name="oss-generation",
    num_instructions=1,
    input_batch_size=8,
    llm=MistralLLM(
        model="mistral-medium",
        api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
    ),
    pipeline=Pipeline(name="oss-instruct-pipeline")
)

# remember to call .load() if testing outside of a Pipeline context
oss_instruct.load()
```

And we can generate our own pair of problem and solution by calling the process method:

```python
code_snippet = """with Pipeline(name="quick-example") as pipeline:
    load_dataset = LoadHubDataset(
        name="load_dataset", output_mappings={"prompt": "instruction"}
    )"""
result = next(oss_instruct.process([{"code_snippet": code_snippet}]))
```

And we will print next the synthetic problem and solution generated starting from the code snippet:

```python
print(result[0]["problem"])
# In this problem, you will be working with the `Pipeline` class from a hypothetical machine learning library, called `mlflow`. The goal is to create a pipeline that takes a dataset from HubDataset, processes it, and saves the output to a specified location. The dataset contains instructions that a text generation model should follow to generate text. Your task is to:

# 1. Load the dataset using the `LoadHubDataset` component and map the column containing the instructions to the key "prompt".
# 2. Create a custom component named `TextGeneration` that uses a pre-trained text generation model to generate text based on the prompts. The component should take the prompts as input and output the generated text.
# 3. Save the generated text to a CSV file using the `SaveToCSV` component.

# The final pipeline should be named "text-generation-pipeline". Here is some additional information about the components:

# - `LoadHubDataset`: A component that loads a dataset from the `mlflow` HubDataset. It takes no arguments, but you can specify the output mappings to rename or select specific columns.
# - `TextGeneration`: A custom component that you need to create. It should take prompts as input and output the generated text. In this problem, you can assume that you have access to a pre-trained text generation model called `generator` that takes a prompt as input and returns the generated text.
# - `SaveToCSV`: A component that saves the input data to a CSV file. It takes a `path` argument to specify the output file location.

# Your task is to implement the `TextGeneration` component and create the complete pipeline as described above.

print(result[0]["solution"])
# First, let's define the `TextGeneration` component. This component will take the prompts as input and use a pre-trained text generation model to generate text.

# ```python
# from mlflow.components import func_to_component
# import numpy as np

# def generate_text(prompts: np.ndarray) -> np.ndarray:
#     # Assuming you have a pre-trained text generation model called `generator`
#     text = generator.generate(prompts)
#     return np.array(text)

# TextGeneration = func_to_component(generate_text)
# ```

# Now, let's create the complete pipeline.

# ```python
# from mlflow.pipeline import Pipeline
# from mlflow.components.builtin import LoadHubDataset, SaveToCSV

# with Pipeline(name="text-generation-pipeline") as pipeline:
#     load_dataset = LoadHubDataset(
#         name="load_dataset", output_mappings={"prompt": "instruction"}
#     )

#     generate_text_task = TextGeneration(name="generate_text_task")

#     save_to_csv = SaveToCSV(name="save_to_csv", path="generated_text.csv")

#     # Set up the pipeline flow
#     load_dataset >> generate_text_task >> save_to_csv
# ```

# This pipeline first loads the dataset using the `LoadHubDataset` component, maps the "instruction" column to the key "prompt", and passes it to the `TextGeneration` component. The `TextGeneration` component uses the pre-trained text generation model to generate text based on the prompts. Finally, the generated text is saved to a CSV file using the `SaveToCSV` component.
```

One can get as creative as needed with their tasks.

## Types of Tasks

The following are the different sections in which we can distinguish the types of `Tasks` you can expect in `distilabel`:

### The general text generation

The main task of an `LLM` is the text generation, but this text generation can be guided to follow our instructions. The [text generation](./text_generation.md) tasks include the broad text generation classes to make custom implementations and some more specific `Tasks` that have appeared in the literature that can be used to generate text according to some given criteria.

### Feedback tasks

Another type of tasks are those whose intent is to return some kind of feedback, generally in the form of a numeric score or rate, and optionally an explanation of how it got to that. Take a look at the feedback tasks [here](./feedback_tasks.md).

### Special kinds of tasks

There is a different type of task, in the sense that they don't inherit from `Task` but from `Step` (in general due to some kind of restriction), but they can be considered `Tasks` anyway. This block includes the embedding generation for a posterior process, or some specific implementation of a task like ranking, but that is for example restricted to a specific framework or library and cannot work like the more general `Task` classes. More information [here](./special_tasks.md).

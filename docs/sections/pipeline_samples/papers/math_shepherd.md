---
hide: toc
---

# Create datasets to train a Process Reward Model using Math-Shepherd

This example will introduce [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935), an innovative math process reward model (PRM) which assigns reward scores to each step of math problem solutions. Specifically, we will present a recipe to create datasets to train such models.

## Replica

Unlike traditional models that only look at final answers (Output Reward Models or ORM), this system evaluates each step of a mathematical solution and assigns reward scores to individual solution steps. Let's see the Figure 2 from the paper, which makes a summary of the labelling approach presented in their work.

![Math-Shepherd framework](../../../assets/tutorials-assets/math-sheperd.png)

In the traditional ORM approach, the annotation was done depending on the final outcome, while the Process Reward Model (PRM) allows labelling the different steps that lead to a solution, making for a richer set of information.

### Steps involved

- [`MathShepherdGenerator`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdgenerator/): This step is in charge of generating solutions for the instruction. Depending on the value set for the `M`, this step can be used to generate both the `golden_solution`, to be used as a reference for the labeller, or the set of `solutions` to be labelled. For the `solutions` column we want some diversity, to allow the model to reach both good and bad solutions, so we have a representative sample for the labeller, so it may be better to use a "weaker" model.

- [`MathShepherdCompleter`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdcompleter/). This task does the job of the `completer` in the paper, generating completions as presented in Figure 2, section 3.3.2. It doesn't generate a column on it's own, but updates the steps generated in the `solutions` column from the [`MathShepherdGenerator`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdgenerator/), using as reference to label the data, the `golden_solution`. So in order for this step to work, we need both of this columns in our dataset. Depending on the type of dataset, we may already have access to the `golden_solution`, even if it's with a different name, but it's not the same for the `solutions`.

- [`FormatPRM`](https://distilabel.argilla.io/dev/components-gallery/task/formatprm/). This step does the auxiliary job of preparing the data to follow the format defined in the paper of having two columns `input` and `label`. After running the [`MathShepherdCompleter`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdcompleter/), we have raw data that can be formatted as the user want. Using [`ExpandColumns`](https://distilabel.argilla.io/latest/components-gallery/steps/expandcolumns/) and this step, one can directly obtain the same format presented in the dataset shared in the paper: [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0).

## Data preparation

For this example, just as the original paper, we are using the [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) dataset. We only need a dataset with instructions to be solved (in this case it corresponds to the `question` column), and we can generate everything else using our predefined steps.

## Building the pipeline

The pipeline uses `openai/gsm8k` as reference, but the pipeline can be applied to different datasets, keep in mind the prompts can be modified with the current definition, by tweaking the `extra_rules` and `few_shots` in each task:

```python
from datasets import load_dataset

from distilabel.steps.tasks import MathShepherdCompleter, MathShepherdGenerator, FormatPRM
from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import CombineOutputs, ExpandColumns

ds_name = "openai/gsm8k"

ds = load_dataset(ds_name, "main", split="test").rename_column("question", "instruction").select(range(3))  # (1)

with Pipeline(name="Math-Shepherd") as pipe:
    model_id_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    model_id_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    llm_70B = InferenceEndpointsLLM(
        model_id=model_id_70B,
        tokenizer_id=model_id_70B,
        generation_kwargs={"max_new_tokens": 1024, "temperature": 0.6},
    )
    llm_8B = InferenceEndpointsLLM(
        model_id=model_id_8B,
        tokenizer_id=model_id_8B,
        generation_kwargs={"max_new_tokens": 2048, "temperature": 0.6},
    )  # (2)

    generator_golden = MathShepherdGenerator(
        name="golden_generator",
        llm=llm_70B,
    )  # (3)
    generator = MathShepherdGenerator(
        name="generator",
        llm=llm_8B,
        M=5
    )  # (4)
    completer = MathShepherdCompleter(
        name="completer",
        llm=llm_8B,
        N=4
    )  # (5)

    combine = CombineOutputs()

    expand = ExpandColumns(
        name="expand_columns",
        columns=["solutions"],
        split_statistics=True,
    )  # (6)
    formatter = FormatPRM(name="format_prm")  # (7)

    [generator_golden, generator] >> combine >> completer >> expand >> formatter # (8)
```

1. Will use just 3 rows from the sample dataset, and rename the "question" to "instruction", to set the expected value for the [`MathShepherdGenerator`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdgenerator/).

2. We will use 2 different LLMs, `meta-llama/Meta-Llama-3.1-70B-Instruct` (a stronger model for the `golden_solution`) and `meta-llama/Meta-Llama-3.1-8B-Instruct` (a weaker one to generate candidate solutions, and the completions).

3. This [`MathShepherdGenerator`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdgenerator/) task, that uses the *stronger* model, will generate the `golden_solution` for us, the step by step solution for the task.

4. Another [`MathShepherdGenerator`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdgenerator/) task, but in this case using the *weaker* model will generate candidate `solutions` (`M=5` in total).

5. Now the [`MathShepherdCompleter`](https://distilabel.argilla.io/dev/components-gallery/task/mathshepherdcompleter/) task will generate `n=4` *completions* for each step of each candidate solution in the `solutions` column, and label them using the `golden_solution` as shown in Figure 2 in the paper. This step will add the label (it uses [+ and -] tags following the implementation in the paper, but these values can be modified) to the `solutions` column in place, instead of generating an additional column, but the intermediate completions won't be shown at the end.

6. The [`ExpandColumns`](https://distilabel.argilla.io/latest/components-gallery/steps/expandcolumns/) step expands the solution to match the instruction, so if we had set M=5, we would now have 5x instruction-pair solutions. We set the `split_statistics` to True to ensure the `distilabel_metadata` is split accordingly, othwerwise the number of tokens for each solution would count as the tokens needed for the whole list of solutions generated. One can omit both this and the following step and process the data for training as preferred.

7. And finally, the [`FormatPRM`](https://distilabel.argilla.io/dev/components-gallery/task/formatprm/) generates two columns: `input` and `label` which prepare the data for training as presented in the original Math-Shepherd dataset.

8. Both the `generator_golden` and `generator` can be run in parallel as there's no dependency between them, and after that we combine the results and pass them to the `completer`. Finally, we use the `expand` and `formatter` prepare the data in the expected format to train the Process Reward Model as defined in the original paper.

## Script and final dataset

To see all the pieces in place, take a look at the full pipeline:

??? Run

    ```python
    python examples/pipe_math_shepherd.py
    ```

??? "Full pipeline"

    ```python title="pipe_math_shepherd.py"
    --8<-- "examples/pipe_math_shepherd.py"
    ```

    The resulting dataset can be seen at: [plaguss/test_math_shepherd_prm](https://huggingface.co/datasets/plaguss/test_math_shepherd_prm).

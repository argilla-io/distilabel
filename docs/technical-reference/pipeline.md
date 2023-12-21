# Pipelines

This section will detail the [`Pipeline`][distilabel.pipeline.Pipeline], providing guidance on creating and using them.

## Pipeline

The [Pipeline][distilabel.pipeline.Pipeline] class is a central component in `distilabel`, responsible for crafting datasets. It manages the generation of datasets and oversees the interaction between the generator and labeller `LLMs`.

You create an instance of the [`Pipeline`][distilabel.pipeline.Pipeline] by providing a *generator* and an optional *labeller* [LLM][distilabel.llm.base.LLM]. Interactions with it are facilitated through its `generate` method. This method requires a [`dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset), specifies the *num_generations* to determine the number of examples to be created, and includes additional parameters for controlling the *batch_size* and managing the generation process.

Let's start by a Pipeline with a single `LLM` as a generator.

### Generator

We will create a [`Pipeline`][distilabel.pipeline.Pipeline] that will use [Notus](https://huggingface.co/argilla/notus-7b-v1) from a Huggingface [Inference Endpoint][distilabel.llm.InferenceEndpointsLLM]. For this matter, we need to create a [TextGenerationTask][distilabel.tasks.TextGenerationTask], and specify the format we want to use for our `Prompt`, in this case *notus*, which corresponds to the same for *zephyr*.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_generator_1.py"
```

We've set up our pipeline using a specialized [`TextGenerationTask`](distilabel.tasks.text_generation.base.TextGenerationTask) (refer to the [tasks section](./tasks.md) for more task details), and an [InferenceEndpointsLLM][distilabel.llm.huggingface.inference_endpoints.InferenceEndpointsLLM] configured for [`notus-7b-v1`](https://huggingface.co/argilla/notus-7b-v1), although any of the available `LLMs` will work.

To utilize the [Pipeline][distilabel.pipeline.Pipeline] for dataset generation, we call the generate method. We provide it with the input dataset and specify the desired number of generations. In this example, we've prepared a `Dataset` with a single row to illustrate the process. This dataset contains one row, and we'll trigger 2 generations from it:

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_generator_2.py"
```

Now, let's examine the dataset that was generated. It's a [`CustomDataset`][distilabel.dataset.CustomDataset], equipped with additional features for seamless interaction with [`Argilla`](https://github.com/argilla-io/argilla).

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_generator_3.py"
```

### Labeller

Next, we move on to labelLing a dataset. Just as before, we need an `LLM` for our `Pipeline`. In this case we will use [`OpenAILLM`][distilabel.llm.openai.OpenAILLM] with `gpt-4`, and a `PreferenceTask`, [UltraFeedbackTask][distilabel.tasks.preference.ultrafeedback.UltraFeedbackTask] for instruction following.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_labeller_1.py"
```

For this example dataset, we've extracted 2 sample rows from the [UltraFeedback binarized dataset](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences), formatted as expected by the default `LLM` and `Task`.

We've selected two distinct examples, one correctly labeled and the other incorrectly labeled in the original dataset. In this instance, the `dataset` being generated includes two columns: the *input*, as seen in the generator, and a *generations* column containing the model's responses.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_labeller_2.py"
```

Let's select the relevant columns from the labelled dataset, and take a look at the first record. This allows us to observe the *rating* and the accompanying *rationale* that provides an explanation.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_labeller_3.py"
```

### Generator and Labeller

In the final scenario, we have a [`Pipeline`][distilabel.pipeline.Pipeline] utilizing both a *generator* and a *labeller* `LLM`. Once more, we'll employ the [Inference Endpoint][distilabel.llm.InferenceEndpointsLLM] with `notus-7b-v1` for the *generator*, using a different *system prompt* this time. As for the labeller, we'll use `gpt-3.5-turbo`, which will label the examples for *instruction following*.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_labeller_generator_1.py"
```

For this example, we'll set up a pipeline to generate and label a dataset of short stories inspired by [XKCD](https://xkcd.com/). To do this, we'll define the *system_prompt* for the `NotusTextGenerationTask`. The dataset will follow the same format we used for the generator scenario, featuring an *input* column with the examples, in this case, just one.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_labeller_generator_2.py"
```

We will now take a look to one of the *generations*, along with the *rating* and *rational* given by our *labeller* `LLM`:

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_labeller_generator_3.py"
```
### Running several generators in parallel

`distilabel` also allows to use several `LLM`s as generators in parallel, thanks to the `ProcessLLM` and `LLMPool` classes. This comes handy for the cases where we want to use several `LLM`s and fed them with the same input, allowing us to later compare their outputs (to see which one is better) or even creating a Preference dataset, following a similar process to UltraFeedback dataset generation.

For this example, we will load four 7B `LLM`s using `vLLM` and a machine with 4 GPUs (to load each `LLM` in a different GPU). Then we will give instructions to all of them, and we will use GPT-4 to label the generated instructions using the `UltraFeedbackTask` for instruction-following.

First of all, we will need to load each `LLM` using a `ProcessLLM`. `ProcessLLM` will create a child process which will load the `LLM` using the `load_llm_fn`.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_llmpool_processllm_1.py"
```

1. The `ProcessLLM` will create a child process in which the `LLM` will be loaded. Therefore, we will need to define a function that will be executed by the child process to load the `LLM`. The child process will pass the provided `Task` to the `load_llm_fn`.
2. We set a value for `CUDA_VISIBLE_DEVICES` environment variable to make sure that each `LLM` is loaded in a different GPU.

We will repeat this pattern 4 times, each time with a different `LLM` and a different GPU.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_llmpool_processllm_2.py"
```

In order to distribute the generations among the different `LLM`s, we will use a `LLMPool`. This class expects a list of `ProcessLLM`. Calling the `generate` method of the `LLMPool` will call the `generate` method of each `LLMProcess` in parallel, and will wait for all of them to finish, returning a list of lists of `LLMOutput`s with the generations.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_llmpool_processllm_3.py"
```

We will use this `LLMPool` as the generator for our pipeline and we will use GPT-4 to label the generated instructions using the `UltraFeedbackTask` for instruction-following.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_llmpool_processllm_4.py"
```

1. We also will execute the calls to OpenAI API in a different process using the `ProcessLLM`. This will allow to not block the main process GIL, and allowing the generator to continue with the next batch.  

Then, we will load the dataset and call the `generate` method of the pipeline. For each input in the dataset, the `LLMPool` will randomly select two `LLM`s and will generate two generations for each of them. The generations will be labelled by GPT-4 using the `UltraFeedbackTask` for instruction-following. Finally, we will push the generated dataset to Argilla, in order to review the generations and labels that were automatically generated, and to manually correct them if needed.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipeline_llmpool_processllm_5.py"
```

With a few lines of code, we have easily generated a dataset with 2 generations per input, using 4 different `LLM`s, and labelled the generations using GPT-4. You can check the full code [here](https://github.com/argilla-io/distilabel/blob/main/examples/pipeline-preference-dataset-llmpool.py).

## pipeline

Considering recurring patterns in dataset creation, we can facilitate the process by utilizing the [`Pipeline`][distilabel.pipeline.Pipeline]. This is made simpler through the [`pipeline`][distilabel.pipeline.pipeline] function, which provides the necessary parameters for creating a `Pipeline`.

In the code snippet below, we use the [`pipeline`][distilabel.pipeline.pipeline] function to craft a `pipeline` tailored for a *preference task*, specifically focusing on *text-quality* as the *subtask*. If we don't initially provide a *labeller* [`LLM`][distilabel.llm.base.LLM], we can specify the subtask we want our `pipeline` to address. By default, this corresponds to [`UltraFeedbackTask`][distilabel.tasks.preference.ultrafeedback.UltraFeedbackTask]. It's mandatory to specify the generator of our choice; however, the labeller defaults to `gpt-3.5-turbo`. Optional parameters required for [OpenAILLM][distilabel.llm.openai.OpenAILLM] can also be passed as optional keyword arguments.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipe_1.py"
```

For the dataset, we'll begin with three rows from [HuggingFaceH4/instruction-dataset](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset). We'll request two generations with checkpoints enabled to safeguard the data in the event of any failures, which is the default behavior.

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipe_2.py"
```

Finally, let's see one of the examples from the dataset:

```python
--8<-- "docs/snippets/technical-reference/pipeline/pipe_3.py"
```

The API reference can be found here: [pipeline][distilabel.pipeline.pipeline]

## Argilla integration

The [CustomDataset][distilabel.dataset.CustomDataset] generated entirely by AI models may require some additional human processing. To facilitate human feedback, the dataset can be uploaded to [`Argilla`](https://github.com/argilla-io/argilla). This process involves logging into an [`Argilla`](https://docs.argilla.io/en/latest/getting_started/cheatsheet.html#connect-to-argilla) instance, converting the dataset to the required format using `CustomDataset.to_argilla()`, and subsequently using push_to_argilla on the resulting dataset:

```python
--8<-- "docs/snippets/technical-reference/pipeline/argilla.py"
```

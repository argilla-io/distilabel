# Tasks

In this section we will see what's a `Task` and the list of tasks available in `distilabel`.

## Task

The `Task` class takes charge of setting how the LLM behaves, deciding whether it acts as a *generator* or a *labeller*. To accomplish this, the `Task` class creates a prompt using a template that will be sent to the [`LLM`](../technical-reference/llms.md). It specifies the necessary input arguments for generating the prompt and identifies the output arguments to be extracted from the `LLM` response. The `Task` class yields a `Prompt` that can generate a string with the format needed, depending on the specific `LLM` used. 

All the `Task`s defines a `system_prompt` which serves as the initial instruction given to the LLM, guiding it on what kind of information or output is expected, and the following methods:

- `generate_prompt`: This method will be used by the `LLM` to create the prompts that will be fed to the model.
- `parse_output`: After the `LLM` has generated the content, this method will be called on the raw outputs of the model to extract the relevant content (scores, rationales, etc).
- `input_args_names` and `output_args_names`: These methods are used in the [`Pipeline`](../technical-reference/pipeline.md) to process the datasets. The first one defines the columns that will be extracted from the dataset to build the prompt in case of a `LLM` that acts as a generator or labeller alone, or the columns that should be placed in the dataset to be processed by the *labeller* `LLM`, in the case of a `Pipeline` that has both a *generator* and a *labeller*. The second one is in charge of inserting the defined fields as columns of the dataset generated dataset.

After defining a task, the only action required is to pass it to the corresponding `LLM`. All the intricate processes are then handled internally:

```python
--8<-- "docs/snippets/technical-reference/tasks/generic_transformersllm.py"
```

Given this explanation, `distilabel` distinguishes between two primary categories of tasks: those focused on text generation and those centered around labelling. These `Task` classes delineate the LLM's conduct, be it the creation of textual content or the assignment of labels to text, each with precise guidelines tailored to their respective functionalities. Users can seamlessly leverage these distinct task types to tailor the LLM's behavior according to their specific application needs.

## Text Generation

These set of classes are designed to steer a `LLM` in generating text with specific guidelines. They provide a structured approach to instruct the LLM on generating content in a manner tailored to predefined criteria.

### TextGenerationTask

This is the base class for *text generation*, and includes the following fields for guiding the generation process: 

- `system_prompt`, which serves as the initial instruction or query given to the LLM, guiding it on what kind of information or output is expected. 
- A list of `principles` to inject on the `system_prompt`, which by default correspond to those defined in the UltraFeedback paper[^1], 
- and lastly a distribution for these principles so the `LLM` can be directed towards the different principles with a more customized behaviour.

[^1]:
    The principles can be found [here][distilabel.tasks.text_generation.principles] in the codebase. More information on the *Principle Sampling* can be found in the [UltraFeedfack repository](https://github.com/OpenBMB/UltraFeedback#principle-sampling).


For the API reference visit [TextGenerationTask][distilabel.tasks.text_generation.base.TextGenerationTask].

### Llama2TextGenerationTask

This class inherits from the `TextGenerationTask` and it's specially prepared to deal with prompts in the form of the *Llama2* model, so it should be the go to task for `LLMs` intented for text generation that were trained using this prompt format. The specific prompt formats can be found in the source code of the [Prompt][distilabel.tasks.prompt.Prompt] class.

```python
--8<-- "docs/snippets/technical-reference/tasks/generic_llama2_textgeneration.py"
```

For the API reference visit [Llama2TextGenerationTask][distilabel.tasks.text_generation.llama.Llama2TextGenerationTask].

### OpenAITextGenerationTask

The OpenAI task for text generation is similar to the `Llama2TextGenerationTask`, but with the specific prompt format expected by the *chat completion* task from OpenAI.

```python
--8<-- "docs/snippets/technical-reference/tasks/generic_openai_textgeneration.py"
```

For the API reference visit [OpenAITextGenerationTask][distilabel.tasks.text_generation.openai.OpenAITextGenerationTask].

### SelfInstructTask

The task specially designed to build the prompts following the Self-Instruct paper: [SELF-INSTRUCT: Aligning Language Models
with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560.pdf).

From the original [repository](https://github.com/yizhongw/self-instruct/tree/main#how-self-instruct-works): *The Self-Instruct process is an iterative bootstrapping algorithm that starts with a seed set of manually-written instructions and uses them to prompt the language model to generate new instructions and corresponding input-output instances*, so this `Task` is specially interesting for generating new datasets from a set of predefined topics.

```python
--8<-- "docs/snippets/technical-reference/tasks/generic_openai_self_instruct.py"
```

For the API reference visit  [SelfInstructTask][distilabel.tasks.text_generation.self_instruct.SelfInstructTask].

## Labelling

Instead of generating text, you can instruct the `LLM` to label datasets. The existing tasks are designed specifically for creating `Preference` datasets.

### Preference

Preference datasets for Language Models (LLMs) are sets of information that show how people rank or prefer one thing over another in a straightforward and clear manner. These datasets help train language models to understand and generate content that aligns with user preferences, enhancing the model's ability to generate contextually relevant and preferred outputs.

Contrary to the `TextGenerationTask`, the `PreferenceTask` is not intended for direct use. It implements the default methods `input_args_names` and `output_args_names`, but `generate_prompt` and `parse_output` are specific to each `PreferenceTask`. Examining the `output_args_names` reveals that the generation will encompass both the rating and the rationale that influenced that rating.

#### UltraFeedbackTask

This task is specifically designed to build the prompts following the format defined in the ["UltraFeedback: Boosting Language Models With High Quality Feedback"](https://arxiv.org/pdf/2310.01377.pdf) paper.

From the original [repository](https://github.com/OpenBMB/UltraFeedback): *To collect high-quality preference and textual feedback, we design a fine-grained annotation instruction, which contains 4 different aspects, namely instruction-following, truthfulness, honesty and helpfulness*. This `Task` is designed to label datasets following the different aspects defined for the UltraFeedback dataset creation.

The following snippet can be used as a simplified UltraFeedback Task, for which we define 3 different ratings, but take into account the predefined versions are intended to be used out of the box:

```python
--8<-- "docs/snippets/technical-reference/tasks/ultrafeedback.py"
```

=== "Text Quality"

    The following example uses a `LLM` to examinate the data for text quality criteria, which includes the different criteria from UltraFeedback (Correctness & Informativeness, Honesty & Uncertainty, Truthfulness & Hallucination and Instruction Following):

    ```python
    --8<-- "docs/snippets/technical-reference/tasks/openai_for_text_quality.py"
    ```

=== "Helpfulness"

    The following example creates a UltraFeedback task to emphasize helpfulness, that is overall quality and correctness of the output:

    ```python
    --8<-- "docs/snippets/technical-reference/tasks/openai_for_helpfulness.py"
    ```

=== "Truthfulness"

    The following example creates a UltraFeedback task to emphasize truthfulness and hallucination assessment:

    ```python
    --8<-- "docs/snippets/technical-reference/tasks/openai_for_truthfulness.py"
    ```

=== "Honesty"

    The following example creates a UltraFeedback task to emphasize honesty and uncertainty expression assessment:

    ```python
    --8<-- "docs/snippets/technical-reference/tasks/openai_for_honesty.py"
    ```

=== "Instruction Following"

    The following example creates a UltraFeedback task to emphasize the evaluation of alignment between output and intent:

    ```python
    --8<-- "docs/snippets/technical-reference/tasks/openai_for_instruction_following.py"
    ```

For the API reference visit [UltraFeedbackTask][distilabel.tasks.preference.ultrafeedback.UltraFeedbackTask].

#### JudgeLMTask

The task specially designed to build the prompts following the UltraFeedback paper: [JudgeLM: Fine-tuned Large Language Models Are Scalable Judges](https://arxiv.org/pdf/2310.17631.pdf). This task is designed to evaluate the performance of AI assistants.

```python
--8<-- "docs/snippets/technical-reference/tasks/openai_judgelm.py"
```

For the API reference visit [JudgeLMTask][distilabel.tasks.preference.judgelm.JudgeLMTask].

#### UltraJudgeTask

This class implements a `PreferenceTask` specifically for a better evaluation using AI Feedback. The task is defined based on both UltraFeedback and JudgeLM, but with several improvements / modifications.

It introduces an additional argument to differentiate various areas for processing. While these areas can be customized, the default values are as follows:

```python
--8<-- "docs/snippets/technical-reference/tasks/ultrajudge.py"
```

Which can be directly used in the following way:

```python
--8<-- "docs/snippets/technical-reference/tasks/openai_ultrajudge.py"
```

For the API reference visit [UltraJudgeTask][distilabel.tasks.preference.ultrajudge.UltraJudgeTask].

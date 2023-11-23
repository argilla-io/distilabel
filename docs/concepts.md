This page aims to get you familiarized with the basic concepts of the framework, describing the most important
components or classes and how they work together. The following sections will guide you through the primary components
of the framework: `Pipeline`, `LLM` (both generator and labeller), and the `Task`.

<figure markdown>
  ![distilabel](assets/images/distilabel-diagram.svg#only-light){ width="600" }
  ![distilabel](assets/images/distilabel-diagram-dark.svg#only-dark){ width="600" }
  <figcaption>distilabel flow diagram</figcaption>
</figure>

## Components

### Task

The `Task` class in the one in charge of defining the behaviour of the `LLM`, and therefore it can define if an LLM is
a `generator` or a `labeller`. To do so, the `Task` class generates the prompt that will be sent to the `LLM` from a template.
It also defines, which input arguments are required to generate the prompt, and which output arguments will be extracted
from the `LLM` response. It's worth mentioning that the `Task` class doesn't return a `str`, but a `Prompt` class which
will generate the `str` format depending on the `LLM` that is going to be used (Zephyr, Llama, OpenAI, etc).

```python
--8<-- "docs/snippets/concepts/task.py"
```

### LLM

The `LLM` class represents a language model and implements the way to interact with it. It also defines the generation
parameters that can be passed to the model to tweak the generations. As mentioned above, the `LLM` will have a `Task`
associated that will use to generate the prompt and extract the output from the generation.

```python
--8<-- "docs/snippets/concepts/llm.py"
```

!!! note
    To run the script successfully, ensure you have assigned your OpenAI API key to the `OPENAI_API_KEY` environment variable.


### Pipeline

The `Pipeline` class orchestrates the whole generation and labelling process, and it's in charge of the batching of the
input dataset, as well as reporting the generation progress. It's worth mentioning that is not mandatory to pass both
a generator `LLM` and a labeller `LLM` to the `Pipeline` class, as it can also be used only for generation or labelling.

!!! Pipelines

    === "Generator and labeller"

        ```python
        --8<-- "docs/snippets/concepts/pipeline.py"
        ```

        !!! note
            To run the script successfully, ensure you have assigned your OpenAI API key to the `OPENAI_API_KEY` environment variable
            and that you have download the file [llama-2-7b-chat.Q4_O.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf)
            in the same folder as the script.

    === "Only generator"

        ```python
        --8<-- "docs/snippets/concepts/pipeline-only-generator.py"
        ```

    === "Only labeller"

        ```python
        --8<-- "docs/snippets/concepts/pipeline-only-labeller.py"
        ```


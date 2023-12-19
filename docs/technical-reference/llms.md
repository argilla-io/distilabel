# LLMs

## LLM

The [`LLM`][distilabel.llm.base.LLM] class encapsulates the functionality for interacting with a language model.

It distinguishes between *task* specifications and configurable parameters that influence the LLM's behavior.

For illustration purposes, we employ the [`TextGenerationTask`][distilabel.tasks.text_generation.base.TextGenerationTask] in this section and guide you to the dedicated [`Tasks`](../technical-reference/tasks.md) section for comprehensive details.

LLM classes share several general parameters and define implementation-specific ones. Let's explain the general parameters first and the generate method, and then the specifics for each class.

### General Parameters

Let's briefly introduce the general parameters we may find[^1]:

[^1]:
    You can take a look at this blog post from [cohere](https://txt.cohere.com/llm-parameters-best-outputs-language-ai/) for a thorough explanation of the different parameters.

- `max_new_tokens`:

    This parameter controls the maximum number of tokens the LLM is allowed to use.

- `temperature`: 

    Parameter associated to the creativity of the model, a value close to 0 makes the model more deterministic, while higher values make the model more "creative".

- `top_k` and `top_p`:

    `top_k` limits the number of tokens the model is allowed to use to generate the following token sorted by probability, while `top_p` limits the number of tokens the model can use for the next token, but in terms of the sum of their probabilities.

- `frequency_penalty` and `presence_penalty`:

    The frequency penalty penalizes tokens that have already appeard in the generated text, limiting the possibility of those appearing again, and the `presence_penalty` penalizes regardless of hte frequency.

- `prompt_format` and `prompt_formatting_fn`:

    These two parameters allow to tweak the prompt of our models, for example we can direct the `LLM` to format the prompt according to one of the defined formats, while `prompt_formatting_fn` allows to pass a function that will be applied to the prompt before the generation, for extra control of what we ingest to the model.

### Generate method

Once you create an `LLM`, you use the `generate` method to interact with it. This method requires inputs for text generation and specifies the number of desired generations. The output will be in the form of lists containing `LLMOutput`[^2] objects, which act as a general container for the LLM's results.

[^2]:
    Or it can also return lists of *Futures* containing the lists of these `LLMOutputs`, if we deal with an asynchronous or thread based API.

Let's see the different LLMs that are implemented in `distilabel` (we can think of them in terms of the engine that generates the text for us):

## OpenAI

These may be the default choice for your ambitious tasks.

For the API reference visit [OpenAILLM][distilabel.llm.openai.OpenAILLM].

```python
--8<-- "docs/snippets/technical-reference/llm/openai_generate.py"
```

## Llama.cpp

Applicable for local execution of Language Models (LLMs). Utilize this LLM when you have access to the quantized weights of your selected model for interaction.

Let's see an example using [notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1). First, you can download the weights from the following [link](https://huggingface.co/TheBloke/notus-7B-v1-GGUF):

```python
--8<-- "docs/snippets/technical-reference/llm/llamacpp_generate.py"
```

For the API reference visit [LlammaCppLLM][distilabel.llm.llama_cpp.LlamaCppLLM].

## vLLM

For the API reference visit [vLLM][distilabel.llm.vllm.vLLM].

## Huggingface LLMs

This section explains two different ways to use HuggingFace models:

### Transformers

This is the option to utilize a model deployed on Hugging Face's model hub. Load the model and tokenizer in the standard manner as done locally, and proceed to instantiate your class.

For the API reference visit [TransformersLLM][distilabel.llm.huggingface.transformers.TransformersLLM].

Let's see an example using [notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1):

```python
--8<-- "docs/snippets/technical-reference/llm/transformers_generate.py"
```

### Inference Endpoints

Hugging Face provides a streamlined approach for deploying models through [inference endpoints](https://huggingface.co/inference-endpoints) on their infrastructure. Opt for this solution if your model is hosted on Hugging Face.

For the API reference visit [InferenceEndpointsLLM][distilabel.llm.huggingface.inference_endpoints.InferenceEndpointsLLM].

Let's see how to interact with these LLMs:

```python
--8<-- "docs/snippets/technical-reference/llm/inference_endpoint_generate.py"
```

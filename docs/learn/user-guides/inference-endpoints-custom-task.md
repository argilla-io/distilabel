# Create an LLM from a huggingface Inference Endpoint for a Question Answering task

This guide shows you how to create a custom task for question answering and use it with the LLM of our choice, in this case `InferenceEndpointsLLM`.

Let's see the code:

!!! note
    To run this example you will need to set the `HF_INFERENCE_ENDPOINT_NAME` env var.

```python
--8<-- "docs/snippets/learn/llms/inference-endpoints-llm.py"
```

## Create our custom task for Question Answering

We start create our custom task by inheriting from `Llama2TextGenerationTask` and overriding the necessary methods:

- Update the `generate_prompt` to make it more sound for our use case.

- Update `parse_output` to return the desired format.

- Return `input_args_names` and `output_arg_names` specific for our task, `[question]` and `[answer]` respectively.

## Instantiate the LLM with out brand new task

Now we are ready to create our LLM. In this case we are using `InferenceEndpointsLLM`, so we can instantiate the class and pass the previously created `Llama2QuestionAnsweringTask`, and call the `generate` method from our LLM with a question to see it in action.

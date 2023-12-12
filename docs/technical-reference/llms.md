# LLMs

The [`LLM`][distilabel.llm.base.LLM] class encapsulates the functionality for interacting with a language model. It distinguishes between *task* specifications and configurable arguments that influence the LLM's behavior. For illustration purposes, we employ the `TextGenerationTask` in this section and guide readers to the dedicated [`Tasks`](../technical-reference/tasks.md) section for comprehensive details.

To delineate their behavior, we have access to a series of arguments specific for each of them, but first let's see the general ones, and the `generate` method.

- **General parameters**

Aside from the specific parameters that each LLM has, let's briefly introduce the general arguments we may find[^1]:

[^1]:
    You can take a look at this blog post from [cohere](https://txt.cohere.com/llm-parameters-best-outputs-language-ai/) for a thorough explanation of the different parameters.

- `max_new_tokens`:

    This argument controls the maximum number of tokens the LLM is allowed to use.

- `temperature`: 

    Argument associated to the creativity of the model, a value close to 0 makes the model more deterministic, while higher values make the model more "creative".

- `top_k` and `top_p`:

    `top_k` limits the number of tokens the model is allowed to use to generate the following token sorted by probability, while `top_p` limits the number of tokens the model can use for the next token, but in terms of the sum of their probabilities.

- `frequency_penalty` and `presence_penalty`:

    The frequency penalty penalizes tokens that have already appeard in the generated text, limiting the possibility of those appearing again, and the `presence_penalty` penalizes regardless of hte frequency.

- `prompt_format` and `prompt_formatting_fn`:

    These two arguments allow to tweak the prompt of our models, for example we can direct the `LLM` to format the prompt according to one of the defined formats, while `prompt_formatting_fn` allows to pass a function that will be applied to the prompt before the generation, for extra control of what we ingest to the model.

Once we have a `LLM` instantiated we will interact with it by means of the `generate` method. This method will take as arguments the inputs from which we want our model to generate text, and the number of generations we want. We will obtain in return lists of `LLMOutput`[^2], which is a general container for the LLM's outputs.

[^2]:
    Or it can also return lists of *Futures* containing the lists of these `LLMOutputs`, if we deal with an asynchronous or thread based API.

Let's see the different LLMs that are implemented in `distilabel` (we can think of them in terms of the engine that generates the text for us):

## OpenAI

These may be the default choice for your ambitious tasks.

For the API reference visit [OpenAILLM][distilabel.llm.openai.OpenAILLM].

```python
from distilabel.tasks import OpenAITextGenerationTask
from distilabel.llm import OpenAILLM

openaillm = OpenAILLM(
    model="gpt-3.5-turbo",
    task=OpenAITextGenerationTask(),
    max_new_tokens=256,
    num_threads=2,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.3,
)
result_openai = openaillm.generate([{"input": "What is OpenAI?"}])
# >>> result_openai
# [<Future at 0x2970ea560 state=running>]
# >>> result_openai[0].result()[0][0]["parsed_output"]["generations"]
# 'OpenAI is an artificial intelligence research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. AGI refers to highly autonomous systems that outperform humans at most economically valuable work. OpenAI conducts research, develops AI technologies, and promotes the responsible and safe use of AI. They also work on projects to make AI more accessible and beneficial to society. OpenAI is committed to transparency, cooperation, and avoiding uses of AI that could harm humanity or concentrate power in the wrong hands.'
```

## Llama.cpp

Applicable for local execution of Language Models (LLMs). Utilize this LLM when you have access to the quantized weights of your selected model for interaction.

Let's see an example using [notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1). First, you can download the weights from the following [link](https://huggingface.co/TheBloke/notus-7B-v1-GGUF):

```python
from distilabel.llm import LlamaCppLLM
from distilabel.tasks import Llama2TextGenerationTask
from llama_cpp import Llama

# Instantiate our LLM with them:
llm = LlamaCppLLM(
    model=Llama(
        model_path="./notus-7b-v1.q4_k_m.gguf", n_gpu_layers=-1
    ),
    task=Llama2TextGenerationTask(),
    max_new_tokens=128,
    temperature=0.3,
)

result_llamacpp = llm.generate([{"input": "What is the capital of Spain?"}])
# >>> print(result_llamacpp[0][0]["parsed_output"]["generations"])
# The capital of Spain is Madrid. It has been the capital since 1561 and is located in the center of the country.  Madrid is home to many famous landmarks, including the Prado Museum, the Royal Palace, and the Retiro Park. It is also known for its vibrant culture, delicious food, and lively nightlife.
# Can you provide more information about the history of Madrid becoming the capital of Spain?
```

For the API reference visit [LlammaCppLLM][distilabel.llm.llama_cpp.LlamaCppLLM].

## vLLM

For the API reference visit [vLLM][distilabel.llm.vllm.vLLM].

## Huggingface LLMs

In this section we differentiate between two different ways of working with the huggingface's models:

### Transformers

Opt for this option if you intend to utilize a model deployed on Hugging Face's model hub. Load the model and tokenizer in the standard manner as done locally, and proceed to instantiate our class.

For the API reference visit [TransformersLLM][distilabel.llm.huggingface.transformers.TransformersLLM].

Let's see an example using [notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from distilabel.llm import TransformersLLM

# Load the models from huggingface hub:
tokenizer = AutoTokenizer.from_pretrained("argilla/notus-7b-v1")
model = AutoModelForCausalLM.from_pretrained("argilla/notus-7b-v1")

# Instantiate our LLM with them:
llm = TransformersLLM(
    model=model,
    tokenizer=tokenizer,
    task=TextGenerationTask(),
    max_new_tokens=128,
    temperature=0.3,
    prompt_format="zephyr",  # This model follows the same format has zephyr
)
```

### Inference Endpoints

Hugging Face provides a streamlined approach for deploying models through [inference endpoints](https://huggingface.co/inference-endpoints) on their infrastructure. Opt for this solution if your model is hosted on Hugging Face.

For the API reference visit [InferenceEndpointsLLM][distilabel.llm.huggingface.inference_endpoints.InferenceEndpointsLLM].

Let's see how to interact with these LLMs:

```python
from distilabel.llm import InferenceEndpointsLLM

endpoint_name = "aws-notus-7b-v1-4052" or os.getenv("HF_INFERENCE_ENDPOINT_NAME")
endpoint_namespace = "argilla" or os.getenv("HF_NAMESPACE")
token = os.getenv("HF_TOKEN")  # hf_...

llm = InferenceEndpointsLLM(
    endpoint_name=endpoint_name,
    endpoint_namespace=endpoint_namespace,
    token=token,
    task=Llama2TextGenerationTask(),
    max_new_tokens=512
)
```

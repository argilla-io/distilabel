---
hide:
  - navigation
---
# vLLM


`vLLM` library LLM implementation.







### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **dtype**: the data type to use for the model. Defaults to `auto`.

- **trust_remote_code**: whether to trust the remote code when loading the model. Defaults  to `False`.

- **quantization**: the quantization mode to use for the model. Defaults to `None`.

- **revision**: the revision of the model to load. Defaults to `None`.

- **tokenizer**: the tokenizer Hugging Face Hub repo id or a path to a directory containing  the tokenizer files. If not provided, the tokenizer will be loaded from the  model directory. Defaults to `None`.

- **tokenizer_mode**: the mode to use for the tokenizer. Defaults to `auto`.

- **tokenizer_revision**: the revision of the tokenizer to load. Defaults to `None`.

- **skip_tokenizer_init**: whether to skip the initialization of the tokenizer. Defaults  to `False`.

- **chat_template**: a chat template that will be used to build the prompts before  sending them to the model. If not provided, the chat template defined in the  tokenizer config will be used. If not provided and the tokenizer doesn't have  a chat template, then ChatML template will be used. Defaults to `None`.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.

- **seed**: the seed to use for the random number generator. Defaults to `0`.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `LLM` class of `vllm` library. Defaults to `{}`.

- **_model**: the `vLLM` model instance. This attribute is meant to be used internally  and should not be accessed directly. It will be set in the `load` method.

- **_tokenizer**: the tokenizer instance used to format the prompt before passing it to  the `LLM`. This attribute is meant to be used internally and should not be  accessed directly. It will be set in the `load` method.

- **use_magpie_template**: a flag used to enable/disable applying the Magpie pre-query  template. Defaults to `False`.

- **magpie_pre_query_template**: the pre-query template to be applied to the prompt or  sent to the LLM to generate an instruction or a follow up user message. Valid  values are "llama3", "qwen2" or another pre-query template provided. Defaults  to `None`.





### Runtime Parameters

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to  the `LLM` class of `vllm` library.




### Examples


#### Generate text
```python
from distilabel.llms import vLLM

# You can pass a custom chat_template to the model
llm = vLLM(
    model="prometheus-eval/prometheus-7b-v2.0",
    chat_template="[INST] {{ messages[0]"content" }}\n{{ messages[1]"content" }}[/INST]",
)

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate structured data
```python
from pathlib import Path
from distilabel.llms import vLLM

class User(BaseModel):
    name: str
    last_name: str
    id: int

llm = vLLM(
    model="prometheus-eval/prometheus-7b-v2.0"
    structured_output={"format": "json", "schema": Character},
)

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
```




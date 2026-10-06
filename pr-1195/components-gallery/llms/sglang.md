---
hide:
  - navigation
---
# SGLang


`SGLang` library LLM implementation.







### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **dtype**: the data type to use for the model. Defaults to `auto`.

- **trust_remote_code**: whether to trust the remote code when loading the model. Defaults  to `False`.

- **quantization**: the quantization mode to use for the model. Defaults to `None`.

- **revision**: the revision of the model to load. Defaults to `None`.

- **tokenizer**: the tokenizer Hugging Face Hub repo id or a path to a directory containing  the tokenizer files. Defaults to `model`.

- **tokenizer_mode**: the mode to use for the tokenizer. Defaults to `auto`.

- **skip_tokenizer_init**: whether to skip the initialization of the tokenizer. Defaults  to `False`.

- **chat_template**: a chat template that will be used to build the prompts before  sending them to the model. If not provided, the chat template defined in the  tokenizer config will be used. If not provided and the tokenizer doesn't have  a chat template, then ChatML template will be used. Defaults to `None`.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.

- **seed**: the seed to use for the random number generator. Defaults to `0`.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `Engine` class of `sglang` library. Defaults to `{}`.

- **_model**: the `SGLang` model instance. This attribute is meant to be used internally  and should not be accessed directly. It will be set in the `load` method.

- **_tokenizer**: the tokenizer instance used to format the prompt before passing it to  the `LLM`. It will be set in the `load` method.

- **use_magpie_template**: a flag used to enable/disable applying the Magpie pre-query  template. Defaults to `False`.

- **magpie_pre_query_template**: the pre-query template to be applied to the prompt or  sent to the LLM to generate an instruction or a follow up user message. Valid  values are "llama3", "qwen2" or another pre-query template provided. Defaults  to `None`.





### Runtime Parameters

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to  the `LLM` class of `SGLang` library.




### Examples


#### Generate text
```python
from distilabel.models.llms import SGLang
if __name__ == "__main__":
    llm = SGLang(
        model="Qwen/Qwen2.5-Coder-3B-Instruct",
        chat_template="[INST] {{ messages[0]['content']}} [/INST]"
    )

    llm.load()
    # Call the model
    output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate structured data
```python
from distilabel.models.llms import SGLang
from pydantic import BaseModel

if __name__ == "__main__":

    class User(BaseModel):
        name: str
        last_name: str
        id: int

    llm = SGLang(
        model="Qwen/Qwen2.5-Coder-3B-Instruct",
        structured_output={"format": "json", "schema": User},
    )

    llm.load()
    # Call the model
    output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
```




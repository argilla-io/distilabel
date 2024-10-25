---
hide:
  - navigation
---
# TransformersLLM


Hugging Face `transformers` library LLM implementation using the text generation



pipeline.





### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **revision**: if `model` refers to a Hugging Face Hub repository, then the revision  (e.g. a branch name or a commit id) to use. Defaults to `"main"`.

- **torch_dtype**: the torch dtype to use for the model e.g. "float16", "float32", etc.  Defaults to `"auto"`.

- **trust_remote_code**: whether to allow fetching and executing remote code fetched  from the repository in the Hub. Defaults to `False`.

- **model_kwargs**: additional dictionary of keyword arguments that will be passed to  the `from_pretrained` method of the model.

- **tokenizer**: the tokenizer Hugging Face Hub repo id or a path to a directory containing  the tokenizer config files. If not provided, the one associated to the `model`  will be used. Defaults to `None`.

- **use_fast**: whether to use a fast tokenizer or not. Defaults to `True`.

- **chat_template**: a chat template that will be used to build the prompts before  sending them to the model. If not provided, the chat template defined in the  tokenizer config will be used. If not provided and the tokenizer doesn't have  a chat template, then ChatML template will be used. Defaults to `None`.

- **device**: the name or index of the device where the model will be loaded. Defaults  to `None`.

- **device_map**: a dictionary mapping each layer of the model to a device, or a mode  like `"sequential"` or `"auto"`. Defaults to `None`.

- **token**: the Hugging Face Hub token that will be used to authenticate to the Hugging  Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package  local configuration will be used. Defaults to `None`.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.

- **use_magpie_template**: a flag used to enable/disable applying the Magpie pre-query  template. Defaults to `False`.

- **magpie_pre_query_template**: the pre-query template to be applied to the prompt or  sent to the LLM to generate an instruction or a follow up user message. Valid  values are "llama3", "qwen2" or another pre-query template provided. Defaults  to `None`.







### Examples


#### Generate text
```python
from distilabel.models.llms import TransformersLLM

llm = TransformersLLM(model="microsoft/Phi-3-mini-4k-instruct")

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```




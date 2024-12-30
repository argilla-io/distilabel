---
hide:
  - navigation
---
# MlxLLM


Hugging Face `transformers` library LLM implementation using the text generation



pipeline.





### Attributes

- **path_or_hf_repo**: the path to the model or the Hugging Face Hub repo id.

- **tokenizer_config**: the tokenizer configuration.

- **model_config**: the model configuration.

- **adapter_path**: the path to the adapter.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.

- **use_magpie_template**: a flag used to enable/disable applying the Magpie pre-query  template. Defaults to `False`.

- **magpie_pre_query_template**: the pre-query template to be applied to the prompt or  sent to the LLM to generate an instruction or a follow up user message. Valid  values are "llama3", "qwen2" or another pre-query template provided. Defaults  to `None`.







### Examples


#### Generate text
```python
from distilabel.models.llms import MlxLLM

llm = MlxLLM(model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```




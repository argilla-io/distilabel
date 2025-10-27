---
hide:
  - navigation
---
# LiteLLM


LiteLLM implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "gpt-3.5-turbo" or "mistral/mistral-large",  etc.

- **verbose**: whether to log the LiteLLM client's logs. Defaults to `False`.

- **structured_output**: a dictionary containing the structured output configuration configuration  using `instructor`. You can take a look at the dictionary structure in  `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.





### Runtime Parameters

- **verbose**: whether to log the LiteLLM client's logs. Defaults to `False`.




### Examples


#### Generate text
```python
from distilabel.models.llms import LiteLLM

llm = LiteLLM(model="gpt-3.5-turbo")

llm.load()

# Call the model
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

Generate structured data:
```




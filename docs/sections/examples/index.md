
This section contains different example pipelines.

### RPG Characters with LlamaCpp

This script makes use of [`LlamaCppLLM`][distilabel.llms.llamacpp.LlamaCppLLM] and the structured output capabilities thanks to [`outlines`](https://outlines-dev.github.io/outlines/welcome/) to generate RPG characters that adhere to a JSON schema.

It makes use of a local model which can be downloaded using curl (explained in the script itself), and can be exchanged with other `LLMs` like [`vLLM`][distilabel.llms.vllm.vLLM].

!!! Run

    ```python
    python examples/structured_generation_with_outlines.py
    ```

```python title="structured_generation_with_outlines.py"
--8<-- "examples/structured_generation_with_outlines.py"
```

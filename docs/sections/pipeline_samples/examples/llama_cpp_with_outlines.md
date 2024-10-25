---
hide: toc
---
# Structured generation with `outlines`

Generate RPG characters following a `pydantic.BaseModel` with `outlines` in `distilabel`.

This script makes use of [`LlamaCppLLM`][distilabel.models.llamacpp.LlamaCppLLM] and the structured output capabilities thanks to [`outlines`](https://outlines-dev.github.io/outlines/welcome/) to generate RPG characters that adhere to a JSON schema.

![Arena Hard](../../../assets/pipelines/knowledge_graphs.png)

It makes use of a local model which can be downloaded using curl (explained in the script itself), and can be exchanged with other `LLMs` like [`vLLM`][distilabel.models.vllm.vLLM].

??? Run

    ```python
    python examples/structured_generation_with_outlines.py
    ```

```python title="structured_generation_with_outlines.py"
--8<-- "examples/structured_generation_with_outlines.py"
```
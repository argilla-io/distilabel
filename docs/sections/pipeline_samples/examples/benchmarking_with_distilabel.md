---
hide: toc
---
# Benchmarking with `distilabel`: Arena Hard

Benchmark LLMs with `distilabel`: reproducing the Arena Hard benchmark.

The script below first defines both the `ArenaHard` and the `ArenaHardResults` tasks, so as to generate responses for a given collection of prompts/questions with up to two LLMs, and then calculate the results as per the original implementation, respectively. Additionally, the second part of the example builds a `Pipeline` to run the generation on top of the prompts with `InferenceEndpointsLLM` while streaming the rest of the generations from a pre-computed set of GPT-4 generations, and then evaluate one against the other with `OpenAILLM` generating an alternate response, a comparison between the responses, and a result as A>>B, A>B, B>A, B>>A, or tie.

To run this example you will first need to install the Arena Hard optional dependencies, being `pandas`, `scikit-learn`, and `numpy`.

??? Run

    ```python
    python examples/arena_hard.py
    ```

```python title="arena_hard.py"
--8<-- "examples/arena_hard.py"
```
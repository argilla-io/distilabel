---
hide:
  - navigation
---
# MixtureOfAgentsLLM


`Mixture-of-Agents` implementation.



An `LLM` class that leverages `LLM`s collective strenghts to generate a response,
    as described in the "Mixture-of-Agents Enhances Large Language model Capabilities"
    paper. There is a list of `LLM`s proposing/generating outputs that `LLM`s from the next
    round/layer can use as auxiliary information. Finally, there is an `LLM` that aggregates
    the outputs to generate the final response.





### Attributes

- **aggregator_llm**: The `LLM` that aggregates the outputs of the proposer `LLM`s.

- **proposers_llms**: The list of `LLM`s that propose outputs to be aggregated.

- **rounds**: The number of layers or rounds that the `proposers_llms` will generate  outputs. Defaults to `1`.







### Examples


#### Generate text
```python
from distilabel.models.llms import MixtureOfAgentsLLM, InferenceEndpointsLLM

llm = MixtureOfAgentsLLM(
    aggregator_llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    proposers_llms=[
        InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        ),
        InferenceEndpointsLLM(
            model_id="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            tokenizer_id="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        ),
        InferenceEndpointsLLM(
            model_id="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
            tokenizer_id="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        ),
    ],
    rounds=2,
)

llm.load()

output = llm.generate_outputs(
    inputs=[
        [
            {
                "role": "user",
                "content": "My favorite witty review of The Rings of Power series is this: Input:",
            }
        ]
    ]
)
```




### References

- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)


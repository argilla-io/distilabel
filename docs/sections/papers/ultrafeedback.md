## UltraFeedback: Boosting Language Models with High-quality Feedback

UltraFeedback is a large-scale, fine-grained, diverse preference dataset, used for training powerful reward models and critic models.

UltraFeedback collects about 64k prompts from diverse resources (including UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FalseQA, and FLAN), then they use these prompts to query multiple LLMs (commercial models, Llama models ranging 7B to 70B, and non-Llama models) and generate four different responses for each prompt, resulting in a total of 256k samples i.e. the UltraFeedback will rate four responses on every OpenAI request.

To collect high-quality preference and textual feedback, they design a fine-grained annotation instruction, which contains four different aspects, namely instruction-following, truthfulness, honesty and helpfulness (even though within the paper they also mention a fifth one named verbalized calibration). Finally, GPT-4 is used to generate the ratings for the generated responses to the given prompt using the previously mentioned aspects.

### Replication

To replicate the paper we will be using `distilabel` and a smaller dataset created by the Hugging Face H4 team named [`HuggingFaceH4/instruction-dataset`](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset) for testing purposes.

Also for testing purposes we will just show how to evaluate the generated responses for a given prompt using a new global aspect named `overall-rating` defined by Argilla, that computes the average of the four aspects, so as to reduce number of requests to be sent to OpenAI, but note that all the aspects are implemented within `distilabel` and can be used instead for a more faithful reproduction. Besides that we will generate two responses i.e. run the text generation on top of two LLMs instead of four, to reduce the compute required too.

#### Installation

To replicate UltraFeedback one will need to install `distilabel` as it follows:

```bash
pip install "distilabel[argilla,openai,vllm]>=1.0.0"
```

And since we will be using `vllm` we will need to use a VM with at least 2 NVIDIA GPUs with at least 16GB of memory each to run the text generation, and set the `OPENAI_API_KEY` environment variable value.

#### Building blocks

* `LoadHubDataset`: Generator Step to load a dataset from the Hugging Face Hub.
* `TextGeneration`: Task to generate responses for a given instruction using an LLM.
    * `vLLM`: LLM that loads a model from the Hugging Face Hub using `vLLM`.
* `CombineColumns`: Task that combines multiple columns into a single one i.e. from string to list of strings. Useful when there are multiple parallel steps that are connected to the same node.
* `UltraFeedback`: Task that generates ratings for the responses of a given instruction using the UltraFeedback prompt.
    * `OpenAILLM`: LLM that loads a model from OpenAI using `OpenAILLM`.
* `KeepColumns`: Task to keep the desired columns while removing the not needed ones, as well as defining the order for those. 
* `PreferenceToArgilla`: Task to optionally push the generated dataset to Argilla to do some further analysis and human annotation.

#### Code

As mentioned before, we will put the previously mentioned building blocks together to replicate UltraFeedback.

```python
from distilabel.llms import OpenAILLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    CombineColumns,
    KeepColumns,
    LoadHubDataset,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback


with Pipeline(name="ultrafeedback-pipeline") as pipeline:
    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    text_generation_with_notus = TextGeneration(
        name="text_generation_with_notus",
        llm=vLLM(model="argilla/notus-7b-v1"),
        input_batch_size=10,
        output_mappings={"model_name": "generation_model"},
    )
    text_generation_with_zephyr = TextGeneration(
        name="text_generation_with_zephyr",
        llm=vLLM(model="HuggingFaceH4/zephyr-7b-gemma-v0.1"),
        input_batch_size=10,
        output_mappings={"model_name": "generation_model"},
    )
    load_hub_dataset.connect(text_generation_with_notus)
    load_hub_dataset.connect(text_generation_with_zephyr)

    combine_columns = CombineColumns(
        name="combine_columns",
        columns=["generation", "generation_model"],
        output_columns=["generations", "generation_models"],
    )
    text_generation_with_notus.connect(combine_columns)
    text_generation_with_zephyr.connect(combine_columns)

    ultrafeedback = UltraFeedback(
        name="ultrafeedback_openai",
        llm=OpenAILLM(model="gpt-4"),
        aspect="overall-rating",
        output_mappings={"model_name": "ultrafeedback_model"},
    )
    combine_columns.connect(ultrafeedback)

    keep_columns = KeepColumns(
        name="keep_columns",
        columns=[
            "instruction",
            "generations",
            "generation_models",
            "ratings",
            "rationales",
            "ultrafeedback_model",
        ],
    )
    ultrafeedback.connect(keep_columns)

    # # Optional: Push the generated dataset to Argilla
    # push_to_argilla = PreferenceToArgilla(
    #     name="push_to_argilla",
    #     api_url="<ARGILLA_API_URL>",
    #     api_key="<ARGILLA_API_KEY>",  # type: ignore
    #     dataset_name="ultrafeedback",
    #     dataset_workspace="admin",
    #     num_generations=2,
    # )
    # keep_columns.connect(push_to_argilla)
```

Then we need to call `pipeline.run` with the runtime parameters so that the pipeline can be launched.

```python
dataset = pipeline.run(
    parameters={
        "load_dataset": {
            "repo_id": "HuggingFaceH4/instruction-dataset",
            "split": "test",
        },
        "text_generation_with_notus": {
            "generation_kwargs": {
                "max_new_tokens": 512,
                "temperature": 0.7,
            },
        },
        "text_generation_with_zephyr": {
            "generation_kwargs": {
                "max_new_tokens": 512,
                "temperature": 0.7,
            },
        },
        "ultrafeedback_overall_rating": {
            "generation_kwargs": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
            },
        },
    }
)
```

Finally, we can optionally push the generated dataset, named `Distiset`, to the Hugging Face Hub via the `push_to_hub` method, so that each subset generated in the leaf steps is pushed to the Hub.

```python
dataset.push_to_hub("distilabel-internal-testing/ultrafeedback-instruction-dataset", private=True)
```

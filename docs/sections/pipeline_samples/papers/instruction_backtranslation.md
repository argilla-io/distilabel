# Instruction Backtranslation

["Self Alignment with Instruction Backtranslation"](https://arxiv.org/abs/2308.06259) presents a scalable method to build high-quality instruction following a language model by automatically labeling human-written text with corresponding instructions. Their approach, named instruction backtranslation, starts with a language model finetuned on a small amount of seed data, and a given web corpus. The seed model is used to construct training examples by generating instruction prompts for web documents (self-augmentation), and then selecting high-quality examples from among these candidates (self-curation). This data is then used to finetune a stronger model.

Their self-training approach assumes access to a base language model, a small amount of seed data, and a collection of unlabelled examples, e.g. a web corpus. The unlabelled data is a large, diverse set of human-written documents that includes writing about all manner of topics humans are interested in – but crucially is not paired with instructions.

A first key assumption is that there exists some subset of this very large human-written text that would be suitable as gold generations for some user instructions. A second key assumption is that they can predict instructions for these candidate gold answers that can be used as high-quality example pairs to train an instruction-following model.

Their overall process, called instruction back translation performs two core steps:

1. Self-augment: Generate instructions for unlabelled data, i.e. the web corpus, to produce candidate training data of (instruction, output) pairs for instruction tuning.

2. Self-curate: Self-select high-quality demonstration examples as training data to finetune the base model to follow instructions. This approach is done iteratively where a better intermediate instruction-following model can improve on selecting data for finetuning in the next iteration.

This replication covers the self-curation step i.e. the second/latter step as mentioned above, so as to be able to use the proposed prompting approach to rate the quality of the generated text, which can either be synthetically generated or real human-written text.

### Replication

To replicate the paper we will be using `distilabel` and a smaller dataset created by the Hugging Face H4 team named [`HuggingFaceH4/instruction-dataset`](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset) for testing purposes.

#### Installation

To replicate Self Alignment with Instruction Backtranslation one will need to install `distilabel` as it follows:

```bash
pip install "distilabel[hf-inference-endpoints,openai]>=1.0.0"
```

And since we will be using [`InferenceEndpointsLLM`][distilabel.llms.InferenceEndpointsLLM] (installed via the extra `hf-inference-endpoints`) we will need deploy those in advance either locally or in the Hugging Face Hub (alternatively also the serverless endpoints can be used, but most of the times the inference times are slower, and there's a limited quota to use those as those are free) and set both the `HF_TOKEN` (to use the [`InferenceEndpointsLLM`][distilabel.llms.InferenceEndpointsLLM]) and the `OPENAI_API_KEY` environment variable value (to use the [`OpenAILLM`][distilabel.llms.OpenAILLM]).

#### Building blocks

- [`LoadDataFromHub`][distilabel.steps.LoadDataFromHub]: Generator Step to load a dataset from the Hugging Face Hub.
- [`TextGeneration`][distilabel.steps.tasks.TextGeneration]: Task to generate responses for a given instruction using an LLM.
    - [`InferenceEndpointsLLM`][distilabel.llms.InferenceEndpointsLLM]: LLM that runs a model from an Inference Endpoint in the Hugging Face Hub.
- [`InstructionBacktranslation`][distilabel.steps.tasks.InstructionBacktranslation]: Task that generates a score and a reason for a response for a given instruction using the Self Alignment with Instruction Backtranslation prompt.
    - [`OpenAILLM`][distilabel.llms.OpenAILLM]: LLM that loads a model from OpenAI.

#### Code

As mentioned before, we will put the previously mentioned building blocks together to replicate Self Alignment with Instruction Backtranslation.

```python
from distilabel.llms import InferenceEndpointsLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps.tasks import InstructionBacktranslation, TextGeneration


with Pipeline(name="self-alignment-with-instruction-backtranslation") as pipeline:
    load_hub_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=InferenceEndpointsLLM(
            base_url="<INFERENCE_ENDPOINT_URL>",
            tokenizer_id="argilla/notus-7b-v1",
            model_display_name="argilla/notus-7b-v1",
        ),
        input_batch_size=10,
        output_mappings={"model_name": "generation_model"},
    )

    instruction_backtranslation = InstructionBacktranslation(
        name="instruction_backtranslation",
        llm=OpenAILLM(model="gpt-4"),
        input_batch_size=10,
        output_mappings={"model_name": "scoring_model"},
    )

    keep_columns = KeepColumns(
        name="keep_columns",
        columns=[
            "instruction",
            "generation",
            "generation_model",
            "score",
            "reason",
            "scoring_model",
        ],
    )

    load_hub_dataset >> text_generation >> instruction_backtranslation >> keep_columns
```

Then we need to call `pipeline.run` with the runtime parameters so that the pipeline can be launched.

```python
distiset = pipeline.run(
    parameters={
        load_hub_dataset.name: {
            "repo_id": "HuggingFaceH4/instruction-dataset",
            "split": "test",
        },
        text_generation.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 1024,
                    "temperature": 0.7,
                },
            },
        },
        instruction_backtranslation.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 1024,
                    "temperature": 0.7,
                },
            },
        },
    },
)
```

Finally, we can optionally push the generated dataset, named [`Distiset`][distilabel.distiset.Distiset], to the Hugging Face Hub via the `push_to_hub` method, so that each subset generated in the leaf steps is pushed to the Hub.

```python
distiset.push_to_hub(
    "instruction-backtranslation-instruction-dataset",
    private=True,
)
```

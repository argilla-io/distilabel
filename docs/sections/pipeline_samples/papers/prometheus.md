# Prometheus 2

["Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models"](https://arxiv.org/pdf/2405.01535) presents Prometheus 2, a new and more powerful evaluator LLM compared to Prometheus (its predecessor) presented in ["Prometheus: Inducing Fine-grained Evaluation Capability in Language Models"](https://arxiv.org/abs/2310.08491); since GPT-4, as well as other proprietary LLMs, are commonly used to assess the quality of the responses for various LLMs, but there are concerns about transparency, controllability, and affordability, that motivate the need of open-source LLMs specialized in evaluations.

![Prometheus 2 pipeline overview](../../../assets/pipelines/prometheus.png)

Existing open evaluator LMs exhibit critical shortcomings:

1. They issue scores that significantly diverge from those assigned by humans.
2. They lack the flexibility to perform both direct assessment and pairwise ranking, the two most prevalent forms of assessment.

Additionally, they do not possess the ability to evaluate based on custom evaluation criteria, focusing instead on general attributes like helpfulness and harmlessness. Prometheus 2 is capable of processing both direct assessment and pair-wise ranking formats grouped with user-defined evaluation criteria.

Prometheus 2 released two variants:

- [`prometheus-eval/prometheus-7b-v2.0`](https://hf.co/prometheus-eval/prometheus-7b-v2.0): fine-tuned on top of [`mistralai/Mistral-7B-Instruct-v0.2`](https://hf.co/mistralai/Mistral-7B-Instruct-v0.2)
- [`prometheus-eval/prometheus-8x7b-v2.0`](https://hf.co/prometheus-eval/prometheus-8x7b-v2.0): fine-tuned on top of [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://hf.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

Both models have been fine-tuned for both direct assessment and pairwise ranking tasks i.e. assessing the quality of a single isolated response for a given instruction with or without a reference answer and assessing the quality of one response against another one for a given instruction with or without a reference answer, respectively.

On four direct assessment benchmarks and four pairwise ranking benchmarks, Prometheus 2 scores the highest correlation and agreement with humans and proprietary LM judges among all tested open evaluator LMs. Their models, code, and data are all publicly available at [`prometheus-eval/prometheus-eval`](https://github.com/prometheus-eval/prometheus-eval).

### Replication

!!! NOTE
    The section is named `Replication` but in this case we're not replicating the Prometheus 2 paper per se, but rather showing how to use the [`PrometheusEval`][distilabel.steps.tasks.PrometheusEval] task implemented within `distilabel` to evaluate the quality of the responses from a given instruction using the Prometheus 2 model.

To showcase Prometheus 2 we will be using the [`PrometheusEval`][distilabel.steps.tasks.PrometheusEval] task implemented in `distilabel` and a smaller dataset created by the Hugging Face H4 team named [`HuggingFaceH4/instruction-dataset`](https://hf.co/datasets/HuggingFaceH4/instruction-dataset) for testing purposes.

#### Installation

To reproduce the code below, one will need to install `distilabel` as it follows:

```bash
pip install "distilabel[vllm]>=1.1.0"
```

Alternatively, it's recommended to install [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) to benefit from Flash Attention 2 speed ups during inference via `vllm`.

```bash
pip install flash-attn --no-build-isolation
```

!!! NOTE
    The installation notes above assume that you are using a VM with one GPU accelerator with at least the required VRAM to fit [`prometheus-eval/prometheus-7b-v2.0`](https://hf.co/prometheus-eval/prometheus-7b-v2.0) in bfloat16 (28GB); but if you have enough VRAM to fit their 8x7B model in bfloat16 (~90GB) you can use [`prometheus-eval/prometheus-8x7b-v2.0`](https://hf.co/prometheus-eval/prometheus-8x7b-v2.0) instead.

#### Building blocks

- [`LoadDataFromHub`][distilabel.steps.LoadDataFromHub]: [`GeneratorStep`][distilabel.steps.GeneratorStep] to load a dataset from the Hugging Face Hub.

- [`PrometheusEval`][distilabel.steps.tasks.PrometheusEval]: [`Task`][distilabel.steps.tasks.Task] that assesses the quality of a response for a given instruction using any of the Prometheus 2 models.
    - [`vLLM`][distilabel.models.vLLM]: [`LLM`][distilabel.models.LLM] that loads a model from the Hugging Face Hub via [vllm-project/vllm](https://github.com/vllm-project/vllm).

    !!! NOTE
        Since the Prometheus 2 models use a slightly different chat template than [`mistralai/Mistral-7B-Instruct-v0.2`](https://hf.co/mistralai/Mistral-7B-Instruct-v0.2), we need to set the `chat_template` parameter to `[INST] {{ messages[0]['content'] }}\n{{ messages[1]['content'] }}[/INST]` so as to properly format the input for Prometheus 2.

- (Optional) [`KeepColumns`][distilabel.steps.KeepColumns]: [`Task`][distilabel.steps.tasks.Task] that keeps only the specified columns in the dataset, used to remove the undesired columns.

#### Code

As mentioned before, we will put the previously mentioned building blocks together to see how Prometheus 2 can be used via `distilabel`.

```python
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns, LoadDataFromHub
from distilabel.steps.tasks import PrometheusEval

if __name__ == "__main__":
    with Pipeline(name="prometheus") as pipeline:
        load_dataset = LoadDataFromHub(
            name="load_dataset",
            repo_id="HuggingFaceH4/instruction-dataset",
            split="test",
            output_mappings={"prompt": "instruction", "completion": "generation"},
        )

        task = PrometheusEval(
            name="task",
            llm=vLLM(
                model="prometheus-eval/prometheus-7b-v2.0",
                chat_template="[INST] {{ messages[0]['content'] }}\n{{ messages[1]['content'] }}[/INST]",
            ),
            mode="absolute",
            rubric="factual-validity",
            reference=False,
            num_generations=1,
            group_generations=False,
        )

        keep_columns = KeepColumns(
            name="keep_columns",
            columns=["instruction", "generation", "feedback", "result", "model_name"],
        )

        load_dataset >> task >> keep_columns
```

Then we need to call `pipeline.run` with the runtime parameters so that the pipeline can be launched.

```python
distiset = pipeline.run(
    parameters={
        task.name: {
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
    "instruction-dataset-prometheus",
    private=True,
)
```

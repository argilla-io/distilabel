# Special Tasks

This section covers some tasks that don't implement the [`Task`][distilabel.steps.tasks.base.Task] API, but can be thought of as tasks, instead they inherit from [`Step`][distilabel.steps.base.Step].

## Embedding Generation

The [`DEITA`](../../papers/deita.md) paper needs to tackle the challenge of ensuring diversity in the final dataset, and they propose an embedding-based method to filter the dataset. For this end, the [`GenerateEmbeddings`][distilabel.steps.tasks.generate_embeddings] step is in charge of generating embeddings for the datasets' text.

```python
from distilabel.llms.huggingface.transformers import TransformersLLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.generate_embeddings import GenerateEmbeddings

llm = TransformersLLM(
    model="TaylorAI/bge-micro-v2",
    model_kwargs={"is_decoder": True},
)
llm.load()

task = GenerateEmbeddings(
    name="task",
    llm=llm,
    pipeline=Pipeline(name="unit-test-pipeline"),
)
```

This step needs an `LLM` to generate the embeddings, we have chosen to use a `TransformersLLM` with `TaylorAI/bge-micro-v2` in this case. Upon call, this step will compute the embedding for the input text and add it to the row:

```python
result = next(task.process([{"text": "Hello, how are you?"}]))
print(result[0]["embedding"])
# [-8.12729941, -5.24642847, -6.34003029, ...]
```

## Ranking LLM Responses

Jian et al. present in their paper [`LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion`](https://arxiv.org/abs/2306.02561) a "small" model that is able to take a instruction and a **pair** output candidates, and output a score for each candidate to measure their **relative** quality, hence **ranking** the responses. You can use [`PairRM`][distilabel.steps.tasks.pair_rm] in distilabel to accomplish this task, let's see how it works:

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.pair_rm import PairRM

ranker = PairRM(
    name="pair_rm_ranker", pipeline=Pipeline(name="ranking-pipeline")
)
# NOTE: Keep in mind this call will automatically try to load an LLM internally
ranker.load()
```

For this step the model is fixed by default contrary to other steps, as the implementation relies completely on [`LLM-Blender`](https://github.com/yuchenlin/LLM-Blender) for it to work.

To ingest data for this task you would need an input, which corresponds to the instruction, and a list of candidates to compare, that the model will rank working on pairs:

```python
result = next(
    ranker.process(
        [
            {"input": "Hello, how are you?", "candidates": ["fine", "good", "bad"]},
        ]
    )
)
```

Let's see what the result looks like:

```python
import json
print(json.dumps(result, indent=2))
# [
#   {
#     "input": "Hello, how are you?",
#     "candidates": [
#       "fine",
#       "good",
#       "bad"
#     ],
#     "ranks": [
#       2,
#       1,
#       3
#     ],
#     "ranked_candidates": [
#       "good",
#       "fine",
#       "bad"
#     ]
#   }
# ]
```

We see we have both the `ranks`, that determine the position that would order the `candidates` field, and the `ranked_candidates` in case these want to be used directly.

## Filtering data to ensure diversity

We have already mentioned a global step that appeared in the `Global Steps` section, but it was quite specific to be introduced at that time. This `Task` is the [`DeitaFiltering`][distilabel.steps.deita.DeitaFiltering] step.

It's a special type of step developed to reproduce the [`DEITA`](../../papers/deita.md) paper, in charge of filtering responses according to a predefined score. Let's see how it is defined:

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.deita import DeitaFiltering

deita_filtering = DeitaFiltering(
    name="deita_filtering",
    data_budget=1,
    pipeline=Pipeline(name="deita-filtering-pipeline"),
)
# Remember to call the load method if working outside of a Pipeline context
deita_filtering.load()
```

This step is prepared to work on `DEITA` outputs:
It expects instructions evolved following the `Evol Instruct` procedure, with a score assigned to the complexity of the instruction and the quality of the response ([`ComplexityScorer`][distilabel.steps.tasks.complexity_scorer] and [`QualityScorer`][distilabel.steps.tasks.quality_scorer] respectively), and embeddings computed on the responses. The following is a random example following the structure of the input needed from the process method:

```python
result = next(
    deita_filtering.process(
        [
            {
                "evol_instruction_score": 0.5,
                "evol_response_score": 0.5,
                "embedding": [-8.12729941, -5.24642847, -6.34003029],
            },
            {
                "evol_instruction_score": 0.6,
                "evol_response_score": 0.6,
                "embedding": [2.99329242, 0.7800932, 0.7799726],
            },
            {
                "evol_instruction_score": 0.7,
                "evol_response_score": 0.7,
                "embedding": [10.29041806, 14.33088073, 13.00557506],
            },
        ]
    )
)
```

And this is what we could expect from the output:

```python
import json

print(json.dumps(result, indent=2))
# [
#   {
#     "evol_instruction_score": 0.5,
#     "evol_response_score": 0.5,
#     "embedding": [
#       -8.12729941,
#       -5.24642847,
#       -6.34003029
#     ],
#     "deita_score": 0.25,
#     "deita_score_computed_with": [
#       "evol_instruction_score",
#       "evol_response_score"
#     ],
#     "nearest_neighbor_distance": 1.9042812683723933
#   }
# ]
```

We would obtain the dataset size expected for our `data_budget` and `diversity_threshold` set. For more information on how this `Task` works take a look at the `API Reference`.

# Review claims against their cited documentation

A URL in a generated answer does not establish that the linked text supports the
answer. This sample keeps a public software documentation excerpt alongside a
review claim, then asks a second task to judge the claim against that excerpt.
It builds on the [custom Step guide](../../how_to_guides/basic/step/index.md#defining-custom-steps)
and the custom-template support in `TextGeneration`.

The pipeline is:

```mermaid
flowchart LR
    A[Saved documentation] --> B[Generate a cited claim]
    A --> R[Optional MCP retrieval]
    R --> B
    B --> C[Judge citation support]
```

## Run with saved evidence

From a repository checkout, using Python 3.11:

```sh
python -m pip install -e . 'datasets==3.6.0'
HF_DATASETS_OFFLINE=1 python docs/sections/pipeline_samples/examples/source_grounded_review.py
```

[Download the script](source_grounded_review.py). Its default evidence is a saved
paragraph from the Apache-2.0 Distilabel caching guide, with a permalink to the
original revision. This path needs no retrieval dependency or service credentials.
`HF_DATASETS_OFFLINE=1` also disables the dataset library's remote lookup when
Distilabel loads its local pipeline output.

The runnable default uses two deterministic `QuoteDemoLLM` instances: one copies
the excerpt into a claim and the other recognizes an exact quotation. This checks
the pipeline wiring; it does **not** demonstrate semantic evaluation quality.
For generating new review claims and applying the full rubric, pass your own
configured generation and rubric `LLM` objects to `build_pipeline`. Their provider
credentials, generation parameters, timeouts and task error handling remain with
the existing Distilabel LLM implementation. See
[executing tasks with LLMs](../../how_to_guides/basic/llm/index.md).

The resulting dataset retains these columns:

| Column | Meaning |
| --- | --- |
| `question` | The public documentation question |
| `source_url`, `source_excerpt` | The URL and exact text supplied to both tasks |
| `retrieval_mode` | `saved` or `live` |
| `answer` | Raw JSON generation, retained for inspection |
| `claim`, `citation_url` | Parsed claim and its citation |
| `citation_matches` | Whether the citation exactly matches the supplied URL |
| `judgment`, `explanation` | Rubric feedback and its reasoning |
| `generation_model`, `rubric_model` | Distilabel's identifiers for the two LLMs |

The rubric distinguishes `supported` (the excerpt supports the entire claim),
`contradicted` (the excerpt explicitly conflicts with it), and `not_established`
(neither). A missing or different citation is always `not_established`, even if
the judge returns `supported`. A matching URL alone does not imply support.
The judgment concerns only the supplied excerpt, which can be incomplete or stale.
Treat generated judgments as feedback for human review, not verified truth.
Malformed task output follows Distilabel's task error handling and must not be
counted as a valid judgment.

## Optionally retrieve current documentation

Install the optional MCP SDK, then explicitly enable the example-local retrieval
step:

```sh
python -m pip install 'mcp==1.26.0'
HF_DATASETS_OFFLINE=1 python docs/sections/pipeline_samples/examples/source_grounded_review.py --live
```

See [Parallel MCP setup](https://docs.parallel.ai/integrations/mcp/quickstart) for
anonymous search and its rate limits. No Parallel key is used by this example.
The public question and a fixed documentation search query are sent to Parallel's
Unified MCP server. Keep private inputs out of this sample. The script makes no
changes to Distilabel's integrations or defaults.

`RetrieveDocumentation` uses the MCP client SDK and a Distilabel-specific
User-Agent on its HTTP client. It accepts only HTTPS results from
`distilabel.argilla.io`, takes the first usable result, and bounds the supplied
excerpt to 4,000 characters. It retains that bounded text without summarizing it.
The generation and rubric tasks receive the returned URL and excerpt in place of
the saved evidence.

Retrieval has HTTP and MCP read timeouts of 60 seconds and a total limit of 75
seconds per question. A tool error, transport failure, or empty usable result
stops retrieval without falling back to saved evidence. The example uses a
`GlobalStep` so all rows, including connection cleanup, must succeed before any
row reaches generation. This also prevents a late failure from producing partial
citations. Keep this sample small: the global step collects all input rows.

## Check the pipeline

```sh
python -m pip install pytest 'mcp==1.26.0'
HF_DATASETS_OFFLINE=1 python -m pytest -q docs/sections/pipeline_samples/examples/test_source_grounded_review.py
```

[The checks](test_source_grounded_review.py) run native
`Pipeline.run(use_cache=False)` with deterministic local generation and rubric
fixtures. They cover supported, contradicted and not established claims, a
mismatched citation, evidence preservation, empty results and a late retrieval
failure. A separate HTTP test double exercises the actual MCP SDK, checks the
User-Agent and absence of an Authorization header, and passes retrieved evidence
through the native pipeline. These deterministic checks do not establish live
service availability or judge accuracy. Run `--live` separately to check anonymous
retrieval and inspect its source URL and excerpt.

## Source

```python
--8<-- "sections/pipeline_samples/examples/source_grounded_review.py"
```

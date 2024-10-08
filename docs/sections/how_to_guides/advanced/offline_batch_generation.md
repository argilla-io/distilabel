The [offline batch generation](../basic/llm/index.md#offline-batch-generation) is a feature that some `LLM`s implemented in `distilabel` offers, allowing to send the inputs to a LLM-as-a-service platform and waiting for the outputs in a asynchronous manner. LLM-as-a-service platforms offer this feature as it allows them to gather many inputs and creating batches as big as the hardware allows, maximizing the hardware utilization and reducing the cost of the service. In exchange, the user has to wait certain time for the outputs to be ready but the cost per token is usually much lower.

`distilabel` pipelines are able to handle `LLM`s that offer this feature in the following way:

* The first time the pipeline gets executed, the `LLM` will send the inputs to the platform. The platform will return jobs ids that can be used later to check the status of the jobs and retrieve the results. The `LLM` will save these jobs ids in its `jobs_ids` attribute and raise an special exception [DistilabelOfflineBatchGenerationNotFinishedException][distilabel.exceptions.DistilabelOfflineBatchGenerationNotFinishedException] that will be handled by the `Pipeline`. The jobs ids will be saved in the pipeline cache, so they can be used in subsequent calls.
* The second time and subsequent calls will recover the pipeline execution and the `LLM` won't send the inputs again to the platform. This time as it has the `jobs_ids` it will check if the jobs have finished, and if they have then it will retrieve the results and return the outputs. If they haven't finished, then it will raise again `DistilabelOfflineBatchGenerationNotFinishedException` again.
* In addition, LLMs with offline batch generation can be specified to do polling until the jobs have finished, blocking the pipeline until they are done. If for some reason the polling needs to be stopped, one can press ++ctrl+c++ or ++cmd+c++ depending on your OS (or send a `SIGINT` to the main process) which will stop the polling and raise `DistilabelOfflineBatchGenerationNotFinishedException` that will be handled by the pipeline as described above.

!!! WARNING

    In order to recover the pipeline execution and retrieve the results, the pipeline cache must be enabled. If the pipeline cache is disabled, then it will send the inputs again and create different jobs incurring in extra costs.


## Example pipeline using `OpenAILLM` with offline batch generation

```python
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    load_data = LoadDataFromHub(output_mappings={"prompt": "instruction"})

    text_generation = TextGeneration(
        llm=OpenAILLM(
            model="gpt-3.5-turbo",
            use_offline_batch_generation=True,  # (1)
        )
    )

    load_data >> text_generation


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_data.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset",
                "split": "test",
                "batch_size": 500,
            },
        }
    )
```

1. Indicate that the `OpenAILLM` should use offline batch generation.

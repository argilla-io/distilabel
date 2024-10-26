---
description: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
hide:
  - toc
---

# Frequent Asked Questions (FAQ)

??? faq "How can I rename the columns in a batch?"
    Every [`Step`][distilabel.steps.base.Step] has both `input_mappings` and `output_mappings` attributes that can be used to rename the columns in each batch.

    But `input_mappings` will only map, meaning that if you have a batch with the column `A` and you want to rename it to `B`, you should use `input_mappings={"A": "B"}`, but that will only be applied to that specific [`Step`][distilabel.steps.base.Step] meaning that the next step in the pipeline will still have the column `A` instead of `B`.

    While `output_mappings` will indeed apply the rename, meaning that if the [`Step`][distilabel.steps.base.Step] produces the column `A` and you want to rename to `B`, you should use `output_mappings={"A": "B"}`, and that will be applied to the next [`Step`][distilabel.steps.base.Step] in the pipeline.

??? faq "Will the API Keys be exposed when sharing the pipeline?"
    No, those will be masked out using `pydantic.SecretStr`, meaning that those won't be exposed when sharing the pipeline.

    This also means that if you want to re-run your own pipeline and the API keys have not been provided via environment variable but either via an attribute or runtime parameter, you will need to provide them again.

??? faq "Does it work for Windows?"

    Yes, but you may need to set the `multiprocessing` context in advance to ensure that the `spawn` method is used since the default method `fork` is not available on Windows.

    ```python
    import multiprocessing as mp

    mp.set_start_method("spawn")
    ```

??? faq "Will the custom Steps / Tasks / LLMs be serialized too?"
    No, at the moment, only the references to the classes within the `distilabel` library will be serialized, meaning that if you define a custom class used within the pipeline, the serialization won't break, but the deserialize will fail since the class won't be available unless used from the same file.

??? faq "What happens if `Pipeline.run` fails? Do I lose all the data?"
    No, indeed, we're using a cache mechanism to store all the intermediate results in the disk so, if a [`Step`][distilabel.steps.base.Step] fails; the pipeline can be re-run from that point without losing the data, only if nothing is changed in the `Pipeline`.

    All the data will be stored in `.cache/distilabel`, but the only data that will persist at the end of the `Pipeline.run` execution is the one from the leaf step/s, so bear that in mind.

    For more information on the caching mechanism in `distilabel`, you can check the [Learn - Advanced - Caching](../how_to_guides/advanced/caching.md) section.

    Also, note that when running a [`Step`][distilabel.steps.base.Step] or a [`Task`][distilabel.steps.tasks.Task] standalone, the cache mechanism won't be used, so if you want to use that, you should use the `Pipeline` context manager.

??? faq "How can I use the same `LLM` across several tasks without having to load it several times?"
    You can serve the LLM using a solution like TGI or vLLM, and then connect to it using an `AsyncLLM` client like `InferenceEndpointsLLM` or `OpenAILLM`. Please refer to [Serving LLMs guide](../how_to_guides/advanced/serving_an_llm_for_reuse.md) for more information.

??? faq "Can `distilabel` be used with [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)?"
    Yes, `distilabel` is integrated with OpenAI Batch API via [OpenAILLM][distilabel.models.llms.openai.OpenAILLM]. Check [LLMs - Offline Batch Generation](../how_to_guides/basic/llm/index.md#offline-batch-generation) for a small example on how to use it and [Advanced - Offline Batch Generation](../how_to_guides/advanced/offline_batch_generation.md) for a more detailed guide.

??? faq "Prevent overloads on [Free Serverless Endpoints][distilabel.models.llms.huggingface.InferenceEndpointsLLM]"
    When running a task using the [InferenceEndpointsLLM][distilabel.models.llms.huggingface.InferenceEndpointsLLM] with Free Serverless Endpoints, you may be facing some errors such as `Model is overloaded` if you let the batch size to the default (set at 50). To fix the issue, lower the value or even better set `input_batch_size=1` in your task. It may take a longer time to finish, but please remember this is a free service.

    ```python
    from distilabel.models import InferenceEndpointsLLM
    from distilabel.steps import TextGeneration

    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        input_batch_size=1
    )
    ```

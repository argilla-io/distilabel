# Frequent Asked Questions (FAQ)

??? faq "How can I rename the columns in a batch?"
    Every [`Step`][distilabel.steps.base.Step] has both `input_mappings` and `output_mappings` attributes, that can be used to rename the columns in each batch.

    But `input_mappings` will only map, meaning that if you have a batch with the column `A` and you want to rename to `B`, you should use `input_mappings={"A": "B"}`, but that will only be applied to that specific [`Step`][distilabel.steps.base.Step] meaning that the next step in the pipeline will still have the column `A` instead of `B`.

    While `output_mappings` will indeed apply the rename, meaning that if the [`Step`][distilabel.steps.base.Step] produces the column `A` and you want to rename to `B`, you should use `output_mappings={"A": "B"}`, and that will be applied to the next [`Step`][distilabel.steps.base.Step] in the pipeline.

??? faq "Will the API Keys be exposed when sharing the pipeline?"
    No, those will be masked out using `pydantic.SecretStr`, meaning that those won't be exposed when sharing the pipeline.

    This also means that if you want to re-run your own pipeline and the API keys have not been provided via environment variable but either via attribute or runtime parameter, you will need to provide them again.

??? faq "Does it work for Windows?"

    Yes, but you may need to set the `multiprocessing` context in advance, to ensure that the `spawn` method is used, since the default method `fork` is not available on Windows.

    ```python
    import multiprocessing as mp

    mp.set_start_method("spawn")
    ```

??? faq "Will the custom Steps / Tasks / LLMs be serialized too?"
    No, at the moment only the references to the classes within the `distilabel` library will be serialized, meaning that if you define a custom class used within the pipeline, the serialization won't break, but the deserialize will fail since the class won't be available, unless used from the same file.

??? faq "What happens if `Pipeline.run` fails? Do I lose all the data?"
    No, indeed we're using a cache mechanism to store all the intermediate results in disk, so that if a [`Step`][distilabel.steps.base.Step] fails, the pipeline can be re-run from that point without losing the data, only if nothing is changed in the `Pipeline`.

    All the data will be stored in `.cache/distilabel`, but the only data that will persist at the end of the `Pipeline.run` execution is the one from the leaf step/s, so bear that in mind.

    For more information on the caching mechanism in `distilabel`, you can check the [Learn - Advanced - Caching](./learn/advanced/caching.md) section.

    Also note that when running a [`Step`][distilabel.steps.base.Step] or a [`Task`][distilabel.steps.tasks.Task] standalone, the cache mechanism won't be used, so if you want to use that, you should use the `Pipeline` context manager.

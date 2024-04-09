# Pipeline

The [`Pipeline`][distilabel.pipeline.local.Pipeline] is the central point in `distilabel`, the way to organize the steps to create your datasets. Up to this point we've seen how we can define different [`Steps`](../steps/index.md) and [`Tasks`](../tasks/index.md), that together with an [`LLM`](.) are the building blocks of our datasets, in this section we will take a look at how all these blocks are organized inside a `Pipeline`.

!!! Note
    Currently `distilabel` implements a *local* implementation of a `Pipeline`, and will assume that's the only definition, but this can be extended in the future to include remote execution of the `Pipeline`.

## Sample pipeline

```python
WRITE HERE pipe_docs.py
```

THEN EXPLAIN STEP BY STEP

## How to create a Pipeline

The most common way a `Pipeline` should be created is by making use of the context manager, we just need to give our `Pipeline` a **name**, and optionally a **description**, and that's it[^1]:

```python
from distilabel.pipeline import Pipeline

with Pipeline("pipe-name", description="My first pipe") as pipeline:  # (1)
    ...

```

1. Create a `Task` for generating text given an instruction.

This way, we ensure all the steps we define there are connected with each other under the same `Pipeline`.

[^1]: We also have the *cache_dir* argument to pass, for more information on this parameter, we refer the reader to the [caching](../caching.md) section.

## Creating the steps of our Pipeline


TODO: WRITE THE WARNING WHEN WE NEED TO CALL THE RUN METHOD

!!! Warning
    Due to `multiprocessing`, the `pipeline.run` method **must** be run inside `__main__`:

    ```python
    with Pipeline("pipeline") as pipe:
        ...

    if __name__ == "__main__":
        distiset = pipe.run()
    ```

    Otherwise an `EOFError` will raise.

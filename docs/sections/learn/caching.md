# Caching

Distilabel `Pipelines` automatically save all the intermediate steps to to avoid loosing any data in case of error.

## Cache directory

Out of the box, the `Pipeline` will use the `~/.cache/distilabel/pipelines` directory to store the different pipelines:

```python
from distilabel.pipeline.local import Pipeline

with Pipeline("cache_testing") as pipeline:
    ...
```

This directory can be modified by setting the `DISTILABEL_CACHE_DIR` environment variable (`export DISTILABEL_CACHE_DIR=my_cache_dir`) or by explicitely passing the `cache_dir` variable to the `Pipeline` constructor like so:

```python
with Pipeline("cache_testing", cache_dir="~/my_cache_dir") as pipeline:
    ...
```

## How does it work?

Let's take a look at the logging messages from a sample pipeline.

When we run a `Pipeline` for the first time

![Initial Pipeline](/assets/images/sections/caching/caching_pipe_1.png)

If we decide to stop the pipeline (say we kill the run altogether via `ctrl+C` or `cmd+c` in *macos*), we will see the signal sent to the different workers:

![Initial Pipeline](/assets/images/sections/caching/caching_pipe_2.png)

After this step, when we run again the pipeline, the first log message we see corresponds to "Load pipeline from cache", which will restart processing from where it stopped:

![Initial Pipeline](/assets/images/sections/caching/caching_pipe_3.png)

Finally, if we decide to run the same `Pipeline` after it has finished completely, it won't start again but resume the process, as we already have all the data processed:

![Initial Pipeline](/assets/images/sections/caching/caching_pipe_4.png)

### Serialization

!!! NOTE
    WORK IN PROGRESS

- Folder layout
- When are those saved and when are they loaded back
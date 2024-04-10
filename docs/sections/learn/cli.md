# Command Line Interface

`Distilabel` offers a [`CLI`][distilabel.cli.pipeline.utils] to initially *explore* and *rerun* `Pipelines`, let's take a look.

## Available commands

We have two commands under the `CLI` app, `distilabel pipeline`:

```bash
$ distilabel pipeline --help

 Usage: distilabel pipeline [OPTIONS] COMMAND [ARGS]...

 Commands to run and inspect Distilabel pipelines.

╭─ Options ───────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────╮
│ info      Get information about a Distilabel pipeline.                                  │
│ run       Run a Distilabel pipeline.                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
```

Will run them using as an example the following [dataset](https://huggingface.co/datasets/distilabel-internal-testing/ultrafeedback-mini) for testing purposes:

### Pipeline info

The first command is `distilabel pipeline info`:

```bash
$ distilabel pipeline info --help

 Usage: distilabel pipeline info [OPTIONS]

 Get information about a Distilabel pipeline.

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ *  --config        TEXT  Path or URL to the Distilabel pipeline configuration file. │
│                          [default: None]                                            │
│                          [required]                                                 │
│    --help                Show this message and exit.                                │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

As we can see from the help message, we need to pass either a `Path` or a `URL`. This second option comes handy for datasets stored in HuggingFace hub, for example:

```bash
distilabel pipeline info --config "https://huggingface.co/datasets/distilabel-internal-testing/ultrafeedback-mini/raw/main/pipeline.yaml"
```

If we take a look (this `Pipeline` is a bit long, so the following captures are shortened for brevity):

![CLI 1](../../assets/images/sections/cli/cli_pipe_1.png)

![CLI 2](../../assets/images/sections/cli/cli_pipe_2.png)

The pipeline information includes the steps used in the `Pipeline` along with the `Runtime Parameter` that was used, as well as a description of each of them, and also the connections between these steps. These can be helpful to explore the Pipeline locally.

### Running a Pipeline

We can also run a `Pipeline` from the CLI just pointing to the same `pipeline.yaml` file or an URL pointing to it and calling `distilabel pipeline run`:

```bash
$ distilabel pipeline run --help

 Usage: distilabel pipeline run [OPTIONS]

 Run a Distilabel pipeline.

╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config                                 TEXT                 Path or URL to the Distilabel pipeline configuration file.   │
│                                                                  [default: None]                                              │
│                                                                  [required]                                                   │
│    --param                                  PARSE_RUNTIME_PARAM  [default: (dynamic)]                                         │
│    --ignore-cache      --no-ignore-cache                         Whether to ignore the cache and re-run the pipeline from     │
│                                                                  scratch.                                                     │
│                                                                  [default: no-ignore-cache]                                   │
│    --repo-id                                TEXT                 The Hugging Face Hub repository ID to push the resulting     │
│                                                                  dataset to.                                                  │
│                                                                  [default: None]                                              │
│    --commit-message                         TEXT                 The commit message to use when pushing the dataset.          │
│                                                                  [default: None]                                              │
│    --private           --no-private                              Whether to make the resulting dataset private on the Hub.    │
│                                                                  [default: no-private]                                        │
│    --token                                  TEXT                 The Hugging Face Hub API token to use when pushing the       │
│                                                                  dataset.                                                     │
│                                                                  [default: None]                                              │
│    --help                                                        Show this message and exit.                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

To specify the runtime parameters of the steps we will need to use the `--param` option and the value of the parameter in the following format:

```bash
distilabel pipeline run --config "https://huggingface.co/datasets/distilabel-internal-testing/ultrafeedback-mini/raw/main/pipeline.yaml" \
  --param load_dataset.repo_id=HuggingFaceH4/instruction-dataset \
  --param load_dataset.split=test \
  --param text_generation_with_notus.generation_kwargs.max_new_tokens=512 \
  --param text_generation_with_notus.generation_kwargs.temperature=0.7 \
  --param text_generation_with_zephyr.generation_kwargs.max_new_tokens=512 \
  --param text_generation_with_zephyr.generation_kwargs.temperature=0.7 \
  --param ultrafeedback_overall_rating.generation_kwargs.max_new_tokens=1024 \
  --param ultrafeedback_overall_rating.generation_kwargs.temperature=0.7
```

Again, this helps with the reproducibility of the results, and simplifies sharing not only the final dataset but also the process to generate it.

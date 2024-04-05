# Command Line Interface

This section contains the API reference for the command line interface.

## CLI commands

This section shows the CLI commands:

### distilabel pipeline run

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

### distilabel pipeline info

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

## Utility functions for the pipeline commands

Here are some utility functions to help working with the pipelines in the console.

::: distilabel.cli.pipeline.utils

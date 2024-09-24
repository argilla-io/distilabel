---
description: This is a step-by-step guide to help you develop distilabel.
hide:
  - footer
---

Thank you for investing your time in contributing to the project!

If you don't have the repository locally, and need any help, go to the [contributor guide](../community/contributor.md) and read the contributor workflow with Git and GitHub first.

## Set up the Python environment

To work on the `distilabel`, you must install the package on your system.

!!! Tip
    This guide will use `uv`, but `pip` and `venv` can be used as well, this guide can work quite similar with both options.

From the root of the cloned Distilabel repository, you should move to the distilabel folder in your terminal.

```bash
cd distilabel
```

### Create a virtual environment

The first step will be creating a virtual environment to keep our dependencies isolated. Here we are choosing `python 3.11` ([uv venv](https://docs.astral.sh/uv/pip/environments/) documentation), and then activate it:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

### Install the project

Installing from local (we are using [`uv pip`](https://docs.astral.sh/uv/pip/packages/)):

```bash
uv pip install -e .
```

We have extra dependencies with their name, depending on the part you are working on, you may want to install some dependency (take a look at `pyproject.toml` in the repo to see all the extra dependencies):

```bash
uv pip install -e ".[vllm,outlines]"
```

### Linting and formatting

To maintain a consistent code format, install the pre-commit hooks to run before each commit automatically (we rely heavily on [`ruff`](https://docs.astral.sh/ruff/)):

```bash
uv pip install -e ".[dev]"
pre-commit install
```

### Running tests

All the changes you add to the codebase should come with tests, either `unit` or `integration` tests, depending on the type of change, which are placed under `tests/unit` and `tests/integration` respectively.

Start by installing the tests dependencies:

```bash
uv pip install ".[tests]"
```

Running the whole tests suite may take some time, and you will need all the dependencies installed, so just run your tests, and the whole tests suite will be run for you in the CI:

```bash
# Run specific tests
pytest tests/unit/steps/generators/test_data.py
```

## Set up the documentation

To contribute to the documentation and generate it locally, ensure you have installed the development dependencies:

```bash
uv pip install -e ".[docs]"
```

And run the following command to create the development server with `mkdocs`:

```bash
mkdocs serve
```

### Documentation guidelines

As mentioned, we use mkdocs to build the documentation. You can write the documentation in `markdown` format, and it will automatically be converted to HTML. In addition, you can include elements such as tables, tabs, images, and others, as shown in this guide. We recommend following these guidelines:

- Use clear and concise language: Ensure the documentation is easy to understand for all users by using straightforward language and including meaningful examples. Images are not easy to maintain, so use them only when necessary and place them in the appropriate folder within the docs/assets/images directory.

- Verify code snippets: Double-check that all code snippets are correct and runnable.

- Review spelling and grammar: Check the spelling and grammar of the documentation.

- Update the table of contents: If you add a new page, include it in the relevant index.md or the mkdocs.yml file.

### Components gallery

The components gallery section of the documentation is automatically generated thanks to a custom plugin, it will be run when `mkdocs serve` is called. This guide to the steps helps us visualize each step, as well as examples of use.

!!! Note
    Changes done to the docstrings of `Steps/Tasks/LLMs` won't appear in the components gallery automatically, you will have to stop the `mkdocs` server and run it again to see the changes, everything else is reloaded automatically.

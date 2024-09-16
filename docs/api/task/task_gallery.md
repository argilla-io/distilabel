# Task Gallery

This section contains the existing [`Task`][distilabel.steps.tasks.Task] subclasses implemented in `distilabel`.

## Types of synthetic data generation

There are a lot of different Tasks and Steps available in Distilabel, and you can find more information about them in the [Components](../../components-gallery) section. Here is a summary of the most common categories:

| Name | Emoticon | Description |
|------|----------|-------------|
| Text Generation | :material-text-box-edit: | Text generation steps are used to generate text based on a given prompt. |
| Evol | :material-dna: | Evol steps are used to rewrite input text and evolve it to a higher quality. |
| Text Manipulation | :material-receipt-text-edit: | Text manipulation steps are used to manipulate or rewrite an input text. |
| Critique | :material-comment-edit: | Critique steps are used to provide feedback on the quality of the data with a written explanation. |
| Scorer | :octicons-number-16: | Scorer steps are used to evaluate and score the data with a numerical value. |
| Preference | :material-poll: | Preference steps are used to collect preferences on the data with numerical values or ranks. |
| Embedding | :material-vector-line: | Embedding steps are used to generate embeddings for the data. |
| Columns | :material-table-column: | Columns steps are used to manipulate columns in the data. |
| Filtering | :material-filter: | Filtering steps are used to filter the data based on some criteria. |
| Format | :material-format-list-bulleted: | Format steps are used to format the data. |
| Load | :material-file-download: | Load steps are used to load the data. |
| Save | :material-content-save: | Save steps are used to save the data. |

::: distilabel.steps.tasks
    options:
        filters:
        - "!Task"
        - "!_Task"
        - "!GeneratorTask"
        - "!ChatType"
        - "!typing"
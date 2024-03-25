---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

<p align="left">
  <a href="https://github.com/argilla-io/distilabel">
    <img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>
  </a>
</p>

# Dataset Card for {{ repo_id.split("/")[-1] }}

This dataset has been created with [Distilabel](https://distilabel.argilla.io/).

### Dataset Summary

This dataset contains a `pipeline.yaml` which can be used to reproduce the pipeline that generated it in distilabel using the CLI.

### Load with `datasets`

To load this dataset with `datasets`, you'll just need to install `datasets` as `pip install datasets --upgrade` and then use the following code:

```python
from datasets import load_dataset

ds = load_dataset("{{ repo_id }}")
```

### Languages

{{ languages_section | default("[More Information Needed]", true)}}

<!-- ## Dataset Structure

TODO: Include a sample record per configuration

### Sample record

An example of a record looks like the following:

```json
{{ huggingface_record | tojson(indent=4) }}
``` -->

### Other Known Limitations

{{ known_limitations_section | default("[More Information Needed]", true)}}

## Additional Information

### Licensing Information

{{ licensing_information_section | default("[More Information Needed]", true)}}

### Citation Information

{{ citation_information_section | default("[More Information Needed]", true)}}

### Contributions

{{ contributions_section | default("[More Information Needed]", true)}}
---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

<p align="left">
  <a href="https://github.com/argilla-io/distilabel">
    <img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>
  </a>
</p>

# Dataset Card for {{ repo_id.split("/")[-1] }}

This dataset has been created with [distilabel](https://distilabel.argilla.io/).

{% if include_script%}
The pipeline script was uploaded to easily reproduce the dataset:
[{{ filename_py }}](https://huggingface.co/datasets/{{ repo_id }}/raw/main/{{ filename_py }}).

It can be run directly using the CLI:

```console
distilabel pipeline run --script "https://huggingface.co/datasets/{{ repo_id }}/raw/main/{{ filename_py }}"
```
{% endif %}

## Dataset Summary

This dataset contains a `pipeline.yaml` which can be used to reproduce the pipeline that generated it in distilabel using the `distilabel` CLI:

```console
distilabel pipeline run --config "https://huggingface.co/datasets/{{ repo_id }}/raw/main/pipeline.yaml"
```

or explore the configuration:

```console
distilabel pipeline info --config "https://huggingface.co/datasets/{{ repo_id }}/raw/main/pipeline.yaml"
```

## Dataset structure

The examples have the following structure per configuration:

{% for config_name, record in sample_records.items() %}
<details><summary> Configuration: {{ config_name }} </summary><hr>

```json
{{ record | tojson(indent=4) }}
```

This subset can be loaded as:

```python
from datasets import load_dataset

ds = load_dataset("{{ repo_id }}", "{{ config_name }}")
```
{% if config_name == "default" %}
Or simply as it follows, since there's only one configuration and is named `default`: 

```python
from datasets import load_dataset

ds = load_dataset("{{ repo_id }}")
```
{% endif %}

</details>
{% endfor %}

{% if artifacts %}
## Artifacts

{% for step_name, artifacts in artifacts.items() %}
* **Step**: `{{ step_name }}`
  {% for artifact in artifacts %}
    * **Artifact name**: `{{ artifact.name }}`
      {% for name, value in artifact.metadata.items() %}
        * `{{ name }}`: {{ value }}
      {% endfor %}
  {% endfor %}
{% endfor %}

{% endif %}

{% if statistics %}
## Dataset Statistics

{% for leaf_name, table in statistics.items() %}
* **Summary statistics**: `{{ leaf_name }}`

{{ table }}
{% endfor %}

{% endif %}

{% if references %}
## References

{% for reference in references %}
```
{{ reference }}
```

{% endfor %}
{% endif %}

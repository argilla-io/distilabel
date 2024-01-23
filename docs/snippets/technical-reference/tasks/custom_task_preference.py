from dataclasses import dataclass
from typing import List, Dict

from distilabel.tasks import PreferenceTask
from distilabel.tasks.prompt import Prompt

input_prompt = """{{ task_description }}
{%- for rating in ratings %}
{{ rating.value }}. {{ rating.description }}
{%- endfor %}

---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts:
{% for index in range(responses|length) %}
<text {{ index + 1}}> [Text {{ index + 1}}]
{%- endfor %}

### Output
{%- for index in range(responses|length) %}

#### Output for Text {{ index + 1}}
Rating: [Rating for text {{ index + 1}}]
Rationale: [Rationale for the rating in short sentences]

{%- endfor %}

---

## Annotation

### Input
Instruction: {{ input }}

Texts:
{% for response in responses %}
<text {{ loop.index }}> {{ response }}
{%- endfor %}

### Output 
"""


@dataclass
class CustomPreference(PreferenceTask):
    system_prompt: str = "You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions."

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input_prompt.format(code=input),
        )

    def parse_output(self, output: str) -> List[Dict]:
        """Parses the output of the model into the desired format."""
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = float(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append({"rating": rating, "rationale": rationale})
        return parsed_output

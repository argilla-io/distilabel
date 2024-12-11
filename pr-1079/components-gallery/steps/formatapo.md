---
hide:
  - navigation
---
# FormatAPO

Format the output of `CLAIR` task for Anchored Preference Optimization (APO).



`FormatAPO` is a `Step` that formats the output of a `CLAIR` task for
    Anchored Preference Optimization (APO) following the standard formatting from `TRL`.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[prompt]
			ICOL1[response]
			ICOL2[revision]
		end
		subgraph New columns
			OCOL0[prompt]
			OCOL1[chosen]
			OCOL2[rejected]
			OCOL3[prompt_id]
		end
	end

	subgraph FormatAPO
		StepInput[Input Columns: prompt, response, revision]
		StepOutput[Output Columns: prompt, chosen, rejected, prompt_id]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepInput --> StepOutput

```


#### Inputs


- **prompt** (`str`): The instruction used to generate the `generation` with the `LLM`.

- **response** (`str`): The generation produced by the `LLM`.

- **revision** (`str`): The revised text.




#### Outputs


- **prompt** (`str`): The instruction used to generate the `generation` with the `LLM`.

- **chosen** (`List[Dict[str, str]]`): The `chosen` generation based on the `ratings`.

- **rejected** (`List[Dict[str, str]]`): The `rejected` generation based on the `ratings`.

- **prompt_id** (`str`): The `SHA256` hash of the `prompt`.





### Examples


#### Format your dataset for APO fine tuning
```python
from distilabel.steps import FormatAPO

formatter = FormatAPO()
formatter.load()

result = next(
    formatter.process(
        [
            {
                "prompt": "How many gaps are there between the earth and the moon?",
                "response": '''There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."

The Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.

So, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.''',
                "revision": '''There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."

The Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon's orbital path, not the presence of any gaps.

In summary, the Moon's orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.''',
            }
        ]
    )
)
# >>> result
# [{'prompt': 'How many gaps are there between the earth and the moon?',
# 'response': 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."

The Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.

So, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.',
# 'revision': 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."

The Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon's orbital path, not the presence of any gaps.

In summary, the Moon's orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
# 'prompt_id': 'd5e8924f2856fe7180c0aef3ec186f7a421b2ba11551b9ebfffeb7638ec5b021',
# 'chosen': [{'role': 'user',
#     'content': 'How many gaps are there between the earth and the moon?'},
# {'role': 'assistant',
#     'content': 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."

The Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon's orbital path, not the presence of any gaps.

In summary, the Moon's orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.'}],
# 'rejected': [{'role': 'user',
#     'content': 'How many gaps are there between the earth and the moon?'},
# {'role': 'assistant',
#     'content': 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."

The Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.

So, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.'}]}]
```





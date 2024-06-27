# Add requirements to run a Pipeline

When sharing a `Pipeline` that contains custom `Step`s or `Task`s, you may want to add the specific requirements that are needed to run them. `distilabel` will take this list of requirements and warn the user if any are missing.

You can now add requirements at the step level as well as on the pipeline. The following is a sample pipeline that adds requirements `nltk` as a dependency to a sample `CustomStep`, and we also inform the pipeline must run using `distilabel>=1.3.0`:

```python
from typing import List

from distilabel.steps import Step
from distilabel.steps.base import StepInput
from distilabel.steps.typing import StepOutput
from distilabel.steps import LoadDataFromDicts
from distilabel.utils.requirements import requirements
from distilabel.pipeline import Pipeline


@requirements(["nltk"])
class CustomStep(Step):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    @property
    def outputs(self) -> List[str]:
        return ["response"]

    def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
        for input in inputs:
            input["response"] = nltk.word_tokenize(input)
        yield inputs


with Pipeline(
    name="pipeline-with-requirements", requirements=["distilabel>=1.3.0"]
) as pipeline:
    loader = LoadDataFromDicts(data=[{"instruction": "sample sentence"}])
    step1 = CustomStep()
    loader >> step1

pipeline.run()
```

Once we call `pipeline.run()`, if any of the requirements informed at the `Step` or `Pipeline` level isn't met, a `ValueError` will be raised telling us that we should install the list of dependencies:

```python
>>> pipeline.run()
[06/27/24 11:07:33] ERROR    ['distilabel.pipeline'] Please install the following requirements to run the pipeline:                                                                                                                                     base.py:350
                             distilabel>=1.3.0
...
ValueError: Please install the following requirements to run the pipeline:
distilabel>=1.3.0
```

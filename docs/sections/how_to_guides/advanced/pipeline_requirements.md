# Add requirements to run a Pipeline

When sharing a `Pipeline` that contains custom `Step`s or `Task`s, you may want to add the specific requirements that are needed to run them. `distilabel` will take this list of requirements and warn the user if any are missing.

Let's see how we can add additional requirements with an example. The first thing we're going to do is to add requirements for our `CustomStep`. To do so we use the `requirements` decorator to specify that the step has `nltk>=3.8` as dependency (we can use [version specifiers](https://peps.python.org/pep-0440/#version-specifiers)). In addition, we're going to specify at `Pipeline` level that we need `distilabel>=1.3.0` to run it.

```python
from typing import List

from distilabel.steps import Step
from distilabel.steps.base import StepInput
from distilabel.typing import StepOutput
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

if __name__ == "__main__":
    pipeline.run()
```

Once we call `pipeline.run()`, if any of the requirements informed at the `Step` or `Pipeline` level is missing, a `ValueError` will be raised telling us that we should install the list of dependencies:

```python
>>> pipeline.run()
[06/27/24 11:07:33] ERROR    ['distilabel.pipeline'] Please install the following requirements to run the pipeline:                                                                                                                                     base.py:350
                             distilabel>=1.3.0
...
ValueError: Please install the following requirements to run the pipeline:
distilabel>=1.3.0
```

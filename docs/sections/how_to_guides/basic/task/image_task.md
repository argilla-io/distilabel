# ImageTask to work with Image Generation Models

## Working with ImageTasks

The [`ImageTask`][distilabel.steps.tasks.ImageTask] is a custom implementation of a [`Task`][distilabel.steps.tasks.Task] special to deal images. These tasks behave exactly as any other [`Task`][distilabel.steps.tasks.Task], but instead of relying on an [`LLM`][distilabel.models.llms.LLM], they work with a [`ImageGenerationModel`][distilabel.models.image_generation.ImageGenerationModel].

!!! info "New in version 1.5.0"
    This task is new and is expected to work with Image Generation Models.

These tasks take as attribute an `image_generation_model` instead of `llm` as we would have with the standard `Task`, but everything else remains the same. Let's see an example with [`ImageGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/imagegeneration/):

```python
from distilabel.steps.tasks import ImageGeneration
from distilabel.models.image_generation import InferenceEndpointsImageGeneration

task = ImageGeneration(
    name="image-generation",
    image_generation_model=InferenceEndpointsImageGeneration(model_id="black-forest-labs/FLUX.1-schnell"),
)
task.load()

next(task.process([{"prompt": "a white siamese cat"}]))
# [{'image": "iVBORw0KGgoAAAANSUhEUgA...", "model_name": "black-forest-labs/FLUX.1-schnell"}]
```

!!! info "Visualize the image in a notebook"
    If you are testing the `ImageGeneration` task in a notebook, you can do the following
    to see the rendered image:

    ```python
    from distilabel.models.image_generation.utils import image_from_str

    result = next(task.process([{"prompt": "a white siamese cat"}]))
    image_from_str(result[0]["image"])  # Returns a `PIL.Image.Image` that renders directly
    ```

!!! tip "Running ImageGeneration in a Pipeline"
    This transformation between image as string and as PIL object can be done for the whole dataset if running a pipeline, by calling the method `transform_columns_to_image` on the final distiset and passing the name (or list of names) of the column image.

## Defining custom ImageTasks

We can define a custom generator task by creating a new subclass of the [`ImageTask`][distilabel.steps.tasks.ImageTask] and defining the following:

- `process`: is a method that generates the data based on the [`ImageGenerationModel`][distilabel.models.image_generation.ImageGenerationModel] and the `prompt` provided within the class instance, and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`.

- `inputs`: is a property that returns a list of strings with the names of the required input fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `outputs`: is a property that returns a list of strings with the names of the output fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not. This property should always include `model_name` as one of the outputs since that's automatically injected from the LLM.

- `format_input`: is a method that receives a dictionary with the input data and returns a *prompt* to be passed to the model.

- `format_output`: is a method that receives the output from the [`ImageGenerationModel`][distilabel.models.image_generation.ImageGenerationModel] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`.

```python
from typing import TYPE_CHECKING

from distilabel.models.image_generation.utils import image_from_str, image_to_str
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import ImageTask

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class MyCustomImageTask(ImageTask):
    @override
    def process(self, offset: int = 0) -> GeneratorOutput:
        formatted_inputs = self._format_inputs(inputs)

        outputs = self.llm.generate_outputs(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.llm.get_generation_kwargs(),
        )

        task_outputs = []
        for input, input_outputs in zip(inputs, outputs):
            formatted_outputs = self._format_outputs(input_outputs, input)
            for formatted_output in formatted_outputs:
                task_outputs.append(
                    {**input, **formatted_output, "model_name": self.llm.model_name}
                )
        yield task_outputs

    @property
    def inputs(self) -> "StepColumns":
        return ["prompt"]

    @property
    def outputs(self) -> "StepColumns":
        return ["image", "model_name"]

    def format_input(self, input: dict[str, any]) -> str:
        return input["prompt"]

    def format_output(
        self, output: Union[str, None], input: dict[str, any]
    ) -> Dict[str, Any]:
        # Extract/generate/modify the image from the output
        return {"image": ..., "model_name": self.llm.model_name}
```

!!! Warning
    Note the fact that in the `process` method we are not dealing with the `image_generation` attribute but with the `llm`. This is not a bug, but intended, as internally we rename the `image_generation` to `llm` to reuse the code. 

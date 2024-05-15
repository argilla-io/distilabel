# Task

The [`Task`][distilabel.steps.tasks.Task] is an implementation on top of [`Step`][distilabel.steps.Step] that includes the [`LLM`][distilabel.llms.LLM] as a mandatory argument, so that the [`Task`][distilabel.steps.tasks.Task] defines both the input and output format via the `format_input` and `format_output` abstract methods, respectively; and calls the [`LLM`][distilabel.llms.LLM] to generate the text. We can see the [`Task`][distilabel.steps.tasks.Task] as an [`LLM`][distilabel.llms.LLM] powered [`Step`][distilabel.steps.Step].

## Working with Tasks

The subclasses of [`Task`][distilabel.steps.tasks.Task] are intended to be used within the scope of a [`Pipeline`][distilabel.pipeline.Pipeline], which will orchestrate the different tasks defined; but nonetheless, they can be used standalone if needed too.

For example, the most basic task is the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task, which generates text based on a given instruction, and it can be used standalone as well as within a [`Pipeline`][distilabel.pipeline.Pipeline].

```python
from distilabel.steps.tasks import TextGeneration

task = TextGeneration(
    name="text-generation",
    llm=OpenAILLM(model="gpt-4"),
)
task.load()

next(task.process([{"instruction": "What's the capital of Spain?"}]))
# [{'instruction': "What's the capital of Spain?", "generation": "The capital of Spain is Madrid.", "model_name": "gpt-4"}]
```

!!! NOTE
    The `load` method needs to be called ALWAYS if using the tasks as standalone, otherwise, if the [`Pipeline`][distilabel.pipeline.Pipeline] context manager is used, there's no need to call that method, since it will be automatically called on `Pipeline.run`; but in any other case the method `load` needs to be called from the parent class e.g. a [`Task`][distilabel.steps.tasks.Task] with an [`LLM`][distilabel.llms.LLM] will need to call `Task.load` to load both the task and the LLM.

## Defining custom Tasks

In order to define custom tasks, we need to inherit from the [`Task`][distilabel.steps.tasks.Task] class and implement the `format_input` and `format_output` methods, as well as setting the properties `inputs` and `outputs`, as for [`Step`][distilabel.steps.Step] subclasses.

So on, the following will need to be defined:

- `inputs`: is a property that returns a list of strings with the names of the required input fields.

- `format_input`: is a method that receives a dictionary with the input data and returns a [`ChatType`][distilabel.steps.tasks.ChatType], which is basically a list of dictionaries with the input data formatted for the [`LLM`][distilabel.llms.LLM] following [the chat-completion OpenAI formatting](https://platform.openai.com/docs/guides/text-generation). It's important to note that the [`ChatType`][distilabel.steps.tasks.ChatType] is a list of dictionaries, where each dictionary represents a turn in the conversation, and it must contain the keys `role` and `content`, and this is done like this since the [`LLM`][distilabel.llms.LLM] subclasses will format that according to the LLM used, since it's the most standard formatting.

- `outputs`: is a property that returns a list of strings with the names of the output fields. Note that since all the [`Task`][distilabel.steps.tasks.Task] subclasses are designed to work with a single [`LLM`][distilabel.llms.LLM], this property should always include `model_name` as one of the outputs, since that's automatically injected from the LLM.

- `format_output`: is a method that receives the output from the [`LLM`][distilabel.llms.LLM] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that there's no need to include the `model_name` in the output, since that's automatically injected from the LLM in the `process` method of the [`Task`][distilabel.steps.tasks.Task].

Once those methods have been implemented, the task can be used as any other task, and it will be able to generate text based on the input data.

```python
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType


class MyCustomTask(Task):
    @property
    def inputs(self) -> List[str]:
        return ["input_field"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": input["input_field"],
            },
        ]

    @property
    def outputs(self) -> List[str]:
        return ["output_field", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```

## Available Tasks

Here's a list with the available tasks that can be used within the `distilabel` library:

### [`ChatGeneration`][distilabel.steps.tasks.ChatGeneration]

Generates the follow up assistant message in the `generation` column based on the provided `messages`.

### [`ComplexityScorer`][distilabel.steps.tasks.ComplexityScorer]

Ranks the provided `instructions` based on their complexity generating the `scores` column with each instruction score, based on the paper "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning".

### [`EvolInstruct`][distilabel.steps.tasks.EvolInstruct]

Evolves the provided `instruction` to make it more complex, according to a set of pre-defined mutation templates following an evolutionary approach, based on the paper "WizardLM: Empowering Large Language Models to Follow Complex Instructions".

### [`EvolComplexity`][distilabel.steps.tasks.EvolComplexity]

Evolves the provided `instruction` to make it more complex, according to a set of pre-defined mutation templates (based on [`EvolInstruct`][distilabel.steps.tasks.EvolInstruct]).

### [`EvolQuality`][distilabel.steps.tasks.EvolQuality]

Evolves the provided `response` based on the `instruction` to make it of higher quality, ensuring that the evolved `response` is still compliant with the provided `instruction` (based on the same approach as [`EvolInstruct`][distilabel.steps.tasks.EvolInstruct]).

### [`GenerateEmbeddings`][distilabel.steps.tasks.GenerateEmbeddings]

Generates embeddings for the provided `text` using the [`LLM`][distilabel.llms.LLM] provided (which should have the `get_last_hidden_state` method implemented) in the `embedding` column.

### [`Genstruct`][distilabel.steps.tasks.Genstruct]

Generates a `user` and `assistant` single-turn conversation based on the provided `title` and `content`, so as to synthetically generate a conversation between an user and the assistant; based on the model "Genstruct 7B" by Nous Research, which at the same time is based on the paper "Ada-Instruct: Adapting Instruction Generators for Complex Reasoning".

### [`InstructionBacktranslation`][distilabel.steps.tasks.InstructionBacktranslation]

Generates a `score` and a `reason` backing the generated `score`, for a given `generation` based on an `instruction`, based on the paper "Self Alignment with Instruction Backtranslation", specifically in the self-curation stage.

### [`PairRM`][distilabel.steps.tasks.PairRM]

Rewards a set of `candidates` i.e. generations, for a given `input` i.e. instruction, and then ranks those (assuming the greater the better) generating the `ranks`, based on the paper "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion", and powered by their custom [PairRM](https://huggingface.co/llm-blender/PairRM) model and framework: [LLM-Blender](https://github.com/yuchenlin/LLM-Blender).

### [`PrometheusEval`][distilabel.steps.tasks.PrometheusEval]

Evaluates either the provided `generation` if `mode=absolute` or the `generations` if `mode=relative` based on an `instruction` and optionally also based on a `reference` i.e. golden answer, using the Prometheus 2.0 prompts and pre-defined rubrics, based on the paper "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models".

### [`QualityScorer`][distilabel.steps.tasks.QualityScorer]

Scores the provided `generations` based on the provided `instruction` generating the `scores` column with each generation score, based on the paper "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning".

### [`SelfInstruct`][distilabel.steps.tasks.SelfInstruct]

Generates `instructions` based on a seed `input`, which is conditioned by the `criteria_for_query_generation` and `application_description` args provided, based on the paper "Self-Instruct: Aligning Language Models with Self-Generated Instructions".

### [`TextGeneration`][distilabel.steps.tasks.TextGeneration]

Generates a `generation` based on the provided `instruction`, alternatively, also the `system_prompt` may be used to generate the `generation` if `use_system_prompt=True`.

### [`UltraFeedback`][distilabel.steps.tasks.UltraFeedback]

Generates a `rating` and a `rationale` for each of the `generations` based on the provided `instruction`, based on the paper "UltraFeedback: Boosting Language Models with High-quality Feedback".

---

Additionally, there is another subclass of [`Task`][distilabel.steps.tasks.Task] that is based on the [`GeneratorStep`][distilabel.steps.GeneratorStep] instead of the standard [`Step`][distilabel.steps.Step] as in this case; which is the [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask]. More information about it at [Components -> Task -> GeneratorTask](/components/task/generator-task).

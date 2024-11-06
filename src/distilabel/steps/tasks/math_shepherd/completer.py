# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Union

from jinja2 import Template
from pydantic import PositiveInt

from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.math_shepherd.utils import split_solution_steps
from distilabel.utils.itertools import batched

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns, StepOutput


SYSTEM_PROMPT = """\
You are a math teacher who helps students by breaking down word problems into clear, logical steps.
When given a problem statement and any number of initial step, generate the remaining steps needed to reach the final answer.
Each step should:

- Build logically on previous steps
- Explain the reasoning in natural language
- Lead to the final answer
- Multiple solution paths are acceptable
- Steps should be concise but clear
- Each calculation should be shown explicitly
- The final answer must be clearly stated
- The number of steps may vary based on the solution approach

# Format requirements:
- Each step should be numbered sequentially, continuing from the last given step
- The final step should clearly state "The answer is: [result]"
- Each step can use different approaches but must be mathematically valid

{{ extra_rules }}{{ few_shots }}{{ errors }}"""

RULES_GSM8K: Final[str] = """\
# Rules:
- Calculations should be embedded within the explanatory text using double angle brackets: <<calculation>>
- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction
- Write the full calculation and result, e.g., <<5*10=50>>50
"""

FEW_SHOTS_GSM8K: Final[str] = """
# Examples:
## Input
Krystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?
Step 1: On Friday, Krystian borrows 40 * 0.4 = <<40*0.4=16>>16 more books than on a regular day.

## Output 1
Step 2: On Friday, Krystian borrows 40 + 16 = <<40+16=56>>56 books in total.
Step 3: For the other 4 days (Monday to Thursday), he borrows 40 * 4 = <<40*4=160>>160 books.
Step 4: The total books for the week is 160 + 56 = <<160+56=216>>216. The answer is: 216

## Output 2
Step 2: In total, he borrows 40 + 16 = <<40+16=56>>56 books on Friday.
Step 3: For the whole week (4 regular days plus Friday), the total is (40 * 4) + 56 = <<(40*4)+56=216>>216. The answer is: 216

## Output 3
Step 2: On Friday, he borrows 40 + 40/100 * 40 = <<40+40/100*40=56>>56 books.
Step 3: In a week, he borrows 5.7 * 7 = <<5.7*7=40>>40 books. The answer is: 40"""


TEMPLATE = """{{ instruction }}
Generate {{ M }} example solutions to the same problem, separated by a single `---` and nothing else"""


# Type aliases
StepSolution = List[str]
Completions = List[StepSolution]


class MathShepherdCompleter(Task):
    """Math Shepherd Completer and auto-labeller task.

    This task is in charge of, given a list of solutions to an instruction, and a golden solution,
    as reference, generate completions for the solutions, and label them according to the golden
    solution using the hard estimation method from figure 2 in the reference paper, Eq. 3.
    The attributes make the task flexible to be used with different types of dataset and LLMs, and
    allow making use of different fields to modify the system and user prompts for it. Before modifying
    them, review the current defaults to ensure the completions are generated correctly.

    Attributes:
        system_prompt: The system prompt to be used in the completions. The default one has been
            checked and generates good completions using Llama 3.1 with 8B and 70B,
            but it can be modified to adapt it to the model and dataset selected.
        extra_rules: This field can be used to inser extra rules relevant to the type of dataset.
            For example, in the original paper they used GSM8K and MATH datasets, and this field
            can be used to insert the rules for the GSM8K dataset.
        few_shots: Few shots to help the model generating the completions, write them in the
            format of the type of solutions wanted for your dataset.
        N: Number of completions to generate for each step, correspond to N in the paper.
            They used 8 in the paper, but it can be adjusted.
        tags: List of tags to be used in the completions, the default ones are ["+", "-"] as in the
            paper, where the first is used as a positive label, and the second as a negative one.
            This can be updated, but it MUST be a list with 2 elements, where the first is the
            positive one, and the second the negative one.

    Input columns:
        - instruction (`str`): The task or instruction.
        - solutions (`str`): JSON formatted list of solutions to the task.
        - golden_solution (`str`): The reference solution to the task, will be used
            to annotate the candidate solutions.

    Output columns:
        - solutions (`str`): The same columns that were used as input, the "solutions" is modified.
        - model_name (`str`): The name of the model used to generate the revision.

    Categories:
        - text-generation
        - labelling  # TODO: Add it to components_gallery.py

    References:
        - [`Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations`](https://arxiv.org/abs/2312.08935)

    Examples:
        Annotate your steps with the Math Shepherd Completer:

        ```python
        from distilabel.steps.tasks import MathShepherdCompleter
        from distilabel.models import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            generation_kwargs={
                "temperature": 0.6,
                "max_new_tokens": 1024,
            },
        )
        task = MathShepherdCompleter(
            llm=llm,
            N=3
        )

        task.load()

        result = next(
            task.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                        "golden_solution": json.dumps(["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.", "The answer is: 18"]),
                        "solutions": json.dumps(
                            [
                                ["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.", "The answer is: 18"],
                                ['Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking.', 'Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day.', 'Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.', 'The answer is: 18'],
                            ]
                        )
                    },
                ]
            )
        )
        # [[{'instruction': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        # 'golden_solution': '["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\\u2019s market.", "The answer is: 18"]',
        # 'solutions': '[["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. -", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\\u2019s market.", "The answer is: 18"], ["Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +", "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +", "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.", "The answer is: 18"]]'}]]
        ```

    Citations:

        ```
        @misc{wang2024mathshepherdverifyreinforcellms,
            title={Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations},
            author={Peiyi Wang and Lei Li and Zhihong Shao and R. X. Xu and Damai Dai and Yifei Li and Deli Chen and Y. Wu and Zhifang Sui},
            year={2024},
            eprint={2312.08935},
            archivePrefix={arXiv},
            primaryClass={cs.AI},
            url={https://arxiv.org/abs/2312.08935},
        }
        ```
    """

    system_prompt: Optional[str] = SYSTEM_PROMPT
    extra_rules: Optional[str] = RULES_GSM8K
    few_shots: Optional[str] = FEW_SHOTS_GSM8K
    N: PositiveInt = 1
    tags: list[str] = ["+", "-"]

    def load(self) -> None:
        super().load()

        if self.system_prompt is not None:
            self.system_prompt = Template(self.system_prompt).render(
                extra_rules=self.extra_rules or "",
                few_shots=self.few_shots or "",
            )
        self._template = Template(TEMPLATE)

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction", "solutions", "golden_solution"]

    @property
    def outputs(self) -> "StepColumns":
        return ["completions", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        messages = [
            {
                "role": "user",
                "content": self._template.render(
                    instruction=input["instruction"], N=self.N
                ),
            }
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def _parse_output(self, output: Union[str, None]) -> List[Union[str, None]]:
        """Parses the output from the LLM, and returns a list of completions.

        It does some extra checking in the case of N>1, to ensure that the completions
        are in the expected number, adding "" if there are less completions than expected,
        or shortening the list if there are more completions than expected.

        Args:
            output: The output from the LLM.

        Returns:
            List of examples
        """
        if output is None:
            return [None]

        if self.N > 1:
            examples = [split_solution_steps(o) for o in output.split("---")]
            # In case there aren't the expected number of completions, we fill it with "", or short the list.
            # This shoulnd't happen if the LLM works as expected, but it's a safety measure as it can be
            # difficult to debug if the completions don't match the solutions.
            if len(examples) < self.N:
                examples.extend([""] * (self.N - len(examples)))
            elif len(examples) > self.N:
                examples = examples[: self.N]
        else:
            examples = [split_solution_steps(output)]
        return examples

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        """Does nothing."""
        return {}

    def process(self, inputs: StepInput) -> "StepOutput":
        """Does the processing of generation completions for the solutions, and annotate
        each step with the logic found in Figure 2 of the paper, with the hard estimation (Eq. (3)).

        Args:
            inputs: Inputs to the step

        Yields:
            Annotated inputs with the completions.
        """

        # A list with all the inputs to be passed to the LLM. Needs another structure to
        # find them afterwards
        prepared_inputs = []
        # Data structure with the indices of the elements.
        # (i, j, k) where i is the input, j is the solution, and k is the completion
        input_positions = []
        golden_answers = []
        for i, input in enumerate(inputs):
            instruction = input["instruction"]
            # This is a single solution
            golden_solution = json.loads(input["golden_solution"])
            golden_answers.append(golden_solution[-1])
            # This contains a list of solutions  (that should later be flattened?)
            solutions = json.loads(input["solutions"])
            for j, solution in enumerate(solutions):
                # For each solution, that has K steps, we have to generate N completions
                # for the first K-2 steps (-2 because the last 2 steps are the last step, and
                # the answer itself, which can be directly compared against golden answer)
                prepared_completions = self._prepare_completions(instruction, solution)
                prepared_inputs.extend(prepared_completions)
                input_positions.extend(
                    [(i, j, k) for k in range(len(prepared_completions))]
                )

        # Send the elements in batches to the LLM to speed up the process
        final_outputs = []
        for i, inner_batch in enumerate(
            batched(prepared_inputs, self.input_batch_size)
        ):
            self._logger.info(f"📦 Processing internal batch of completions {i}...")
            outputs = self.llm.generate_outputs(
                inputs=inner_batch,
                num_generations=1,
                **self.llm.get_generation_kwargs(),  # type: ignore
            )
            formatted_outputs = [self._parse_output(output[0]) for output in outputs]

            final_outputs.extend(formatted_outputs)

        yield self._auto_label(inputs, final_outputs, input_positions, golden_answers)

    def _prepare_completions(
        self, instruction: str, steps: List[str]
    ) -> List["ChatType"]:
        """Helper method to create, given a solution (a list of steps), and a instruction, the
        texts to be completed by the LLM.

        Args:
            instruction: Instruction of the problem.
            steps: List of steps that are part of the solution.

        Returns:
            List of ChatType, where each ChatType is the prompt corresponding to one of the steps
            to be completed.
        """
        prepared_inputs = []
        # Use the number of completions that correspond to a given instruction/steps pair
        # to find afterwards the input that corresponds to a given completion (to do the labelling)
        num_completions = len(steps[:-2])
        for i in range(1, num_completions + 1):
            to_complete = instruction + " " + "\n".join(steps[:i])
            prepared_inputs.append(self.format_input({"instruction": to_complete}))

        return prepared_inputs

    def _auto_label(
        self,
        inputs: StepInput,
        final_outputs: list[Completions],
        input_positions: list[tuple[int, int, int]],
        golden_answers: list[str],
    ) -> StepInput:
        """Labels the steps inplace (in the inputs), and returns the inputs.

        Args:
            inputs: The original inputs
            final_outputs: List of generations from the LLM.
                It's organized as a list where the elements sent to the LLM are
                grouped together, then each element contains the completions, and
                each completion is a list of steps.
            input_positions: A list with tuples generated in the process method
                that contains (i, j, k) where i is the index of the input, j is the
                index of the solution, and k is the index of the completion.
            golden_answers: List of golden answers for each input.

        Returns:
            Inputs annotated.
        """
        for i, (instruction_i, solution_i, step_i) in enumerate(input_positions):
            input = inputs[instruction_i]
            solutions = json.loads(input["solutions"])

            n_completions = final_outputs[i]

            label = f" {self.tags[1]}"
            for completion in n_completions:
                if completion[-1] == golden_answers[instruction_i]:
                    label = f" { self.tags[0]}"
                    # If we found one, it's enough as we are doing Hard Estimation
                    continue

            solutions[solution_i][step_i] += label
            inputs[instruction_i]["solutions"] = json.dumps(solutions)

        for i, input in enumerate(inputs):
            solutions = json.loads(input["solutions"])
            new_solutions = []
            for solution in solutions:
                answer = solution.pop()
                label = (
                    f" {self.tags[0]}"
                    if answer == golden_answers[i]
                    else f" {self.tags[1]}"
                )
                solution[-1] += " " + answer + label
                new_solutions.append(solution)
            input["solutions"] = json.dumps(new_solutions)
            input["model_name"] = self.llm.model_name

        return inputs

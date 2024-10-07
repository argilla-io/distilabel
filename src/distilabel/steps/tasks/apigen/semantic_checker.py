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

import importlib.resources as importlib_resources
from typing import TYPE_CHECKING, Any, Dict, Final, Union

import orjson
from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.apigen.utils import remove_fences
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT_SEMANTIC_CHECKER: Final[str] = """\
As a data quality evaluator, you must assess the alignment between a user query, corresponding function calls, and their execution results.
These function calls and results are generated by other models, and your task is to ensure these results accurately reflect the user’s intentions.

Do not pass if:
1. The function call does not align with the query’s objective, or the input arguments appear incorrect.
2. The function call and arguments are not properly chosen from the available functions.
3. The number of function calls does not correspond to the user’s intentions.
4. The execution results are irrelevant and do not match the function’s purpose.
5. The execution results contain errors or reflect that the function calls were not executed successfully.
""".rstrip()


class APIGenSemanticChecker(Task):
    r"""Generate queries and answers for the given functions in JSON format.

    The `APIGenGenerator` is inspired by the APIGen pipeline, which was designed to generate
    verifiable and diverse function-calling datasets. The task generates a set of diverse queries
    and corresponding answers for the given functions in JSON format.

    Attributes:
        system_prompt: System prompt for the task. Has a default one.
        exclude_failed_execution: Whether to exclude failed executions (won't run on those
            rows that have a False in `keep_row_after_execution_check` column, which
            comes from running `APIGenExecutionChecker`). Defaults to True.

    Input columns:
        - func_desc (`str`): Description of what the function should do.
        - query (`str`): Instruction from the user.
        - answers (`str`): JSON encoded list with arguments to be passed to the function/API.
            Should be loaded using `json.loads`.
        - execution_result (`str`): Result of the function/API executed.

    Output columns:
        - thought (`str`): Reasoning for the output on whether to keep this output or not.
        - keep_row_after_semantic_check (`bool`): True or False, can be used to filter
            afterwards.

    Categories:
        - filtering
        - text-generation

    References:
        - [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)
        - [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

    Examples:

        Semantic checker for generated function calls (original implementation):

        ```python
        from distilabel.steps.tasks import APIGenSemanticChecker
        from distilabel.llms import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
        )
        semantic_checker = APIGenSemanticChecker(
            use_default_structured_output=False,
            llm=llm
        )
        semantic_checker.load()

        res = next(
            semantic_checker.process(
                [
                    {
                        "func_desc": "Fetch information about a specific cat breed from the Cat Breeds API.",
                        "query": "What information can be obtained about the Maine Coon cat breed?",
                        "answers": json.dumps([{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]),
                        "execution_result": "The Maine Coon is a big and hairy breed of cat",
                    }
                ]
            )
        )
        res
        # [{'func_desc': 'Fetch information about a specific cat breed from the Cat Breeds API.',
        # 'query': 'What information can be obtained about the Maine Coon cat breed?',
        # 'answers': [{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}],
        # 'execution_result': 'The Maine Coon is a big and hairy breed of cat',
        # 'thought': '',
        # 'keep_row_after_semantic_check': True,
        # 'raw_input_a_p_i_gen_semantic_checker_0': [{'role': 'system',
        #     'content': 'As a data quality evaluator, you must assess the alignment between a user query, corresponding function calls, and their execution results.\nThese function calls and results are generated by other models, and your task is to ensure these results accurately reflect the user’s intentions.\n\nDo not pass if:\n1. The function call does not align with the query’s objective, or the input arguments appear incorrect.\n2. The function call and arguments are not properly chosen from the available functions.\n3. The number of function calls does not correspond to the user’s intentions.\n4. The execution results are irrelevant and do not match the function’s purpose.\n5. The execution results contain errors or reflect that the function calls were not executed successfully.\n'},
        #     {'role': 'user',
        #     'content': 'Given Information:\n- All Available Functions:\nFetch information about a specific cat breed from the Cat Breeds API.\n- User Query: What information can be obtained about the Maine Coon cat breed?\n- Generated Function Calls: [{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]\n- Execution Results: The Maine Coon is a big and hairy breed of cat\n\nNote: The query may have multiple intentions. Functions may be placeholders, and execution results may be truncated due to length, which is acceptable and should not cause a failure.\n\nThe main decision factor is wheather the function calls accurately reflect the query\'s intentions and the function descriptions.\nProvide your reasoning in the thought section and decide if the data passes (answer yes or no).\nIf not passing, concisely explain your reasons in the thought section; otherwise, leave this section blank.\n\nYour response MUST strictly adhere to the following JSON format, and NO other text MUST be included.\n```\n{\n   "thought": "Concisely describe your reasoning here",\n   "pass": "yes" or "no"\n}\n```\n'}]},
        # 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```

        Semantic checker for generated function calls (structured output):

        ```python
        from distilabel.steps.tasks import APIGenSemanticChecker
        from distilabel.llms import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
        )
        semantic_checker = APIGenSemanticChecker(
            use_default_structured_output=True,
            llm=llm
        )
        semantic_checker.load()

        res = next(
            semantic_checker.process(
                [
                    {
                        "func_desc": "Fetch information about a specific cat breed from the Cat Breeds API.",
                        "query": "What information can be obtained about the Maine Coon cat breed?",
                        "answers": json.dumps([{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]),
                        "execution_result": "The Maine Coon is a big and hairy breed of cat",
                    }
                ]
            )
        )
        res
        # [{'func_desc': 'Fetch information about a specific cat breed from the Cat Breeds API.',
        # 'query': 'What information can be obtained about the Maine Coon cat breed?',
        # 'answers': [{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}],
        # 'execution_result': 'The Maine Coon is a big and hairy breed of cat',
        # 'keep_row_after_semantic_check': True,
        # 'thought': '',
        # 'raw_input_a_p_i_gen_semantic_checker_0': [{'role': 'system',
        #     'content': 'As a data quality evaluator, you must assess the alignment between a user query, corresponding function calls, and their execution results.\nThese function calls and results are generated by other models, and your task is to ensure these results accurately reflect the user’s intentions.\n\nDo not pass if:\n1. The function call does not align with the query’s objective, or the input arguments appear incorrect.\n2. The function call and arguments are not properly chosen from the available functions.\n3. The number of function calls does not correspond to the user’s intentions.\n4. The execution results are irrelevant and do not match the function’s purpose.\n5. The execution results contain errors or reflect that the function calls were not executed successfully.\n'},
        #     {'role': 'user',
        #     'content': 'Given Information:\n- All Available Functions:\nFetch information about a specific cat breed from the Cat Breeds API.\n- User Query: What information can be obtained about the Maine Coon cat breed?\n- Generated Function Calls: [{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]\n- Execution Results: The Maine Coon is a big and hairy breed of cat\n\nNote: The query may have multiple intentions. Functions may be placeholders, and execution results may be truncated due to length, which is acceptable and should not cause a failure.\n\nThe main decision factor is wheather the function calls accurately reflect the query\'s intentions and the function descriptions.\nProvide your reasoning in the thought section and decide if the data passes (answer yes or no).\nIf not passing, concisely explain your reasons in the thought section; otherwise, leave this section blank.\n'}]},
        # 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```
    """

    system_prompt: str = SYSTEM_PROMPT_SEMANTIC_CHECKER
    use_default_structured_output: bool = False

    _format_inst: Union[str, None] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the template for the generator prompt."""
        super().load()
        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "apigen"
            / "semantic_checker.jinja2"
        )

        self._template = Template(open(_path).read())
        self._format_inst = self._set_format_inst()

    def _set_format_inst(self) -> str:
        """Prepares the function to generate the formatted instructions for the prompt.

        If the default structured output is used, returns an empty string because nothing
        else is needed, otherwise, returns the original addition to the prompt to guide the model
        to generate a formatted JSON.
        """
        return (
            "\nYour response MUST strictly adhere to the following JSON format, and NO other text MUST be included.\n"
            "```\n"
            "{\n"
            '   "thought": "Concisely describe your reasoning here",\n'
            '   "passes": "yes" or "no"\n'
            "}\n"
            "```\n"
        )

    @property
    def inputs(self) -> "StepColumns":
        """The inputs for the task."""
        return {
            "func_desc": True,
            "query": True,
            "answers": True,
            "execution_result": True,
            "keep_row_after_execution_check": True,
        }

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType`."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._template.render(
                    func_desc=input["func_desc"],
                    query=input["query"] or "",
                    func_call=input["answers"] or "",
                    execution_result=input["execution_result"],
                    format_inst=self._format_inst,
                ),
            },
        ]

    @property
    def outputs(self) -> "StepColumns":
        """The output for the task are the queries and corresponding answers."""
        return ["keep_row_after_semantic_check", "thought"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the score of each instruction.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with the queries and answers pairs.
            The answers are an array of answers corresponding to the query.
            Each answer is represented as an object with the following properties:
                - name (string): The name of the tool used to generate the answer.
                - arguments (object): An object representing the arguments passed to the tool to generate the answer.
            Each argument is represented as a key-value pair, where the key is the parameter name and the
            value is the corresponding value.
        """
        if output is None:
            return self._default_error(input)

        output = remove_fences(output)

        try:
            result = orjson.loads(output)
            # Update the column name and change to bool
            result["keep_row_after_semantic_check"] = (
                result.pop("passes").lower() == "yes"
            )
            input.update(**result)
            return input
        except orjson.JSONDecodeError:
            return self._default_error(input)

    def _default_error(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Default error message for the task."""
        input.update({"thought": None, "keep_row_after_semantic_check": None})
        return input

    @override
    def get_structured_output(self) -> Dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        The schema corresponds to the following:

        ```python
        from typing import Literal
        from pydantic import BaseModel
        import json

        class Checker(BaseModel):
            thought: str
            passes: Literal["yes", "no"]

        json.dumps(Checker.model_json_schema(), indent=4)
        ```

        Returns:
            JSON Schema of the response to enforce.
        """
        return {
            "properties": {
                "thought": {"title": "Thought", "type": "string"},
                "passes": {"enum": ["yes", "no"], "title": "Passes", "type": "string"},
            },
            "required": ["thought", "passes"],
            "title": "Checker",
            "type": "object",
        }

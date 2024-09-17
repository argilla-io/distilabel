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
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Union

import orjson
from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT_API_GEN: Final[str] = """\
You are a data labeler. Your responsibility is to generate a set of diverse queries and corresponding answers for the given functions in JSON format.

Construct queries and answers that exemplify how to use these functions in a practical scenario. Include in each query specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.

Ensure the query:
- Is clear and concise
- Demonstrates typical use cases
- Includes all necessary parameters in a meaningful way. For numerical parameters, it could be either numbers or words
- Across a variety level of difficulties, ranging from beginner and advanced use cases
- The corresponding result's parameter types and ranges match with the function's descriptions

Ensure the answer:
- Is a list of function calls in JSON format
- The length of the answer list should be equal to the number of requests in the query
- Can solve all the requests in the query effectively
"""


class APIGenGenerator(Task):
    """Generate queries and answers for the given functions in JSON format.

    The `APIGenGenerator` is inspired by the APIGen pipeline, which was designed to generate
    verifiable and diverse function-calling datasets. The task generates a set of diverse queries
    and corresponding answers for the given functions in JSON format.

    Input columns:
        - examples (`str`): Examples used as few shots to guide the model.
        - func_name (`str`): Name for the function to generate.
        - func_desc (`str`): Description of what the function should do.

    Output columns:
        - queries (`List[str]`): The list of queries.
        - answers (`List[str]`): The list of answers.

    Categories:
        - text-generation

    References:
        - [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)
        - [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

    Examples:

        Generate without structured output (original implementation):

        ```python
        from distilabel.steps.tasks import ApiGenGenerator
        from distilabel.llms import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
        )
        apigen = ApiGenGenerator(
            use_default_structured_output=False,
            llm=llm
        )
        apigen.load()

        res = next(
            apigen.process(
                [
                    {
                        "examples": 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
                        "func_name": "getrandommovie",
                        "func_desc": "Returns a list of random movies from a database by calling an external API."
                    }
                ]
            )
        )
        res
        # [{'examples': 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
        # 'number': 1,
        # 'func_name': 'getrandommovie',
        # 'func_desc': 'Returns a list of random movies from a database by calling an external API.',
        # 'queries': ['I want to watch a movie tonight, can you recommend a random one from your database?',
        # 'Give me 5 random movie suggestions from your database to plan my weekend.'],
        # 'answers': [[{'name': 'getrandommovie', 'arguments': {}}],
        # [{'name': 'getrandommovie', 'arguments': {}},
        #     {'name': 'getrandommovie', 'arguments': {}},
        #     {'name': 'getrandommovie', 'arguments': {}},
        #     {'name': 'getrandommovie', 'arguments': {}},
        #     {'name': 'getrandommovie', 'arguments': {}}]],
        # 'distilabel_metadata': {'raw_output_api_gen_generator_0': '[\n   {\n       "query": "I want to watch a movie tonight, can you recommend a random one from your database?",\n       "answers": [\n           {\n               "name": "getrandommovie",\n               "arguments": {}\n           }\n       ]\n   },\n   {\n       "query": "Give me 5 random movie suggestions from your database to plan my weekend.",\n       "answers": [\n           {\n               "name": "getrandommovie",\n               "arguments": {}\n           },\n           {\n               "name": "getrandommovie",\n               "arguments": {}\n           },\n           {\n               "name": "getrandommovie",\n               "arguments": {}\n           },\n           {\n               "name": "getrandommovie",\n               "arguments": {}\n           },\n           {\n               "name": "getrandommovie",\n               "arguments": {}\n           }\n       ]\n   }\n]',
        # 'raw_input_api_gen_generator_0': [{'role': 'system',
        #     'content': "You are a data labeler. Your responsibility is to generate a set of diverse queries and corresponding answers for the given functions in JSON format.\n\nConstruct queries and answers that exemplify how to use these functions in a practical scenario. Include in each query specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.\n\nEnsure the query:\n- Is clear and concise\n- Demonstrates typical use cases\n- Includes all necessary parameters in a meaningful way. For numerical parameters, it could be either numbers or words\n- Across a variety level of difficulties, ranging from beginner and advanced use cases\n- The corresponding result's parameter types and ranges match with the function's descriptions\n\nEnsure the answer:\n- Is a list of function calls in JSON format\n- The length of the answer list should be equal to the number of requests in the query\n- Can solve all the requests in the query effectively"},
        #     {'role': 'user',
        #     'content': 'Here are examples of queries and the corresponding answers for similar functions:\nQUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]\n\nNote that the query could be interpreted as a combination of several independent requests.\nBased on these examples, generate 2 diverse query and answer pairs for the function `getrandommovie`\nThe detailed function description is the following:\nReturns a list of random movies from a database by calling an external API.\n\nThe output MUST strictly adhere to the following JSON format, and NO other text MUST be included:\n```json\n[\n   {\n       "query": "The generated query.",\n       "answers": [\n           {\n               "name": "api_name",\n               "arguments": {\n                   "arg_name": "value"\n                   ... (more arguments as required)\n               }\n           },\n           ... (more API calls as required)\n       ]\n   }\n]\n```\n\nNow please generate 2 diverse query and answer pairs following the above format.'}]},
        # 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```

        Generate with structured output:

        ```python
        from distilabel.steps.tasks import ApiGenGenerator
        from distilabel.llms import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
        )
        apigen = ApiGenGenerator(
            use_default_structured_output=True,
            llm=llm
        )
        apigen.load()

        res_struct = next(
            apigen.process(
                [
                    {
                        "examples": 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
                        "func_name": "getrandommovie",
                        "func_desc": "Returns a list of random movies from a database by calling an external API."
                    }
                ]
            )
        )
        res_struct
        # [{'examples': 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
        # 'number': 1,
        # 'func_name': 'getrandommovie',
        # 'func_desc': 'Returns a list of random movies from a database by calling an external API.',
        # 'queries': ["I'm bored and want to watch a movie. Can you suggest some movies?",
        # "My family and I are planning a movie night. We can't decide on what to watch. Can you suggest some random movie titles?"],
        # 'answers': [[{'arguments': {}, 'name': 'getrandommovie'}],
        # [{'arguments': {}, 'name': 'getrandommovie'}]],
        # 'distilabel_metadata': {'raw_output_api_gen_generator_0': '{ \n  "pairs": [\n    {\n      "answers": [\n        {\n          "arguments": {},\n          "name": "getrandommovie"\n        }\n      ],\n      "query": "I\'m bored and want to watch a movie. Can you suggest some movies?"\n    },\n    {\n      "answers": [\n        {\n          "arguments": {},\n          "name": "getrandommovie"\n        }\n      ],\n      "query": "My family and I are planning a movie night. We can\'t decide on what to watch. Can you suggest some random movie titles?"\n    }\n  ]\n}',
        # 'raw_input_api_gen_generator_0': [{'role': 'system',
        #     'content': "You are a data labeler. Your responsibility is to generate a set of diverse queries and corresponding answers for the given functions in JSON format.\n\nConstruct queries and answers that exemplify how to use these functions in a practical scenario. Include in each query specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.\n\nEnsure the query:\n- Is clear and concise\n- Demonstrates typical use cases\n- Includes all necessary parameters in a meaningful way. For numerical parameters, it could be either numbers or words\n- Across a variety level of difficulties, ranging from beginner and advanced use cases\n- The corresponding result's parameter types and ranges match with the function's descriptions\n\nEnsure the answer:\n- Is a list of function calls in JSON format\n- The length of the answer list should be equal to the number of requests in the query\n- Can solve all the requests in the query effectively"},
        #     {'role': 'user',
        #     'content': 'Here are examples of queries and the corresponding answers for similar functions:\nQUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]\n\nNote that the query could be interpreted as a combination of several independent requests.\nBased on these examples, generate 2 diverse query and answer pairs for the function `getrandommovie`\nThe detailed function description is the following:\nReturns a list of random movies from a database by calling an external API.\n\nNow please generate 2 diverse query and answer pairs following the above format.'}]},
        # 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```
    """

    system_prompt: str = SYSTEM_PROMPT_API_GEN
    use_default_structured_output: bool = False
    is_parallel: Union[bool, float] = False
    number: Union[int, List[int]] = 1

    _number: Union[int, None] = PrivateAttr(None)
    _fn_parallel_queries: Union[Callable[[], str], None] = PrivateAttr(None)
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
            / "generator.jinja2"
        )
        self._template = Template(open(_path).read())
        self._fn_parallel_queries = self._set_parallel_queries()
        self._format_inst = self._set_format_inst()

    def _set_parallel_queries(self) -> Callable[[], str]:
        """Prepares the function to generate update the parallel queries guide in the prompt.

        Raises:
            ValueError: if `is_parallel` is not a boolean or a list of floats.

        Returns:
            The function to generate the parallel queries guide.
        """
        parallel_queries_guide: str = (
            "It can contain multiple parallel queries in natural language for the given functions. "
            "They could use either the same function with different arguments or different functions.\n"
        )
        if isinstance(self.is_parallel, float):

            def fn_parallel_queries() -> str:
                return random.choices(
                    [parallel_queries_guide, ""],
                    weights=[self.is_parallel, 1 - self.is_parallel],
                )[0]
        elif isinstance(self.is_parallel, bool):
            if self.is_parallel:

                def fn_parallel_queries() -> str:
                    return parallel_queries_guide
            else:

                def fn_parallel_queries() -> str:
                    return ""
        else:
            # TODO: Update to DistilabelUserError
            raise ValueError("`is_parallel` must be a boolean or a list of floats.")
        return fn_parallel_queries

    def _get_number(self) -> int:
        """Generates the number of queries to generate in a single call.
        The number must be set to `_number` to avoid changing the original value
        when calling `_default_error`.
        """
        if isinstance(self.number, list):
            self._number = random.choice(self.number)
        else:
            self._number = self.number
        return self._number

    def _set_format_inst(self) -> str:
        """Prepares the function to generate the formatted instructions for the prompt.

        If the default structured output is used, returns an empty string because nothing
        else is needed, otherwise, returns the original addition to the prompt to guide the model
        to generate a formatted JSON.
        """
        if self.use_default_structured_output:
            return ""
        return (
            "\nThe output MUST strictly adhere to the following JSON format, and NO other text MUST be included:\n"
            "```json\n"
            "[\n"
            "   {\n"
            '       "query": "The generated query.",\n'
            '       "answers": [\n'
            "           {\n"
            '               "name": "api_name",\n'
            '               "arguments": {\n'
            '                   "arg_name": "value"\n'
            "                   ... (more arguments as required)\n"
            "               }\n"
            "           },\n"
            "           ... (more API calls as required)\n"
            "       ]\n"
            "   }\n"
            "]\n"
            "```\n"
        )

    @property
    def inputs(self) -> "StepColumns":
        """The inputs for the task."""
        return {
            "examples": True,
            "func_name": True,
            "func_desc": True,
        }

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType`."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._template.render(
                    examples=input["examples"],
                    parallel_queries=self._fn_parallel_queries(),
                    number=self._get_number(),
                    func_name=input["func_name"],
                    func_desc=input["func_desc"],
                    format_inst=self._format_inst,
                ),
            },
        ]

    @property
    def outputs(self) -> "StepColumns":
        """The output for the task are the queries and corresponding answers."""
        return ["queries", "answers", "model_name"]

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
            return self._default_error()

        try:
            pairs = orjson.loads(output)
        except orjson.JSONDecodeError:
            return self._default_error()

        if self.use_default_structured_output:
            pairs = pairs["pairs"]
        return self._format_output(pairs, input)

    def _format_output(
        self, pairs: Dict[str, Any], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parses the response, returning a dictionary with queries and answers.

        Args:
            pairs: The parsed dictionary from the LLM's output.
            input: The input from the `LLM`.

        Returns:
            Formatted output, where the `queries` are a list of strings, and the `answers`
            are a list of objects.
        """
        try:
            result = {"queries": [], "answers": []}
            for pair in pairs:
                result["queries"].append(pair["query"])
                result["answers"].append(pair["answers"])
            return result
        except Exception as e:
            self._logger.error(f"Error formatting output: {e}")
            return self._default_error()

    def _default_error(self) -> Dict[str, Any]:
        """Returns a default error output, to fill the responses in case of failure."""
        return {
            "queries": [None] * self._number,
            "answers": [None] * self._number,
        }

    @override
    def get_structured_output(self) -> Dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        The schema corresponds to the following:

        ```python
        from typing import Dict, List
        from pydantic import BaseModel


        class Answer(BaseModel):
            name: str
            arguments: Dict[str, str]

        class QueryAnswer(BaseModel):
            query: str
            answers: List[Answer]

        class QueryAnswerPairs(BaseModel):
            pairs: List[QueryAnswer]

        json.dumps(QueryAnswerPairs.model_json_schema(), indent=4)
        ```

        Returns:
            JSON Schema of the response to enforce.
        """
        return {
            "$defs": {
                "Answer": {
                    "properties": {
                        "name": {"title": "Name", "type": "string"},
                        "arguments": {
                            "additionalProperties": {"type": "string"},
                            "title": "Arguments",
                            "type": "object",
                        },
                    },
                    "required": ["name", "arguments"],
                    "title": "Answer",
                    "type": "object",
                },
                "QueryAnswer": {
                    "properties": {
                        "query": {"title": "Query", "type": "string"},
                        "answers": {
                            "items": {"$ref": "#/$defs/Answer"},
                            "title": "Answers",
                            "type": "array",
                        },
                    },
                    "required": ["query", "answers"],
                    "title": "QueryAnswer",
                    "type": "object",
                },
            },
            "properties": {
                "pairs": {
                    "items": {"$ref": "#/$defs/QueryAnswer"},
                    "title": "Pairs",
                    "type": "array",
                }
            },
            "required": ["pairs"],
            "title": "QueryAnswerPairs",
            "type": "object",
        }


# NOTE: This step seems unnecessary, use PrepareExamples from utils instead.
# class APIGenTransform(Step):
#     """Helper step to transform a dataset like
#     https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k in into examples
#     for the `APIGenGenerator` task.

#     Given the rows in formatted as in that dataset, this step prepares the input to be
#     passed to the `APIGenGenerator` task by sampling at the batch size.

#     Attributes:
#         example_template: String template to format the examples, comes with a default.

#     Input columns:
#         - query (`str`): The query that requires an answer in tool format.
#         - answers (`str`): String formatted dict.
#         - tools (`str`): String with formatted list of dictionaries containing the available
#             tools.

#     Output columns:
#         - examples (`str`): Query and answer formatted as an example to be feed
#             to the prompt.
#         - func_name (`str`): Example name for a function.
#         - func_desc (`str`): Description of the function `func_name`.

#     Categories:
#         - text-manipulation

#     References:
#         - [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)
#         - [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

#     Examples:

#         Transform the data for APIGenGenerator:

#         ```python
#         from datasets import load_dataset
#         from distilabel.steps.tasks.apigen.base import APIGenTransform

#         samples = load_dataset("Salesforce/xlam-function-calling-60k", split="train").select(range(3)).to_list()
#         transform = APIGenTransform()
#         transform.load()
#         outputs = next(transform.process(samples))
#         outputs
#         # [{'examples': '## Query:\nWhat is the T3MA for \'ETH/BTC\' using a 1h interval and a time period of 14?\n## Answer:\n[{"name": "t3ma", "arguments": {"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}}]',
#         # 'func_name': 'live_giveaways_by_type',
#         # 'func_desc': 'Retrieve live giveaways from the GamerPower API based on the specified type.'},
#         # {'examples': '## Query:\nWhere can I find live giveaways for beta access and games?\n## Answer:\n[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}]',
#         # 'func_name': 'web_chain_details',
#         # 'func_desc': 'python'},
#         # {'examples': '## Query:\nWhere can I find live giveaways for beta access and games?\n## Answer:\n[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}]',
#         # 'func_name': 't3ma',
#         # 'func_desc': 'Fetches the Triple Exponential Moving Average (T3MA) for a given financial instrument.'}]
#         ```

#     Citations:

#         ```
#         @misc{liu2024apigenautomatedpipelinegenerating,
#             title={APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
#             author={Zuxin Liu and Thai Hoang and Jianguo Zhang and Ming Zhu and Tian Lan and Shirley Kokane and Juntao Tan and Weiran Yao and Zhiwei Liu and Yihao Feng and Rithesh Murthy and Liangwei Yang and Silvio Savarese and Juan Carlos Niebles and Huan Wang and Shelby Heinecke and Caiming Xiong},
#             year={2024},
#             eprint={2406.18518},
#             archivePrefix={arXiv},
#             primaryClass={cs.CL},
#             url={https://arxiv.org/abs/2406.18518},
#         }
#         ```
#     """

#     example_template: str = "## Query:\n{query}\n## Answer:\n{answers}"

#     @property
#     def inputs(self) -> "StepColumns":
#         """The inputs for the task are those found in the original dataset."""
#         return ["query", "answers", "tools"]

#     @property
#     def outputs(self) -> "StepColumns":
#         """The outputs are the columns required by `APIGenGenerator` task."""
#         return ["examples", "func_name", "func_desc"]

#     @override
#     def process(self, inputs: StepInput) -> "StepOutput":
#         """The process prepares the data for the `APIGenGenerator` task.

#         If a single example is provided, it is copied to avoid raising an error.

#         Args:
#             inputs: A list of dictionaries with the input data.

#         Yields:
#             A list of dictionaries with the output data.
#         """
#         if len(inputs) < 2:
#             self._logger.warning(
#                 "The batch must have at least 2 examples, copying to avoid raising an error."
#             )
#             inputs = inputs * 2

#         outputs = []
#         for _ in range(len(inputs)):
#             # Selects 2 random examples without replacement
#             selection = random.sample(inputs, 2)
#             tools = orjson.loads(selection[1]["tools"])
#             tool = random.choice(tools)
#             outputs.append(
#                 {
#                     "examples": self.example_template.format(
#                         query=selection[0]["query"], answers=selection[0]["answers"]
#                     ),
#                     "func_name": tool["name"],
#                     "func_desc": tool["description"],
#                 }
#             )
#         yield outputs

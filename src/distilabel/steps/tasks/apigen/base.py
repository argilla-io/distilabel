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

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import orjson
from typing_extensions import override

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


# TODO: Move these to jinja2 templates
SYSTEM_PROMPT_API_GEN: str = (
    "You are a data labeler. Your responsibility is to generate a set of diverse queries and corresponding answers for the given functions in JSON format.\n\n"
    "Construct queries and answers that exemplify how to use these functions in a practical scenario. Include in each query specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.\n\n"
    "Ensure the query:\n"
    "- Is clear and concise\n"
    "- Demonstrates typical use cases\n"
    "- Includes all necessary parameters in a meaningful way. For numerical parameters, it could be either numbers or words\n"
    "- Across a variety level of difficulties, ranging from beginner and advanced use cases\n"
    "- The corresponding result's parameter types and ranges match with the function's descriptions\n\n"
    "Ensure the answer:\n"
    "- Is a list of function calls in JSON format\n"
    "- The length of the answer list should be equal to the number of requests in the query\n"
    "- Can solve all the requests in the query effectively"
)

API_GEN_PROMPT: str = (
    "Here are examples of queries and the corresponding answers for similar functions:\n"
    "{examples}\n\n"
    "Note that the query could be interpreted as a combination of several independent requests.\n"
    "{parallel_queries}"
    "Based on these examples, generate {number} diverse query and answer pairs for the function `{func_name}`\n"
    "The detailed function description is the following:\n"
    "{func_desc}\n"
    "{format_inst}\n"
    "Now please generate {number} diverse query and answer pairs following the above format."
)


class APIGenGenerator(Task):
    """_summary_

    Examples:

        Generate without structured output (original implementation):

        ```python
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
                        "number": 2,
                        "func_name": "getrandommovie",
                        "func_desc": "Returns a list of random movies from a database by calling an external API."
                    }
                ]
            )
        )
        res
        # [{'examples': 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
        # 'number': 2,
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
        from distilabel.llms import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct"
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
                        "number": 2,
                        "func_name": "getrandommovie",
                        "func_desc": "Returns a list of random movies from a database by calling an external API."
                    }
                ]
            )
        )
        res_struct
        # [{'examples': 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
        # 'number': 2,
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
    is_parallel: Union[bool, List[float]] = False
    number: Optional[int] = 1

    # TODO: Move _get_parallel_queries here to avoid recomputing always on _format_input
    # def model_post_init(self, __context: Any) -> None:
    #     return super().model_post_init(__context)

    def _get_parallel_queries(self) -> str:
        parallel_queries_guide = "It can contain multiple parallel queries in natural language for the given functions. They could use either the same function with different arguments or different functions.\n"
        if isinstance(self.is_parallel, list):
            return random.choices(
                [parallel_queries_guide, ""], weights=self.is_parallel
            )
        elif isinstance(self.is_parallel, bool):
            return parallel_queries_guide if self.is_parallel else ""
        # TODO: Update to DistilabelUserError
        raise ValueError("`is_parallel` must be a boolean or a list of floats.")

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task."""
        return ["examples", "number", "func_name", "func_desc"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType`."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": API_GEN_PROMPT.format(
                    examples=input["examples"],
                    parallel_queries=self._get_parallel_queries(),
                    number=input.get("number", self.number),
                    func_name=input["func_name"],
                    func_desc=input["func_desc"],
                    format_inst=f"\n{self._steer_output()}\n"
                    if not self.use_default_structured_output
                    else "",
                ),
            },
        ]

    @property
    def outputs(self) -> List[str]:
        """The output for the task are the queries and corresponding answers."""
        return ["queries", "answers"]

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

        try:
            pairs = orjson.loads(output)
        except orjson.JSONDecodeError:
            return self._default_error(input)

        if self.use_default_structured_output:
            pairs = pairs["pairs"]
        return self._format_output(pairs, input)

    def _steer_output(self) -> str:
        """Steers the output to adhere to the following JSON format."""
        return (
            "The output MUST strictly adhere to the following JSON format, and NO other text MUST be included:\n"
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
            "```"
        )

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
            print("Error formatting output:", e)
            return self._default_error(input)

    def _default_error(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a default error output, to fill the responses in case of failure."""
        return {
            "queries": [None] * input["number"],
            "answers": [None] * input["number"],
        }

    # @staticmethod
    # def prepare_example(python_function: Callable) -> str:
    #     # TODO: We should check the transfomers version for this
    #     from transformers.utils import get_json_schema
    #     pass

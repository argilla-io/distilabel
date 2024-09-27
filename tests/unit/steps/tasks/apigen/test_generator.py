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
from typing import TYPE_CHECKING, List, Union

import pytest

from distilabel.steps.tasks.apigen.generator import APIGenGenerator
from tests.unit.conftest import DummyLLM

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import FormattedInput

import json


class DummyAPIGenLLM(DummyLLM):
    use_structured_output: bool = False
    number: int = 1

    def generate(
        self, inputs: List["FormattedInput"], num_generations: int = 1
    ) -> "GenerateOutput":
        query_answers = [
            {
                "query": "What information can be obtained about the Maine Coon cat breed?",
                "answers": [
                    {
                        "name": "get_breed_information",
                        "arguments": {"breed": "Maine Coon"},
                    }
                ]
                * self.number,
            }
        ]
        if self.use_structured_output:
            query_answers = {"pairs": query_answers}
        # return [json.dumps(query_answers) for _ in range(len(inputs))]
        return [
            [json.dumps(query_answers) for _ in range(num_generations)]
            for _ in range(len(inputs))
        ]


# Example of 3 rows from Salesforce/xlam-function-calling-60k
SAMPLE_DATA = [
    {
        "answers": '[{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]',
        "query": "What information can be obtained about the Maine Coon cat breed?",
        "id": 3493,
        "tools": '[{"name": "get_breed_information", "description": "Fetch information about a specific cat breed from the Cat Breeds API.", "parameters": {"breed": {"description": "The name of the cat breed to fetch information for.", "type": "str", "default": "aegean"}}}, {"name": "country_region_cities", "description": "Fetches a list of cities within a specified region of a given country from the GeoDB API.", "parameters": {"countryid": {"description": "An ISO-3166 country code or WikiData ID.", "type": "str", "default": "US"}, "regioncode": {"description": "An ISO-3166 or FIPS region code.", "type": "str", "default": "CA"}, "limit": {"description": "The maximum number of results to retrieve. Defaults to None.", "type": "int, optional", "default": ""}, "hateoasmode": {"description": "Include HATEOAS-style links in results. Defaults to None.", "type": "bool, optional", "default": ""}, "asciimode": {"description": "Display results using ASCII characters. Defaults to None.", "type": "bool, optional", "default": ""}, "nameprefixdefaultlangresults": {"description": "Match on names in the default language if a non-default language is requested when prefix-matching. Defaults to None.", "type": "bool, optional", "default": ""}, "timezoneids": {"description": "Only include cities in these time zones. Comma-separated values. Defaults to None.", "type": "str, optional", "default": ""}, "nameprefix": {"description": "Only include cities whose names start with this prefix. If languagecode is set, the prefix will be matched on the name as it appears in that language. Defaults to None.", "type": "str, optional", "default": ""}, "types": {"description": "Only include cities of these types (comma-separated): CITY, ADM2. Defaults to None.", "type": "str, optional", "default": ""}, "minpopulation": {"description": "Only include cities with at least this population. Defaults to None.", "type": "int, optional", "default": ""}, "languagecode": {"description": "Display results in this language. Defaults to None.", "type": "str, optional", "default": ""}, "offset": {"description": "The zero-based offset into the results. Defaults to None.", "type": "int, optional", "default": ""}, "maxpopulation": {"description": "Only include cities with no more than this population. Defaults to None.", "type": "int, optional", "default": ""}, "includedeleted": {"description": "Whether to include any cities marked deleted. Options are: ALL, SINCE_YESTERDAY, SINCE_LAST_WEEK, NONE. Defaults to None.", "type": "str, optional", "default": ""}, "sort": {"description": "How to sort the results. Format: \\u00b1SORT_FIELD,\\u00b1SORT_FIELD where SORT_FIELD = elevation, name, population. Defaults to None.", "type": "str, optional", "default": ""}}}, {"name": "company_details", "description": "Fetch details of a company from Indeed\'s API.", "parameters": {"company_id": {"description": "The unique identifier of the company to fetch details for.", "type": "str", "default": "Microsoft"}, "locality": {"description": "The locality or country code for Indeed\'s subdomain. Default is \'us\' if not provided.", "type": "str, optional", "default": ""}}}]',
    },
    {
        "answers": '[{"name": "mailcheck", "arguments": {"domain": "protonmail.com"}}, {"name": "mailcheck", "arguments": {"domain": "mail.com"}}, {"name": "get_products_in_category", "arguments": {"skip": 20, "limit": 25, "category": "furniture"}}]',
        "query": "Check if the email domains 'protonmail.com' and 'mail.com' are valid and not temporary. Get the products from category 'furniture' in my store, skipping the first 20 items and limiting to 25 items.",
        "id": 57546,
        "tools": '[{"name": "mailcheck", "description": "Checks if an email domain is valid or a disposable/temporary address.", "parameters": {"domain": {"description": "The email or domain to check for validity. It is recommended to enter just the domain for user privacy.", "type": "str", "default": "mailinator.com"}}}, {"name": "get_products_in_category", "description": "Fetches a list of products from a specified category in a store with pagination.", "parameters": {"skip": {"description": "The number of items to skip before starting to collect the result set.", "type": "int", "default": ""}, "limit": {"description": "The number of items to return in the result set.", "type": "int", "default": ""}, "category": {"description": "The category from which to fetch products.", "type": "str", "default": ""}}}, {"name": "product_by_id", "description": "Fetches detailed information about a specific product from the AliExpress API using the provided product ID.", "parameters": {"product_id": {"description": "The unique identifier for the product on AliExpress.", "type": "int", "default": "32841070485"}}}]',
    },
    {
        "answers": '[{"name": "navigations_get_node_content", "arguments": {"is_id": 8899, "cat_id": 8899, "language": "en"}}, {"name": "navigations_get_node_content", "arguments": {"is_id": 7766, "cat_id": 7766, "language": "en"}}, {"name": "navigations_get_node_content", "arguments": {"is_id": 5544, "cat_id": 5544, "language": "fr"}}, {"name": "navigations_get_node_content", "arguments": {"is_id": 3322, "cat_id": 3322, "language": "fr"}}]',
        "query": "What are the node contents for category IDs 8899 and 7766 in English and for category IDs 5544 and 3322 in French?",
        "id": 8815,
        "tools": '[{"name": "navigations_get_node_content", "description": "Fetches the content of a node in a navigation hierarchy.", "parameters": {"is_id": {"description": "The \'id\' field value returned from the /navigations/get-root endpoint.", "type": "int", "default": "26066300130"}, "cat_id": {"description": "The \'cat_id\' field value returned from the /navigations/get-tabs endpoint.", "type": "int", "default": "2026"}, "language": {"description": "The 2-letter language code (default is \'en\').", "type": "str, optional", "default": "en"}, "currency": {"description": "The 3-letter currency code (default is \'USD\').", "type": "str, optional", "default": "USD"}, "country": {"description": "The 2-letter country code (default is \'US\').", "type": "str, optional", "default": "US"}}}, {"name": "products_get_reviews", "description": "Fetches brief reviews of a product from the Shein API.", "parameters": {"goods_spu": {"description": "The value of \'productRelationID\' returned in the /products/list or /products/search endpoints. Defaults to \'m22022854841\'.", "type": "str, optional", "default": "m22022854841"}, "cat_id": {"description": "The value of \'cat_id\' returned in the /products/list or /products/search endpoints. Defaults to \'1727\'.", "type": "str, optional", "default": "1727"}, "sku": {"description": "The value of \'goods_sn\' returned in the /products/list or /products/search endpoints. Defaults to \'rm2202285484176751\'.", "type": "str, optional", "default": "rm2202285484176751"}, "currency": {"description": "The 3-letter currency code. Defaults to \'USD\'.", "type": "str, optional", "default": "USD"}, "goods_id": {"description": "The value of \'goods_id\' field returned in the /products/list or /products/search endpoints. Defaults to \'10196865\'.", "type": "str, optional", "default": "10196865"}, "language": {"description": "The 2-letter language code. Defaults to \'en\'.", "type": "str, optional", "default": "en"}, "country": {"description": "The 2-letter country code. Defaults to \'US\'.", "type": "str, optional", "default": "US"}}}]',
    },
]


class TestApiGenGenerator:
    @pytest.mark.parametrize("number", [1, 2, [3]])
    @pytest.mark.parametrize("use_default_structured_output", [True, False])
    @pytest.mark.parametrize("use_tools", [True, False])
    def test_format_input(
        self,
        number: Union[int, List[int]],
        use_default_structured_output: bool,
        use_tools: bool,
    ) -> None:
        random.seed(42)
        task = APIGenGenerator(
            llm=DummyLLM(),
            number=number,
            use_tools=use_tools,
            use_default_structured_output=use_default_structured_output,
        )
        task.load()
        formatted = task.format_input(
            input={
                "examples": '## Query:\nWhat information can be obtained about the Maine Coon cat breed?\n## Answer:\n[{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]',
                "func_name": "get_breed_information",
                "func_desc": "Fetch information about a specific cat breed from the Cat Breeds API.",
                "tools": '[{"name": "navigations_get_node_content", "description": "Fetches the content of a node in a navigation hierarchy.", "parameters": {"is_id": {"description": "The \'id\' field value returned from the /navigations/get-root endpoint.", "type": "int", "default": "26066300130"}, "cat_id": {"description": "The \'cat_id\' field value returned from the /navigations/get-tabs endpoint.", "type": "int", "default": "2026"}, "language": {"description": "The 2-letter language code (default is \'en\').", "type": "str, optional", "default": "en"}, "currency": {"description": "The 3-letter currency code (default is \'USD\').", "type": "str, optional", "default": "USD"}, "country": {"description": "The 2-letter country code (default is \'US\').", "type": "str, optional", "default": "US"}}}, {"name": "products_get_reviews", "description": "Fetches brief reviews of a product from the Shein API.", "parameters": {"goods_spu": {"description": "The value of \'productRelationID\' returned in the /products/list or /products/search endpoints. Defaults to \'m22022854841\'.", "type": "str, optional", "default": "m22022854841"}, "cat_id": {"description": "The value of \'cat_id\' returned in the /products/list or /products/search endpoints. Defaults to \'1727\'.", "type": "str, optional", "default": "1727"}, "sku": {"description": "The value of \'goods_sn\' returned in the /products/list or /products/search endpoints. Defaults to \'rm2202285484176751\'.", "type": "str, optional", "default": "rm2202285484176751"}, "currency": {"description": "The 3-letter currency code. Defaults to \'USD\'.", "type": "str, optional", "default": "USD"}, "goods_id": {"description": "The value of \'goods_id\' field returned in the /products/list or /products/search endpoints. Defaults to \'10196865\'.", "type": "str, optional", "default": "10196865"}, "language": {"description": "The 2-letter language code. Defaults to \'en\'.", "type": "str, optional", "default": "en"}, "country": {"description": "The 2-letter country code. Defaults to \'US\'.", "type": "str, optional", "default": "US"}}}]',
            }
        )

        assert isinstance(formatted, list)
        # Check only the user prompt, the system one should be fixed
        formatted_prompt = formatted[1]["content"]

        if isinstance(number, list):
            # Fix the number for the tests for simplicity
            number = 3
        assert f"Now please generate {number} diverse" in formatted_prompt

        assert (
            "The output MUST strictly adhere to the following JSON format, and NO other text MUST be included:"
            in formatted_prompt
        )

        tools_entry = "This is the available tool to guide you (respect the order of the parameters):"
        if use_tools:
            assert tools_entry in formatted_prompt
        else:
            assert tools_entry not in formatted_prompt

        is_parallel_check = "It can contain multiple parallel queries in natural language for the given functions. They could use either the same function with different arguments or different functions."
        if number > 1:
            assert is_parallel_check in formatted_prompt
        else:
            assert is_parallel_check not in formatted_prompt

    @pytest.mark.parametrize("number", [1, 2])
    @pytest.mark.parametrize("use_default_structured_output", [True, False])
    @pytest.mark.parametrize("use_tools", [True, False])
    def test_format_process(
        self,
        # is_parallel: Union[bool, List[float]],
        number: Union[int, List[int]],
        use_default_structured_output: bool,
        use_tools: bool,
    ) -> None:
        # Is parallel is not relevant in this case, it's only relevant for the format_input
        # as it will be multiple questions in the prompt
        random.seed(42)
        task = APIGenGenerator(
            llm=DummyAPIGenLLM(
                use_structured_output=use_default_structured_output, number=number
            ),
            number=number,
            use_tools=use_tools,
            use_default_structured_output=use_default_structured_output,
        )
        task.load()
        result = next(
            task.process(
                [
                    {
                        "examples": '## Query:\nWhat information can be obtained about the Maine Coon cat breed?\n## Answer:\n[{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]',
                        "func_name": "get_breed_information",
                        "func_desc": "Fetch information about a specific cat breed from the Cat Breeds API.",
                        "tools": '[{"name": "navigations_get_node_content", "description": "Fetches the content of a node in a navigation hierarchy.", "parameters": {"is_id": {"description": "The \'id\' field value returned from the /navigations/get-root endpoint.", "type": "int", "default": "26066300130"}, "cat_id": {"description": "The \'cat_id\' field value returned from the /navigations/get-tabs endpoint.", "type": "int", "default": "2026"}, "language": {"description": "The 2-letter language code (default is \'en\').", "type": "str, optional", "default": "en"}, "currency": {"description": "The 3-letter currency code (default is \'USD\').", "type": "str, optional", "default": "USD"}, "country": {"description": "The 2-letter country code (default is \'US\').", "type": "str, optional", "default": "US"}}}, {"name": "products_get_reviews", "description": "Fetches brief reviews of a product from the Shein API.", "parameters": {"goods_spu": {"description": "The value of \'productRelationID\' returned in the /products/list or /products/search endpoints. Defaults to \'m22022854841\'.", "type": "str, optional", "default": "m22022854841"}, "cat_id": {"description": "The value of \'cat_id\' returned in the /products/list or /products/search endpoints. Defaults to \'1727\'.", "type": "str, optional", "default": "1727"}, "sku": {"description": "The value of \'goods_sn\' returned in the /products/list or /products/search endpoints. Defaults to \'rm2202285484176751\'.", "type": "str, optional", "default": "rm2202285484176751"}, "currency": {"description": "The 3-letter currency code. Defaults to \'USD\'.", "type": "str, optional", "default": "USD"}, "goods_id": {"description": "The value of \'goods_id\' field returned in the /products/list or /products/search endpoints. Defaults to \'10196865\'.", "type": "str, optional", "default": "10196865"}, "language": {"description": "The 2-letter language code. Defaults to \'en\'.", "type": "str, optional", "default": "en"}, "country": {"description": "The 2-letter country code. Defaults to \'US\'.", "type": "str, optional", "default": "US"}}}]',
                    }
                ]
            )
        )[0]
        assert "query" in result
        assert "answers" in result
        query = result["query"]
        assert isinstance(query, str)
        answers = json.loads(result["answers"])
        assert isinstance(answers, list)
        assert len(answers) == number

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

from typing import List, Union

import argilla as rg
import pytest
from distilabel.dataset import CustomDataset
from distilabel.tasks.preference.complexity_scorer import ComplexityScorerTask
from distilabel.tasks.preference.quality_scorer import QualityScorerTask

prompt_complexity_1 = """Ranking the following questions according to the difficulty and complexity. Score 1-2.
You can give a score of 3 if the question is too complex for you to answer it. You should
respond with the format:
[1] Score: 1
[2] Score: 2
...

[1] instruct 1
[2] instruct 2"""

prompt_complexity_2 = """Ranking the following questions according to the difficulty and complexity. Score 1-4.
You can give a score of 5 if the question is too complex for you to answer it. You should
respond with the format:
[1] Score: 1
[2] Score: 2
...

[1] instruct 1
[2] instruct 2
[3] instruct 3
[4] instruct 4"""


@pytest.mark.parametrize(
    "input, expected",
    [
        (["instruct 1", "instruct 2"], prompt_complexity_1),
        ([f"instruct {i+1}" for i in range(4)], prompt_complexity_2),
    ],
)
def test_evol_complexity_scorer_task(input: List[str], expected: str):
    task = ComplexityScorerTask()
    result = task.generate_prompt(input).formatted_prompt
    assert result == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("[1] Score: 4\n[2] Score: 3\n[3] Score: 2\n[4] Score: 1", [4, 3, 2, 1]),
    ],
)
def test_evolcomplexity_scorer_task_parsing(input: str, expected: str):
    task = ComplexityScorerTask()
    result = task.parse_output(input)
    assert result["rating"] == expected


prompt_quality_1 = """Rank the following responses provided by different AI assistants to the user’s question
according to the quality of their response. Score each response from 1 to 2, with 3
reserved for responses that are already very well written and cannot be improved further.
Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth,
creativity, and level of detail of the response.
Use the following format:
[Response 1] Score:
[Response 2] Score:
...
#Question#: instruction
#Response List#:

[Response 1] response 1
[Response 2] response 2"""


prompt_quality_2 = """Rank the following responses provided by different AI assistants to the user’s question
according to the quality of their response. Score each response from 1 to 4, with 5
reserved for responses that are already very well written and cannot be improved further.
Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth,
creativity, and level of detail of the response.
Use the following format:
[Response 1] Score:
[Response 2] Score:
...
#Question#: instruction
#Response List#:

[Response 1] response 1
[Response 2] response 2
[Response 3] response 3
[Response 4] response 4"""


@pytest.mark.parametrize(
    "instruction, responses, expected",
    [
        ("instruction", ["response 1", "response 2"], prompt_quality_1),
        ("instruction", [f"response {i+1}" for i in range(4)], prompt_quality_2),
    ],
)
def test_evol_quality_scorer_task(
    instruction: str, responses: List[str], expected: str
):
    task = QualityScorerTask()
    result = task.generate_prompt(instruction, responses).formatted_prompt
    assert result == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("[Response 1] Score: 2\n[Response 2] Score: 3", [2, 3]),
    ],
)
def test_evol_quality_scorer_task_parsing(input: str, expected: str):
    task = QualityScorerTask()
    result = task.parse_output(input)
    assert result["rating"] == expected


@pytest.fixture
def custom_dataset() -> CustomDataset:
    ds = CustomDataset.from_dict(
        {
            "input": [
                "You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:",
                "Generate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One",
            ],
            "generations": [
                [
                    '[\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\n]',
                    " Sure, I'd be happy to help! Here are the RDF triplets for the input sentence:\n\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst]\n\nExplanation:\n\n* AFC Ajax (amateurs) is the subject of the first triplet, and hasGround is the predicate that describes the relationship between AFC Ajax (amateurs) and Sportpark De Toekomst.\n* Ajax Youth Academy is the subject of the second triplet, and playsAt is the predicate that describes the relationship between Ajax Youth Academy and Sportpark De Toekomst.\n\nNote that there may be other possible RDF triplets that could be derived from the input sentence, but the above triplets capture the main relationships present in the sentence.",
                ],
                [
                    "Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One.",
                    ' Sure! Here\'s a sentence that describes all the data you provided:\n\n"Midsummer House is a moderately priced Chinese restaurant with a customer rating of 3 out of 5, located near All Bar One, offering a variety of delicious dishes."',
                ],
            ],
            "rating": [[1.0, 2.0], [1.0, 2.0]],
        }
    )
    return ds


@pytest.mark.parametrize("task", [ComplexityScorerTask(), QualityScorerTask()])
def test_evol_scorers_task_to_argilla_dataset(
    custom_dataset: CustomDataset,
    task: Union[ComplexityScorerTask, QualityScorerTask],
):
    custom_dataset.task = task
    ds_row = custom_dataset[0]
    task = custom_dataset.task
    rg_dataset = task.to_argilla_dataset(ds_row)
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 0


@pytest.mark.parametrize(
    "task, num_fields", [(ComplexityScorerTask(), 2), (QualityScorerTask(), 3)]
)
def test_evol_complexity_scorer_task_to_argilla_record(
    custom_dataset: CustomDataset,
    task: Union[ComplexityScorerTask, QualityScorerTask],
    num_fields: int,
):
    custom_dataset.task = task
    ds_row = custom_dataset[0]
    task = custom_dataset.task
    record = task.to_argilla_record(ds_row)
    assert isinstance(record, rg.FeedbackRecord)
    assert len(record.fields) == num_fields


@pytest.mark.parametrize("task", [ComplexityScorerTask(), QualityScorerTask()])
def test_evol_complexity_scorer_task_to_argilla(
    custom_dataset: CustomDataset,
    task: Union[ComplexityScorerTask, QualityScorerTask],
):
    custom_dataset.task = task
    rg_dataset = custom_dataset.to_argilla(vector_strategy=False, metric_strategy=False)
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 2

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

import tempfile
from pathlib import Path

import pytest
from distilabel.dataset import CustomDataset, DatasetCheckpoint
from distilabel.tasks import UltraFeedbackTask
from distilabel.tasks.text_generation.self_instruct import SelfInstructTask


@pytest.fixture
def custom_dataset():
    ds = CustomDataset.from_dict({"input": ["a", "b"], "generations": ["c", "d"]})
    ds.task = UltraFeedbackTask.for_overall_quality()
    return ds


def test_dataset_save_to_disk(custom_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_name = Path(tmpdir) / "dataset_folder"
        custom_dataset.save_to_disk(ds_name)
        assert ds_name.is_dir()
        assert (ds_name / "task.pkl").is_file()


def test_dataset_load_disk(custom_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_name = Path(tmpdir) / "dataset_folder"
        custom_dataset.save_to_disk(ds_name)
        ds_from_disk = CustomDataset.load_from_disk(ds_name)
        assert isinstance(ds_from_disk, CustomDataset)
        assert isinstance(ds_from_disk.task, UltraFeedbackTask)


@pytest.mark.parametrize(
    "save_frequency, dataset_len, batch_size, expected",
    [
        (1, 10, 1, 10),
        (3, 10, 1, 3),
        (8, 32, 8, 4),
        (8, 64, 16, 4),
        (20, 100, 7, 5),
    ],
)
def test_do_checkpoint(
    save_frequency: int, dataset_len: int, batch_size: int, expected: int
):
    ds = CustomDataset.from_dict(
        {"input": ["a"] * dataset_len, "generations": ["a"] * dataset_len}
    )
    ds.task = UltraFeedbackTask.for_overall_quality()
    chk = DatasetCheckpoint(save_frequency=save_frequency)
    ctr = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_name = Path(tmpdir) / "dataset_folder"
        for batch_i, _ in enumerate(ds.iter(batch_size=batch_size), start=1):
            step = batch_i * batch_size
            if chk.do_checkpoint(step):
                ds.save_to_disk(ds_name)
                ds_from_disk = CustomDataset.load_from_disk(ds_name)
                assert ds_from_disk.to_pandas()["generations"].isna().sum() == 0

                ctr += 1
    assert ctr == expected == chk._total_checks


@pytest.fixture
def dataset_with_self_instruct() -> CustomDataset:
    ds = CustomDataset.from_dict(
        {
            "input": ["Algebraic Expressions", "Linear Equations"],
            "generation_model": [
                ["gpt-3.5-turbo", "gpt-3.5-turbo"],
                ["gpt-3.5-turbo", "gpt-3.5-turbo"],
            ],
            "generation_prompt": [
                [
                    [
                        {
                            "content": "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.",
                            "role": "system",
                        },
                        {
                            "content": '# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nA question-answering assistant for engaging and challenging math quizzes and problems\n\n# Context\nAlgebraic Expressions\n\n# Output\n',
                            "role": "user",
                        },
                    ],
                    [
                        {
                            "content": "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.",
                            "role": "system",
                        },
                        {
                            "content": '# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nA question-answering assistant for engaging and challenging math quizzes and problems\n\n# Context\nAlgebraic Expressions\n\n# Output\n',
                            "role": "user",
                        },
                    ],
                ],
                [
                    [
                        {
                            "content": "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.",
                            "role": "system",
                        },
                        {
                            "content": '# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nA question-answering assistant for engaging and challenging math quizzes and problems\n\n# Context\nLinear Equations\n\n# Output\n',
                            "role": "user",
                        },
                    ],
                    [
                        {
                            "content": "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.",
                            "role": "system",
                        },
                        {
                            "content": '# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nA question-answering assistant for engaging and challenging math quizzes and problems\n\n# Context\nLinear Equations\n\n# Output\n',
                            "role": "user",
                        },
                    ],
                ],
            ],
            "raw_generation_responses": [
                [
                    "1. How do I simplify the given algebraic expression?\n2. Can you explain the steps to expand this algebraic expression fully?\n3. Provide a detailed explanation of how to factorize this algebraic expression.\n4. What is the purpose of combining like terms in algebraic expressions?\n5. Show me the process of solving this algebraic equation step by step.",
                    '1. "Explain the concept of simplifying algebraic expressions using variables and constants."\n2. "Detail the steps involved in evaluating algebraic expressions with multiple variables."\n3. "What are the different properties of algebraic expressions and how do they affect simplification?"\n4. "Provide an example of a complex algebraic expression and guide me through the process of simplifying it."\n5. "How can algebraic expressions be used to model real-life situations and problems?"',
                ],
                [
                    '1. Seek a solution: What is the unknown variable that solves the equation "2x + 3 = 9"?\n\n2. Investigate a relation: Detail how the slope of a line can be determined using two points on the line.\n\n3. Explore real-world applications: Explain the significance of linear equations in modeling the growth of a population over time.\n\n4. Uncover a connection: How does solving a system of linear equations relate to finding the intersection point(s) of two lines?\n\n5. Analyze a scenario: Provide a step-by-step explanation of how to graphically solve the equation "y = 2x + 1" and determine its solution(s).',
                    "1. How can I solve a linear equation using the substitution method?\n2. Show an example of solving a linear equation using the elimination method.\n3. Explain the concept of graphing linear equations and its relation to solving them.\n4. Provide a step-by-step guide on solving a linear equation with fractions.\n5. How do I determine the slope and y-intercept of a linear equation?",
                ],
            ],
            "instructions": [
                [
                    [
                        "How do I simplify the given algebraic expression?",
                        "Can you explain the steps to expand this algebraic expression fully?",
                        "Provide a detailed explanation of how to factorize this algebraic expression.",
                        "What is the purpose of combining like terms in algebraic expressions?",
                    ],
                    [
                        '"Explain the concept of simplifying algebraic expressions using variables and constants."',
                        '"Detail the steps involved in evaluating algebraic expressions with multiple variables."',
                        '"What are the different properties of algebraic expressions and how do they affect simplification?"',
                        '"Provide an example of a complex algebraic expression and guide me through the process of simplifying it."',
                    ],
                ],
                [
                    [
                        'Seek a solution: What is the unknown variable that solves the equation "2x + 3 = 9"?',
                        "Investigate a relation: Detail how the slope of a line can be determined using two points on the line.",
                        "Explore real-world applications: Explain the significance of linear equations in modeling the growth of a population over time.",
                        "Uncover a connection: How does solving a system of linear equations relate to finding the intersection point(s) of two lines?",
                    ],
                    [
                        "How can I solve a linear equation using the substitution method?",
                        "Show an example of solving a linear equation using the elimination method.",
                        "Explain the concept of graphing linear equations and its relation to solving them.",
                        "Provide a step-by-step guide on solving a linear equation with fractions.",
                    ],
                ],
            ],
        }
    )
    ds.task = SelfInstructTask()
    return ds


def test_dataset_with_self_instruct_to_argilla(dataset_with_self_instruct):
    rg_dataset = dataset_with_self_instruct.to_argilla()
    assert (
        rg_dataset[0].fields["instruction"]
        == "How do I simplify the given algebraic expression?"
    )
    assert len(rg_dataset) == 16

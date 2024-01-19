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

import re
import tempfile
from pathlib import Path
from typing import List

import pytest
from distilabel.dataset import CustomDataset, DatasetCheckpoint
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from distilabel.tasks.text_generation.self_instruct import SelfInstructTask
from distilabel.utils.dataset import prepare_dataset


@pytest.fixture
def custom_dataset():
    ds = CustomDataset.from_dict({"input": ["a", "b"], "generations": ["c", "d"]})
    ds.task = UltraFeedbackTask.for_overall_quality()
    return ds


@pytest.fixture
def sample_preference_dataset():
    ds = CustomDataset.from_dict(
        {
            "input": ["input 1", "input 2", "input 3"],
            "generation_model": [
                [
                    "argilla/notus-7b-v1",
                    "WizardLM/WizardCoder-15B-V1.0",
                    "ise-uiuc/Magicoder-S-DS-6.7B",
                    "gpt-3.5-turbo",
                ],
                [
                    "argilla/notus-7b-v1",
                    "ise-uiuc/Magicoder-S-DS-6.7B",
                    "WizardLM/WizardCoder-15B-V1.0",
                    "gpt-3.5-turbo",
                ],
                [
                    "argilla/notus-7b-v1",
                    "ise-uiuc/Magicoder-S-DS-6.7B",
                    "WizardLM/WizardCoder-15B-V1.0",
                    "gpt-3.5-turbo",
                ],
            ],
            "generations": [
                [
                    "generation 1 1",
                    "generation 1 2",
                    "generation 1 3",
                    "generation 1 4",
                ],
                [
                    "generation 2 1",
                    "generation 2 2",
                    "generation 2 3",
                    "generation 2 4",
                ],
                [
                    "generation 3 1",
                    "generation 3 2",
                    "generation 3 3",
                    "generation 3 4",
                ],
            ],
            "labelling_model": [
                "gpt-4-1106-preview",
                "gpt-4-1106-preview",
                "gpt-4-1106-preview",
            ],
            "labelling_prompt": [
                [
                    {
                        "content": "Your role is to evaluate text quality based on given criteria.",
                        "role": "system",
                    },
                    {"content": "content", "role": "user"},
                ],
                [
                    {
                        "content": "Your role is to evaluate text quality based on given criteria.",
                        "role": "system",
                    },
                    {"content": "content", "role": "user"},
                ],
                [
                    {
                        "content": "Your role is to evaluate text quality based on given criteria.",
                        "role": "system",
                    },
                    {"content": "content", "role": "user"},
                ],
            ],
            "raw_labelling_response": ["response", "response", "response"],
            "rating": [
                [2.0, 5.0, 4.0, 5.0],
                [2.0, 3.0, 1.0, 4.0],
                [4.0, 3.0, 5.0, 3.0],
            ],
            "rationale": [
                ["rationale 1", "rationale 2", "rationale 3", "rationale 4"],
                ["rationale 1", "rationale 2", "rationale 3", "rationale 4"],
                ["rationale 1", "rationale 2", "rationale 3", "rationale 4"],
            ],
        }
    )
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


@pytest.mark.parametrize(
    "with_generation_model",
    [True],
)
@pytest.mark.parametrize(
    "strategy, chosen, rejected, chosen_model, rejected_model, keep_ties",
    [
        (
            "random",
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 2", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 4", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 3", "role": "assistant"},
                ],
            ],
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 1", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 3", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 4", "role": "assistant"},
                ],
            ],
            [
                "WizardLM/WizardCoder-15B-V1.0",
                "gpt-3.5-turbo",
                "WizardLM/WizardCoder-15B-V1.0",
            ],
            ["argilla/notus-7b-v1", "WizardLM/WizardCoder-15B-V1.0", "gpt-3.5-turbo"],
            True,
        ),
        (
            "worst",
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 2", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 4", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 3", "role": "assistant"},
                ],
            ],
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 1", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 3", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 2", "role": "assistant"},
                ],
            ],
            [
                "WizardLM/WizardCoder-15B-V1.0",
                "gpt-3.5-turbo",
                "WizardLM/WizardCoder-15B-V1.0",
            ],
            [
                "argilla/notus-7b-v1",
                "WizardLM/WizardCoder-15B-V1.0",
                "ise-uiuc/Magicoder-S-DS-6.7B",
            ],
            True,
        ),
    ],
)
def test_prepare_dataset(
    sample_preference_dataset: CustomDataset,
    strategy: str,
    chosen: List[str],
    rejected: List[str],
    chosen_model: List[str],
    rejected_model: List[str],
    with_generation_model: bool,
    keep_ties: bool,
):
    if not with_generation_model:
        sample_preference_dataset = sample_preference_dataset.remove_columns(
            ["generation_model"]
        )
    ds = prepare_dataset(
        sample_preference_dataset, strategy=strategy, seed=42, keep_ties=keep_ties
    )
    assert isinstance(ds, CustomDataset)
    assert ds.column_names == [
        "prompt",
        "chosen",
        "rejected",
        "rating_chosen",
        "rating_rejected",
        "chosen_model",
        "rejected_model",
    ]
    for i, row in enumerate(ds):
        assert row["chosen"] == chosen[i]
        assert row["rejected"] == rejected[i]
        assert row["chosen_model"] == chosen_model[i]
        assert row["rejected_model"] == rejected_model[i]


def test_prepare_dataset_wrong_task(sample_preference_dataset: CustomDataset):
    sample_preference_dataset.task = TextGenerationTask()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "This functionality is currently implemented for `PreferenceTask` only."
        ),
    ):
        prepare_dataset(sample_preference_dataset)


def test_dataset_wrong_strategy(sample_preference_dataset: CustomDataset):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Strategy `wrong_strategy` is not implemented, it must be one of: ('random', 'worst')"
        ),
    ):
        prepare_dataset(sample_preference_dataset, strategy="wrong_strategy")


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

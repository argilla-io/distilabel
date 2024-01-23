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

import functools
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, get_args

from distilabel.logger import get_logger

if TYPE_CHECKING:
    from distilabel.dataset import CustomDataset


logger = get_logger()

BinarizationStrategies = Literal["random", "worst"]


def _get_best_response(
    example: Any, rating_column: str = "rating", responses_column: str = "generations"
) -> Tuple[str, int, str, str]:
    """Helper function to get the best response from an example, this can be used
    independent on the method to chose the rejected response.

    Also, it removes the best response from the example.

    Args:
        example (Any): Each row in the dataset as passed when calling the map function on a datasets.Dataset.
        rating_column (str, optional):
            Column containing the rating in the CustomDataset. Defaults to "rating".
        responses_column (str, optional):
            Column containing the responses from a model in a CustomDataset. Defaults to "generations".

    Returns:
        Tuple[str, int, str, str]: Contains the prompt, best rating, chosen response, and chosen model.
    """
    # Pick the highest rating
    prompt = example["input"]
    best_rating = max(example[rating_column])
    best_response_idx = example[rating_column].index(best_rating)
    chosen_response = example[responses_column][best_response_idx]
    chosen_model = example["generation_model"][best_response_idx]

    # Remove best response
    example[rating_column].pop(best_response_idx)
    example[responses_column].pop(best_response_idx)
    example["generation_model"].pop(best_response_idx)
    return prompt, best_rating, chosen_response, chosen_model


def _format_message(prompt: str, response: str) -> List[Dict[str, str]]:
    """Helper function to format the messages (chosen/rejected) in OpenAI format.

    Returns:
        message: List of dictionaries with the OpenAI format.
    """
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _binarize_dataset(
    dataset: "CustomDataset",
    seed: int = None,
    strategy: BinarizationStrategies = "random",
    keep_ties: bool = False,
    rating_column: str = "rating",
    responses_column: str = "generations",
    **kwargs: Any,
) -> "CustomDataset":
    """Binarizes a distilabel dataset.

    Args:
        dataset (CustomDataset): The distilabel dataset to binarize.
        seed (int, optional): Random seed. Defaults to 42.
        strategy (BinarizationStrategies, optional): Method to binarize the data. Defaults to "random".
        keep_ties (bool, optional):
            Whether to keep ties in case the binarization method generated the chosen
            and rejected responses to have the same rating. Defaults to False.
        kwargs: Extra parameters passed to `datasets.Dataset.map`.

    Raises:
        ValueError: If the strategy is not implemented.

    Returns:
        CustomDataset: Dataset binarized.
    """
    get_best_response = functools.partial(
        _get_best_response,
        rating_column=rating_column,
        responses_column=responses_column,
    )

    def binarize_random(example):
        prompt, best_rating, chosen_response, chosen_model = get_best_response(example)
        random.seed(seed)

        # Then you pick the rejected from the list of candidates with lower scores.
        example_lower = defaultdict(list)
        for i, rating in enumerate(example[rating_column]):
            if rating < best_rating:
                example_lower[responses_column].append(example[responses_column][i])
                example_lower[rating_column].append(rating)

        # Otherwise you declare that a tie
        if len(example_lower[rating_column]) == 0:
            # In this case we don't have any response with a lower rating, so we just
            # let the original example (we have a tie)
            example_lower = example

        random_response = random.choice(example_lower[responses_column])
        random_response_idx = example_lower[responses_column].index(random_response)
        random_rating = example_lower[rating_column][random_response_idx]

        random_model = example["generation_model"][random_response_idx]

        return {
            "prompt": prompt,
            "chosen": _format_message(prompt, chosen_response),
            "rejected": _format_message(prompt, random_response),
            "rating_chosen": int(best_rating),
            "rating_rejected": int(random_rating),
            "chosen_model": chosen_model,
            "rejected_model": random_model,
        }

    def binarize_worst(example):
        prompt, best_rating, chosen_response, chosen_model = get_best_response(example)

        worst_rating = min(example[rating_column])
        worst_response_idx = example[rating_column].index(worst_rating)
        worst_response = example[responses_column][worst_response_idx]
        worst_model = example["generation_model"][worst_response_idx]

        return {
            "prompt": prompt,
            "chosen": _format_message(prompt, chosen_response),
            "rejected": _format_message(prompt, worst_response),
            "rating_chosen": int(best_rating),
            "rating_rejected": int(worst_rating),
            "chosen_model": chosen_model,
            "rejected_model": worst_model,
        }

    if strategy == "random":
        binarization_method = binarize_random
    elif strategy == "worst":
        binarization_method = binarize_worst
    else:
        raise ValueError(
            f"Strategy `{strategy}` is not implemented, it must be one of: {get_args(BinarizationStrategies)}"
        )

    if "generation_model" not in dataset.column_names:
        # Ensure generation model is found in the dataset, even if empty, to avoid
        # erros when calling map
        dataset = dataset.add_column(
            "generation_model", [[""] * len(dataset[0]["generations"])] * len(dataset)
        )

    dataset = dataset.map(binarization_method, **kwargs)

    if not keep_ties:
        dataset = dataset.filter(
            lambda example: example["rating_chosen"] != example["rating_rejected"]
        )
    return dataset


def prepare_dataset(
    dataset: "CustomDataset",
    strategy: BinarizationStrategies = "random",
    seed: Optional[int] = None,
    keep_ties: bool = False,
    **kwargs: Any,
) -> "CustomDataset":
    """Helper function to prepare a distilabel dataset for training with the standard formats.

    Currently supports the `PreferenceTask`, and binarizes the responses assuming
    one of two strategies:

    - `random`: Selects the *chosen* response based on the highest rating, and for the
        *rejected* selects a random response from the remaining ones. Filters the examples in which
        the chosen rating is equal to the rejected one.
    - `worst`: Selects the *chosen* response based on the highest rating, and for the
        *rejected* selects the response with the lowest rating. Filters the examples in which the
        chosen rating is equal to the rejected one.

    Take a look at [argilla/ultrafeedback-binarized-preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences)
    for more information on binarizing a dataset to prepare it for DPO fine-tuning.

    Expected format for a dataset to be trained with DPO as defined in trl's
    [dpo trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#expected-dataset-format).

    Note:
        Take a look at the
        [Prepare datasets for fine-tuning](https://distilabel.argilla.io/latest/technical-reference/pipeline/#prepare-datasets-for-fine-tuning)
        section in the Concept guides for more information on the binarization process.

    Args:
        dataset (CustomDataset):
            CustomDataset with a PreferenceTask to prepare for Direct Preference Optimization.
        strategy (BinarizationStrategies, optional):
            Strategy to binarize the data. Defaults to "random".
        seed (int, optional): Seed for the random generator, in case of `random` strategy. Defaults to None.
        keep_ties (bool, optional):
            Whether to keep ties in case the binarization method generated the chosen
            and rejected responses to have the same rating. Defaults to False.
        kwargs: Extra parameters passed to `datasets.Dataset.map`.

    Returns:
        CustomDataset: Dataset formatted for training with DPO.

    Examples:
        >>> from datasets import load_dataset
        >>> from distilabel.tasks import UltraFeedbackTask
        >>> import os
        >>> dataset = load_dataset("argilla/DistiCoder-dpo", token=os.getenv("HF_API_TOKEN"), split="train")
        >>> dataset.task = UltraFeedbackTask.for_instruction_following()
        >>> dataset_binarized = prepare_dataset(dataset, strategy="worst")
    """
    from distilabel.tasks.preference.base import PreferenceTask

    if not isinstance(dataset.task, PreferenceTask):
        raise ValueError(
            "This functionality is currently implemented for `PreferenceTask` only."
        )

    remove_columns = [
        "input",
        "generation_model",
        "generations",
        "rating",
        "labelling_model",
        "labelling_prompt",
        "raw_labelling_response",
        "rationale",
    ]
    # Remove the rows for which there is no rating
    initial_length = len(dataset)
    dataset = dataset.filter(lambda example: example["rating"])
    if len(dataset) != initial_length:
        logger.info(
            f"Found {initial_length - len(dataset)} examples with no rating, removing them."
        )

    if len(dataset[0]["generations"]) < 2:
        raise ValueError("The dataset must contain at least 2 generations per example.")

    ds = _binarize_dataset(
        dataset,
        strategy=strategy,
        seed=seed,
        keep_ties=keep_ties,
        rating_column="rating",
        responses_column="generations",
        **kwargs,
    )

    # Imported here to avoid circular imports
    from distilabel.dataset import CustomDataset

    ds = ds.remove_columns(remove_columns)
    ds.__class__ = CustomDataset
    ds.task = dataset.task
    return ds

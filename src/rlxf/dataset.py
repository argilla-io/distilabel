from typing import TYPE_CHECKING, Union

from datasets import Dataset

if TYPE_CHECKING:
    from rlxf.prompts.integrations.argilla import ArgillaTemplate

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False


class CustomDataset(Dataset):
    prompt_template: Union["ArgillaTemplate", None] = None

    def to_argilla(self) -> None:
        if _argilla_installed is False:
            raise ImportError(
                "The argilla library is not installed. Please install it with `pip install argilla`."
            )
        if self.prompt_template is None:
            raise ValueError(
                "The prompt template is not set. Please set it with `dataset.prompt_template = <prompt_template>`."
            )

        rg_dataset = rg.FeedbackDataset(
            fields=self.prompt_template.to_argilla_fields(dataset_row=self[0]),
            questions=self.prompt_template.to_argilla_questions(dataset_row=self[0]),
        )
        rg_dataset.add_records(
            [
                self.prompt_template.to_argilla_record(dataset_row=dataset_row)
                for dataset_row in self
            ]
        )
        return rg_dataset

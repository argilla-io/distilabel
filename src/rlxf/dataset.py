from datasets import Dataset

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False


class CustomDataset(Dataset):
    argilla_fields: list
    argilla_input_args: dict  # TODO: mapping from HF rows to Argilla fields
    argilla_questions: list
    argilla_output_args: list  # TODO: from HF row values with index to responses (questions in argilla)

    def to_argilla(self) -> None:
        if _argilla_installed is False:
            raise ImportError(
                "The argilla library is not installed. Please install it with `pip install argilla`."
            )
        rg_dataset = rg.FeedbackDataset(
            fields=self.argilla_fields,
            questions=self.argilla_questions,
            guidelines="These are the guidelines",
        )

        for item in self:
            fields = {}
            for input_arg_key, input_arg_value in self.argilla_input_args.items():
                if isinstance(input_arg_value, str):
                    fields.update({input_arg_key: item[input_arg_value]})
                elif isinstance(input_arg_value, dict):
                    for input_arg_subkey, input_arg_subvalue in input_arg_value.items():
                        fields.update(
                            {input_arg_subvalue: item[input_arg_key][input_arg_subkey]}
                        )
            response_values = {}
            for output_arg_key, output_arg_value in self.argilla_output_args.items():
                for output_arg_subkey, output_arg_subvalue in output_arg_value.items():
                    response_values.update(
                        {
                            output_arg_subvalue: {
                                "value": item[output_arg_key][output_arg_subkey]
                            }
                        }
                    )
            rg_dataset.add_records(
                [
                    rg.FeedbackRecord(
                        fields=fields,
                        responses=[{"values": response_values, "status": "submitted"}],
                    )
                ]
            )
        return rg_dataset

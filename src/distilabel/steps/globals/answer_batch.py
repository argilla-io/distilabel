import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from typing_extensions import override

from distilabel.steps import GlobalStep, StepOutput
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepOutput

from openai import OpenAI
from openai.types.batch import Batch


class AnswerBatch(GlobalStep, ABC):
    """Use the OpenAI Batch API to process a large amount of requests asynchronously.
    Requests may take up to 24 hours to complete. See https://platform.openai.com/docs/guides/batch/getting-started for more info on the batch API.
    This Step behaves exactly like a task would except you set the model_id and generation_kwargs as separate attributes instead of in the LLM definition.

    It works by sending a batch request to the API and then polling for maximum 24 hours until the batch job is completed. The results are then processed and returned.
    """

    model_id: str = "gpt-4o"
    generation_kwargs: Dict[str, Any] = {"max_tokens": 4096, "temperature": 0.7}
    openai_client_kwargs: Dict[str, Any] = {}
    batch_description: Optional[str] = "Processing Distilabel Step"
    continue_batch: bool = False
    poll_every_n_seconds: int = 10
    batch_id: Optional[str] = None
    batch_metadata: Dict[str, Any] = {}
    input_batch_size: int = 10000000  # process all documents in one batch
    _client: OpenAI = None

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Abstract method to format the inputs of the task. It needs to receive an input
        as a Python dictionary, and generates an OpenAI chat-like list of dicts."""
        pass

    def _format_inputs(self, inputs: List[Dict[str, Any]]) -> List["ChatType"]:
        """Formats the inputs of the task using the `format_input` method.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list containing the formatted inputs, which are `ChatType`-like following
            the OpenAI formatting.
        """
        return [self.format_input(input) for input in inputs]

    @override
    def load(self) -> None:
        super().load()
        self._client = OpenAI(**self.openai_client_kwargs)
        if self.continue_batch:
            if self.batch_id is None:
                raise ValueError(
                    "continue_batch is set to True but no batch ID was provided."
                )
            self._logger.info(f"Continuing batch job with ID: {self.batch_id}")

    def _check_batch_status(self) -> Tuple[Optional[str], Batch]:
        try:
            batch = self._client.batches.retrieve(self.batch_id)
            batch_status = batch.status
            self._logger.info(f"Batch job status: {batch_status}")
        except Exception as e:
            self._logger.error(f"Batch job not found. Error: {e}")
            batch_status = None
        return batch_status, batch

    def _create_batch(self, formatted_inputs: List["ChatType"]) -> str:
        self._logger.debug("Creating batch job.")
        formatted_docs = []
        for i, input in enumerate(formatted_inputs, start=1):
            formatted_doc = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "messages": input,
                    **self.generation_kwargs,
                },
            }
            formatted_docs.append(json.dumps(formatted_doc))
        jsonlines = "\n".join(formatted_docs)
        self._logger.info("Uploading batch file to OpenAI API...")
        batch_input_file = self._client.files.create(
            file=jsonlines.encode("utf-8"), purpose="batch"
        )
        self._logger.info(
            f"Batch file uploaded to OpenAI API. ID: {batch_input_file.id}"
        )

        self._logger.info("Creating batch job...")
        batch_input_file_id = batch_input_file.id
        batch = self._client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": self.batch_description, **self.batch_metadata},
        )
        self._logger.info(f"Batch job with ID {batch.id} created.")
        return batch.id

    def _process_results(
        self, content: str, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        self._logger.debug("Processing batch results.")
        outputs = [json.loads(line) for line in content.split("\n") if line]
        # sort outputs by custom_id to match to input index. This is necessary because the order of the outputs is not guaranteed.
        outputs = sorted(outputs, key=lambda x: x["custom_id"])
        formatted_outputs = []
        for i, (input, output) in enumerate(zip(inputs, outputs), start=1):
            assert input["custom_id"] == f"request-{i}"
            output_message = output["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            formatted_outputs.append(
                {
                    **input,
                    **self.format_output(output_message, input),
                    "model_name": self.model_id,
                }
            )
        return formatted_outputs

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the OpenAI Batch API.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            A list of Python dictionaries with the outputs of the task.
        """
        self._logger.info("Starting batch processing.")
        formatted_inputs = self._format_inputs(inputs)

        if not self.continue_batch:
            self.batch_id = self._create_batch(formatted_inputs)

        polls_per_24h = int(24 * 60 * 60 / self.poll_every_n_seconds)
        for i in range(int(polls_per_24h)):
            self._logger.debug(f"Polling batch job status, attempt {i + 1}.")
            batch_status, batch = self._check_batch_status()

            if batch_status == "completed":
                self._logger.info("Batch job completed successfully.")

                # Get the contents of the output file (jsonlines format)
                content = self._client.files.content(
                    batch.output_file_id
                ).content.decode("utf-8")
                yield self._process_results(content, inputs)
                return
            elif batch_status in ["in_progress", "finalizing", "validating"]:
                # batch is still running
                remaining_time_seconds = (polls_per_24h - i) * self.poll_every_n_seconds
                remaining_hours = remaining_time_seconds // 3600
                remaining_minutes = (remaining_time_seconds % 3600) // 60
                self._logger.info(
                    f"Batch job is still running (status: {batch_status}). Time left: {remaining_hours}:{remaining_minutes:2f} hours."
                )
            elif batch_status in ["failed", "expired", "cancelled", "cancelling"]:
                # batch failed, expired, cancelled, or is being cancelled
                raise Exception(
                    f"Batch job failed (status: {batch_status}). Please check the OpenAI API for more information."
                )
            elif batch_status is None:
                raise Exception(f"Batch job not found with ID: {self.batch_id}.")
            time.sleep(self.poll_every_n_seconds)

    @abstractmethod
    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Abstract method to format the outputs of the task. It needs to receive an output
        as a string, and generates a Python dictionary with the outputs of the task. In
        addition the `input` used to generate the output is also received just in case it's
        needed to be able to parse the output correctly.
        """
        pass

    def _format_outputs(
        self, outputs: "GenerateOutput", inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Formats the outputs of the task using the `format_output` method. If the output
        is `None` (i.e. the LLM failed to generate a response), then the outputs will be
        set to `None` as well.

        Args:
            outputs: The outputs of the LLM.
            inputs: The inputs used to generate the outputs.

        Returns:
            A list containing a dictionary with the outputs of the task for each input.
        """
        formatted_outputs = []
        for output, input in zip(outputs, inputs * len(outputs)):
            try:
                formatted_outputs.append(self.format_output(output, input))
            except Exception as e:
                self._logger.warning(  # type: ignore
                    f"Task '{self.name}' failed to format output: {e}. Using empty dict."  # type: ignore
                )
                formatted_outputs.append(self._outputs_empty_dict())
        return formatted_outputs

    def _outputs_empty_dict(self) -> Dict[str, None]:
        """Returns a dictionary with the outputs of the task set to `None`."""
        return {output: None for output in self.outputs}  # type: ignore

from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Any, List, Union, TYPE_CHECKING

from pydantic import Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps import GeneratorStep, GeneratorStepOutput

if TYPE_CHECKING:
    from argilla import Argilla


class EventsListener:
    """This class is a wrapper around the Argilla webhook listener that allows for messages to be retrieved from a queue."""

    def __init__(self, events: List[str], client: Union["Argilla", None] = None):
        """
        Initialize the EventsListener.

        Args:
            events: The list of events to listen for.
            client: The Argilla client to use. Defaults to the default client.
        """
        self.queue = Queue()

        from argilla.webhooks import webhook_listener

        @webhook_listener(events=events, client=client)
        async def events_handler(**kwargs):
            self.put_message(kwargs)

    def get_message(self) -> Optional[Any]:
        """Get a message from the queue, or return None if the queue is empty."""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    def put_message(self, message: Any):
        """Put a message into the queue."""
        self.queue.put(message)


class ArgillaListenerStep(GeneratorStep):
    """
    A step that listens for events from the Argilla webhook listener.

    Attributes:

        events (List[str]): The list of events to listen for.
        batch_timeout (int): The maximum number of seconds to wait for a batch to be filled before
        yielding the current batch. Defaults to 10 seconds.

    """
    events: RuntimeParameter[List[str]] = Field(description="The list of events to listen for")

    batch_timeout: RuntimeParameter[int] = Field(
        default=10,
        description="The maximum number of seconds to wait for a batch to be filled before yielding the current batch",
    )

    _listener: EventsListener = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        self._listener = EventsListener(self.events)

    def load(self) -> None:
        """Start the Argilla webhook server."""
        from argilla.webhooks import start_webhook_server

        start_webhook_server()

    def unload(self) -> None:
        """Stop the Argilla webhook server."""
        from argilla.webhooks import stop_webhook_server

        stop_webhook_server()

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        if offset > 0:
            self.logger.warning(f"Offset of {offset} is not supported by ArgillaListener")

        current_batch = []
        last_delivery_time = datetime.utcnow()
        while True:
            # Wait for a message from the Argilla listener
            message = self._listener.get_message()

            if message is not None:
                current_batch.append(message)

            elapsed_time = (datetime.utcnow() - last_delivery_time).total_seconds()

            if len(current_batch) >= self.batch_size or elapsed_time >= self.batch_timeout:
                if current_batch:
                    yield current_batch, False

                current_batch = []
                last_delivery_time = datetime.utcnow()


if __name__ == "__main__":

    listener = ArgillaListenerStep(batch_size=50, events=["record.created"])
    listener.load()
    for batch, end in listener.process():
        print(len(batch))

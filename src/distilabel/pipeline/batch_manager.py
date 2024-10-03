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

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from distilabel.constants import (
    RECEIVES_ROUTED_BATCHES_ATTR_NAME,
    STEP_ATTR_NAME,
)
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.batch import _Batch
from distilabel.steps.base import _Step
from distilabel.utils.files import list_files_in_dir
from distilabel.utils.serialization import (
    StrOrPath,
    _check_is_dir,
    _Serializable,
    read_json,
)

if TYPE_CHECKING:
    from distilabel.utils.serialization import StrOrPath


@dataclass
class _BatchManagerStep(_Serializable):
    """A class that will accumulate data for a step from the predecessors and create
    batches for the step to process when there is enough data.

    Attributes:
        step_name: The name of the step that will process the data.
        accumulate: A flag to indicate if the data should be accumulated and create a
            batch with all the data received from the predecessors instead of creating
            batches with the `input_batch_size`.
        input_batch_size: The size of the batch to be created for the step to process.
            If `None`, then `accumulate` must be `True`. Defaults to `None`.
        data: A dictionary with the predecessor step name as the key and a list of
            dictionaries (rows) as the value.
        built_batches: A list with the batches that were built and sent to the step queue,
            but the step was stopped before processing the batch, so the batch doesn't get
            lost. Defaults to an empty list.
        seq_no: The sequence number of the next batch to be created. It will be
            incremented for each batch created.
        last_batch_received: A list with the names of the steps that sent the last
            batch of data.
        convergence_step: A flag to indicate if the step is a convergence step. An
            `Step` is a convergence step if all its predecessors are receiving routed
            batches. Defaults to `False`.
        convergence_step_batches_consumed: A dictionary in which the key is the `seq_no`
            of the batch created by step A, that was used by step B and C and obtained from
            the `created_from` of the batches created by them. It's used to know if all
            the batches from B and C steps created from batches of A have been consumed
            by D, in order to not mess up the order of the batches. Only used if `convergence_step=True`.
            Defaults to an empty dictionary.
        next_expected_created_from_batch_seq_no: The next expected sequence number of the
            batch from step A used by steps B and C and obtained from the `created_from`
            of the batches created by them. It's used to avoid messing up the order of the
            batches. Only used if `convergence_step=True`. Defaults to `0`.
        step_signature: The signature that defines a given `Step`. It will be used for the
            caching mechanism.
        use_cache: Flag from the original `Step` to indicate whether this step should make use of
            the cached data.
        step_offset: Dictionary with each key the predecessor/s step/s and as value a dict
            with keys `batch` and `offset`, containing the name of the file for the corresponding
            batch, and the number of rows that were read from that step, respectively. Used
            for caching mechanism.
    """

    step_name: str
    accumulate: bool
    input_batch_size: Union[int, None] = None
    data: Dict[str, List[_Batch]] = field(default_factory=dict)
    built_batches: List[_Batch] = field(default_factory=list)
    seq_no: int = 0
    last_batch_received: List[str] = field(default_factory=list)
    convergence_step: bool = False
    convergence_step_batches_consumed: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    next_expected_created_from_batch_seq_no: int = 0
    next_expected_seq_no: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    step_signature: Optional[str] = None
    use_cache: bool = False
    step_offset: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def add_batch(self, batch: _Batch, prepend: bool = False) -> None:
        """Add a batch of data from `batch.step_name` to the step. It will accumulate the
        data and keep track of the last batch received from the predecessors.

        Args:
            batch: The output batch of an step to be processed by the step.
            prepend: If `True`, the content of the batch will be added to the `built_batches`
                list. This is done so if a `_Batch` was already built and send to the step
                queue, and the step is stopped before processing the batch, the batch doesn't
                get lost. Defaults to `False`.
        """
        from_step = batch.step_name

        if prepend:
            self.built_batches.append(batch)
        else:
            self.data[from_step].append(batch)
            self.data[from_step].sort(key=lambda batch: batch.seq_no)

        if batch.last_batch:
            self.last_batch_received.append(from_step)

    def get_batch(self) -> Union[_Batch, None]:
        """Create a new batch of data for the step to process. It will return `None` if
        there is not enough data to create a batch.

        Returns:
            A `_Batch` instance if there is enough data to create a batch. Otherwise,
            `None`.
        """
        # If there are batches in the `built_batches` list, then return the first one
        # and remove it from the list.
        if self.built_batches:
            return self.built_batches.pop(0)

        if not self._ready_to_create_batch():
            return None

        seq_no = self._get_seq_no()

        # `_last_batch` must be called before `_get_data`, as `_get_data` will update the
        # list of data which is used to determine if the batch to be created is the last one.
        last_batch = self._last_batch()

        # Get the batch data and the information from which batches of the upstream steps
        # the data was taken.
        data, created_from, batch_routed_to = self._get_data()

        # Update the step offset i.e. which is the last batch and last row index from that
        # batch that the step has consumed
        self._update_offset(created_from)

        return _Batch(
            seq_no=seq_no,
            step_name=self.step_name,
            last_batch=last_batch,
            data=data,
            accumulated=self.accumulate,
            created_from=created_from,
            batch_routed_to=batch_routed_to,
        )

    def empty_buffers(self) -> List[str]:
        """Checks if the input buffer for the step is empty.

        Returns:
            The name of the previous steps for which the input buffer for this step is
            empty.
        """
        if self.accumulate:
            return [
                previous_step
                for previous_step in self.data.keys()
                if previous_step not in self.last_batch_received
            ]

        return [
            previous_step
            for previous_step, batches in self.data.items()
            if previous_step not in self.last_batch_received
            and sum(len(batch.data[0]) for batch in batches) < self.input_batch_size  # type: ignore
        ]

    def set_next_expected_seq_no(
        self, from_step: str, next_expected_seq_no: int
    ) -> None:
        """Sets the next expected sequence number of a `_Batch` received by the step coming
        from `from_step`.

        Args:
            from_step: The name of the step from which its next expected sequence number
                in step has to be updated.
            next_expected_seq_no: the next expected sequence number of a `_Batch` coming
                from `from_step`.
        """

        if not self.data[from_step] or (
            self.data[from_step]
            and self.data[from_step][0].seq_no >= next_expected_seq_no
        ):
            self.next_expected_seq_no[from_step] = (
                next_expected_seq_no,
                next_expected_seq_no,
            )
        else:
            self.next_expected_seq_no[from_step] = (
                self.next_expected_seq_no[from_step][0],
                next_expected_seq_no,
            )

    @classmethod
    def from_step(
        cls, step: "_Step", predecessors: Iterable[str], convergence_step: bool = False
    ) -> "_BatchManagerStep":
        """Creates a `_BatchManagerStep` instance from a `_Step` instance and its
        predecessors.

        Args:
            step: The `_Step` instance.
            predecessors: The names of the predecessors of the step.
            convergence_step: A flag to indicate if the step is a convergence step. An
                `Step` is a convergence step if all its predecessors are receiving routed
                batches. Defaults to `False`.

        Returns:
            A `_BatchManagerStep` instance.
        """
        return cls(
            step_name=step.name,  # type: ignore
            accumulate=step.is_global,
            input_batch_size=getattr(step, "input_batch_size", None),
            data={predecessor: [] for predecessor in predecessors},
            convergence_step=convergence_step,
            next_expected_seq_no={predecessor: (0, 0) for predecessor in predecessors},
            step_signature=step.signature,
            use_cache=step.use_cache,
            step_offset={predecessor: (0, 0) for predecessor in predecessors},
        )

    def _get_seq_no(self) -> int:
        """Gets the sequence number for the next batch to be created and increments it.

        Returns:
            The sequence number for the next batch to be created.
        """
        seq_no = self.seq_no
        self.seq_no += 1
        return seq_no

    def _get_data(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]], List[str]]:
        """Gets the data needed to create a batch for the step to process. If the step is
        accumulating data, then it will return a list with all the data received from the
        predecessors. Otherwise, it will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the
        batch from the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process, a dictionary with the sequence numbers of the batches that were used
            to create the batch and the list of steps to which the batch was routed to if
            the step is a normal step.
        """
        if self.accumulate:
            # Steps accumulating cannot receive routed batches
            return self._get_data_for_accumulate() + ([],)

        if self.convergence_step:
            # Convergence steps will receive routed batches, but we need to clean the
            # `batch_routed_to` list
            return self._get_data_for_convergence_step() + ([],)

        return self._get_data_normal()

    def _get_data_for_accumulate(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]]]:
        """Gets the data needed to create a batch for the step to process when the step
        is accumulating data. It will return a list with all the data received from the
        predecessors. In addition, it will remove the data used to create the batch from
        the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process and a dictionary with the sequence numbers of the batches that were
            used to create the batch.
        """
        data = []
        batches_used = {}
        for step_name, batches in self.data.items():
            batches_used[step_name] = []
            for batch in batches:
                batches_used[step_name].append((batch.seq_no, batch.size))
            data.append([row for batch in batches for row in batch.get_data()])
        # Reset the data buffer
        self.data = {step_name: [] for step_name in self.data}
        return data, batches_used

    def _get_data_for_convergence_step(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]]]:
        """Gets the data needed to create a batch for the step to process when the step is
        a convergence step.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process and a dictionary with the sequence numbers of the batches that were
            used to create the batch.
        """
        grouped_batches = self._group_batches_by_created_from()
        seq_no, batches = grouped_batches[0]
        str_seq_no = str(seq_no)

        remaining_rows_per_step: Dict[str, int] = {
            step_name: self.input_batch_size
            for step_name in self.data  # type: ignore
        }
        batches_used = defaultdict(list)
        data = defaultdict(list)
        for batch, batch_size in batches:
            remaining_rows = remaining_rows_per_step[batch.step_name]
            selected_data = batch.get_data(remaining_rows)
            data[batch.step_name].extend(selected_data)

            # If A -> [B, C] -> D, then in D (this step) we keep track of the remaining
            # rows from the batches of A that B and C used to create the `batches`.
            batch_size = self.convergence_step_batches_consumed.setdefault(
                str_seq_no, {}
            ).get(batch.step_name, batch_size)
            remaining_rows_in_batch = batch_size - len(selected_data)
            self.convergence_step_batches_consumed[str_seq_no].update(
                {batch.step_name: remaining_rows_in_batch}
            )

            # Update the remaining rows
            num_rows = len(selected_data)
            remaining_rows_per_step[batch.step_name] -= num_rows  # type: ignore

            # Keep track of the batches used to create the batch
            batches_used[batch.step_name].append((batch.seq_no, len(selected_data)))

            # If the batch was entirely consumed, then remove it from the buffer
            if len(batch.data[0]) == 0:
                self.data[batch.step_name].remove(batch)
                continue

        # If all the batches grouped by the `seq_no` in `created_from` were consumed, then
        # we can update the `next_expected_created_from_batch_seq_no` to the next one
        # to avoid skipping batches.
        no_remaining_rows = all(
            count == 0
            for count in self.convergence_step_batches_consumed[str_seq_no].values()
        )
        if no_remaining_rows:
            self.next_expected_created_from_batch_seq_no += 1

        return list(data.values()), dict(batches_used)

    def _get_data_normal(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]], List[str]]:
        """Gets the data needed to create a batch for the step to process when the step is
        not accumulating data. It will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the batch
        from the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process, a dictionary with the sequence numbers of the batches that were used
            to create the batch and the list of steps to which the batch was routed to if
            the step is a convergence step.
        """
        data = []
        batches_used = defaultdict(list)
        batch_routed_to = []
        for step_name in self.data:
            # For each step batches buffer, we will create a batch with the `input_batch_size`
            # using the data from the buffer. We will remove the consumed batches (no data
            # left) and update the batch data with the remaining data.
            step_data = []
            idx_drop_batches = []
            remaining_rows: int = self.input_batch_size  # type: ignore
            next_expected_seq_no = None
            for idx, batch in enumerate(self.data[step_name]):
                if remaining_rows == 0:
                    break

                # Get `remaining_rows` or the remaining rows in the batch and add it to
                # the step data that will be used to create the batch
                selected_data = batch.get_data(remaining_rows)
                step_data.extend(selected_data)
                batch_routed_to = batch.batch_routed_to

                # Update the remaining rows
                num_rows = len(selected_data)
                remaining_rows -= num_rows

                # Keep track of the batches used to create the batch
                batches_used[step_name].append((batch.seq_no, len(selected_data)))

                next_expected_seq_no = batch.seq_no

                # If the batch was entirely consumed, then remove it from the buffer
                if len(batch.data[0]) == 0:
                    next_expected_seq_no += 1
                    idx_drop_batches.append(idx)
                    continue

            # Remove the batches that were entirely consumed
            idx_drop_batches.reverse()
            for idx in idx_drop_batches:
                self.data[step_name].pop(idx)

            # Update the `next_expected_seq_no` from `step_name`. It can happen that:
            # 1. This step didn't receive one batch because it was routed to other batches
            # and `set_next_expected_seq_no` method was called. If the first element
            # is not equal to the second, that means there is a potential `next_expected_seq_no`
            # from `step_name`. If there is no data left, then we set that as the `next_expected_seq_no`.
            # 2. `set_next_expected_seq_no` has not been called, so we set the `next_expected_seq_no`
            # taking into account the data left in the step.
            step_next_expected_seq_no = self.next_expected_seq_no[step_name]
            if step_next_expected_seq_no[0] != step_next_expected_seq_no[1] and (
                not self.data[step_name]
                or (
                    self.data[step_name]
                    and self.data[step_name][0].seq_no >= step_next_expected_seq_no[1]
                )
            ):
                self.next_expected_seq_no[step_name] = (
                    step_next_expected_seq_no[1],
                    step_next_expected_seq_no[1],
                )
            elif next_expected_seq_no:
                self.next_expected_seq_no[step_name] = (
                    next_expected_seq_no,
                    next_expected_seq_no
                    if next_expected_seq_no > step_next_expected_seq_no[1]
                    else step_next_expected_seq_no[1],
                )

            data.append(step_data)

        return data, dict(batches_used), batch_routed_to

    def _ready_to_create_batch(self) -> bool:
        """Checks if there is enough data to create a batch for the step.

        Returns:
            `True` if there is enough data to create a batch for the step. Otherwise,
            `False`.
        """

        if self.accumulate:
            return self._ready_to_create_batch_accumulate()

        if self.convergence_step:
            return self._ready_to_create_batch_convergence_step()

        return self._ready_to_create_batch_normal()

    def _ready_to_create_batch_accumulate(self) -> bool:
        """Checks if there is enough data for an step accumulating data. It will return
        `True` if the last batch was received from all the predecessors.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        return all(
            step in self.last_batch_received
            and sum(len(batch.data[0]) for batch in batches) > 0
            for step, batches in self.data.items()
        )

    def _ready_to_create_batch_convergence_step(self) -> bool:
        """Checks if there is enough data for creating a batch for an step in which output
        batches that were generated by steps that received routed batches are received.
        It will return `True`, if all the output batches that were generated from a routed
        batch have been received.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        grouped_batches = self._group_batches_by_created_from()
        if not grouped_batches:
            return False
        seq_no, batches = grouped_batches[0]

        # If the `seq_no` from the `created_from` field is not the expected one, then
        # we cannot create a batch yet or the order will be messed up
        if seq_no != self.next_expected_created_from_batch_seq_no:
            return False

        # Not all output batches to which the input batch was routed to haven't been
        # received
        batch_routed_to = batches[0][0].batch_routed_to
        batches_received_from = {batch.step_name for batch, _ in batches}
        if any(step_name not in batches_received_from for step_name in batch_routed_to):
            return False

        # There are output batches to which the input batch was routed to from all
        # the steps. Check if there is enough data for creating a batch with `input_batch_size`
        rows_per_step = defaultdict(lambda: 0)
        for batch, _ in batches:
            num_rows = len(batch.data[0])
            rows_per_step[batch.step_name] += num_rows

        # If there aren't at least `input_batch_size` rows from each step, then there
        # isn't enough data to create a batch
        if not all(
            num_rows >= self.input_batch_size or step_name in self.last_batch_received  # type: ignore
            for step_name, num_rows in rows_per_step.items()
        ):
            return False

        return True

    def _ready_to_create_batch_normal(self) -> bool:
        """Checks if there is enough data for creating a batch for a normal step. It will
        be `True` it there are at least `input_batch_size` rows from each predecessor step.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        for step_name, batches in self.data.items():
            # Depending on the number of replicas of the `Step` it can happen that some
            # replica is faster and send batch with `seq_no==1` faster than the other that
            # sends the batch with `seq_no==0`. We need to check which `seq_no` was expected
            # next to not mess up the ordering of the rows.
            next_expected_seq_no = self.next_expected_seq_no[step_name][0]

            # `batches` are sorted by `seq_no`
            num_rows = 0
            is_batch_in_order = True
            for batch in batches:
                # Need to create batches using the data from batches with sequential `seq_no`
                if batch.seq_no != next_expected_seq_no:
                    is_batch_in_order = False
                    break
                # There are enough rows to create a batch
                num_rows += len(batch.data[0])
                if self.input_batch_size and num_rows >= self.input_batch_size:
                    break
                next_expected_seq_no += 1

            # If there are now rows but the last batch was already received, then there
            # are no more batches to be created
            if num_rows == 0 and step_name in self.last_batch_received:
                return False

            # If there are not enough rows and the last batch was not received yet, then
            # there is not enough data yet to create a batch
            # If the last batch was received, the batch preceding it must be in order
            if (
                self.input_batch_size
                and num_rows < self.input_batch_size
                and not (step_name in self.last_batch_received and is_batch_in_order)
            ):
                return False

        return True

    def _last_batch(self) -> bool:
        """Checks if the batch to be created is the last one i.e. if the last batch was
        received from all the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        if self.accumulate:
            return self._last_batch_accumulate()

        if self.convergence_step:
            return self._last_batch_convergence_step()

        return self._last_batch_normal()

    def _update_offset(self, created_from: Dict[str, List[Tuple[int, int]]]) -> None:
        """Update the offset for the batch buffers of the upstream steps.

        Args:
            created_from: a dictionary containing the names of the steps and the batches
                that were used to create the batch for this step. The key is the name of
                one step and the value is a list of tuples in which the first value is the
                seq no of the batch used and the second value is the number of rows from
                that batch used.
        """
        for predecessor, seq_no_and_batch in created_from.items():
            prev_last_batch_seq_no, prev_last_batch_offset = self.step_offset[
                predecessor
            ]
            last_batch_seq_no, last_batch_size = seq_no_and_batch[-1]
            batch_offset = (
                prev_last_batch_offset + last_batch_size
                if prev_last_batch_seq_no == last_batch_seq_no
                else last_batch_size
            )
            last_batch_seq_no = (
                last_batch_seq_no
                if last_batch_seq_no > prev_last_batch_seq_no
                else prev_last_batch_seq_no
            )
            self.step_offset[predecessor] = (last_batch_seq_no, batch_offset)

    def _last_batch_accumulate(self) -> bool:
        """Checks if the batch to be created is the last one for an step accumulating data.
        `True` if the last batch was received from all the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        return all(step in self.last_batch_received for step in self.data.keys())

    def _last_batch_convergence_step(self) -> bool:
        """Checks if the batch to be created is the last one for a convergence step. `True`
        if the last batch of all the steps (`batch_routed_to`) in the last routed batch
        have been received.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        grouped_batches = self._group_batches_by_created_from()
        if not grouped_batches:
            return False
        _, batches = grouped_batches[0]

        for batch, _ in batches:
            if not batch.last_batch:
                return False

            if len(batch.data[0]) > self.input_batch_size:  # type: ignore
                return False

        return True

    def _last_batch_normal(self) -> bool:
        """Checks if the batch to be created is the last one for a normal step. `True` if
        there is no more data to be received from the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        for step_name, batches in self.data.items():
            if step_name not in self.last_batch_received:
                return False

            num_rows = sum(len(batch.data[0]) for batch in batches)

            if self.input_batch_size and num_rows > self.input_batch_size:
                return False

        return True

    def _group_batches_by_created_from(
        self,
    ) -> List[Tuple[int, List[Tuple["_Batch", int]]]]:
        """Group the batches by the first key of `created_from` field. This method is
        meant to be used only with a `convergence_step`.

        Returns:
            A list of the batches grouped by the `seq_no` of the first step name in `created_from`.
            The list is sorted by the `seq_no`.
        """
        grouped_batches: Dict[int, List[Tuple["_Batch", int]]] = defaultdict(list)
        for batches in self.data.values():
            for batch in batches:
                first_key = next(iter(batch.created_from))
                batch_seq_no, batch_size = batch.created_from[first_key][0]
                grouped_batches[batch_seq_no].append((batch, batch_size))
        return sorted((seq_no, batches) for seq_no, batches in grouped_batches.items())

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManagerStep` to a dictionary.

        Args:
            obj: Unused, just kept to match the signature of the parent method.
            kwargs: Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Internal representation of the `_BatchManagerStep`.
        """
        return {
            "step_name": self.step_name,
            "accumulate": self.accumulate,
            "input_batch_size": self.input_batch_size,
            "data": {
                step_name: [batch.dump(**kwargs) for batch in batches]
                for step_name, batches in self.data.items()
            },
            "built_batches": [batch.dump(**kwargs) for batch in self.built_batches],
            "seq_no": self.seq_no,
            "last_batch_received": self.last_batch_received,
            "convergence_step": self.convergence_step,
            "convergence_step_batches_consumed": self.convergence_step_batches_consumed,
            "next_expected_created_from_batch_seq_no": self.next_expected_created_from_batch_seq_no,
            "next_expected_seq_no": self.next_expected_seq_no,
            "step_signature": self.step_signature,
            "use_cache": self.use_cache,
            "step_offset": self.step_offset,
        }

    @property
    def signature(self) -> str:
        return f"{self.step_name}_{self.step_signature}"


class _BatchManager(_Serializable):
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns new batches for the steps to process based on their
    input batch size and the batches received from the predecessors.

    Attributes:
        steps: A dictionary with the step name as the key and a `_BatchManagerStep`
            instance as the value.
        last_batch_received: A dictionary with the step name as the key and a flag to
            indicate whether we received the last batch from the step.
    """

    def __init__(
        self,
        steps: Dict[str, _BatchManagerStep],
        last_batch_received: Dict[str, Union[_Batch, None]],
        last_batch_sent: Dict[str, Union[_Batch, None]],
        last_batch_flag_sent_to: List[str],
    ) -> None:
        """Initialize the `_BatchManager` instance.

        Args:
            steps: A dictionary with the step name as the key and a dictionary with the
                predecessor step name as the key and a list of batches as the value.
            last_batch_received: A dictionary with the step name as the key and the last
                `_Batch` received from the step.
            last_batch_sent: A dictionary with the step name as the key and the last
                `_Batch` sent to the step.
            last_batch_flag_sent_to: A list with the names of the steps to which `LAST_BATCH_SENT_FLAG`
                was sent.
        """

        self._steps = steps
        self._last_batch_received = last_batch_received
        self._last_batch_sent = last_batch_sent
        self._last_batch_flag_sent_to = last_batch_flag_sent_to

    def can_generate(self) -> bool:
        """Checks if there are still batches to be processed by the steps.

        Returns:
            `True` if there are still batches to be processed by the steps. Otherwise,
            `False`.
        """
        for step_name, batch in self._last_batch_received.items():
            if step_name not in self._last_batch_flag_sent_to:
                if not batch:
                    return True

                if not batch.last_batch:
                    return True

                if not self.get_last_batch_sent(step_name):
                    return True

        return False

    def register_batch(
        self, batch: _Batch, steps_data_path: Optional["StrOrPath"] = None
    ) -> None:
        """Method to register a batch received from a step. It will keep track of the
        sequence number and the last batch received from the step in the internal maps.

        Args:
            batch: _Batch from which we will register the sequence number and the last batch received.
            steps_data_path: The path where the outputs of each `Step` (considering its
                signature) will be saved for later reuse in another pipelines executions.
        """
        last_batch = self._last_batch_received[batch.step_name]
        if not last_batch or (last_batch and last_batch.seq_no < batch.seq_no):
            self._last_batch_received[batch.step_name] = batch

        if steps_data_path:
            self.write_batch_data(batch, steps_data_path)

    def write_batch_data(self, batch: _Batch, steps_data_path: Path) -> None:
        """Writes the batch to the steps data directory.

        Argument:
            batch: the batch to be written.
            steps_data_path: the steps data base directory.
        """
        step = self._steps[batch.step_name]
        batch_manager_data_dir = Path(steps_data_path) / step.signature
        batch_manager_data_dir.mkdir(parents=True, exist_ok=True)
        filename = batch_manager_data_dir / f"batch_{batch.seq_no}.json"
        if not filename.exists():
            self.save(path=filename, format="json", dump=batch.dump())

    def get_last_batch(self, step_name: str) -> Union[_Batch, None]:
        """Gets the last batch received from a step.

        Args:
            step_name: The name of the step.

        Returns:
            The last batch received from the step or `None` if no batch was received.
        """
        return self._last_batch_received.get(step_name)

    def add_batch(
        self,
        to_step: str,
        batch: _Batch,
        prepend: bool = False,
    ) -> None:
        """Add an output batch from `batch.step_name` to `to_step`.

        Args:
            to_step: The name of the step that will process the batch.
            batch: The output batch of an step to be processed by `to_step`.
            prepend: If `True`, the content of the batch will be added at the start of
                the buffer.

        Raises:
            ValueError: If `to_step` is not found in the batch manager.
        """
        if to_step not in self._steps:
            raise ValueError(f"Step '{to_step}' not found in the batch manager.")
        step = self._steps[to_step]
        step.add_batch(batch, prepend)

    def add_batch_to_recover_offline_batch_generation(
        self, to_step: str, data: List[List[Dict[str, Any]]]
    ) -> None:
        """Add a batch to recover pipeline execution from an `_Step` that used an `LLM`
        with offline batch generation. It will add the batch to the start of the buffer
        of the step  and set the last batch received of the step to `None`.

        Args:
            to_step: The name of the step that will process the batch.
            data: The data that was used with the offline batch generation.
        """
        self.add_batch(
            to_step=to_step,
            batch=_Batch(seq_no=0, step_name=to_step, last_batch=True, data=data),
            prepend=True,
        )
        self._last_batch_received[to_step] = None

    def get_batch(self, step_name: str) -> Union[_Batch, None]:
        """Get the next batch to be processed by the step.

        Args:
            step_name: The name of the step that will process the batch.

        Returns:
            A `_Batch` instance if there is a batch to be processed by the step. Otherwise,
            `None`.
        """
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' not found in the batch manager.")

        return self._steps[step_name].get_batch()

    def step_empty_buffers(self, step_name: str) -> List[str]:
        """Checks if the input buffer for a step is empty.

        Args:
            step_name: The name of the step.

        Returns:
            The name of the previous steps for which the input buffer for this step is
            empty.
        """
        return self._steps[step_name].empty_buffers()

    def set_last_batch_sent(self, batch: "_Batch") -> None:
        """Set the last batch sent to a step.

        Args:
            batch: The last batch sent to a step.
        """
        self._last_batch_sent[batch.step_name] = batch

    def get_last_batch_sent(self, step_name: str) -> Union["_Batch", None]:
        """Get the last batch sent to a step.

        Args:
            step_name: The name of the step.

        Returns:
            The last batch sent to a step or `None` if no batch was sent.
        """
        return self._last_batch_sent.get(step_name, None)

    def set_last_batch_flag_sent_to(self, step_name: str) -> None:
        """Set the flag to indicate that the last batch was sent to a step.

        Args:
            step_name: The name of the step.
        """
        self._last_batch_flag_sent_to.append(step_name)

    def set_next_expected_seq_no(
        self, step_name: str, from_step: str, next_expected_seq_no: int
    ) -> None:
        """Sets the next expected sequence number of a `_Batch` received by `step` coming
        from `from_step`.

        Args:
            step_name: The step name whose next expected sequence number for `from_step`
                has to be updated.
            from_step: The name of the step from which its next expected sequence number
                in step has to be updated.
            next_expected_seq_no: the next expected sequence number of a `_Batch` coming
                from `from_step`.
        """
        self._steps[step_name].set_next_expected_seq_no(from_step, next_expected_seq_no)

    def step_has_finished(self, step_name: str) -> bool:
        """Indicates if the step has finished by checking if it sent a batch with `last_batch==True`
        or it was sent the `LAST_BATCH_SENT_FLAG`.

        Args:
            step_name: the name of the step to be checked.

        Returns:
            `True` if step has finished generating batches, `False` otherwise.
        """
        return step_name in self._last_batch_flag_sent_to or (
            self._last_batch_received[step_name] is not None
            and self._last_batch_received[step_name].last_batch  # type: ignore
        )

    @classmethod
    def from_dag(  # noqa: C901
        cls, dag: "DAG", use_cache: bool = False, steps_data_path: Optional[Path] = None
    ) -> "_BatchManager":
        """Create a `_BatchManager` instance from a `DAG` instance.

        Args:
            dag: The `DAG` instance.
            use_cache: whether or not to try loading outputs from steps of previous pipelines
                executions. Defaults to `False`.
            steps_data_path: The path where the outputs of each `Step` (considering its
                signature) will be saved for later reuse in another pipelines executions.

        Returns:
            A `_BatchManager` instance.
        """
        steps = {}
        last_batch_received = {}
        last_batch_sent = {}
        last_batch_flag_sent_to = []

        load_batches = {}
        steps_to_load_data_from_previous_executions: Dict[str, Union[Path, None]] = {}
        for step_name in dag:
            step: "_Step" = dag.get_step(step_name)[STEP_ATTR_NAME]
            last_batch_received[step.name] = None
            last_batch_sent[step.name] = None
            predecessors = list(dag.get_step_predecessors(step_name))
            convergence_step = all(
                dag.get_step(predecessor).get(RECEIVES_ROUTED_BATCHES_ATTR_NAME, False)
                for predecessor in predecessors
            )
            batch_manager_step = _BatchManagerStep.from_step(
                step=step,
                predecessors=predecessors,
                convergence_step=convergence_step,
            )

            all_step_precessors_use_cache = all(
                dag.get_step(step_name)[STEP_ATTR_NAME].use_cache
                for step_name in predecessors
            )
            if use_cache and step.use_cache and all_step_precessors_use_cache:
                step_data_path = steps_data_path / batch_manager_step.signature
                if step_data_path.exists():
                    steps_to_load_data_from_previous_executions[step_name] = (
                        step_data_path
                    )
                    # We only want to load the outputs that are directly needed by the added
                    # steps, so if we need to load the outputs of one step and one of its
                    # predecessors it's in the list, then we remove it.
                    for predecessor in predecessors:
                        if predecessor in steps_to_load_data_from_previous_executions:
                            steps_to_load_data_from_previous_executions[predecessor] = (
                                None
                            )

            steps[step_name] = batch_manager_step

        for (
            step_name,
            step_outputs_path,
        ) in steps_to_load_data_from_previous_executions.items():
            last_batch_flag_sent_to.append(step_name)
            if step_outputs_path is None:
                continue
            load_batches[step_name] = sorted(
                [
                    _Batch.from_json(batch_file)
                    for batch_file in step_outputs_path.glob("*.json")
                    if batch_file.is_file() and batch_file.suffix == ".json"
                ],
                key=lambda x: x.seq_no,
            )
            last_batch_received[step_name] = load_batches[step_name][-1]

        # Load batches from previous steps in batch manager steps
        for step_name, batch_manager_step in steps.items():
            for predecessor in dag.get_step_predecessors(step_name):
                if predecessor in load_batches:
                    batch_manager_step.data[predecessor] = deepcopy(
                        load_batches[predecessor]
                    )
                    batch_manager_step.last_batch_received.append(predecessor)

        return cls(steps, last_batch_received, last_batch_sent, last_batch_flag_sent_to)

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManager` to a dictionary.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the `_BatchManager`.
        """
        return {
            "steps": {name: step.dump(**kwargs) for name, step in self._steps.items()},
            "last_batch_received": {
                step_name: batch.dump(**kwargs) if batch is not None else None
                for step_name, batch in self._last_batch_received.items()
            },
            "last_batch_sent": {
                step_name: batch.dump(**kwargs) if batch is not None else None
                for step_name, batch in self._last_batch_sent.items()
            },
            "last_batch_flag_sent_to": self._last_batch_flag_sent_to,
        }

    def cache(self, path: Path, steps_data_path: Path) -> None:  # noqa: C901
        """Cache the `_BatchManager` to a file.

        Args:
            path: The path to the file where the `_BatchManager` will be cached. If `None`,
                then the `_BatchManager` will be cached in the default cache folder.
            steps_data_path: The path where the outputs of each `Step` (considering its
                signature) will be saved for later reuse in another pipelines executions.
        """

        def save_batch(
            batches_dir: Path, batch_dump: Dict[str, Any], batch_list: List[_Batch]
        ) -> Path:
            seq_no = batch_dump["seq_no"]
            data_hash = batch_dump["data_hash"]
            batch_file = batches_dir / f"batch_{seq_no}_{data_hash}.json"

            # Save the batch if it doesn't exist
            if not batch_file.exists():
                # Get the data of the batch before saving it
                batch = next(batch for batch in batch_list if batch.seq_no == seq_no)
                batch_dump["data"] = batch.data
                self.save(path=batch_file, format="json", dump=batch_dump)

            return batch_file

        def remove_files(keep_files: List[str], dir: Path) -> None:
            files = list_files_in_dir(dir, key=None)
            remove = set(files) - {Path(file) for file in keep_files}
            for file in remove:
                file.unlink()

        path = Path(path)

        # Do not include `_Batch` data so `dump` is fast
        dump = self.dump(include_batch_data=False)
        batch_manager_step_files = {}

        # Do this to avoid modifying the dictionary while iterating over it
        batch_manager_steps = set(dump["steps"].keys())
        for step_name in batch_manager_steps:
            step_dump = dump["steps"].pop(step_name)

            # Create a directory for each batch manager step to store their batches
            batch_manager_step_dir = path.parent / "batch_manager_steps" / step_name
            batch_manager_step_dir.mkdir(parents=True, exist_ok=True)

            # Store each built `_Batch` in a separate file
            built_batches_dir = batch_manager_step_dir / "built_batches"
            built_batches_dir.mkdir(parents=True, exist_ok=True)
            step_dump["built_batches"] = [
                str(
                    save_batch(
                        batches_dir=built_batches_dir,
                        batch_dump=batch_dump,
                        batch_list=self._steps[step_name].built_batches,
                    )
                )
                for batch_dump in step_dump["built_batches"]
            ]
            # Remove built `_Batch`es that were consumed from cache
            remove_files(step_dump["built_batches"], built_batches_dir)

            # Store the `_BatchManagerStep` info
            batch_manager_step_file = str(
                path.parent / f"batch_manager_steps/{step_name}/batch_manager_step.json"
            )
            self.save(path=batch_manager_step_file, format="json", dump=step_dump)

            # Store the path to the `_BatchManagerStep` file
            batch_manager_step_files[step_name] = batch_manager_step_file

        dump["steps"] = batch_manager_step_files
        self.save(path=path, format="json", dump=dump)

    @classmethod
    def load_from_cache(
        cls, dag: "DAG", batch_manager_path: "StrOrPath", steps_data_path: "StrOrPath"
    ) -> "_BatchManager":
        """Loads the `_BatchManager` from a cache file.

        Args:
            path: The path to the cache file.
        """
        _check_is_dir(batch_manager_path)
        content = read_json(batch_manager_path)

        # Read each `_BatchManagerStep` from file
        steps = {}
        for step_name, step_file in content["steps"].items():
            steps[step_name] = read_json(step_file)

            # When reading back from JSON, `next_expected_seq_no` is a list (because JSON
            # files do not have tuples).
            steps[step_name]["next_expected_seq_no"] = {
                k: tuple(v) for k, v in steps[step_name]["next_expected_seq_no"].items()
            }

            # TODO: where are we writing built batches now? xD
            # Read each `_Batch` from file
            steps[step_name]["built_batches"] = [
                read_json(batch) for batch in steps[step_name]["built_batches"]
            ]

            # Read the batches from the `steps_data` directory to populate back the `_BatchManagerStep`
            step_offset = steps[step_name]["step_offset"]
            for successor_step_name, offset in step_offset.items():
                batch_offset, batch_row_offset = offset
                step: "_Step" = dag.get_step(successor_step_name)[STEP_ATTR_NAME]
                successor_step_data_path = (
                    steps_data_path / f"{step.name}_{step.signature}"
                )

                # read batches from successor step from the step data directory taking into
                # account offset
                batches = []
                for batch_file in successor_step_data_path.glob("*.json"):
                    if not batch_file.is_file() or batch_file.suffix != ".json":
                        continue

                    # If the batch number is lower than the batch offset then we should
                    # skip it as it has already been processed by the step
                    batch_no = int(batch_file.stem.split("batch_")[1])
                    if batch_no < batch_offset:
                        continue

                    # read the batch and skip the first N rows of the first batch
                    batch = read_json(batch_file)
                    if batch_no == batch_offset:
                        batch["data"][0] = batch["data"][0][batch_row_offset:]

                    batches.append(batch)

                # sort batches by `seq_no` as it's a requirement for checking if ready to
                # create next batch
                batches.sort(key=lambda batch: batch["seq_no"])
                steps[step_name]["data"][successor_step_name] = batches

        content["steps"] = steps
        return cls.from_dict(content)

    def invalidate_cache_for(
        self, step_name: str, dag: "DAG", steps_data_path: Path
    ) -> None:
        """Invalidate the step that changed and dependent ones.

        Once we have loaded the `_BatchManager` from cache in the pipeline,
        if there are steps that changed (the signature is different), we will
        make the step and successors invalid, meaning the `_BatchManagerStep`
        will be regenerated, and the batches of the first step that was invalidated
        will be loaded from where we left.

        Args:
            step_name: Name of the `_BatchManagerStep` that changed.
        """
        # Loop over the steps sorted, so we can invalidate successor steps
        invalidate_after = False
        for sorted_step in dag:
            if (sorted_step == step_name) or invalidate_after:
                predecessors = list(dag.get_step_predecessors(sorted_step))
                convergence_step = dag.is_convergence_step(step_name)
                step = dag.get_step(sorted_step)[STEP_ATTR_NAME]
                self._steps[sorted_step] = _BatchManagerStep.from_step(
                    step, predecessors=predecessors, convergence_step=convergence_step
                )

                invalidate_after = True
                self._last_batch_received[sorted_step] = None
                self._last_batch_sent[sorted_step] = None
                if sorted_step in self._last_batch_flag_sent_to:
                    self._last_batch_flag_sent_to.remove(sorted_step)

        # Loop on the predecessors to load their batches
        for predecessor in dag.get_step_predecessors(step_name):
            step_predecessor = dag.get_step(predecessor)[STEP_ATTR_NAME]
            predecessor_step_data_path = (
                steps_data_path
                / f"{step_predecessor.name}_{step_predecessor.signature}"
            )
            batch_files = list_files_in_dir(
                predecessor_step_data_path, key=lambda x: int(x.stem.split("_")[-1])
            )
            for file in batch_files:
                batch = _Batch.from_file(file)
                if batch.last_batch:
                    self._steps[step_name].last_batch_received.append(batch.step_name)
                self._steps[step_name].data[predecessor].append(batch)

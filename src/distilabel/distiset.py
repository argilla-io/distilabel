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

import json
import logging
import os.path as posixpath
import re
import sys
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

import fsspec
import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.filesystems import is_remote_filesystem
from huggingface_hub import DatasetCardData, HfApi, upload_file, upload_folder
from huggingface_hub.file_download import hf_hub_download
from pyarrow.lib import ArrowInvalid
from typing_extensions import Self

from distilabel.constants import (
    DISTISET_ARTIFACTS_FOLDER,
    DISTISET_CONFIG_FOLDER,
    PIPELINE_CONFIG_FILENAME,
    PIPELINE_LOG_FILENAME,
    STEP_ATTR_NAME,
    STEPS_ARTIFACTS_PATH,
    STEPS_OUTPUTS_PATH,
)
from distilabel.utils.card.dataset_card import (
    DistilabelDatasetCard,
    size_categories_parser,
)
from distilabel.utils.docstring import get_bibtex, parse_google_docstring
from distilabel.utils.files import list_files_in_dir
from distilabel.utils.huggingface import get_hf_token

if TYPE_CHECKING:
    from distilabel.pipeline._dag import DAG


class Distiset(dict):
    """Convenient wrapper around `datasets.Dataset` to push to the Hugging Face Hub.

    It's a dictionary where the keys correspond to the different leaf_steps from the internal
    `DAG` and the values are `datasets.Dataset`.

    Attributes:
        _pipeline_path: Optional path to the `pipeline.yaml` file that generated the dataset.
            Defaults to `None`.
        _artifacts_path: Optional path to the directory containing the generated artifacts
            by the pipeline steps. Defaults to `None`.
        _log_filename_path: Optional path to the `pipeline.log` file that generated was written
            by the pipeline. Defaults to `None`.
        _citations: Optional list containing citations that will be included in the dataset
            card. Defaults to `None`.
    """

    _pipeline_path: Optional[Path] = None
    _artifacts_path: Optional[Path] = None
    _log_filename_path: Optional[Path] = None
    _citations: Optional[List[str]] = None

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        generate_card: bool = True,
        include_script: bool = False,
        **kwargs: Any,
    ) -> None:
        """Pushes the `Distiset` to the Hugging Face Hub, each dataset will be pushed as a different configuration
        corresponding to the leaf step that generated it.

        Args:
            repo_id:
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.
            private:
                Whether the dataset repository should be set to private or not. Only affects repository creation:
                a repository that already exists will not be affected by that parameter.
            token:
                An optional authentication token for the Hugging Face Hub. If no token is passed, will default
                to the token saved locally when logging in with `huggingface-cli login`. Will raise an error
                if no token is passed and the user is not logged-in.
            generate_card:
                Whether to generate a dataset card or not. Defaults to True.
            include_script:
                Whether you want to push the pipeline script to the hugging face hub to share it.
                If set to True, the name of the script that was run to create the distiset will be
                automatically determined, and that will be the name of the file uploaded to your
                repository. Take into account, this operation only makes sense for a distiset obtained
                from calling `Pipeline.run()` method. Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to the `push_to_hub` method of the `datasets.Dataset` object.

        Raises:
            ValueError: If no token is provided and couldn't be retrieved automatically.
        """
        script_filename = sys.argv[0]
        filename_py = (
            script_filename.split("/")[-1]
            if "/" in script_filename
            else script_filename
        )
        script_path = Path.cwd() / script_filename

        if token is None:
            token = get_hf_token(self.__class__.__name__, "token")

        for name, dataset in self.items():
            dataset.push_to_hub(
                repo_id=repo_id,
                config_name=name,
                private=private,
                token=token,
                **kwargs,
            )

        if self.artifacts_path:
            upload_folder(
                repo_id=repo_id,
                folder_path=self.artifacts_path,
                path_in_repo="artifacts",
                token=token,
                repo_type="dataset",
                commit_message="Include pipeline artifacts",
            )

        if include_script and script_path.exists():
            upload_file(
                path_or_fileobj=script_path,
                path_in_repo=filename_py,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Include pipeline script",
            )

        if generate_card:
            self._generate_card(
                repo_id, token, include_script=include_script, filename_py=filename_py
            )

    def _get_card(
        self,
        repo_id: str,
        token: Optional[str] = None,
        include_script: bool = False,
        filename_py: Optional[str] = None,
    ) -> DistilabelDatasetCard:
        """Generates the dataset card for the `Distiset`.

        Note:
            If `repo_id` and `token` are provided, it will extract the metadata from the README.md file
            on the hub.

        Args:
            repo_id: Name of the repository to push to, or the path for the distiset if saved to disk.
            token: The token to authenticate with the Hugging Face Hub.
                We assume that if it's provided, the dataset will be in the Hugging Face Hub,
                so the README metadata will be extracted from there.
            include_script: Whether to upload the script to the hugging face repository.
            filename_py: The name of the script. If `include_script` is True, the script will
                be uploaded to the repository using this name, otherwise it won't be used.

        Returns:
            The dataset card for the `Distiset`.
        """
        sample_records = {}
        for name, dataset in self.items():
            record = (
                dataset[0] if not isinstance(dataset, dict) else dataset["train"][0]
            )
            from PIL import ImageFile

            for key, value in record.items():
                # If the value is an image, we set it to an empty string to avoid the `README.md` to huge
                if isinstance(value, ImageFile.ImageFile):
                    value = ""
                # If list is too big, the `README.md` generated will be huge so we truncate it
                elif isinstance(value, list):
                    length = len(value)
                    if length < 10:
                        continue
                    record[key] = value[:10]
                    record[key].append(
                        f"... (truncated - showing 10 of {length} elements)"
                    )
            sample_records[name] = record

        readme_metadata = {}
        if repo_id and token:
            readme_metadata = self._extract_readme_metadata(repo_id, token)

        metadata = {
            **readme_metadata,
            "size_categories": size_categories_parser(
                max(len(dataset) for dataset in self.values())
            ),
            "tags": ["synthetic", "distilabel", "rlaif"],
        }

        card = DistilabelDatasetCard.from_template(
            card_data=DatasetCardData(**metadata),
            repo_id=repo_id,
            sample_records=sample_records,
            include_script=include_script,
            filename_py=filename_py,
            artifacts=self._get_artifacts_metadata(),
            references=self.citations,
        )

        return card

    def _get_artifacts_metadata(self) -> Dict[str, List[Dict[str, Any]]]:
        """Gets a dictionary with the metadata of the artifacts generated by the pipeline steps.

        Returns:
            A dictionary in which the key is the name of the step and the value is a list
            of dictionaries, each of them containing the name and metadata of the step artifact.
        """
        if not self.artifacts_path:
            return {}

        def iterdir_ignore_hidden(path: Path) -> Generator[Path, None, None]:
            return (f for f in Path(path).iterdir() if not f.name.startswith("."))

        artifacts_metadata = defaultdict(list)
        for step_artifacts_dir in iterdir_ignore_hidden(self.artifacts_path):
            step_name = step_artifacts_dir.stem
            for artifact_dir in iterdir_ignore_hidden(step_artifacts_dir):
                artifact_name = artifact_dir.stem
                metadata_path = artifact_dir / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                artifacts_metadata[step_name].append(
                    {"name": artifact_name, "metadata": metadata}
                )

        return dict(artifacts_metadata)

    def _extract_readme_metadata(
        self, repo_id: str, token: Optional[str]
    ) -> Dict[str, Any]:
        """Extracts the metadata from the README.md file of the dataset repository.

        We have to download the previous README.md file in the repo, extract the metadata from it,
        and generate a dict again to be passed thorough the `DatasetCardData` object.

        Args:
            repo_id: The ID of the repository to push to, from the `push_to_hub` method.
            token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.

        Returns:
            The metadata extracted from the README.md file of the dataset repository as a dict.
        """
        import requests

        try:
            readme_path = Path(
                hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token)
            )
        except requests.exceptions.HTTPError:
            # This can fail when using the checkpoint step
            return {}
        # Remove the '---' from the metadata
        metadata = re.findall(r"---\n(.*?)\n---", readme_path.read_text(), re.DOTALL)[0]
        metadata = yaml.safe_load(metadata)
        return metadata

    def _generate_card(
        self,
        repo_id: str,
        token: str,
        include_script: bool = False,
        filename_py: Optional[str] = None,
    ) -> None:
        """Generates a dataset card and pushes it to the Hugging Face Hub, and
        if the `pipeline.yaml` path is available in the `Distiset`, uploads that
        to the same repository.

        Args:
            repo_id: The ID of the repository to push to, from the `push_to_hub` method.
            token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
            include_script: Whether to upload the script to the hugging face repository.
            filename_py: The name of the script. If `include_script` is True, the script will
                be uploaded to the repository using this name, otherwise it won't be used.
        """
        card = self._get_card(
            repo_id=repo_id,
            token=token,
            include_script=include_script,
            filename_py=filename_py,
        )

        card.push_to_hub(
            repo_id,
            repo_type="dataset",
            token=token,
        )

        if self.pipeline_path:
            # If the pipeline.yaml is available, upload it to the Hugging Face Hub as well.
            HfApi().upload_file(
                path_or_fileobj=self.pipeline_path,
                path_in_repo=PIPELINE_CONFIG_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

        if self.log_filename_path:
            # The same we had with "pipeline.yaml" but with the log file.
            HfApi().upload_file(
                path_or_fileobj=self.log_filename_path,
                path_in_repo=PIPELINE_LOG_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    def train_test_split(
        self,
        train_size: float,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Self:
        """Return a `Distiset` whose values will be a `datasets.DatasetDict` with two random train and test subsets.
        Splits are created from the dataset according to `train_size` and `shuffle`.

        Args:
            train_size:
                Float between `0.0` and `1.0` representing the proportion of the dataset to include in the test split.
                It will be applied to all the datasets in the `Distiset`.
            shuffle: Whether or not to shuffle the data before splitting
            seed:
                A seed to initialize the default BitGenerator, passed to the underlying method.

        Returns:
            The `Distiset` with the train-test split applied to all the datasets.
        """
        assert 0 < train_size < 1, "train_size must be a float between 0 and 1"
        for name, dataset in self.items():
            self[name] = dataset.train_test_split(
                train_size=train_size,
                shuffle=shuffle,
                seed=seed,
            )
        return self

    def save_to_disk(
        self,
        distiset_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        save_card: bool = True,
        save_pipeline_config: bool = True,
        save_pipeline_log: bool = True,
    ) -> None:
        r"""
        Saves a `Distiset` to a dataset directory, or in a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

        In case you want to save the `Distiset` in a remote filesystem, you can pass the `storage_options` parameter
        as you would do with `datasets`'s `Dataset.save_to_disk` method: [see example](https://huggingface.co/docs/datasets/filesystems#saving-serialized-datasets)

        Args:
            distiset_path: Path where you want to save the `Distiset`. It can be a local path
                (e.g. `dataset/train`) or remote URI (e.g. `s3://my-bucket/dataset/train`)
            max_shard_size: The maximum size of the dataset shards to be uploaded to the hub.
                If expressed as a string, needs to be digits followed by a unit (like `"50MB"`).
                Defaults to `None`.
            num_shards: Number of shards to write. By default the number of shards depends on
                `max_shard_size` and `num_proc`. Defaults to `None`.
            num_proc: Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default. Defaults to `None`.
            storage_options: Key/value pairs to be passed on to the file-system backend, if any.
                Defaults to `None`.
            save_card: Whether to save the dataset card. Defaults to `True`.
            save_pipeline_config: Whether to save the pipeline configuration file (aka the `pipeline.yaml` file).
                Defaults to `True`.
            save_pipeline_log: Whether to save the pipeline log file (aka the `pipeline.log` file).
                Defaults to `True`.

        Examples:
            ```python
            # Save your distiset in a local folder:
            distiset.save_to_disk(distiset_path="my-distiset")
            # Save your distiset in a remote storage:
            storage_options = {
                "key": os.environ["S3_ACCESS_KEY"],
                "secret": os.environ["S3_SECRET_KEY"],
                "client_kwargs": {
                    "endpoint_url": os.environ["S3_ENDPOINT_URL"],
                    "region_name": os.environ["S3_REGION"],
                },
            }
            distiset.save_to_disk(distiset_path="my-distiset", storage_options=storage_options)
            ```
        """
        distiset_path = str(distiset_path)
        for name, dataset in self.items():
            dataset.save_to_disk(
                f"{distiset_path}/{name}",
                max_shard_size=max_shard_size,
                num_shards=num_shards,
                num_proc=num_proc,
                storage_options=storage_options,
            )

        distiset_config_folder = posixpath.join(distiset_path, DISTISET_CONFIG_FOLDER)

        fs: fsspec.AbstractFileSystem
        fs, _, _ = fsspec.get_fs_token_paths(
            distiset_config_folder, storage_options=storage_options
        )
        fs.makedirs(distiset_config_folder, exist_ok=True)

        if self.artifacts_path:
            distiset_artifacts_folder = posixpath.join(
                distiset_path, DISTISET_ARTIFACTS_FOLDER
            )
            fs.copy(str(self.artifacts_path), distiset_artifacts_folder, recursive=True)

        if save_card:
            # NOTE: Currently the card is not the same if we write to disk or push to the HF hub,
            # as we aren't generating the README copying/updating the data from the dataset repo.
            card = self._get_card(repo_id=Path(distiset_path).stem, token=None)
            new_filename = posixpath.join(distiset_config_folder, "README.md")
            if storage_options:
                # Write the card the same way as DatasetCard.save does:
                with fs.open(new_filename, "w", newline="", encoding="utf-8") as f:
                    f.write(str(card))
            else:
                card.save(new_filename)

        # Write our internal files to the distiset folder by copying them to the distiset folder.
        if save_pipeline_config and self.pipeline_path:
            new_filename = posixpath.join(
                distiset_config_folder, PIPELINE_CONFIG_FILENAME
            )
            if self.pipeline_path.exists() and (not fs.isfile(new_filename)):
                data = yaml.safe_load(self.pipeline_path.read_text())
                with fs.open(new_filename, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False)

        if save_pipeline_log and self.log_filename_path:
            new_filename = posixpath.join(distiset_config_folder, PIPELINE_LOG_FILENAME)
            if self.log_filename_path.exists() and (not fs.isfile(new_filename)):
                data = self.log_filename_path.read_text()
                with fs.open(new_filename, "w", encoding="utf-8") as f:
                    f.write(data)

    @classmethod
    def load_from_disk(
        cls,
        distiset_path: PathLike,
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        download_dir: Optional[PathLike] = None,
    ) -> Self:
        """Loads a dataset that was previously saved using `Distiset.save_to_disk` from a dataset
        directory, or from a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

        Args:
            distiset_path: Path ("dataset/train") or remote URI ("s3://bucket/dataset/train").
            keep_in_memory: Whether to copy the dataset in-memory, see `datasets.Dataset.load_from_disk``
                for more information. Defaults to `None`.
            storage_options: Key/value pairs to be passed on to the file-system backend, if any.
                Defaults to `None`.
            download_dir: Optional directory to download the dataset to. Defaults to None,
                in which case it will create a temporary directory.

        Returns:
            A `Distiset` loaded from disk, it should be a `Distiset` object created using `Distiset.save_to_disk`.
        """
        original_distiset_path = str(distiset_path)

        fs: fsspec.AbstractFileSystem
        fs, _, [distiset_path] = fsspec.get_fs_token_paths(  # type: ignore
            original_distiset_path, storage_options=storage_options
        )
        dest_distiset_path = distiset_path

        assert fs.isdir(
            original_distiset_path
        ), "`distiset_path` must be a `PathLike` object pointing to a folder or a URI of a remote filesystem."

        has_config = False
        has_artifacts = False
        distiset = cls()

        if is_remote_filesystem(fs):
            src_dataset_path = distiset_path
            if download_dir:
                dest_distiset_path = download_dir
            else:
                dest_distiset_path = Dataset._build_local_temp_path(src_dataset_path)  # type: ignore
            fs.download(src_dataset_path, dest_distiset_path.as_posix(), recursive=True)  # type: ignore

        # Now we should have the distiset locally, so we can read those files
        for folder in Path(dest_distiset_path).iterdir():
            if folder.stem == DISTISET_CONFIG_FOLDER:
                has_config = True
                continue
            elif folder.stem == DISTISET_ARTIFACTS_FOLDER:
                has_artifacts = True
                continue
            distiset[folder.stem] = load_from_disk(
                str(folder),
                keep_in_memory=keep_in_memory,
            )

        # From the config folder we just need to point to the files. Once downloaded we set the path to point to point to the files. Once downloaded we set the path
        # to wherever they are.
        if has_config:
            distiset_config_folder = posixpath.join(
                dest_distiset_path, DISTISET_CONFIG_FOLDER
            )

            pipeline_path = posixpath.join(
                distiset_config_folder, PIPELINE_CONFIG_FILENAME
            )
            if Path(pipeline_path).exists():
                distiset.pipeline_path = Path(pipeline_path)

            log_filename_path = posixpath.join(
                distiset_config_folder, PIPELINE_LOG_FILENAME
            )
            if Path(log_filename_path).exists():
                distiset.log_filename_path = Path(log_filename_path)

        if has_artifacts:
            distiset.artifacts_path = Path(
                posixpath.join(dest_distiset_path, DISTISET_ARTIFACTS_FOLDER)
            )

        return distiset

    @property
    def pipeline_path(self) -> Union[Path, None]:
        """Returns the path to the `pipeline.yaml` file that generated the `Pipeline`."""
        return self._pipeline_path

    @pipeline_path.setter
    def pipeline_path(self, path: PathLike) -> None:
        self._pipeline_path = Path(path)

    @property
    def artifacts_path(self) -> Union[Path, None]:
        """Returns the path to the directory containing the artifacts generated by the steps
        of the pipeline."""
        return self._artifacts_path

    @artifacts_path.setter
    def artifacts_path(self, path: PathLike) -> None:
        self._artifacts_path = Path(path)

    @property
    def log_filename_path(self) -> Union[Path, None]:
        """Returns the path to the `pipeline.log` file that generated the `Pipeline`."""
        return self._log_filename_path

    @log_filename_path.setter
    def log_filename_path(self, path: PathLike) -> None:
        self._log_filename_path = Path(path)

    @property
    def citations(self) -> Union[List[str], None]:
        """Bibtex references to be included in the README."""
        return self._citations

    @citations.setter
    def citations(self, citations_: List[str]) -> None:
        self._citations = sorted(set(citations_))

    def __repr__(self):
        # Copy from `datasets.DatasetDict.__repr__`.
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, count=0, flags=re.M)
        return f"Distiset({{\n{repr}\n}})"

    def transform_columns_to_image(self, columns: Union[str, list[str]]) -> Self:
        """Transforms the columns of the dataset to `PIL.Image` objects.

        Args:
            columns: Column or list of columns to transform.

        Returns:
            Transforms the columns of the dataset to `PIL.Image` objects before pushing,
            so the Hub treats them as Image objects and can be rendered in the dataset
            viewer, and cast them to be automatically transformed when downloading
            the dataset back.
        """
        from datasets import Image

        from distilabel.models.image_generation.utils import image_from_str

        columns = [columns] if isinstance(columns, str) else columns

        def cast_to_image(row: dict) -> dict:
            for column in columns:
                row[column] = image_from_str(row[column])
            return row

        for name, dataset in self.items():
            # In case train_test_split was called
            if isinstance(dataset, DatasetDict):
                for split, dataset_split in dataset.items():
                    dataset_split = dataset_split.map(cast_to_image)
                    for column in columns:
                        if column in dataset_split.column_names:
                            dataset_split = dataset_split.cast_column(
                                column, Image(decode=True)
                            )
                    self[name][split] = dataset_split
            else:
                dataset = dataset.map(cast_to_image)

                for column in columns:
                    if column in dataset.column_names:
                        dataset = dataset.cast_column(column, Image(decode=True))

                self[name] = dataset

        return self


def create_distiset(  # noqa: C901
    data_dir: Path,
    pipeline_path: Optional[Path] = None,
    log_filename_path: Optional[Path] = None,
    enable_metadata: bool = False,
    dag: Optional["DAG"] = None,
) -> Distiset:
    """Creates a `Distiset` from the buffer folder.

    This function is intended to be used as a helper to create a `Distiset` from from the folder
    where the cached data was written by the `_WriteBuffer`.

    Args:
        data_dir: Folder where the data buffers were written by the `_WriteBuffer`.
            It should correspond to `CacheLocation.data`.
        pipeline_path: Optional path to the pipeline.yaml file that generated the dataset.
            Internally this will be passed to the `Distiset` object on creation to allow
            uploading the `pipeline.yaml` file to the repo upon `Distiset.push_to_hub`.
        log_filename_path: Optional path to the pipeline.log file that was generated during the pipeline run.
            Internally this will be passed to the `Distiset` object on creation to allow
            uploading the `pipeline.log` file to the repo upon `Distiset.push_to_hub`.
        enable_metadata: Whether to include the distilabel metadata column in the dataset or not.
            Defaults to `False`.
        dag: DAG contained in a `Pipeline`. If informed, will be used to extract the references/
            citations from it.

    Returns:
        The dataset created from the buffer folder, where the different leaf steps will
        correspond to different configurations of the dataset.

    Examples:
        ```python
        from pathlib import Path
        distiset = create_distiset(Path.home() / ".cache/distilabel/pipelines/path-to-pipe-hashname")
        ```
    """
    from distilabel.constants import DISTILABEL_METADATA_KEY

    logger = logging.getLogger("distilabel.distiset")

    steps_outputs_dir = data_dir / STEPS_OUTPUTS_PATH

    distiset = Distiset()
    for file in steps_outputs_dir.iterdir():
        if file.is_file():
            continue

        files = [str(file) for file in list_files_in_dir(file)]
        if files:
            try:
                ds = load_dataset(
                    "parquet", name=file.stem, data_files={"train": files}
                )
                if not enable_metadata and DISTILABEL_METADATA_KEY in ds.column_names:
                    ds = ds.remove_columns(DISTILABEL_METADATA_KEY)
                distiset[file.stem] = ds
            except ArrowInvalid:
                logger.warning(f"❌ Failed to load the subset from '{file}' directory.")
                continue
        else:
            logger.warning(
                f"No output files for step '{file.stem}', can't create a dataset."
                " Did the step produce any data?"
            )

    # If there's only one dataset i.e. one config, then set the config name to `default`
    if len(distiset.keys()) == 1:
        distiset["default"] = distiset.pop(list(distiset.keys())[0])

    # If there's any artifact set the `artifacts_path` so they can be uploaded
    steps_artifacts_dir = data_dir / STEPS_ARTIFACTS_PATH
    if any(steps_artifacts_dir.rglob("*")):
        distiset.artifacts_path = steps_artifacts_dir

    # Include `pipeline.yaml` if exists
    if pipeline_path:
        distiset.pipeline_path = pipeline_path
    else:
        # If the pipeline path is not provided, try to find it in the parent directory
        # and assume that's the wanted file.
        pipeline_path = steps_outputs_dir.parent / "pipeline.yaml"
        if pipeline_path.exists():
            distiset.pipeline_path = pipeline_path

    # Include `pipeline.log` if exists
    if log_filename_path:
        distiset.log_filename_path = log_filename_path
    else:
        log_filename_path = steps_outputs_dir.parent / "pipeline.log"
        if log_filename_path.exists():
            distiset.log_filename_path = log_filename_path

    if dag:
        distiset._citations = _grab_citations(dag)

    return distiset


def _grab_citations(dag: "DAG") -> List[str]:
    """Extracts the citations from the steps that form the DAG.

    Args:
        dag: `DAG` contained in the pipeline that created the `Distiset`.

    Returns:
        List of citations to add to the `Distiset`.
    """
    citations = []
    for step_name in dag:
        step_info = parse_google_docstring(dag.get_step(step_name)[STEP_ATTR_NAME])
        if cites := step_info["citations"]:
            citations.extend(cites)
            continue
        # If there were no citations but we have references with arxiv URLs, try to extract
        # the bixtex citations from those
        if references := step_info["references"]:
            bibtex_refs = []
            for ref in references.values():
                try:
                    bibtex_refs.append(get_bibtex(ref))
                except ValueError:
                    # No need to inform in this case, it's noise
                    pass
                except AttributeError as e:
                    print(
                        f"Couldn't obtain the bibtex format for the ref: '{ref}', error: {e}"
                    )
                except Exception as e:
                    print(f"Untracked error: {e}")
            citations.extend(bibtex_refs)
    return citations

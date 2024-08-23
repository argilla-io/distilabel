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

from pathlib import Path
from typing import Final

# Steps related constants
DISTILABEL_METADATA_KEY: Final[str] = "distilabel_metadata"

# Pipeline cache
BASE_CACHE_DIR = Path.home() / ".cache" / "distilabel" / "pipelines"

# Pipeline dag related constants
STEP_ATTR_NAME: Final[str] = "step"
INPUT_QUEUE_ATTR_NAME: Final[str] = "input_queue"
RECEIVES_ROUTED_BATCHES_ATTR_NAME: Final[str] = "receives_routed_batches"
ROUTING_BATCH_FUNCTION_ATTR_NAME: Final[str] = "routing_batch_function"
CONVERGENCE_STEP_ATTR_NAME: Final[str] = "convergence_step"
LAST_BATCH_SENT_FLAG: Final[str] = "last_batch_sent"

# Data paths constants
STEPS_OUTPUTS_PATH = "steps_outputs"
STEPS_ARTIFACTS_PATH = "steps_artifacts"

# Distiset related constants
DISTISET_CONFIG_FOLDER: Final[str] = "distiset_configs"
DISTISET_ARTIFACTS_FOLDER: Final[str] = "artifacts"
PIPELINE_CONFIG_FILENAME: Final[str] = "pipeline.yaml"
PIPELINE_LOG_FILENAME: Final[str] = "pipeline.log"

# Docs page for the custom errors
DISTILABEL_DOCS_URL: Final[str] = "https://distilabel.argilla.io/latest/"


__all__ = [
    "STEP_ATTR_NAME",
    "INPUT_QUEUE_ATTR_NAME",
    "RECEIVES_ROUTED_BATCHES_ATTR_NAME",
    "ROUTING_BATCH_FUNCTION_ATTR_NAME",
    "CONVERGENCE_STEP_ATTR_NAME",
    "LAST_BATCH_SENT_FLAG",
    "DISTILABEL_METADATA_KEY",
]

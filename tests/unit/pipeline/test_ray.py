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

from typing import Generator

import pytest

from distilabel.llms.vllm import vLLM
from distilabel.pipeline.ray import RayPipeline
from distilabel.steps.base import StepResources
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.utils.serialization import TYPE_INFO_KEY


@pytest.fixture
def ray_test_cluster() -> Generator[None, None, None]:
    import ray
    from ray.cluster_utils import Cluster

    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_gpus": 8,
        },
    )
    for _ in range(3):
        cluster.add_node(num_gpus=8)

    ray.init(address=cluster.address)

    yield

    ray.shutdown()


@pytest.mark.skip_python_versions(["3.12"])
@pytest.mark.usefixtures("ray_test_cluster")
class TestRayPipeline:
    def test_dump(self) -> None:
        pipeline = RayPipeline(name="unit-test")
        dump = pipeline.dump()

        assert dump["pipeline"][TYPE_INFO_KEY] == {
            "module": "distilabel.pipeline.local",
            "name": "Pipeline",
        }

    def test_get_ray_gpus_per_node(self) -> None:
        pipeline = RayPipeline(name="unit-test")
        pipeline._init_ray()
        assert list(pipeline._ray_node_ids.values()) == [8, 8, 8, 8]

    def test_create_vllm_placement_group(self) -> None:
        with RayPipeline(name="unit-test") as pipeline:
            step_1 = TextGeneration(
                llm=vLLM(
                    model="invented",
                    extra_kwargs={"tensor_parallel_size": 2},
                ),
                resources=StepResources(gpus=8),
            )
            step_2 = TextGeneration(
                llm=vLLM(
                    model="invented",
                    extra_kwargs={
                        "tensor_parallel_size": 8,
                    },
                ),
                resources=StepResources(gpus=8),
            )
            step_3 = TextGeneration(
                llm=vLLM(
                    model="invented",
                    extra_kwargs={
                        "tensor_parallel_size": 2,
                    },
                ),
                resources=StepResources(gpus=8),
            )
            step_4 = TextGeneration(
                llm=vLLM(
                    model="invented",
                    extra_kwargs={
                        "tensor_parallel_size": 4,
                    },
                ),
                resources=StepResources(gpus=8),
            )
            step_5 = TextGeneration(
                llm=vLLM(
                    model="invented",
                    extra_kwargs={
                        "tensor_parallel_size": 2,
                    },
                ),
                resources=StepResources(gpus=8),
            )

        pipeline._init_ray()
        num_gpus = sum(pipeline._ray_node_ids.values())

        allocated_gpus = 2
        pipeline._create_vllm_placement_group(step_1)
        assert sum(pipeline._ray_node_ids.values()) == num_gpus - allocated_gpus

        allocated_gpus += 8
        pipeline._create_vllm_placement_group(step_2)
        assert sum(pipeline._ray_node_ids.values()) == num_gpus - allocated_gpus

        allocated_gpus += 2
        pipeline._create_vllm_placement_group(step_3)
        assert sum(pipeline._ray_node_ids.values()) == num_gpus - allocated_gpus

        allocated_gpus += 4
        pipeline._create_vllm_placement_group(step_4)
        assert sum(pipeline._ray_node_ids.values()) == num_gpus - allocated_gpus

        allocated_gpus += 2
        pipeline._create_vllm_placement_group(step_5)
        assert sum(pipeline._ray_node_ids.values()) == num_gpus - allocated_gpus

    def test_create_vllm_placement_group_raise_valueerror(self) -> None:
        with RayPipeline(name="unit-test") as pipeline:
            step = TextGeneration(
                llm=vLLM(
                    model="invented",
                    extra_kwargs={
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 100,
                    },
                ),
                resources=StepResources(gpus=8),
            )

        with pytest.raises(
            ValueError, match="Ray cluster does not allocate enough GPUs"
        ):
            pipeline._create_vllm_placement_group(step)

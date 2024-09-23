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


from distilabel.steps.clustering.dbscan import DBSCAN


class TestDBSCAN:
    def test_process(self) -> None:
        step = DBSCAN(n_jobs=1, eps=0.5, min_samples=5)
        step.load()

        results = next(
            step.process(
                inputs=[
                    {"projection": [0.1, -0.4]},
                    {"projection": [-0.3, 0.9]},
                    {"projection": [0.6, 0.2]},
                    {"projection": [-0.2, -0.6]},
                    {"projection": [0.9, 0.1]},
                    {"projection": [0.4, -0.7]},
                    {"projection": [-0.5, 0.3]},
                    {"projection": [0.7, 0.5]},
                    {"projection": [-0.1, -0.9]},
                ]
            )
        )
        assert all(result["cluster_label"] == -1 for result in results)

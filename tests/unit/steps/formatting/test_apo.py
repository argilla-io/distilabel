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

from distilabel.steps.formatting.apo import FormatAPO


class TestFormatAPO:
    def test_process(self) -> None:
        step = FormatAPO()
        step.load()

        assert next(
            step.process(
                [
                    {
                        "prompt": "How many gaps are there between the earth and the moon?",
                        "response": """There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.""",
                        "revision": """There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.""",
                    }
                ]
            )
        ) == [
            {
                "prompt": "How many gaps are there between the earth and the moon?",
                "response": 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.',
                "revision": 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
                "prompt_id": "d5e8924f2856fe7180c0aef3ec186f7a421b2ba11551b9ebfffeb7638ec5b021",
                "chosen": [
                    {
                        "role": "user",
                        "content": "How many gaps are there between the earth and the moon?",
                    },
                    {
                        "role": "assistant",
                        "content": 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "How many gaps are there between the earth and the moon?",
                    },
                    {
                        "role": "assistant",
                        "content": 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.',
                    },
                ],
            }
        ]

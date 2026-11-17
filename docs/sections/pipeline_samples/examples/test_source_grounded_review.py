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
#
#

"""Run with pytest; live retrieval is deliberately opt-in via --live in the sample."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
import source_grounded_review as sample

from distilabel.models import LLM

CLAIMS = {
    "supported": "The pipeline saves intermediate step outputs.",
    "contradicted": "The pipeline never saves intermediate step outputs.",
    "not_established": "The pipeline cache expires after exactly seven days.",
    "mismatch": "The pipeline saves intermediate step outputs.",
}


class FixtureLLM(LLM):
    role: str

    @property
    def model_name(self):
        return "local-fixture"

    def generate(self, inputs, num_generations=1):
        results = []
        for messages in inputs:
            payload = json.loads(messages[-1]["content"])
            assert payload["source_excerpt"] == sample.SOURCE_EXCERPT
            assert payload["source_url"] == sample.SOURCE_URL
            if self.role == "generate":
                case = payload["question"]
                output = {
                    "claim": CLAIMS[case],
                    "citation_url": "https://example.com/unrelated"
                    if case == "mismatch"
                    else sample.SOURCE_URL,
                }
            else:
                assert messages[0]["content"] == sample.RUBRIC
                judgments = {
                    CLAIMS["supported"]: (
                        "supported",
                        "The excerpt explicitly says outputs are saved.",
                    ),
                    CLAIMS["contradicted"]: (
                        "contradicted",
                        "The excerpt says outputs are saved, conflicting with never.",
                    ),
                    CLAIMS["not_established"]: (
                        "not_established",
                        "The excerpt says nothing about an expiry period.",
                    ),
                }
                judgment, explanation = judgments[payload["claim"]]
                output = {"judgment": judgment, "explanation": explanation}
            results.append(
                {
                    "generations": [json.dumps(output)] * num_generations,
                    "statistics": {
                        "input_tokens": [0] * num_generations,
                        "output_tokens": [0] * num_generations,
                    },
                }
            )
        return results


def test_native_pipeline_preserves_evidence_and_judgments(tmp_path, monkeypatch):
    # No network calls are permitted in the saved-evidence application check.
    import socket

    original_connect = socket.socket.connect

    def no_network(connection, address):
        if connection.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Unexpected network access")
        return original_connect(connection, address)

    monkeypatch.setattr(socket.socket, "connect", no_network)
    rows = [{**sample.saved_rows()[0], "question": case} for case in CLAIMS]
    pipeline = sample.build_pipeline(
        FixtureLLM(role="generate"),
        FixtureLLM(role="judge"),
        rows=rows,
        cache_dir=tmp_path,
    )
    result = pipeline.run(use_cache=False)["default"]["train"]
    assert len(result) == 4
    for row in result:
        case = row["question"]
        assert row["source_url"] == sample.SOURCE_URL
        assert row["source_excerpt"] == sample.SOURCE_EXCERPT
        assert row["claim"] == CLAIMS[case]
        assert row["retrieval_mode"] == "saved"
        assert row["judgment"] == ("not_established" if case == "mismatch" else case)
        assert row["citation_matches"] == (case != "mismatch")
        assert row["explanation"]


@pytest.mark.parametrize(
    "failure", [RuntimeError("late retrieval failure"), ValueError("empty result")]
)
def test_late_retrieval_failure_yields_no_rows_or_citations(
    tmp_path, monkeypatch, failure
):
    search = AsyncMock(
        side_effect=[
            {
                "source_url": "https://distilabel.argilla.io/latest/",
                "source_excerpt": "first result",
            },
            failure,
        ]
    )
    monkeypatch.setattr(sample, "search_documentation", search)
    rows = [{"question": "first"}, {"question": "second"}]
    step = sample.RetrieveDocumentation(name="retrieve", pipeline=None)
    output = step.process(rows)
    with pytest.raises(type(failure), match=str(failure)):
        next(output)
    assert rows == [{"question": "first"}, {"question": "second"}]
    assert search.await_count == 2
    search.reset_mock(side_effect=True)
    search.side_effect = [
        {
            "source_url": "https://distilabel.argilla.io/latest/",
            "source_excerpt": "first result",
        },
        failure,
    ]
    pipeline = sample.build_pipeline(
        sample.QuoteDemoLLM(role="generate"),
        sample.QuoteDemoLLM(role="judge"),
        rows=rows,
        live=True,
        cache_dir=tmp_path,
    )
    assert not pipeline.run(use_cache=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"results": []},
        {"results": [{"url": "https://distilabel.argilla.io/latest/", "excerpts": []}]},
        {"results": [{"url": "https://example.com/", "excerpts": ["unrelated"]}]},
    ],
)
def test_empty_or_unusable_results_stop(payload):
    with pytest.raises(ValueError, match="nonempty excerpt"):
        sample.select_evidence(payload)


def test_excerpt_bound():
    evidence = sample.select_evidence(
        {
            "results": [
                {
                    "url": "https://distilabel.argilla.io/latest/",
                    "excerpts": ["x" * 5000],
                }
            ]
        }
    )
    assert evidence["source_excerpt"] == "x" * sample.MAX_EXCERPT_CHARS


def test_mcp_sdk_transport_and_live_step_in_native_pipeline(tmp_path, monkeypatch):
    # Exercise the actual MCP SDK over HTTP test doubles, including HTTP headers.
    # This is application evidence, not an anonymous live-service test.
    pytest.importorskip("mcp")
    requests = []
    evidence_url = (
        "https://distilabel.argilla.io/latest/sections/how_to_guides/advanced/caching/"
    )
    excerpt = "Saved intermediate outputs can be reused."

    def handle(request):
        requests.append(request)
        assert request.headers["user-agent"] == sample.USER_AGENT
        assert "authorization" not in request.headers
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(200)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            response = {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif message["method"] == "tools/list":
            response = {
                "tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]
            }
        else:
            assert message["method"] == "tools/call"
            assert message["params"]["name"] == "web_search"
            assert (
                "site:distilabel.argilla.io"
                in message["params"]["arguments"]["search_queries"][0]
            )
            response = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"results": [{"url": evidence_url, "excerpts": [excerpt]}]}
                        ),
                    }
                ]
            }
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": response}
        )

    original_client = httpx.AsyncClient

    def client(**kwargs):
        return original_client(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    # Check transport in this process, so the actual requests are observable.
    evidence = asyncio.run(sample.search_documentation("What does the cache save?"))
    assert evidence == {"source_url": evidence_url, "source_excerpt": excerpt}
    assert any(
        b"web_search" in request.content
        for request in requests
        if request.method == "POST"
    )
    pipeline = sample.build_pipeline(
        sample.QuoteDemoLLM(role="generate"),
        sample.QuoteDemoLLM(role="judge"),
        live=True,
        cache_dir=tmp_path,
    )
    result = pipeline.run(use_cache=False)["default"]["train"]
    assert len(result) == 1
    assert result[0]["source_url"] == evidence_url
    assert result[0]["source_excerpt"] == excerpt
    assert result[0]["retrieval_mode"] == "live"
    assert result[0]["citation_url"] == evidence_url
    assert result[0]["judgment"] == "supported"


def test_missing_citation_cannot_be_supported():
    task = sample.CitationSupport(llm=sample.QuoteDemoLLM(role="judge"), pipeline=None)
    output = task.format_output(
        '{"judgment": "supported", "explanation": "A judge can be wrong."}',
        input={
            "answer": json.dumps({"claim": "The pipeline saves intermediate outputs."}),
            "source_url": sample.SOURCE_URL,
            "source_excerpt": sample.SOURCE_EXCERPT,
        },
    )
    assert output["judgment"] == "not_established"
    assert output["citation_matches"] is False

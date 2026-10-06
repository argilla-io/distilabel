#!/bin/bash

set -euo pipefail

VLLM_REPOSITORY="vllm-project/vllm"
VLLM_VERSION="${VLLM_VERSION:-}"

if [[ -n "${VLLM_VERSION}" ]]; then
    VLLM_TAG="v${VLLM_VERSION#v}"
    RELEASE_URL="https://api.github.com/repos/${VLLM_REPOSITORY}/releases/tags/${VLLM_TAG}"
    echo "Installing vLLM CPU wheel for requested release '${VLLM_TAG}'..."
else
    RELEASE_URL="https://api.github.com/repos/${VLLM_REPOSITORY}/releases/latest"
    echo "Resolving the latest stable vLLM CPU release..."
fi

CURL_ARGS=(--fail --silent --show-error --location --retry 3)
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    CURL_ARGS+=(--header "Authorization: Bearer ${GITHUB_TOKEN}")
fi
RELEASE_JSON=$(curl "${CURL_ARGS[@]}" "${RELEASE_URL}")
VLLM_TAG=$(printf '%s' "${RELEASE_JSON}" | python -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')
ARCHITECTURE=$(uname -m)

VLLM_WHEEL_URL=$(printf '%s' "${RELEASE_JSON}" | python -c '
import json
import re
import sys

release = json.load(sys.stdin)
architecture = sys.argv[1]
pattern = re.compile(
    rf"[+]cpu-cp38-abi3-manylinux_[^/]+_{re.escape(architecture)}[.]whl$"
)
matching_assets = [
    asset for asset in release.get("assets", []) if pattern.search(asset["name"])
]
if len(matching_assets) != 1:
    names = [asset["name"] for asset in release.get("assets", [])]
    raise SystemExit(
        f"Expected exactly one CPU wheel for {architecture}, found "
        f"{len(matching_assets)} in release {release.get('tag_name')}: {names}"
    )
print(matching_assets[0]["browser_download_url"])
' "${ARCHITECTURE}")

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
fi

echo "Installing ${VLLM_TAG} CPU wheel for ${ARCHITECTURE}:"
echo "${VLLM_WHEEL_URL}"
uv pip install --system "${VLLM_WHEEL_URL}" --torch-backend cpu

python -c 'import vllm; print(f"Installed vLLM {vllm.__version__}")'

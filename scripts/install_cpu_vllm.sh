#!/bin/bash

echo "Cloning 'vllm-project/vllm' GitHub repository..."
git clone https://github.com/vllm-project/vllm.git

cd vllm || exit

git fetch --tags

latest_tag=$(git describe --tags "$(git rev-list --tags --max-count=1)")

echo "Checking out to '$latest_tag' tag..."
git checkout "$latest_tag"

echo "Installing vLLM CPU requirements..."
uv pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

echo "Installing vLLM for CPU..."
VLLM_TARGET_DEVICE=cpu python setup.py install
echo "Installed!"

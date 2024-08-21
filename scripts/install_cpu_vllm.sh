#!/bin/bash

echo "Installing system build dependencies..."
apt-get update -y
apt-get install -y gcc-12 g++-12 libnuma-dev
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

echo "Installing Python build dependencies..."
uv pip install --system wheel packaging ninja "setuptools>=49.4.0" numpy

echo "Cloning 'vllm-project/vllm' GitHub repository..."
git clone https://github.com/vllm-project/vllm.git

cd vllm || exit

git fetch --tags

latest_tag=$(git describe --tags "$(git rev-list --tags --max-count=1)")

echo "Checking out to '$latest_tag' tag..."
git checkout "$latest_tag"

echo "Installing vLLM CPU requirements..."
uv pip install --system -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

echo "Installing vLLM for CPU..."
VLLM_TARGET_DEVICE=cpu python setup.py install
echo "Installed!"

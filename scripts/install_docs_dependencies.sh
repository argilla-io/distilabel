#!/bin/bash

set -e

python_version=$(python -c "import sys; print(sys.version_info[:2])")

python -m pip install --break-system-packages uv

uv pip install --system -e ".[docs]"

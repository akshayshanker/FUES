#!/bin/bash
# Create a virtual environment mimicking public installation
# Installs dynx from GitHub instead of local dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default venv location
VENV_PATH="${1:-$REPO_ROOT/.venv_public}"

echo "Creating public venv at: $VENV_PATH"

# Remove existing venv if present
if [[ -d "$VENV_PATH" ]]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_PATH"
fi

# Create new venv
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install dependencies from pyproject.toml (without editable install of this package)
pip install numpy numba matplotlib pyyaml

# Install dynx from GitHub (public version)
pip install "dynx @ git+https://github.com/akshayshanker/dynx.git"

# Install consav
pip install consav

echo ""
echo "Public venv created at: $VENV_PATH"
echo "Activate with: source $VENV_PATH/bin/activate"

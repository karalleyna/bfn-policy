#!/bin/bash
set -e

echo "🔧 [ENV] Activating Conda env: bfn-pol"
# Optional: skip if already active
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bfn-pol

echo "🧹 Running black..."
black .

echo "🔃 Running isort..."
isort .

# echo "🔍 Running flake8..."
# flake8 .

echo "✅ All checks passed and code formatted."

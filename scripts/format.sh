#!/bin/bash

# Format all Python files with black
echo "Formatting Python files with black..."
uv run black .
echo "Done!"

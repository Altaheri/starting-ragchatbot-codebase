#!/bin/bash

# Check formatting without making changes
echo "Checking Python formatting with black..."
uv run black --check .
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "All files are properly formatted!"
else
    echo "Some files need formatting. Run './scripts/format.sh' to fix."
fi

exit $exit_code

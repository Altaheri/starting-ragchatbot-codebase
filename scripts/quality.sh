#!/bin/bash

# Run all quality checks
echo "=========================================="
echo "Running Code Quality Checks"
echo "=========================================="
echo ""

# Track overall status
overall_status=0

# Check formatting
echo "1. Checking code formatting (black)..."
echo "------------------------------------------"
uv run black --check .
if [ $? -ne 0 ]; then
    echo "FAILED: Code formatting issues found"
    overall_status=1
else
    echo "PASSED: Code formatting"
fi
echo ""

# Run tests
echo "2. Running tests (pytest)..."
echo "------------------------------------------"
cd backend && uv run pytest
if [ $? -ne 0 ]; then
    echo "FAILED: Tests"
    overall_status=1
else
    echo "PASSED: Tests"
fi
cd ..
echo ""

# Summary
echo "=========================================="
if [ $overall_status -eq 0 ]; then
    echo "All quality checks passed!"
else
    echo "Some quality checks failed!"
fi
echo "=========================================="

exit $overall_status

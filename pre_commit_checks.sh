#!/bin/bash

# Pre-commit checks script for aind-hcr-data-loader
# Run this before submitting a pull request

set -e  # Exit on any error

echo "🔍 Running pre-commit checks..."
echo "=================================="

# Change to the script's directory
cd "$(dirname "$0")"

echo "📁 Working directory: $(pwd)"
echo ""

# 1. Run black (auto-format code)
echo "🎨 Running black to format code..."
black .
echo "✅ Black formatting complete"
echo ""

# 2. Run isort (sort imports)
echo "📦 Running isort to sort imports..."
isort .
echo "✅ Import sorting complete"
echo ""

# 3. Run flake8 (linting)
echo "🔍 Running flake8 for linting..."
if flake8 .; then
    echo "✅ Flake8 checks passed"
else
    echo "❌ Flake8 checks failed"
    exit 1
fi
echo ""

# 4. Run interrogate (documentation coverage)
echo "📚 Running interrogate for documentation coverage..."
if interrogate .; then
    echo "✅ Documentation coverage checks passed"
else
    echo "❌ Documentation coverage checks failed"
    exit 1
fi
echo ""

# 5. Run tests with coverage
echo "🧪 Running tests with coverage..."
if coverage run -m unittest discover && coverage report; then
    echo "✅ Tests and coverage checks passed"
else
    echo "❌ Tests or coverage checks failed"
    exit 1
fi
echo ""

echo "🎉 All pre-commit checks passed!"
echo "Your code is ready for pull request submission."

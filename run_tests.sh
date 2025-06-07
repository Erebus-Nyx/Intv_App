#!/bin/bash
# filepath: /home/nyx/intv/run_tests.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Interview Application Test Suite ===${NC}"
echo ""

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Virtual environment not found${NC}"
    exit 1
fi

# Check if pytest is installed
if ! .venv/bin/python -c "import pytest" &> /dev/null; then
    echo -e "${RED}✗ pytest not found. Installing...${NC}"
    .venv/bin/pip install pytest pytest-cov pytest-timeout
fi

echo ""
echo -e "${BLUE}=== Running Test Suite ===${NC}"
echo ""

# Default test arguments
PYTEST_ARGS="-v --tb=short --color=yes"

# Parse command line arguments
case "${1:-all}" in
    "unit")
        echo -e "${YELLOW}Running unit tests only...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS -m unit"
        ;;
    "integration")
        echo -e "${YELLOW}Running integration tests only...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS -m integration"
        ;;
    "fast")
        echo -e "${YELLOW}Running fast tests only (excluding slow tests)...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
        ;;
    "coverage")
        echo -e "${YELLOW}Running tests with coverage report...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS --cov=intv --cov-report=html --cov-report=term-missing"
        ;;
    "audio")
        echo -e "${YELLOW}Running audio tests only...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS -m audio"
        ;;
    "llm")
        echo -e "${YELLOW}Running LLM tests only...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS -m llm"
        ;;
    "cli")
        echo -e "${YELLOW}Running CLI tests only...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS -m cli"
        ;;
    "all")
        echo -e "${YELLOW}Running all tests...${NC}"
        PYTEST_ARGS="$PYTEST_ARGS --cov=intv --cov-report=term-missing"
        ;;
    *)
        echo -e "${RED}Invalid test category: $1${NC}"
        echo "Available options: unit, integration, fast, coverage, audio, llm, cli, all"
        exit 1
        ;;
esac

echo ""

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run the tests
if .venv/bin/python -m pytest $PYTEST_ARGS tests/test_*.py; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    
    # Show coverage report location if it was generated
    if [[ "$PYTEST_ARGS" == *"--cov-report=html"* ]]; then
        echo -e "${BLUE}Coverage report generated: htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== Test Summary ===${NC}"
echo "To run specific test categories:"
echo "  ./run_tests.sh unit         - Run only unit tests"
echo "  ./run_tests.sh integration  - Run only integration tests"
echo "  ./run_tests.sh fast         - Run fast tests (exclude slow ones)"
echo "  ./run_tests.sh coverage     - Run tests with coverage report"
echo "  ./run_tests.sh audio        - Run audio processing tests"
echo "  ./run_tests.sh llm          - Run language model tests"
echo "  ./run_tests.sh cli          - Run CLI tests"
echo "  ./run_tests.sh all          - Run all tests (default)"
echo ""

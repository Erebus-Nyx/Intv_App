#!/bin/bash
# INTV Local Development Installation Script
# This script installs INTV from the current local directory (for development)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the INTV directory
if [ ! -f "pyproject.toml" ] || [ ! -f "install.py" ]; then
    print_error "Must be run from the INTV project root directory!"
    print_info "Expected files: pyproject.toml, install.py"
    exit 1
fi

print_header "INTV Local Development Installation"

# Parse command line arguments
INSTALL_MODE="development"
PYTHON_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            INSTALL_MODE="production"
            print_info "Production installation mode (will try PyPI/GitHub)"
            shift
            ;;
        --editable)
            PYTHON_ARGS="$PYTHON_ARGS --editable"
            print_info "Editable installation mode"
            shift
            ;;
        --dry-run)
            PYTHON_ARGS="$PYTHON_ARGS --dry-run"
            print_info "Dry run mode enabled"
            shift
            ;;
        --cpu-only)
            PYTHON_ARGS="$PYTHON_ARGS --cpu-only"
            print_info "CPU-only installation mode"
            shift
            ;;
        --skip-native)
            PYTHON_ARGS="$PYTHON_ARGS --skip-native"
            print_info "Skipping native dependency installation"
            shift
            ;;
        --help|-h)
            echo "INTV Local Development Installation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --production          Try to install from PyPI/GitHub (not local)"
            echo "  --editable            Install in editable mode (development)"
            echo "  --dry-run             Show what would be done"
            echo "  --cpu-only            Force CPU-only installation"
            echo "  --skip-native         Skip native dependency installation"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Default: Local development installation from current directory"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    print_error "Python 3.10 or later is required (found: $PYTHON_VERSION)"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Check if pipx is available
if ! command -v pipx &> /dev/null; then
    print_info "Installing pipx..."
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    print_warning "You may need to restart your terminal for pipx to be available"
fi

if [ "$INSTALL_MODE" = "development" ]; then
    print_info "Installing INTV from local directory..."
    
    # Check if already installed and offer to uninstall
    if pipx list | grep -q "intv"; then
        print_warning "INTV is already installed via pipx"
        read -p "Uninstall existing version? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Uninstalling existing INTV..."
            pipx uninstall intv || true
        fi
    fi
    
    # Install from local directory
    print_info "Installing from current directory..."
    if [[ "$PYTHON_ARGS" == *"--editable"* ]]; then
        pipx install -e .
    else
        pipx install .
    fi
    
    print_success "Local INTV installation complete!"
    
else
    # Production mode - use the existing install.py script
    print_info "Running production installation script..."
    python3 install.py $PYTHON_ARGS
fi

# Verify installation
print_info "Verifying installation..."
if intv --version; then
    print_success "INTV is working correctly!"
    echo ""
    print_info "Development workflow:"
    echo "  1. Make changes to the code"
    echo "  2. Test: intv --help"
    echo "  3. For editable installs, changes are live"
    echo "  4. For regular installs, re-run this script to update"
else
    print_error "Installation verification failed"
    exit 1
fi
